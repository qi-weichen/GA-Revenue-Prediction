import pandas as pd
import numpy as np
from datetime import date, timedelta
import json
import os
import re
import gc
import math
import pickle
import xgboost as xgb
from ast import literal_eval
from pandas.io.json import json_normalize
from sklearn.metrics import mean_squared_error as mse
from dateutil.relativedelta import relativedelta

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
gc.enable()


# read data ex-hits
def read_ex_hits(n='train'):
    JSON_COLUMNS = ['totals', 'device', 'geoNetwork', 'trafficSource']
    t = pd.read_csv(
        './data/{}_v2.csv'.format(n),
        dtype={'fullVisitorId': str, 'visitStarTime': int},
        converters={column: json.loads for column in JSON_COLUMNS},
        usecols=[
            'fullVisitorId', 'visitStartTime', 'visitNumber', 'device', 'channelGrouping',
            'geoNetwork', 'totals', 'trafficSource'
        ],

    )

    for col in JSON_COLUMNS:
        column_as_df = json_normalize(t[col])
        column_as_df.columns = ['{}_{}'.format(col, c.replace('.', '_')) for c in column_as_df.columns]
        t = pd.concat([t.drop(col, axis=1), column_as_df], axis=1)

    return t


# "hits" column is too big to read in one time into memory. As a result, the file
# is separted into 1e5 chunksize and concatenate together later
def read_hit(n='train'):
    train_hit = pd.read_csv(
        './data/{}_v2.csv'.format(n),
        usecols=['fullVisitorId', 'visitStartTime', 'hits', 'customDimensions'],
        dtype={'fullVisitorId': 'str'},
        chunksize=1e5,
    )

    for i, df in enumerate(train_hit):
        df.loc[df['hits'] == '[]', 'hits'] = '[{}]'
        df.loc[df['customDimensions'] == '[]', 'customDimensions'] = '[{}]'

        df['hits'] = df['hits'].apply(literal_eval).str[0]
        df['customDimensions'] = df['customDimensions'].apply(literal_eval).str[0]

        c_df = json_normalize(df['customDimensions'])
        c_df.columns = ['customDimensions_{}'.format(c) for c in c_df.columns]

        hits_df = json_normalize(df['hits'])
        hits_df.columns = ['hits_{}'.format(c) for c in hits_df.columns]

        df = df.drop(['hits', 'customDimensions'], axis=1).reset_index(drop=True).join(c_df).join(hits_df)
        df.drop([
            'hits_customDimensions', 'hits_customMetrics', 'hits_customVariables',
            'hits_experiment', 'hits_publisher_infos'
        ], axis=1, inplace=True)

        print("chunk {} null num: {}".format(i + 1, df.isnull().sum().sum()))
        print("chunk {} shape: {}".format(i + 1, df.shape))
        df.to_csv('./data/{}_hit_{}.csv'.format(n, i + 1), index=False)
        print("save chunk {}\n".format(i + 1))
        del df

        gc.collect()


# concatenate all df containing "hits"
def get_hit_df(train_ex_hits, n='train'):
    train_hits_origin = pd.DataFrame()

    if n == 'train':
        max_file_num = 19
    else:
        max_file_num = 6

    for i in range(1, max_file_num):
        train_hits_origin = pd.concat([
            train_hits_origin,
            pd.read_csv(
                './data/{}_hit_{}.csv'.format(n, i),
                dtype={'fullVisitorId': str},
            )
        ], axis=0, sort=True, ignore_index=True)

    train_hits_origin = train_hits_origin.loc[~train_hits_origin.fullVisitorId.isnull(), :]
    train_hits_origin['visitStartTime'] = train_hits_origin['visitStartTime'].astype(int)
    train_hits_origin = train_hits_origin.merge(
        train_ex_hits.loc[:, ['fullVisitorId', 'visitStartTime', 'totals_transactionRevenue']],
        on=['fullVisitorId', 'visitStartTime'], how='left'
    )

    return train_hits_origin


def split_train_test(df, train_start_date=date(2016, 8, 2), val_months_gap=4):
    df['totals_transactionRevenue'] = df['totals_transactionRevenue'].astype(float)
    df['visitTime'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['totals_transactionRevenue'] = df['totals_transactionRevenue'].fillna(0)

    def _get_train(train_start, set_n='train'):
        train_x_start = train_start
        train_x_end = train_x_start + relativedelta(months=5) + timedelta(days=15)
        train_y_start = train_x_start + relativedelta(months=7)
        train_y_end = train_x_start + relativedelta(months=9)

        print("{} x period: {}  ~  {}".format(set_n, train_x_start, train_x_end))
        print("{} y period: {}  ~  {}".format(set_n, train_y_start, train_y_end))

        y_period_user_rev = df.loc[df['visitTime'].dt.date.between(train_y_start, train_y_end), :].groupby(
            'fullVisitorId', as_index=False
        ).agg({'totals_transactionRevenue': 'sum'}).rename(columns={'totals_transactionRevenue': 'label'})
        y_period_user_rev['label'] = np.log1p(y_period_user_rev['label'])

        train_x_y = df.loc[df['visitTime'].dt.date.between(train_x_start, train_x_end), :].merge(
            y_period_user_rev, on='fullVisitorId', how='left'
        )
        train_x_y['label'] = train_x_y['label'].fillna(0)

        return train_x_y

    val_start_date = train_start_date + relativedelta(months=val_months_gap)
    test_start_date = date(2017, 5, 1)

    train_x_y, val_x_y, test_x_y = _get_train(train_start_date), _get_train(val_start_date, 'val'), _get_train(
        test_start_date, 'test')

    return train_x_y, val_x_y, test_x_y


# Gen Features
def prep_data(input_df):
    df = input_df.copy()

    dtype_map = dict(
        totals_bounces=float,
        totals_hits=float,
        totals_newVisits=float,
        totals_pageviews=float,
        totals_timeOnSite=float,
        totals_transactions=float,
        trafficSource_isTrueDirect=float,
        trafficSource_adwordsClickInfo_page=float,
        trafficSource_adwordsClickInfo_isVideoAd=float,
        device_isMobile=float,
    )
    df = df.astype(dtype_map)

    df['totals_transactionRevenue'] = df['totals_transactionRevenue'].astype(float)
    df['totals_transactionRevenue'] = df['totals_transactionRevenue'].fillna(0)
    df['totals_bounces'].fillna(0, inplace=True)
    df['totals_newVisits'].fillna(0, inplace=True)
    df['trafficSource_isTrueDirect'].fillna(0, inplace=True)
    df['trafficSource_adwordsClickInfo_isVideoAd'].fillna(1, inplace=True)

    df['visitTime'] = pd.to_datetime(df['visitStartTime'], unit='s')

    na_re = re.compile(
        '.*not set.*|.*not available.*|.*not provided.*|.*none.*'
    )
    bracket_re = re.compile(r'\(|\)')

    na_str = 'not_available'
    bracket_str = ''

    for c in df.select_dtypes('object').columns:
        df[c] = df[c].str.strip()
        df[c] = df[c].str.lower()
        df[c] = df[c].str.replace(na_re, na_str)
        df[c] = df[c].str.replace(bracket_re, bracket_str)
        df.loc[df[c].isnull(), c] = na_str
    return df


def prepare_features(train_window, is_train=True):
    num_cols = [
        'visitNumber',
        'visitTime',
        'totals_bounces',
        'totals_hits',
        'totals_newVisits',
        'totals_pageviews',
        'totals_transactionRevenue',
        'totals_timeOnSite',
        'totals_transactions',
        'trafficSource_adwordsClickInfo_page',
        'trafficSource_isTrueDirect',
        'trafficSource_adwordsClickInfo_isVideoAd',
    ]

    if is_train:
        train_num = train_window.loc[:, ['fullVisitorId', 'label'] + num_cols].copy()
    else:
        train_num = train_window.loc[:, ['fullVisitorId'] + num_cols].copy()

    train_num['totals_pageviews'].fillna(train_num['totals_pageviews'].mean(), inplace=True)

    train_num = train_num.sort_values(['fullVisitorId', 'visitTime'])
    train_num['first_session_time'] = train_num.groupby('fullVisitorId')['visitTime'].transform('first')
    train_num['hours_diff_last_session'] = (
                                               train_num['visitTime'] - train_num['visitTime'].shift(
                                                   1)).dt.total_seconds() / 3600

    train_num.loc[
        train_num['fullVisitorId'] != train_num['fullVisitorId'].shift(1), 'hours_diff_last_session'
    ] = 0

    train_num['days_since_last_session'] = train_num['hours_diff_last_session'] / 24

    train_num['hits_prev_ratio'] = train_num['totals_hits'] / train_num['totals_hits'].shift(1)
    train_num['pageviews_prev_ratio'] = train_num['totals_pageviews'] / train_num['totals_pageviews'].shift(1)

    train_num.loc[
        train_num['fullVisitorId'] != train_num['fullVisitorId'].shift(1), ['hits_prev_ratio', 'pageviews_prev_ratio']
    ] = 0

    train_num['hits_per_page'] = train_num['totals_hits'] / train_num['totals_pageviews']
    train_num['last_session_time'] = train_num.groupby('fullVisitorId')['visitTime'].transform('last')
    train_num['days_since_pred'] = (
        train_num['visitTime'].max().date() + timedelta(days=15) + relativedelta(months=1) -
        train_num['visitTime'].dt.date).dt.days

    train_num['totals_bounces'] = train_num['totals_bounces'].fillna(0)
    train_num['totals_newVisits'] = train_num['totals_newVisits'].fillna(0)
    train_num['trafficSource_adwordsClickInfo_page'].fillna(0, inplace=True)
    train_num['avg_time_per_page'] = train_num['totals_timeOnSite'] / train_num['totals_pageviews']

    # group
    def has_rev_ratio(se):
        return se[se > 0].shape[0] / se.shape[0]

    def gap_mean(se):
        if len(se) == 1:
            return math.nan
        else:
            return se.sum() / (len(se) - 1)

    def gap_std(se):
        if len(se) == 1:
            return math.nan
        else:
            return np.std(se[1:])

    def mean(se):
        se = se[~se.isnull()]
        if len(se) > 0:
            return np.mean(se)
        else:
            return math.nan

    def std(se):
        se = se[~se.isnull()]
        if len(se) > 1:
            return np.std(se)
        else:
            return math.nan

    def namax(se):
        se = se[~se.isnull()]
        if len(se) > 0:
            return np.max(se)
        else:
            return math.nan

    if is_train:
        user_features = train_num.groupby('fullVisitorId').agg({
            'visitTime': 'count',
            'visitNumber': 'max',
            'totals_bounces': 'mean',  # bounce ratio
            'totals_transactions': [namax, mean],
            'totals_hits': ['max', 'mean'],
            'totals_newVisits': ['max', 'mean'],  # whether contain new visits
            'trafficSource_isTrueDirect': 'mean',
            'trafficSource_adwordsClickInfo_isVideoAd': 'mean',
            'totals_pageviews': ['max', 'mean', 'std'],
            'totals_timeOnSite': [mean, namax, std],
            'avg_time_per_page': [mean, namax, std],
            'totals_transactionRevenue': ['max', has_rev_ratio, 'mean'],
            'trafficSource_adwordsClickInfo_page': [namax, mean, std],
            'days_since_last_session': ['max', gap_mean, gap_std],
            'hits_prev_ratio': ['max', gap_mean, gap_std],
            'pageviews_prev_ratio': ['max', gap_mean, gap_std],
            'hits_per_page': ['max', 'mean', 'std'],
            'days_since_pred': ['min', 'max'],
            'label': 'first'
        })

        user_features.columns = [c[0] + '_' + c[1] for c in user_features.columns.tolist()]

        user_features.rename(columns={
            'totals_bounces_mean': 'bounce_ratio',
            'visitTime_count': 'total_visits',
            'totals_newVisits_max': 'include_first_visit',
            'label_first': 'label'
        }, inplace=True)

    else:
        user_features = train_num.groupby('fullVisitorId').agg({
            'visitTime': 'count',
            'visitNumber': 'max',
            'totals_bounces': 'mean',  # bounce ratio
            'totals_transactions': [namax, mean],
            'totals_hits': ['max', 'mean'],
            'totals_newVisits': ['max', 'mean'],  # whether contain new visits
            'trafficSource_isTrueDirect': 'mean',
            'trafficSource_adwordsClickInfo_isVideoAd': 'mean',
            'totals_pageviews': ['max', 'mean', 'std'],
            'totals_timeOnSite': [mean, namax, std],
            'avg_time_per_page': [mean, namax, std],
            'totals_transactionRevenue': ['max', has_rev_ratio, 'mean'],
            'trafficSource_adwordsClickInfo_page': [namax, mean, std],
            'days_since_last_session': ['max', gap_mean, gap_std],
            'hits_prev_ratio': ['max', gap_mean, gap_std],
            'pageviews_prev_ratio': ['max', gap_mean, gap_std],
            'hits_per_page': ['max', 'mean', 'std'],
            'days_since_pred': ['min', 'max'],

        })

        user_features.columns = [c[0] + '_' + c[1] for c in user_features.columns.tolist()]

        user_features.rename(columns={
            'totals_bounces_mean': 'bounce_ratio',
            'visitTime_count': 'total_visits',
            'totals_newVisits_max': 'include_first_visit',

        }, inplace=True)

    # hits with rev / hits without rev
    def hits_rev_ratio(df):
        try:
            return df.loc[df['totals_transactionRevenue'] == 0, 'totals_hits'].mean() / \
                   df.loc[df['totals_transactionRevenue'] > 0, 'totals_hits'].mean()
        except Exception:
            return math.nan

    # pageviews with rev / pageviews without rev
    def pageviews_rev_ratio(df):
        try:
            return df.loc[df['totals_transactionRevenue'] == 0, 'totals_pageviews'].mean() / \
                   df.loc[df['totals_transactionRevenue'] > 0, 'totals_pageviews'].mean()
        except Exception:
            return math.nan

    # last session days diff with rev
    def last_rev_days_diff(df):
        sp = df.loc[df['totals_transactionRevenue'] > 0, :].shape
        if sp[0] > 0:
            return df.loc[df['totals_transactionRevenue'] > 0, 'days_since_pred'].values[-1]
        else:
            return math.nan

    # first session days diff with rev
    def first_rev_days_diff(df):
        sp = df.loc[df['totals_transactionRevenue'] > 0, :].shape
        if sp[0] > 0:
            return df.loc[df['totals_transactionRevenue'] > 0, 'days_since_pred'].values[0]
        else:
            return math.nan

    # avg days diff between revs
    def avg_days_between_revs(df):
        sp = df.loc[df['totals_transactionRevenue'] > 0, :].shape
        if sp[0] > 0:
            return df.loc[df['totals_transactionRevenue'] > 0, 'days_since_last_session'].mean()
        else:
            return math.nan

    n = train_num.groupby('fullVisitorId').apply(hits_rev_ratio)

    n.name = 'hits_rev_ratio'

    a = train_num.groupby('fullVisitorId').apply(pageviews_rev_ratio)
    a.name = 'pageview_rev_ratio'

    b = train_num.groupby('fullVisitorId').apply(last_rev_days_diff)
    b.name = 'last_rev_days_diff'

    c = train_num.groupby('fullVisitorId').apply(first_rev_days_diff)
    c.name = 'first_rev_days_diff'

    d = train_num.groupby('fullVisitorId').apply(avg_days_between_revs)
    d.name = 'avg_days_between_revs'

    user_features = user_features.join(n).join(a).join(b).join(c).join(d)

    user_features['totals_pageviews_max'].fillna(user_features['totals_pageviews_max'].median(), inplace=True)
    user_features['totals_pageviews_mean'].fillna(user_features['totals_pageviews_mean'].median(), inplace=True)
    user_features['hits_per_page_max'].fillna(user_features['hits_per_page_max'].mean(), inplace=True)
    user_features['hits_per_page_mean'].fillna(user_features['hits_per_page_mean'].mean(), inplace=True)

    return user_features


def preprocess_cate_cols(df):
    train_df = df.copy()

    # String Column Grouping
    # device_operatingSystem
    operatingSystem_map = {
        'nintendo wiiu': 'nintendo wii',
        'nintendo 3ds': 'nintendo wii',
    }
    for k, v in operatingSystem_map.items():
        train_df['device_operatingSystem'] = train_df['device_operatingSystem'].str.replace(k, v, regex=False)

    # trafficSource_adContent
    adContent_map = {}
    adContent_map['daily ad display'] = re.compile('.*\d*/\d*/\d*')
    adContent_map['keyword search'] = re.compile('.*keyword.*')
    adContent_map['full auto ad'] = re.compile('.*full auto.*ad.*')
    adContent_map['google merchandise store'] = re.compile(
        '.*google.*store.*|.*google.*merchandise.*|.*google.*paraphernalia.*')
    for k, v in adContent_map.items():
        train_df['trafficSource_adContent'] = train_df['trafficSource_adContent'].str.replace(v, k)

    # trafficSource_keyword
    source_kw_map = {}
    source_kw_map['google merchandise'] = re.compile('.*google.*merch.*')
    source_kw_map['google store'] = re.compile('.*google.*store.*|.*google.*shop.*|.google.*shop.*')
    source_kw_map['youtube'] = re.compile('.*youtube.*|.you.*tube.*')
    source_kw_map['shirt'] = re.compile('.*shirt.*')
    for k, v in source_kw_map.items():
        train_df['trafficSource_keyword'] = train_df['trafficSource_keyword'].str.replace(v, k)

    # trafficSource_referralPath
    referralPaths_res = [
        '.*yt/about.*',
        '.*analytics.*web.*',
        '.permissions.*using.*logo.*',
        '.*googletopia.*',
        '.*visit.*google.*',
        '.*google.*merch.*',
        '.*forum.*',
        '.*comment.*',
        '.*intl.*',
    ]

    referralPaths_names = [
        'yt_abount',
        'analytics_web',
        'permission_logo',
        'googletopia',
        'visit_google',
        'google_merchandise_store',
        'forum',
        'comment',
        'intl'
    ]
    for k, v in zip(referralPaths_names, referralPaths_res):
        train_df['trafficSource_referralPath'] = train_df['trafficSource_referralPath'].str.replace(re.compile(v), k)

    # trafficSource_source
    train_df['trafficSource_source'] = train_df['trafficSource_source'].str.replace(
        re.compile('^google\..*'), 'google_com')

    # Handle NaN values
    # trafficSource_trafficSource_adwordsClickInfo_isVideoAd
    train_df['trafficSource_adwordsClickInfo_isVideoAd'] = \
        train_df['trafficSource_adwordsClickInfo_isVideoAd'].isna().astype(float)

    # trafficSource_isTrueDirect
    train_df['trafficSource_isTrueDirect'] = train_df['trafficSource_isTrueDirect'].isna().astype(float)

    # trafficSource_adwordsClickInfo_slot
    train_df.loc[train_df['trafficSource_adwordsClickInfo_slot'] == 'google display network',
                 'trafficSource_adwordsClickInfo_slot'] = 'not_available'

    train_df['visit_day'] = train_df['visitTime'].dt.day
    train_df['visit_day_of_week'] = train_df['visitTime'].dt.dayofweek
    train_df['visit_day_of_week'] = train_df['visit_day_of_week'].astype(str)
    train_df['visit_hour'] = train_df['visitTime'].dt.hour

    # train_df.drop(['date', 'sessionId', 'visitId'], axis=1, inplace=True)

    return train_df


def get_cate_values(df):
    train_df = df.copy()
    value_dict = {}
    # channelGrouping
    channelGrouping = list(train_df.channelGrouping.value_counts().head(7).index) + ['others']
    value_dict['channelGrouping'] = channelGrouping

    # device_browser
    device_browser = list(train_df.device_browser.value_counts(dropna=False).head(12).index) + ['others']
    value_dict['device_browser'] = device_browser

    # device_deviceCategory
    value_dict['device_deviceCategory'] = list(train_df.device_deviceCategory.value_counts().index)

    # device_operatingSystem
    value_dict['device_operatingSystem'] = list(
        train_df.device_operatingSystem.value_counts(dropna=False).head(8).index) + ['others']

    # geoNetwork_city
    value_dict['geoNetwork_city'] = list(
        train_df.geoNetwork_city.value_counts(dropna=False).head(200).index) + ['others']

    # geoNetwork_continent
    value_dict['geoNetwork_continent'] = list(train_df.geoNetwork_continent.value_counts(dropna=False).index)

    # geoNetwork_country
    value_dict['geoNetwork_country'] = list(
        train_df.geoNetwork_country.value_counts(dropna=False).head(100).index) + ['others']

    # geoNetwork_metro
    value_dict['geoNetwork_metro'] = list(
        train_df.geoNetwork_metro.value_counts(dropna=False).head(30).index) + ['others']

    # geoNetwork_networkDomain
    value_dict['geoNetwork_networkDomain'] = list(
        train_df.geoNetwork_networkDomain.value_counts(dropna=False).head(200).index) + ['others']

    # region
    value_dict['geoNetwork_region'] = list(
        train_df.geoNetwork_region.value_counts(dropna=False).head(100).index) + ['others']

    # geoNetwork_subContinent
    value_dict['geoNetwork_subContinent'] = list(train_df.geoNetwork_subContinent.value_counts().index)

    # trafficSource_adContent
    value_dict['trafficSource_adContent'] = list(
        train_df.trafficSource_adContent.value_counts(dropna=False).head(5).index) + ['others']

    # trafficSource_adwordsClickInfo_adNetworkType
    value_dict['trafficSource_adwordsClickInfo_adNetworkType'] = ['not_available', 'google search', 'others']

    # trafficSource_adwordsClickInfo_slot
    value_dict['trafficSource_adwordsClickInfo_slot'] = ['not_available', 'top', 'rhs']

    # trafficSource_compaign
    value_dict['trafficSource_campaign'] = [
        'not_available',
        'data share promo',
        'aw - dynamic search ads whole site',
        'aw - accessories',
        'aw - apparel',
        'others'
    ]

    # trafficSource_keyword
    value_dict['trafficSource_keyword'] = list(
        train_df['trafficSource_keyword'].value_counts(dropna=False).head(15).index) + ['others']

    # trafficSource_medium
    value_dict['trafficSource_medium'] = list(
        train_df.trafficSource_medium.value_counts(dropna=False).index)

    # trafficSource_referralPath
    value_dict['trafficSource_referralPath'] = list(
        train_df.trafficSource_referralPath.value_counts(dropna=False).head(15).index) + ['others']

    # trafficSource_source
    value_dict['trafficSource_source'] = list(
        train_df['trafficSource_source'].value_counts(dropna=False).head(15).index) + ['others']

    # visit_day_of_week
    value_dict['visit_day_of_week'] = [str(i) for i in range(0, 7)]

    embed_dict = {}

    for k, v in value_dict.items():
        n = {}
        n['value_lists'] = v
        n['value_lists_encoding'] = {}
        n['value_lists_length'] = len(n['value_lists'])
        for i, e in enumerate(n['value_lists']):
            n['value_lists_encoding'][e] = i + 1

        value_dict[k] = n

    c_cols = list(value_dict.keys())
    train_mean_encoding = train_df.loc[:, ['fullVisitorId', 'label'] + c_cols]

    for k, v in value_dict.items():
        train_mean_encoding.loc[~train_mean_encoding[k].isin(v['value_lists']), k] = 'others'

    train_first = train_mean_encoding.groupby('fullVisitorId').first()
    for k, v in value_dict.items():
        v['value_lists_mean_encoding'] = train_first.groupby(k)['label'].mean().to_dict()
        v['value_lists_frequency_encoding'] = (
            train_first.groupby(k)['label'].count() / train_first.shape[0]
        ).to_dict()

    del train_df, train_mean_encoding, train_first

    gc.collect()
    return value_dict


def gen_num_cate_interaction_feas(df, num_feas, cate_dict):
    cate_cols = list(cate_dict.keys())
    df_cate = df.loc[:, ['fullVisitorId', 'visitTime'] + cate_cols].copy()
    df_cate.sort_values(['fullVisitorId', 'visitTime'], inplace=True)
    cate_first = df_cate.drop('visitTime', axis=1).groupby('fullVisitorId').first()

    feas_to = num_feas.join(cate_first)

    for i in (
            'device_browser', 'device_operatingSystem', 'trafficSource_keyword',
            'geoNetwork_country', 'channelGrouping', 'geoNetwork_networkDomain',
            'visit_day_of_week'
    ):
        feas_to['mean_pageviews_per_{}'.format(i)] = feas_to.groupby(i)['totals_pageviews_mean'].transform('mean')

        feas_to['max_days_gap_from_pred_per_{}'.format(i)] = \
            feas_to.groupby(i)['days_since_last_session_max'].transform('max')

        feas_to['mean_hits_per_page_per_{}'.format(i)] = \
            feas_to.groupby(i)['hits_per_page_mean'].transform('mean')

        feas_to['mean_pageviews_prev_ratio_per_{}'.format(i)] = \
            feas_to.groupby(i)['pageviews_prev_ratio_max'].transform('mean')

        feas_to['mean_total_hits_per_{}'.format(i)] = \
            feas_to.groupby(i)['totals_hits_mean'].transform('mean')

        feas_to['max_hits_prev_ratio_per_{}'.format(i)] = \
            feas_to.groupby(i)['hits_prev_ratio_max'].transform('mean')

        feas_to['mean_total_visits_per_{}'.format(i)] = \
            feas_to.groupby(i)['total_visits'].transform('mean')

        feas_to['mean_max_visitNumber_per_{}'.format(i)] = \
            feas_to.groupby(i)['visitNumber_max'].transform('mean')

        feas_to['mean_revenue_per_{}'.format(i)] = \
            feas_to.groupby(i)['totals_transactionRevenue_max'].transform('mean')

    feas_to.drop(cate_cols, axis=1, inplace=True)

    return feas_to


def gen_cate_feas(df, cate_dict, n=1):
    cate_cols = list(cate_dict.keys())

    df = df.loc[:, ['fullVisitorId', 'visitTime'] + cate_cols]

    df = df.sort_values(['fullVisitorId', 'visitTime'])

    df['visit_count'] = df.groupby('fullVisitorId').cumcount() + 1
    df.loc[df['visit_count'] >= n, 'visit_count'] = n

    for k, v in cate_dict.items():
        df.loc[~df[k].isin(v['value_lists']), k] = 'others'
        df[k + '_factorized'] = df[k].map(v['value_lists_encoding'])
        df[k + '_mean'] = df[k].map(v['value_lists_mean_encoding'])
        df[k + '_freq'] = df[k].map(v['value_lists_frequency_encoding'])

    ret = df.drop(['visitTime'] + cate_cols, axis=1).groupby(['fullVisitorId', 'visit_count']).first().unstack()
    ret.columns = [e[0] + '_' + str(e[1]) for e in ret.columns]

    return ret


# hit
def prep_data_hits(fit_df):
    df = fit_df.copy()

    df['visitStartTime'] = df['visitStartTime'].astype(int)
    df['visitTime'] = pd.to_datetime(df['visitStartTime'], unit='s')
    drop_cols = [
        'hits_contentGroup.contentGroup4',
        'hits_contentGroup.contentGroup5',
        'hits_contentGroup.previousContentGroup1',
        'hits_contentGroup.previousContentGroup2',
        'hits_contentGroup.previousContentGroup3',
        'hits_contentGroup.previousContentGroup4',
        'hits_contentGroup.previousContentGroup5',
        'hits_exceptionInfo.isFatal',
        'hits_item.currencyCode',
        'hits_latencyTracking.domLatencyMetricsSample',
        'hits_latencyTracking.pageLoadSample',
        'hits_latencyTracking.speedMetricsSample',
        'hits_page.searchCategory',
        'hits_social.socialInteractionNetworkAction',
        'hits_time',
        'hits_transaction.affiliation',
        'hits_transaction.currencyCode'
    ]

    df.drop(drop_cols, axis=1, inplace=True)
    bool_type_list = [
        'hits_contentGroup.contentGroupUniqueViews1',
        'hits_contentGroup.contentGroupUniqueViews3',
        'hits_isEntrance',
        'hits_isExit',
        'hits_isInteraction',
        'hits_promotionActionInfo.promoIsClick',
        'hits_promotionActionInfo.promoIsView',

    ]

    bool_type_dict = {k: float for k in bool_type_list}
    bool_type_na_dict = {k: 0 for k in bool_type_list}
    df = df.astype(bool_type_dict)
    df = df.fillna(bool_type_na_dict)

    na_re = re.compile(
        '.*not set.*|.*not available.*|.*not provided.*|.*none.*'
    )
    bracket_re = re.compile(r'\(|\)')

    na_str = 'not_available'
    bracket_str = ''

    for c in df.select_dtypes('object').columns:

        # print(c)
        if c not in ['hits_promotion', 'hits_product']:
            df[c] = df[c].str.strip()
            df[c] = df[c].str.lower()
            df[c] = df[c].str.replace(na_re, na_str)
            df[c] = df[c].str.replace(bracket_re, bracket_str)
            df.loc[df[c].isnull(), c] = na_str
    return df


def p_feas(p):
    if p:
        p_price = [float(i['productPrice']) for i in p]
        p_position = [float(i['productListPosition']) for i in p]
        mean_price = np.mean(p_price)
        max_price = np.max(p_price)
        max_position = np.max(p_position)
        first_p_category = p[0]['v2ProductCategory']
        return [mean_price, max_price, max_position, first_p_category]
    else:
        return [math.nan] * 4


def process_hit_df(df):
    train_hits = df.copy()
    train_hits['hits_product'].fillna('[]', inplace=True)
    train_hits['hits_product'] = train_hits['hits_product'].apply(literal_eval)
    train_hits['hits_product_len'] = train_hits['hits_product'].apply(len)
    train_hits['product_feas'] = train_hits['hits_product'].apply(p_feas)
    train_hits[[
        'mean_hits_product_price', 'max_hits_product_price', 'max_hits_product_num', 'first_hits_product_category'
    ]] = pd.DataFrame(train_hits['product_feas'].values.tolist(), index=train_hits.index)
    train_hits['hits_promotion'].fillna('[]', inplace=True)
    train_hits['hits_promotion'] = train_hits['hits_promotion'].apply(literal_eval)
    train_hits['hits_promotion_len'] = train_hits['hits_promotion'].apply(len)

    train_hits.loc[train_hits['hits_appInfo.screenDepth'] == 'not_available', 'hits_appInfo.screenDepth'] = 1
    train_hits['hits_appInfo.screenDepth'] = train_hits['hits_appInfo.screenDepth'].astype(int)

    for i in (1, 2, 3):
        train_hits['hits_contentGroup.contentGroupUniqueViews{}'.format(i)] = \
            train_hits['hits_contentGroup.contentGroupUniqueViews{}'.format(i)].astype(str)
        train_hits.loc[train_hits['hits_contentGroup.contentGroupUniqueViews{}'.format(i)] == 'not_available',
                       'hits_contentGroup.contentGroupUniqueViews{}'.format(i)] = 0
        train_hits['hits_contentGroup.contentGroupUniqueViews{}'.format(i)] = \
            train_hits['hits_contentGroup.contentGroupUniqueViews{}'.format(i)].astype(float)

    train_hits['hits_hitNumber'].fillna(0, inplace=True)
    return train_hits


def gen_hit_num_feas(df):
    hit_numeric_cols = [
        'hits_appInfo.screenDepth', 'hits_contentGroup.contentGroupUniqueViews1',
        'hits_contentGroup.contentGroupUniqueViews2',
        'hits_contentGroup.contentGroupUniqueViews3', 'hits_hitNumber', 'hits_hour', 'hits_isEntrance',
        'hits_isExit', 'hits_promotionActionInfo.promoIsClick', 'hits_promotionActionInfo.promoIsView',
        'hits_product_len', 'mean_hits_product_price', 'max_hits_product_price', 'hits_promotion_len',
    ]

    df_numeric = df.loc[:, ['fullVisitorId', 'visitTime'] + hit_numeric_cols].copy()
    df_numeric.sort_values(['fullVisitorId', 'visitTime'], inplace=True)

    df_numeric['prev_hitNumber_ratio'] = df_numeric['hits_hitNumber'] / df_numeric['hits_hitNumber'].shift(1)
    df_numeric['hits_product_len_prev_ratio'] = df_numeric['hits_product_len'] / df_numeric['hits_product_len'].shift(1)
    df_numeric['max_hits_product_price_prev_ratio'] = df_numeric['max_hits_product_price'] / \
                                                      df_numeric['max_hits_product_price'].shift(1)

    df_numeric.loc[
        df_numeric['fullVisitorId'] != df_numeric['fullVisitorId'].shift(1),
        ['prev_hitNumber_ratio', 'hits_product_len_prev_ratio', 'max_hits_product_price_prev_ratio']
    ] = 0

    def mean(se):
        return se.sum() / (len(se) - 1)

    fea_numeric = df_numeric.groupby('fullVisitorId').agg({
        'visitTime': 'count',
        'hits_appInfo.screenDepth': 'max',
        'hits_contentGroup.contentGroupUniqueViews1': 'max',
        'hits_contentGroup.contentGroupUniqueViews2': 'max',
        'hits_contentGroup.contentGroupUniqueViews3': 'max',
        'hits_hitNumber': ['mean', 'max', 'min'],
        'hits_hour': ['mean', 'max'],
        'hits_isEntrance': 'mean',
        'hits_isExit': 'mean',
        'hits_promotionActionInfo.promoIsClick': 'mean',
        'hits_promotionActionInfo.promoIsView': 'mean',
        'hits_product_len': ['mean', 'max'],
        'mean_hits_product_price': 'mean',
        'max_hits_product_price': 'max',
        'hits_promotion_len': ['mean', 'max'],
        'prev_hitNumber_ratio': [mean, 'max'],
        'hits_product_len_prev_ratio': [mean, 'max'],
        'max_hits_product_price_prev_ratio': [mean, 'max']

    })

    fea_numeric.columns = [e[0] + '_' + e[1] for e in fea_numeric.columns.tolist()]
    fea_numeric.loc[fea_numeric['visitTime_count'] == 1,
                    ['prev_hitNumber_ratio_mean', 'prev_hitNumber_ratio_max',
                     'hits_product_len_prev_ratio_mean', 'hits_product_len_prev_ratio_max',
                     'max_hits_product_price_prev_ratio_mean', 'max_hits_product_price_prev_ratio_max']] = math.nan

    fea_numeric.drop('visitTime_count', axis=1, inplace=True)

    fea_numeric['hits_hour_mean'].fillna(fea_numeric['hits_hour_mean'].mean(), inplace=True)
    fea_numeric['hits_hour_max'].fillna(fea_numeric['hits_hour_max'].mean(), inplace=True)

    return fea_numeric


def get_hits_cate_dict(train_hits):
    hits_cate_dict = dict()
    for i in (1, 2, 3):
        hits_cate_dict['hits_contentGroup.contentGroup{}'.format(i)] = list(
            train_hits['hits_contentGroup.contentGroup{}'.format(i)].value_counts().index) + ['others']
    hits_cate_dict['customDimensions_value'] = list(
        train_hits['customDimensions_value'].value_counts().index) + ['others']

    hits_cate_dict['customDimensions_index'] = list(
        train_hits['customDimensions_index'].value_counts().index) + ['others']

    hits_cate_dict['hits_appInfo.exitScreenName'] = list(
        train_hits['hits_appInfo.exitScreenName'].value_counts().index) + ['others']

    hits_cate_dict['hits_appInfo.landingScreenName'] = list(
        train_hits['hits_appInfo.landingScreenName'].value_counts().index) + ['others']

    hits_cate_dict['hits_appInfo.screenName'] = list(
        train_hits['hits_appInfo.screenName'].value_counts().index) + ['others']

    hits_cate_dict['hits_eCommerceAction.action_type'] = [
                                                             str(i) for i in list(
            train_hits['hits_eCommerceAction.action_type'].value_counts().index
        )
                                                         ] + ['others']

    hits_cate_dict['hits_eventInfo.eventAction'] = list(
        train_hits['hits_eventInfo.eventAction'].value_counts(dropna=False).index
    ) + ['others']

    hits_cate_dict['hits_eventInfo.eventCategory'] = list(
        train_hits['hits_eventInfo.eventCategory'].value_counts().index
    ) + ['others']

    hits_cate_dict['hits_page.hostname'] = list(
        train_hits['hits_page.hostname'].value_counts(dropna=False).index
    ) + ['others']

    hits_cate_dict['hits_page.pagePath'] = list(
        train_hits['hits_page.pagePath'].value_counts(dropna=False).head(100).index
    ) + ['others']

    hits_cate_dict['hits_page.pagePathLevel1'] = list(
        train_hits['hits_page.pagePathLevel1'].value_counts(dropna=False).head(10).index
    ) + ['others']

    hits_cate_dict['hits_page.pagePathLevel2'] = list(
        train_hits['hits_page.pagePathLevel2'].value_counts(dropna=False).head(30).index
    ) + ['others']

    hits_cate_dict['hits_page.pagePathLevel3'] = list(
        train_hits['hits_page.pagePathLevel3'].value_counts(dropna=False).head(50).index
    ) + ['others']

    hits_cate_dict['hits_page.pagePathLevel4'] = list(
        train_hits['hits_page.pagePathLevel4'].value_counts(dropna=False).head(30).index
    ) + ['others']

    hits_cate_dict['hits_page.pageTitle'] = list(
        train_hits['hits_page.pageTitle'].value_counts(dropna=False).head(100).index
    ) + ['others']

    hits_cate_dict['hits_page.searchKeyword'] = list(
        train_hits['hits_page.searchKeyword'].value_counts(dropna=False).index
    ) + ['others']

    hits_cate_dict['hits_referer'] = list(
        train_hits['hits_referer'].value_counts(dropna=False).head(200).index
    ) + ['others']

    hits_cate_dict['hits_social.hasSocialSourceReferral'] = list(
        train_hits['hits_social.hasSocialSourceReferral'].value_counts(dropna=False).index
    ) + ['others']

    hits_cate_dict['hits_social.socialNetwork'] = list(
        train_hits['hits_social.socialNetwork'].value_counts(dropna=False).head(8).index
    ) + ['others']

    hits_cate_dict['hits_type'] = list(
        train_hits['hits_type'].value_counts().index
    ) + ['others']

    hits_cate_dict['first_hits_product_category'] = list(
        train_hits['first_hits_product_category'].value_counts(dropna=False).head(50).index
    ) + ['others']

    value_dict = {}
    for k, v in hits_cate_dict.items():
        n = {}
        n['value_lists'] = v
        n['value_lists_encoding'] = {}
        for i, u in enumerate(v):
            n['value_lists_encoding'][u] = i + 1
        value_dict[k] = n

    c_cols = list(value_dict.keys())
    train_mean_encoding = train_hits.loc[:, ['fullVisitorId', 'label'] + c_cols]

    for k, v in value_dict.items():
        train_mean_encoding.loc[~train_mean_encoding[k].isin(v['value_lists']), k] = 'others'

    train_first = train_mean_encoding.groupby('fullVisitorId').first()
    for k, v in value_dict.items():
        v['value_lists_mean_encoding'] = train_first.groupby(k)['label'].mean().to_dict()
        v['value_lists_frequency_encoding'] = (
            train_first.groupby(k)['label'].count() / train_first.shape[0]
        ).to_dict()

    del train_mean_encoding, train_first

    gc.collect()

    return value_dict


def extract_feature(df_exhits, df_hits, is_train=True, exhits_value_dict=None,
                    hits_value_dict=None, n_exhits=3, n_hits=1):
    new_value_dict = pickle.load(open('new_value_dict.pkl', 'rb'))
    exhits_new_value_dict, hits_new_value_dict = new_value_dict['exhits'], new_value_dict['hits']
    # exhits
    df_exhits = prep_data(df_exhits)
    num_feas_exhits = prepare_features(df_exhits, is_train)
    df_cate_exhits = preprocess_cate_cols(df_exhits)

    if is_train:
        exhits_value_dict = get_cate_values(df_cate_exhits)

        for k, v in exhits_value_dict.items():
            if k in exhits_new_value_dict:
                for m, n in v['value_lists_encoding'].items():
                    if m in exhits_new_value_dict[k]:
                        v['value_lists_encoding'][m] = exhits_new_value_dict[k][m]

    num_feas_expand_exhits = gen_num_cate_interaction_feas(df_cate_exhits, num_feas_exhits, exhits_value_dict)

    cate_feas_exhits = gen_cate_feas(df_cate_exhits, exhits_value_dict, n=n_exhits)

    # hits
    df_hits = prep_data_hits(df_hits)
    df_hits_process = process_hit_df(df_hits)
    num_feas_hits = gen_hit_num_feas(df_hits_process)
    if is_train:
        hits_value_dict = get_hits_cate_dict(df_hits_process)

        for k, v in hits_value_dict.items():
            if k in hits_new_value_dict:
                for m, n in v['value_lists_encoding'].items():
                    if m in hits_new_value_dict[k]:
                        v['value_lists_encoding'][m] = hits_new_value_dict[k][m]

    cate_feas_hits = gen_cate_feas(df_hits_process, hits_value_dict, n=n_hits)

    assert num_feas_exhits.shape[0] == cate_feas_exhits.shape[0] == num_feas_hits.shape[0] == cate_feas_hits.shape[0]

    feas_all = num_feas_expand_exhits.join(cate_feas_exhits).join(num_feas_hits).join(cate_feas_hits)

    del df_exhits
    del num_feas_exhits, num_feas_expand_exhits, df_cate_exhits, cate_feas_exhits, df_hits, df_hits_process
    del num_feas_hits
    del cate_feas_hits
    gc.collect()

    if is_train:
        return feas_all, exhits_value_dict, hits_value_dict
    else:
        return feas_all


class bulk_pickle(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.n_bytes = 2 ** 31
        self.max_bytes = 2 ** 31 - 1

    def write(self, data):
        bytes_out = pickle.dumps(data)
        with open(self.file_path, 'wb') as f_out:
            for idx in range(0, len(bytes_out), self.max_bytes):
                f_out.write(bytes_out[idx:idx + self.max_bytes])

    def read(self):
        bytes_in = bytearray(0)
        input_size = os.path.getsize(self.file_path)
        with open(self.file_path, 'rb') as f_in:
            for _ in range(0, input_size, self.max_bytes):
                bytes_in += f_in.read(self.max_bytes)
        data2 = pickle.loads(bytes_in)
        return data2


def main():
    read_hit(n='train')
    train_ex_hits = read_ex_hits('train')
    train_hits_origin = get_hit_df(train_ex_hits)

    # read test data
    read_hit(n='test')
    pred_ex_hits = read_ex_hits('test')
    pred_hits_origin = get_hit_df(pred_ex_hits, 'test')

    train_exhits, val_exhits, test_exhits = split_train_test(train_ex_hits)
    train_hits, val_hits, test_hits = split_train_test(train_hits_origin)

    train_x_y, exhits_value_dict, hits_value_dict = extract_feature(train_exhits, train_hits, True)
    val_x_y = extract_feature(val_exhits, val_hits, False, exhits_value_dict, hits_value_dict)
    test_x_y = extract_feature(test_exhits, test_hits, False, exhits_value_dict, hits_value_dict)

    val_label = val_exhits.groupby('fullVisitorId')['label'].first().to_frame('label')
    val_x_y = val_x_y.join(val_label)

    test_label = test_exhits.groupby('fullVisitorId')['label'].first().to_frame('label')
    test_x_y = test_x_y.join(test_label)

    pred_x_y = extract_feature(
        pred_ex_hits, pred_hits_origin, False, exhits_value_dict, hits_value_dict)

    train_columns_seq = list(train_x_y.drop('label', axis=1).columns)
    pred_x_y = pred_x_y[train_columns_seq]

    trainDM = xgb.DMatrix(train_x_y.drop('label', axis=1), train_x_y['label'])
    valDM = xgb.DMatrix(val_x_y.drop('label', axis=1), val_x_y['label'])
    testDM = xgb.DMatrix(test_x_y.drop('label', axis=1), test_x_y['label'])
    predDM = xgb.DMatrix(pred_x_y)

    p3 = {
        'seed': 0,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
        'colsample_bytree': 0.6545,
        'eta': 0.1057,
        'gamma': 14.2827,
        'max_depth': 8,
        'min_child_weight': 7,
        'num_boost_round': 797,
        'reg_alpha': 91.9003,
        'reg_lambda': 1.0696,
        'subsample': 0.7605
    }

    def multiple_xgb(param_single, seeds):
        train_df = pd.DataFrame({'label': train_x_y['label']}, index=train_x_y.index)
        val_df = pd.DataFrame({'label': val_x_y['label']}, index=val_x_y.index)
        test_df = pd.DataFrame({'label': test_x_y['label']}, index=test_x_y.index)
        pred_df = pd.DataFrame(index=pred_x_y.index)

        num_boost_round = param_single['num_boost_round']
        del param_single['num_boost_round']

        for i, s in enumerate(seeds):
            print("round {}".format(i + 1))

            param_single['seed'] = s * 9

            xgb_s = xgb.train(
                param_single, trainDM, num_boost_round=num_boost_round, evals=[(valDM, 'val')],
                early_stopping_rounds=50, verbose_eval=False
            )

            train_pred = xgb_s.predict(trainDM, ntree_limit=xgb_s.best_ntree_limit)
            train_pred[train_pred < 0] = 0

            val_pred = xgb_s.predict(valDM, ntree_limit=xgb_s.best_ntree_limit)
            val_pred[val_pred < 0] = 0

            test_pred = xgb_s.predict(testDM, ntree_limit=xgb_s.best_ntree_limit)
            test_pred[test_pred < 0] = 0

            pred_pred = xgb_s.predict(predDM, ntree_limit=xgb_s.best_ntree_limit)
            pred_pred[pred_pred < 0] = 0

            print(
                "train rmse: {}\nval rmse: {}\ntest rmse: {}\n".format(
                    mse(train_x_y['label'], train_pred) ** 0.5, mse(val_x_y['label'], val_pred) ** 0.5,
                    mse(test_x_y['label'], test_pred) ** 0.5
                )
            )
            print('\n')

            train_df['xgb_{}'.format(i + 1)] = train_pred
            val_df['xgb_{}'.format(i + 1)] = val_pred
            test_df['xgb_{}'.format(i + 1)] = test_pred
            pred_df['xgb_{}'.format(i + 1)] = pred_pred

        return train_df, val_df, test_df, pred_df

    train_df, val_df, test_df, pred_df = multiple_xgb(p3, range(1, 10))

    s = pd.read_csv(
        './data/sample_submission_v2.csv', dtype={'fullVisitorId':str}
    )

    pred_sub = pd.DataFrame(pred_df.index)

    pred_sub = pred_df.mean(axis=1).to_frame('n')

    pred_sub['fullVisitorId'] = pred_sub.index

    pred_sub.reset_index(drop=True, inplace=True)

    result = s.merge(pred_sub, on='fullVisitorId')

    result['PredictedLogRevenue'] = result['n'].copy()

    result.drop('n', axis=1, inplace=True)
    if not os.path.exists('./submission'):
        os.mkdir('./submission')
    result.to_csv('./submission/sub_xgb_mean.csv', index=False)


if __name__ == '__main__':
    main()
