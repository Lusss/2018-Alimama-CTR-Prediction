#from util_pre import featureExtraction
#from baseline_Predata import train,np,val,test,y_train,y_val
# from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
# from keras.layers import BatchNormalization, SpatialDropout1D
# from keras.models import Model
# from keras.layers import Dense, Activation
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import lightgbm as lgb
from bayesSmooth import BayesianSmoothing
import warnings
warnings.filterwarnings("ignore")


np.random.seed(2018)
warnings.filterwarnings("ignore")


# 时间处理
all_data = pd.read_pickle('./data/all_data.pkl')
all_data['context_hour'] = all_data['realtime'].dt.hour
all_data['context_day'] = all_data['realtime'].dt.day
print(all_data.columns)
def time_change(hour):
    hour = hour - 1
    if hour == -1:
        hour = 23
    return hour


def time_change_1(hour):
    hour = hour + 1
    if hour == 24:
        hour = 0
    return hour


all_data['hour_before'] = all_data['context_hour'].apply(time_change)
all_data['hour_after'] = all_data['context_hour'].apply(time_change_1)

# 18 21 19 20 22 23 24 | 25
print(all_data['context_day'].unique())

def c_log_loss(y_t, y_p):
    tmp = np.array(y_t) * np.log(np.array(y_p)) + (1 - np.array(y_t)) * np.log(1 - np.array(y_p))
    return -np.sum(tmp) / len(y_t), False


# # 获取当前时间之前的前x天的转化率特征
# def get_before_cov_radio(all_data, label_data,
#                          cov_list=list(['user_id','shop_id', 'item_id', 'context_hour', 'item_pv_level', 'item_sales_level']),
#                          day_list=list([1, 2, 3])):
#     result = []
#     r = pd.DataFrame()
#     label_data_time = label_data['context_day'].min()
#     label_data_time_set = label_data['context_day'].unique()
#     print('label set day', label_data_time_set)
#     for cov in cov_list:
#         for d in day_list:
#             feat_set = all_data[
#                 (all_data['context_day'] >= label_data_time - d) & (all_data['context_day'] < label_data_time)
#                 ]
#             print("cov feature", feat_set['context_day'].unique())
#             print("cov time", cov)
#
#             tmp = feat_set.groupby([cov], as_index=False).is_trade.agg({'bought': np.sum, 'click': 'count'}).add_suffix("_%s_before_%d_day" % (cov, d))
#             tmp.rename(columns={'%s_%s_before_%d_day' % (cov, cov, d): cov}, inplace=True)
#
#             if d == 1:
#                 r = tmp
#             else:
#                 r = pd.merge(r, tmp, on=[cov], how='outer').fillna(0)
#
#         result.append(r)
#     return result

# 获取当前时间之前的转化率特征
def get_before_cov_radio(all_data,cov_list=list(['user_id','shop_id', 'item_id', 'context_hour', 'item_pv_level', 'item_sales_level'])):
    for cov in cov_list:
        all_data_temp = all_data.loc[all_data['context_day'] < 24]
        print("cov feature", all_data_temp['context_day'].unique())
        print("cov time", cov)
        tmp = all_data_temp.groupby([cov], as_index=False).is_trade.agg({"bought_%s" % (cov): 'sum', "click_%s" % (cov): 'count'})
        #tmp.rename(columns={'%s_%s_before_%d_day' % (cov, cov, d): cov}, inplace=True)
        all_data = pd.merge(all_data, tmp, on=[cov], how='left').fillna(0)

    return all_data

def calc_categry_feat(data):
    for i in range(3):
        data['predict_category_%d' % (i)] = data['predict_category_property'].apply(
            lambda x: int(str(x.split(";")[i]).split(":")[0]) if len(x.split(";")) > i else -1
        )
    # #print(data.columns)
    # for item_cate in ['category_1', 'category_2']:
    #     for pre_item_cate in ['predict_category_0', 'predict_category_1', 'predict_category_2']:
    #         data['%s_%s' % (item_cate, pre_item_cate)] = data[item_cate] == data[pre_item_cate]
    #         data['%s_%s' % (item_cate, pre_item_cate)] = data['%s_%s' % (item_cate, pre_item_cate)].astype(int)

    # del data['item_category_list']
    # del data['item_property_list']
    del data['predict_category_property']
    return data


take_columns = ['instance_id', 'item_id', 'shop_id', 'user_id', 'is_trade']

shop_current_col = [
    'shop_score_description', 'shop_score_delivery', 'shop_score_service',
    'shop_star_level', 'shop_review_positive_rate', 'shop_review_num_level',
    'smoothctr_shop_id'
]

user_col = [
    'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level','smoothctr_user_id'
]

item_col = [
    'item_brand_id', 'item_city_id', 'item_price_level',
    'item_sales_level', 'item_collected_level', 'item_pv_level',
    'category_1','category_2','property_0','property_1','property_2',
    'item_id_vec', 'item_brand_id_vec', 'property_2_vec', 'property_0_vec', 'property_1_vec',

    'item_norm_price_score', 'item_id_user_cnt', 'item_id_user_prob',
    'item_brand_id_user_cnt', 'item_brand_id_user_prob',
    'item_city_id_user_cnt', 'item_city_id_user_prob',
    'item_price_level_user_cnt', 'item_price_level_user_prob',
    'item_sales_level_user_cnt', 'item_sales_level_user_prob',
    'item_collected_level_user_cnt', 'item_collected_level_user_prob',
    'item_pv_level_user_cnt', 'item_pv_level_user_prob', 'item_id_shop_cnt',
    'item_id_shop_prob', 'item_brand_id_shop_cnt',
    'item_brand_id_shop_prob', 'item_city_id_shop_cnt',
    'item_city_id_shop_prob', 'item_price_level_shop_cnt',
    'item_price_level_shop_prob', 'item_sales_level_shop_cnt',
    'item_sales_level_shop_prob', 'item_collected_level_shop_cnt',
    'item_collected_level_shop_prob', 'item_pv_level_shop_cnt',
    'item_pv_level_shop_prob', 'brand_price_mean', 'brand_price_std',
    'smoothctr_item_id','smoothctr_item_pv_level','smoothctr_item_sales_level'

]
time_feat = ['context_hour', 'hour_before', 'hour_after', 'context_timestamp', 'context_day','smoothctr_context_hour']

context_col = ['predict_category_property', 'context_page_id']

feat = take_columns + shop_current_col + time_feat + user_col + item_col + context_col


def get_history_user_feat(all_data, data):
    label_data_time = data['context_day'].min()
    print(label_data_time)

    tmp = all_data[all_data['context_day'] < label_data_time]
    print(tmp['context_day'].unique())

    user_time = tmp.groupby(['user_id'], as_index=False).context_timestamp.agg({'day_begin': 'min', 'day_end': 'max'})
    user_time['alive'] = user_time['day_end'] - user_time['day_begin']

    user_time['s_alive'] = label_data_time - user_time['day_begin']
    user_time['alive/s_alive'] = user_time['alive'] / user_time['s_alive']

    user_time_cov = tmp[tmp['is_trade'] == 1]
    user_time_cov = user_time_cov.groupby(['user_id'], as_index=False).context_timestamp.agg({'day_end_cov': 'max'})

    user_time_cov = pd.DataFrame(user_time_cov).drop_duplicates(['user_id', 'day_end_cov'])

    data = pd.merge(data, user_time[['user_id', 'alive', 's_alive', 'alive/s_alive', 'day_begin', 'day_end']],
                    on=['user_id'], how='left')

    data = pd.merge(data, user_time_cov, on=['user_id'], how='left')
    data['day_end_cov'] = data['day_end_cov'].fillna(data['day_end'])

    data['alive_cov'] = data['day_end_cov'] - data['day_begin']
    data['alive/alive_cov'] = data['alive'] / data['alive_cov']
    # data['s_alive/alive_cov'] = data['s_alive'] / data['alive_cov']

    del data['day_end_cov']
    del data['day_end']
    del data['day_begin']

    # for i in [1,2,3]:
    #     tmp = all_data[(all_data['context_day'] < data['context_day'].min()) & (all_data['context_day'] >= data['context_day'].min() - i)]
    #     user_item_sales_level_day = tmp.groupby(['user_id'], as_index=False)['item_sales_level'] \
    #         .agg({'user_item_sales_level_day_mean': 'mean',
    #               'user_item_sales_level_day_median': 'median',
    #               'user_item_sales_level_day_min': 'min',
    #               'user_item_sales_level_day_max': 'max',
    #               'user_item_sales_level_day_std': 'std',
    #               'user_item_sales_level_day_count': 'count'})
    #     data = pd.merge(data, user_item_sales_level_day, 'left', on=['user_id'])

    # data = data[['user_id','alive','s_alive','alive/s_alive','alive_cov','alive/alive_cov']]

    return data.fillna(-1)


def get_history_shop_feat(all_data, data):
    label_data_time = data['context_day'].min()
    print(label_data_time)
    for i in [1, 2, 3]:
        tmp = all_data[(all_data['context_day'] < label_data_time) & (all_data['context_day'] >= label_data_time - i)]

        shop_score_service_hour = tmp.groupby(['context_hour'], as_index=False)[
            'shop_score_service'] \
            .agg({
            'shop_score_service_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_score_service_hour, 'left', on=['context_hour'])

        shop_score_delivery = tmp.groupby(['context_hour'], as_index=False)[
            'shop_score_delivery'] \
            .agg({
            'shop_score_delivery_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_score_delivery, 'left', on=['context_hour'])

        shop_score_service_hour = tmp.groupby(['context_hour'], as_index=False)[
            'shop_score_description'] \
            .agg({
            'shop_score_description_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_score_service_hour, 'left', on=['context_hour'])

        shop_review_positive_rate = tmp.groupby(['context_hour'], as_index=False)[
            'shop_review_positive_rate'] \
            .agg({
            'shop_review_positive_rate_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_review_positive_rate, 'left', on=['context_hour'])

        shop_star_level = tmp.groupby(['context_hour'], as_index=False)[
            'shop_star_level'] \
            .agg({
            'shop_star_level_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_star_level, 'left', on=['context_hour'])

        shop_review_num_level = tmp.groupby(['context_hour'], as_index=False)[
            'shop_review_num_level'] \
            .agg({
            'shop_review_num_level_hour_std_%d' % (i): 'std',
        })
        data = pd.merge(data, shop_review_num_level, 'left', on=['context_hour'])

        shop_query_day_hour = tmp.groupby(['shop_id', 'context_hour']).size().reset_index().rename(
            columns={0: 'shop_query_day_hour_%d' % (i)})
        data = pd.merge(data, shop_query_day_hour, 'left', on=['shop_id', 'context_hour'])

    return data


def get_history_item_feat(all_data, data):
    for i in [1, 2, 3]:
        tmp = all_data[
            (all_data['context_day'] < data['context_day'].min()) & (all_data['context_day'] >= data['context_day'].min() - i)]

        item_brand_id_day = tmp.groupby(['item_city_id', 'context_hour']).size().reset_index().rename(
            columns={0: 'item_brand_id_day_%d' % (i)})
        data = pd.merge(data, item_brand_id_day, 'left', on=['item_city_id', 'context_hour'])

        item_brand_id_hour = tmp.groupby(['item_brand_id', 'context_hour']).size().reset_index().rename(
            columns={0: 'item_brand_id_hour_%d' % (i)})
        data = pd.merge(data, item_brand_id_hour, 'left', on=['item_brand_id', 'context_hour'])
        item_pv_level_hour = tmp.groupby(['item_pv_level', 'context_hour']).size().reset_index().rename(
            columns={0: 'item_pv_level_hour_%d' % (i)})
        data = pd.merge(data, item_pv_level_hour, 'left', on=['item_pv_level', 'context_hour'])
        #
        # item_pv_level_day = data.groupby(['context_day','context_hour'], as_index=False)['item_pv_level'] \
        #     .agg({'item_pv_level_day_mean_%d'%(i): 'mean',
        #           'item_pv_level_day_median_%d'%(i): 'median',
        #           'item_pv_level_day_std_%d'%(i): 'std'
        #           })
        # data = pd.merge(data, item_pv_level_day, 'left', on=['context_day','context_hour'])
    return data


print('make feat')


def make_feat(data):
    '''
    :param data: 标签数据，当前时刻的用户特征
    :param feat: 特征数据，统计的用户特征
    :return: 拼接后的特征
    '''

    data = calc_categry_feat(data)
    data = get_history_user_feat(all_data, data)
    data = get_history_shop_feat(all_data, data)
    data = get_history_item_feat(all_data, data)
    print('000000000')
    # for f in feat:
    #     data = pd.merge(data, f, on=[f.columns[0]], how='left')

    return data.fillna(0)


#test_a = all_data[train.shape[0]:]

train_val = all_data[(all_data['context_day'] >= 18) & (all_data['context_day'] <= 24)]
train_val = get_before_cov_radio(train_val)
# test = all_data[(all_data['context_day'] == 25)]

# train_val = train_val.reset_index(drop=True)

print(train_val.columns)
# train = get_before_cov_radio(train)
# val = get_before_cov_radio(val)

# 对转化率进行贝叶斯平滑
import os
if(os.path.isfile("./data/train_val.pkl")):
    train_val = pd.read_pickle("./data/train_val.pkl")
else:
    cov_list = list(['user_id','shop_id', 'item_id', 'context_hour', 'item_pv_level', 'item_sales_level'])
    for cov in cov_list:
            click_series = train_val["click_"+cov]
            bought_series = train_val["bought_"+cov]
            if cov in ['user_id','shop_id', 'item_id']:
                bayes_smooter = BayesianSmoothing(1,1)
                bayes_smooter.update(click_series, bought_series, 800, 0.00000001)
                #bayes_smooter.update_from_data_by_moment(click_series, bought_series)
                print("Finished",cov,bayes_smooter.beta,bayes_smooter.alpha)
                train_val["smoothctr_" + cov] = (bayes_smooter.alpha + bought_series)/( bayes_smooter.alpha + bayes_smooter.beta + click_series)
            else:
                train_val["smoothctr_" + cov] = train_val["bought_"+cov]/train_val["click_"+cov]
            del train_val["click_"+cov]
            del train_val["bought_"+cov]

    train_val.to_pickle('./data/train_val.pkl')

val = train_val[train_val['context_day'] == 24]
print(val)
train_a = train_val[train_val['context_day'] == 23]
train_b = train_val[train_val['context_day'] == 22]
train_c = train_val[train_val['context_day'] == 21]


# 传入全部数据和当前标签数据
val = make_feat(val[feat])

train_a = make_feat(train_a[feat])
train_b = make_feat(train_b[feat])
train_c = make_feat(train_c[feat])


# 传入全部数据和当前标签数据

train = pd.concat([train_a, train_b])
train = pd.concat([train, train_c])


y_train = train.pop('is_trade')
train_index = train.pop('instance_id')
X_train = train


y_val = val.pop('is_trade')
val_index = val.pop('instance_id')
X_val = val


category_list = [
    'item_id', 'shop_id', 'user_id', 'user_gender_id', 'user_age_level',
    'user_occupation_id', 'user_star_level',
    'item_brand_id', 'item_city_id', 'item_price_level',
    'item_sales_level', 'item_collected_level', 'item_pv_level',
    'shop_review_num_level', 'shop_star_level', 'category_1', 'category_2',
    'property_0', 'property_1', 'property_2','context_hour',
    'predict_category_0', 'predict_category_1', 'predict_category_2', 'context_page_id'
]
from sklearn.preprocessing import LabelEncoder
def make_cat(data):
    le = LabelEncoder()
    for i in category_list:
        data[i] = le.fit_transform(data[i])
        #data[i] = data[i].astype('category')
    # del data['category_1_vec']
    # del data['property_0_vec']
    # del data['property_1_vec']
    # del data['property_2_vec']
    # del data['item_id']
    # del data['item_brand_id']
    # vector_list = ['item_id_vec', 'item_brand_id_vec', 'property_2_vec', 'property_0_vec', 'property_1_vec']
    # #vector_list = ['category_1_vec', 'property_2_vec', 'property_0_vec', 'property_1_vec', 'item_id_vec', 'item_brand_id_vec']
    # for feat in vector_list:
    #     for i in range(50):
    #         data[feat+'_'+str(i)] = data[feat].apply(lambda x: x[i])
    #     del data[feat]
    return data

X_train = make_cat(X_train)
X_val = make_cat(X_val)

print(X_train.shape)
print(X_val.shape)

del X_train['hour_before']
del X_val['hour_before']

del X_train['hour_after']
del X_val['hour_after']

del X_train['context_day']
del X_val['context_day']

print(X_train.dtypes)
del X_train['context_timestamp']
del X_val['context_timestamp']
X_train = X_train[X_train.columns]
X_val = X_val[X_train.columns]
X_train.to_pickle('./data/X_train.pkl')
X_val.to_pickle('./data/X_val.pkl')
y_train.to_pickle('./data/y_train.pkl')
y_val.to_pickle('./data/y_val.pkl')


#
#
# category_list = [
#     'item_id', 'shop_id', 'user_id', 'user_gender_id', 'user_age_level',
#     'user_occupation_id', 'user_star_level',
#     'item_brand_id', 'item_city_id', 'item_price_level',
#     'item_sales_level', 'item_collected_level', 'item_pv_level',
#     'shop_review_num_level', 'shop_star_level', 'category_1', 'category_2',
#     'property_0', 'property_1', 'property_2',
#     'predict_category_0', 'predict_category_1', 'predict_category_2', 'context_page_id'
# ]
#
# def make_cat(data):
#     for i in category_list:
#         data[i] = data[i].astype('category')
#     del data['category_1_vec']
#     del data['property_0_vec']
#     del data['property_1_vec']
#     del data['property_2_vec']
#     del data['item_id']
#     del data['item_brand_id']
#     vector_list = ['item_id_vec', 'item_brand_id_vec']
#
#     #vector_list = ['category_1_vec', 'property_2_vec', 'property_0_vec', 'property_1_vec', 'item_id_vec', 'item_brand_id_vec']
#     for feat in vector_list:
#         for i in range(50):
#             data[feat+'_'+str(i)] = data[feat].apply(lambda x: x[i])
#         del data[feat]
#     return data
#
# def make_cat2(data):
#     for i in category_list:
#         data[i] = data[i].astype('category')
#     return data
# data = pd.read_pickle("./data/all_data.pkl")
# data = make_cat(data.drop_duplicates(subset='instance_id'))  # 把instance id去重
# print('make feature')
# data['context_hour'] = data['realtime'].dt.hour
# data['context_day'] = data['realtime'].dt.day
#
# train = data[(data['context_day'] >= 18) & (data['context_day'] <= 23)]
# del train['context_hour']
# del train['context_day']
#
# valid = data[(data['context_day'] == 24)]
# del valid['context_hour']
# del valid['context_day']
# best_iter = lgbCV(train,valid)
# print(best_iter)
# #print(train['item_brand_id_vec'].head(5))
# train = data[data.is_trade.notnull()]
# test = data[data.is_trade.isnull()]
# sub(train, test, best_iter)
# print("----------------------------------------------------线上----------------------------------------")
#
# train = data[(data['day'] >= 18) & (data['day'] <= 24)]
# test = data[data['day'] == 25]
#
# sub(train, test, best_iter)
# print(data.is_trade.unique())
# print(test)
#
# test_data = pd.read_csv('./data/test.txt', sep="\s+")
# import time
# def time2cov(value):
#     return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))
# test_data['realtime'] = pd.to_datetime(test_data['context_timestamp'].apply(time2cov))
# test_data['day'] = test_data['realtime'].dt.day
# test_data['hour'] = test_data['realtime'].dt.hour
# print(test_data['day'].head(10))
# print(test_data['day'].unique())

