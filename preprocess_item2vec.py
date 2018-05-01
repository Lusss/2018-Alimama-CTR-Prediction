#coding:utf-8
# NAME : LIN He 2018/3/31
# Hong kong university of science and technology

# 0 . Description : Preprocess the training and testing data, including data washing and feature extraction.
# 1 . Given features:
# instance_id 样本编号
#######################################
# item_id 广告商品编号
# item_category_list 广告商品的的类目列表 分割; item_property_list_0 item_property_list_1 item_property_list_2
# item_property_list 广告商品的属性列表 分割 1 2 3
# item_brand_id 广告商品的品牌编号
# item_city_id 广告商品的城市编号
# item_price_level 广告商品的价格等级
# item_sales_level 广告商品的销量等级
# item_collected_level 广告商品被收藏次数的等级
# item_pv_level 广告商品被展示次数的等级
#######################################
# user_id 用户的编号
# 'user_gender_id', 用户的预测性别编号
# 'user_age_level', 用户的预测年龄等级
# 'user_occupation_id', 用户的预测职业编号
# 'user_star_level' 用户的星级编号
#######################################
# context_id 上下文信息 的编号
#  context_timestamp 广告商品的展示时间
# context_page_id 广告商品的展示页面编号
# predict_category_property

# 2 . Manual construct features
# user_behaviour_series 用户行为序列([商品类别，商品价格，广告商品的展示时间]...）->时间序列模型stack 输出embedded features.（预训练）
# * 假设用户的行为在短时间内存在相关性
# user_age_  用户的购买率 : 用户总浏览量 / 用户购买量（可以考虑的其他features,年龄的购买率，）
#




import numpy as np
import pandas as pd
import time
#from item2Vec import
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import log_loss


from gensim.models import Word2Vec
# ？`Need to construct a item2Vec embedding model that output is both neighbour elements and the is_trade values of it.

# class DataPreprocess{
#
# }
def time2cov(value):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))

def map_hour(x):
    if (x>=7)&(x<=12):
        return 1
    elif (x>=13)&(x<=20):
        return 2
    else:
        return 3


def slide_cnt(data):
    # item_cnt = data.groupby(by='item_id').count()['instance_id'].to_dict()
    # data['item_cnt'] = data['item_id'].apply(lambda x: item_cnt[x])
    # user_cnt = data.groupby(by='user_id').count()['instance_id'].to_dict()
    # data['user_cnt'] = data['user_id'].apply(lambda x: user_cnt[x])
    # shop_cnt = data.groupby(by='shop_id').count()['instance_id'].to_dict()
    # data['shop_cnt'] = data['shop_id'].apply(lambda x: shop_cnt[x])

    print('当前日期前一天的cnt')
    for d in range(19, 26):  # 18到24号
        df1 = data[data['day'] == d - 1]
        df2 = data[data['day'] == d]  # 19到25号
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
        df2['user_cnt1'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cnt1'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cnt1'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
        df2 = df2[['user_cnt1', 'item_cnt1', 'shop_cnt1', 'instance_id']]
        if d == 19:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')
    print('当前日期之前的cnt')
    for d in range(19, 26):
        # 19到25，25是test
        df1 = data[data['day'] < d]
        df2 = data[data['day'] == d]
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
        df2['user_cntx'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cntx'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cntx'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
        df2 = df2[['user_cntx', 'item_cntx', 'shop_cntx', 'instance_id']]
        if d == 19:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')

    print("前一个小时的统计量")

    return data

def convert_data(data):
    data['time'] = data.context_timestamp.apply(time2cov)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    return data

def dataClean(raw_data):
    print("###############  Cleaning & Encoding categorical data to one-hot expression   ################")
    le = LabelEncoder()
    print("###############  item  ###############")
    for i in range(3):
        raw_data['category_%d' % (i)] = raw_data['item_category_list'].apply(
            lambda x: str(x.split(";")[i]) if len(x.split(";")) > i else " ")
    del raw_data['item_category_list']
    for i in range(3):
        raw_data['property_%d' % (i)] = raw_data['item_property_list'].apply(
            lambda x: str(x.split(";")[i]) if len(x.split(";")) > i else " ")
    del raw_data['item_property_list']

    for col in ['item_id', 'item_brand_id', 'item_city_id']:
        raw_data[col] = raw_data[col].apply(lambda x: str(x))

    #raw_data['context_timestamp'] = raw_data['context_timestamp'].apply(time2cov)
    raw_data['realtime'] = raw_data['context_timestamp'].apply(time2cov)

    for i in range(3):
        raw_data['predict_category_%d' % (i)] = raw_data['predict_category_property'].apply(
            lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " ")

    print("###############  user  ###############")
    for col in ['user_id']:
        raw_data[col] = le.fit_transform(raw_data[col])
    print('user 0,1 feature')
    raw_data['gender0'] = raw_data['user_gender_id'].apply(lambda x: 1 if x == -1 else 2)
    raw_data['age0'] = raw_data['user_age_level'].apply(lambda x: 1 if x == 1004 | x == 1005 | x == 1006 | x == 1007  else 2)
    raw_data['occupation0'] = raw_data['user_occupation_id'].apply(lambda x: 1 if x == -1 | x == 2003  else 2)
    raw_data['star0'] = raw_data['user_star_level'].apply(lambda x: 1 if x == -1 | x == 3000 | x == 3001  else 2)

    print("###############  context  ###############")
    raw_data['realtime'] = pd.to_datetime(raw_data['realtime'])
    # raw_data['day'] = raw_data['realtime'].dt.day
    # raw_data['hour'] = raw_data['realtime'].dt.hour

    #raw_data['hour'] = raw_data['realtime'].dt.hour.apply(map_hour)
    raw_data['len_predict_category_property'] = raw_data['predict_category_property'].map(lambda x: len(str(x).split(';')))

    print('context 0,1 feature')
    raw_data['context_page0'] = raw_data['context_page_id'].apply(
        lambda x: 1 if x == 4001 | x == 4002 | x == 4003 | x == 4004 | x == 4007  else 2)

    print("###############  shop  ###############")
    for col in ['shop_id']:
        raw_data[col] = le.fit_transform(raw_data[col])
    raw_data['shop_score_delivery0'] = raw_data['shop_score_delivery'].apply(lambda x: 0 if x <= 0.98 and x >= 0.96  else 1)


    return raw_data


#归一化(价格与最低价差值占总价格的百分比)
def price_normalization(df):
    #return (df['item_price_level'] - df['item_price_level'].min()) / df['item_price_level'].mean()
    df['item_norm_price_score'] = (df['item_price_level'] - df['item_price_level'].min()) / df['item_price_level'].mean()
    del df['item_price_level']
    return df


def featureExtraction(raw_data):
    print("###############  Extracting Training Features   ################")
    #user action series

    #raw_data = raw_data.sort_values(by=['context_timestamp'])
    raw_data = raw_data.sort_values(by=['realtime'])

    print(raw_data.columns)

    # 商品的价格在同类商品中的比值
    print('一个item在同类商品中item_price_level的归一化分数…(在同类商品价格中的分布)(查询词的预测cat0_1_2相同 if( != ' '))')
    #item_norm_price_score = raw_data.groupby(['predict_category_0', 'predict_category_1','predict_category_2'], as_index=False)['item_price_level'].apply({'item_norm_price_score': price_normalization})
    # f = {x: x for x in ['item_id','item_price_level']}
    # f['item_price_level'] = lambda g: g.apply(price_normalization
    item_norm_price_score = raw_data.groupby(['predict_category_0', 'predict_category_1', 'predict_category_2'], as_index=False)['item_id','item_price_level'].apply(price_normalization)
    print(len(item_norm_price_score))
    print(len(raw_data))
    raw_data['item_norm_price_score'] = item_norm_price_score['item_norm_price_score']
    #raw_data = pd.merge(raw_data, item_norm_price_score,on = ['item_id'],how='inner')

    print('一个user有多少item_id,item_brand_id……')
    user_cnt = raw_data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    raw_data = pd.merge(raw_data, user_cnt, on=['user_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = raw_data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg({str(col)+'_user_cnt': 'count'})
        raw_data = pd.merge(raw_data, item_shop_cnt, on=[col, 'user_id'], how='left')
        raw_data[str(col) + '_user_prob'] = raw_data[str(col) + '_user_cnt'] / raw_data['user_cnt']
    del raw_data['user_cnt']
    #
    # print('一个user_gender有多少shop_id,shop_review_num_level……')
    # for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
    #     item_shop_cnt = raw_data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg(
    #         {str(col) + '_user_gender_cnt': 'count'})
    #     raw_data = pd.merge(raw_data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
    #     raw_data[str(col) + '_user_gender_prob'] = raw_data[str(col) + '_user_gender_cnt'] / raw_data['user_gender_cnt']
    # del raw_data['user_gender_cnt']

    print('一个shop有多少item_id,item_brand_id,item_city_id,item_price_level……')
    itemcnt = raw_data.groupby(['shop_id'], as_index=False)['instance_id'].agg({'shop_cnt': 'count'})
    raw_data = pd.merge(raw_data, itemcnt, on=['shop_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = raw_data.groupby([col, 'shop_id'], as_index=False)['instance_id'].agg({str(col)+'_shop_cnt': 'count'})
        raw_data = pd.merge(raw_data, item_shop_cnt, on=[col, 'shop_id'], how='left')
        raw_data[str(col) + '_shop_prob'] = raw_data[str(col) + '_shop_cnt'] / raw_data['shop_cnt']
    del raw_data['shop_cnt']



    # print('一个item在同个品牌商品中item_price_level的归一化分数…(在同品牌商品价格中的分布（所处于的位置）)')
    # item_norm_price_score = raw_data.groupby('item_brand_id', as_index=False)['item_price_level'].apply({'item_norm_price_score': price_normalization})
    # raw_data = pd.merge(raw_data, item_norm_price_score, on=['item_price_level'], how='left')

    # print('一个item在同类商品中item_sales_level的归一化分数…')
    # item_norm_sales_score = raw_data.groupby(['predict_category_0', 'predict_category_1','predict_category_2'], as_index=False)['item_sales_level'].apply({'item_norm_sales_score': price_normalization})
    # raw_data = pd.merge(raw_data, item_norm_sales_score, on=['item_sales_level'], how='left')

    print('一个brand的平均价格指数（平均值，方差，衡量品牌的类型）…')

    brand_price_mean = raw_data.groupby('item_brand_id', as_index=False)['item_price_level'].agg({'brand_price_mean': 'mean'})
    brand_price_std = raw_data.groupby('item_brand_id', as_index=False)['item_price_level'].agg({'brand_price_std': 'std'})
    raw_data = pd.merge(raw_data, brand_price_mean, on=['item_brand_id'], how='left')
    raw_data = pd.merge(raw_data, brand_price_std, on=['item_brand_id'], how='left')


    print("###############  Constructing user_items_shop feature series  ################")
    # 用每个用户的浏览（购买）序列来编码 每个商品的embedding vector，以及每个店铺的embedding vector
    # 1. 构造每个用户的购买序列data frame (过滤掉浏览记录过少的用户)
    user_list = raw_data['user_id'].unique()
    item_list = raw_data['item_id'].unique()

    seq_item_id = [[] for i in range(len(user_list))]
    seq_item_brand_id = [[] for i in range(len(user_list))]
    seq_shop_id = [[] for i in range(len(user_list))]

    #seq_item_category_0 = [[] for i in range(len(user_list))]
    seq_item_category_1 = [[] for i in range(len(user_list))]
    #seq_item_category_2 = [[] for i in range(len(user_list))]

    seq_item_property_0 = [[] for i in range(len(user_list))]
    seq_item_property_1 = [[] for i in range(len(user_list))]
    seq_item_property_2 = [[] for i in range(len(user_list))]


    for i in range(len(user_list)):
        user_instance_series = raw_data.loc[raw_data['user_id'] == user_list[i]]
        seq_item_id[i] = user_instance_series['item_id'].tolist()
        seq_item_brand_id[i] = user_instance_series['item_brand_id'].tolist()
        #seq_shop_id[i] = user_instance_series['shop_id'].tolist()
        #seq_item_category_0[i] = user_instance_series['category_0'].tolist()
        seq_item_category_1[i] = user_instance_series['category_1'].tolist()
        #seq_item_category_2[i] = user_instance_series['category_2'].tolist()

        seq_item_property_0[i] = user_instance_series['property_0'].tolist()
        seq_item_property_1[i] = user_instance_series['property_1'].tolist()
        seq_item_property_2[i] = user_instance_series['property_2'].tolist()
        if(i%5000 == 0):
            print('%.2f%%' % (i/len(user_list) * 100))


    #train the embedding layer : output item2vec dense features
    print("###############  Training the Embedding vector of user items series  ################")
    spare_features = {'item_id','item_brand_id','category_1','property_0','property_1','property_2'}
    seq_data = {'item_id':seq_item_id,'item_brand_id':seq_item_brand_id,'category_1':seq_item_category_1,'property_0':seq_item_property_0,'property_1':seq_item_property_1,'property_2':seq_item_property_2}
    for feat in spare_features:
        if(feat == "category_1"):
            # raw_data[feat + '_vec'] = item2vecTraining2(feat, seq_data, raw_data[feat].unique(), size=5, window=3, min_count=1, workers=1, iter=3, sample=1e-4,negative=20)
            raw_data=raw_data.merge(item2vecTraining2(feat, seq_data, raw_data[feat].unique(), size=5, window=3, min_count=1, workers=1, iter=5, sample=1e-4,negative=20), left_on=feat,right_index=True,how='left')
#iter=3 min_count=1
        else:
            # raw_data[feat + '_vec'] = item2vecTraining2(feat, seq_data, raw_data[feat].unique(), size=50, window=3, min_count=1, workers=1, iter=3, sample=1e-4,negative=20)
            raw_data=raw_data.merge(item2vecTraining2(feat, seq_data, raw_data[feat].unique(), size=50, window=3, min_count=1, workers=1, iter=8, sample=1e-4,negative=20), left_on=feat,right_index=True,how='left')
    return raw_data


def item2vecTraining(feat,seq_data,item_list,**params):
    model = Word2Vec(seq_data[feat], **params)
    # 训练skip-gram模型; 默认window=5
    weights = model.wv.syn0
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    np.save("./embedded_weight/"+feat+".npy", weights)
    np.save("./embedded_weight/vocab_map/"+feat+"_map.npy",vocab)
    del model

def item2vecTraining2(feat,seq_data,key_list,**params):
    model = Word2Vec(seq_data[feat], **params)
    # 训练skip-gram模型; 默认window=5
    item_vec_dict = {k : model[k] for k in key_list}
    item_vec_series = pd.Series(item_vec_dict)
    print(item_vec_series)
    del model
    return item_vec_series.to_frame(name=feat + '_vec')

#def userActionSeries(raw_data):

def readData():
    train_data = dataClean(pd.read_csv('./data/train.txt', sep="\s+"))
    test_data = dataClean(pd.read_csv('./data/test.txt', sep="\s+"))
    all_data = featureExtraction(pd.concat([train_data, test_data]).reset_index(drop=True))
    #train_data = featureExtraction(train_data)  # 把instance id去重
    #all_data = featureExtraction(all_data) # 把instance id去重
    print(type(all_data['item_id'].unique()))

    #np.save("./data/all_data.npy",all_data)
    all_data.to_pickle('./data/all_data.pkl')
    # print(train_data.groupby(['predict_category_0', 'predict_category_1','predict_category_2'], as_index=False)['item_price_level'].apply(price_normalization))
    # print(all_data.groupby(['predict_category_0', 'predict_category_1', 'predict_category_2'], as_index=False)[
    #               'item_price_level'].apply(price_normalization))
    # print(sum(train_data.groupby(['predict_category_0', 'predict_category_1','predict_category_2'], as_index=False)['item_price_level'].apply(price_normalization) != 0))
    # print(train_data)

    return all_data


# majority instance is not trade : has sample imblance problem
readData()

