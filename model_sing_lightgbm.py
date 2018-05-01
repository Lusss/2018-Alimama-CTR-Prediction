import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss


def divide_vector(data):
    del data['item_id']
    del data['item_brand_id']
    vector_list = ['item_id_vec', 'item_brand_id_vec', 'property_2_vec', 'property_0_vec', 'property_1_vec']
    for feat in vector_list:
        for i in range(50):
            data[feat+'_'+str(i)] = data[feat].apply(lambda x: x[i])
        del data[feat]
    return data

X_train = pd.read_pickle('./data/X_train.pkl')
X_val = pd.read_pickle('./data/X_val.pkl')
y_train = pd.read_pickle('./data/y_train.pkl')
y_val = pd.read_pickle('./data/y_val.pkl')
# X_train = divide_vector(X_train)
# X_val = divide_vector(X_val)
print(X_train.columns.tolist())
X_train.to_csv('./data/X_train.csv')
#
#
# def sub(train, test, best_iter):
#     col = [c for c in train if
#            c not in ['category_0','is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id','context_id', 'realtime', 'context_timestamp','item_id_vec', 'item_brand_id_vec', 'property_2_vec', 'property_0_vec', 'property_1_vec']]
#     X = train[col]
#     y = train['is_trade'].values
#     print('Training LGBM model...')
#     lgb0 = lgb.LGBMClassifier(
#         objective='binary',
#         # metric='binary_error',
#         metric='binary_logloss',
#         num_leaves=128,
#         max_depth=8,
#         learning_rate=0.05,
#         seed=2018,
#         colsample_bytree=0.8,
#         # min_child_samples=8,
#         subsample=0.9,
#         n_estimators=best_iter)
#     lgb_model = lgb0.fit(X, y)
#     predictors = [i for i in X.columns]
#     feat_imp = pd.Series(lgb_model.feature_importances_, predictors).sort_values(ascending=False)
#     print(feat_imp)
#     print(feat_imp.shape)
#     # pred= lgb_model.predict(test[col])
#     pred = lgb_model.predict_proba(test[col])[:, 1]
#     test['predicted_score'] = pred
#     sub1 = test[['instance_id', 'predicted_score']]
#     sub=pd.read_csv("./data/test.txt", sep="\s+")
#     sub=pd.merge(sub,sub1,on=['instance_id'],how='left')
#     sub=sub.fillna(0)
#     #sub[['instance_id', 'predicted_score']].to_csv('result/result0320.csv',index=None,sep=' ')
#     sub[['instance_id', 'predicted_score']].to_csv('result/result0420.txt',sep=" ",index=False)
#
#
# def lgbCV(X_train,y_train,X_val,y_val):
#     col = [c for c in X_train if
#            c not in ['category_0','is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id','context_id', 'realtime', 'context_timestamp', 'category_1_vec', 'property_2_vec', 'item_id_vec', 'property_0_vec', 'item_brand_id_vec', 'property_1_vec','smoothctr_user_id']]
#     # cat = ['sale_price', 'gender_star', 'user_age_level', 'item_price_level', 'item_sales_level', 'sale_collect',
#     #        'price_collect', 'item_brand_id', 'user_star_level', 'item_id', 'shop_id',
#     #        'item_city_id', 'context_page_id', 'gender_age', 'shop_star_level', 'item_pv_level', 'user_occupation_id',
#     #        'day', 'gender_occ', 'user_gender_id']
#     # X_train = train[col]
#     # y_train = train['is_trade'].values
#     # X_val = test[col]
#     # y_val = test['is_trade'].values
#     X_train = X_train[col]
#     X_val = X_val[col]
#     print(X_train.columns.tolist())
#     print(list(set((X_train.columns).tolist())-set((X_val.columns).tolist())))
#     print('Training LGBM model...')
#     lgb0 = lgb.LGBMClassifier(
#         objective='binary',
#         metric= 'binary_logloss',
#         is_training_metric=True,
#         #num_leaves=64,
#         num_leaves=64,
#         max_depth=8,
#         #learning_rate=0.05,
#         learning_rate=0.05,
#         seed=2018,
#         colsample_bytree=0.8,
#         # min_child_samples=8,
#         subsample=0.9,
#         n_estimators=1000,
#         #scale_pos_weight = 10
#     )
#     lgb_model = lgb0.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=200)
#     best_iter = lgb_model.best_iteration_
#     predictors = [i for i in X_train.columns]
#     feat_imp = pd.Series(lgb_model.feature_importances_, predictors).sort_values(ascending=False)
#     print(feat_imp)
#     print(feat_imp.shape)
#     # pred= lgb_model.predict(test[col])
#     pred = lgb_model.predict_proba(X_val[col])[:, 1]
#     X_val['pred'] = pred
#     X_val['index'] = range(len(X_val))
#     # print(test[['is_trade','pred']])
#     print('误差 ', log_loss(y_val, X_val['pred']))
#     return best_iter
#
# best_iter = lgbCV(X_train,y_train,X_val,y_val)
# print(best_iter)