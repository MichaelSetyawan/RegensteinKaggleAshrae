import numpy as np
import pandas as pd
import os
import datetime
from reduce_memory import *
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn import preprocessing
from datetime import datetime, timedelta
from math import sqrt

def _date_range_parameters(start, end):
    DATE_FORMAT = '%Y-%m-%d'
    start = datetime.strptime(start, DATE_FORMAT)
    end   = datetime.strptime(end, DATE_FORMAT)
    delta = end - start
    day_period = []
    for i in range(delta.days + 1):
        day = start + timedelta(days=i)
        day_period.append(day)
    return day_period


complete_train_all = pd.read_csv('complete_trainV1.csv')#complete_trainV1
complete_test_all  = pd.read_csv('complete_testV1.csv')#complete_testV1

complete_train_all['timestamp'] = pd.to_datetime(complete_train_all['timestamp'])
complete_test_all['timestamp']  = pd.to_datetime(complete_test_all['timestamp'])

#remove bad rows
remove_dates = _date_range_parameters('2016-01-01','2016-05-20')
complete_train_part = complete_train_all[complete_train_all['building_id'] != 1099 ]
filter1 = (complete_train_part['meter']==0)
filter2 = (complete_train_part['timestamp'].isin(remove_dates))
complete_train  = complete_train_part.drop(complete_train_part[filter1 & filter2].index)

complete_train = reduce_memory(complete_train)
complete_test  = reduce_memory(complete_test_all)

categorical=['site_id', 'primary_use', 'meter', 'weekday', 'month', 'is_holiday_break', 'day_hour', 'season']

train_cats = complete_train[categorical]
test_cats  = complete_test[categorical]
num_train  = train_cats.shape[0]
cat_merged = pd.concat([train_cats,test_cats],axis=0)

encode_df = pd.get_dummies(cat_merged, columns=categorical)
X_train = encode_df.iloc[0:num_train,]
X_test  = encode_df.iloc[num_train:,]

drop_cols = categorical+['building_id','timestamp', 'day']
complete_train.drop(drop_cols, axis=1, inplace=True)
complete_test.drop(drop_cols, axis=1, inplace=True)

X_test = X_test.reset_index()
final_train = pd.concat([X_train,complete_train],axis=1)
final_test  = pd.concat([X_test,complete_test],axis=1)

#modeling
X_final_train = final_train.drop(['log_meter_reading'], axis=1)
final_test.drop('index', inplace=True, axis=1)
final_test_wrowid = final_test.drop('row_id',axis=1)
X_final_test      = final_test_wrowid
X_final_train     = X_final_train.values
Y_final_train     = final_train['log_meter_reading'].values

# split train and validation set into 80 and 20 percent sequentially
train_X = X_final_train[:int(4 * X_final_train.shape[0] / 5)]
valid_X = X_final_train[int(4 * X_final_train.shape[0] / 5):]

train_y = Y_final_train[:int(4 * Y_final_train.shape[0] / 5)]
valid_y = Y_final_train[int(4 * Y_final_train.shape[0] / 5):]

def fit_evaluate_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_valid)
    return sqrt(mean_squared_error(y_valid, y_predicted))

updated_params = {
    "objective": "regression",
    "boosting": "dart",
    "num_leaves": 150,
    "learning_rate": 0.05,
    "feature_fraction": 0.97,
    "reg_lambda": 2,
    "metric": "rmse",
    "max_bin": 100,
    "num_iterations":200
}

lgbm_regressor = lgb.LGBMRegressor(boosting_type='gbdt',
num_leaves=31, max_depth=-1, learning_rate=0.01, n_estimators=1000, max_bin=255, subsample_for_bin=50000, objective=None,
min_split_gain=0, min_child_weight=3,min_child_samples=10, subsample=1, subsample_freq=1, colsample_bytree=1, reg_alpha=0.1,
reg_lambda=0, seed=17,silent=False, nthread=-1)
lgbm_rmse = fit_evaluate_model(lgbm_regressor, train_X, train_y, valid_X, valid_y)
print("RMSE of the light gbm regressor is:", lgbm_rmse)

# tranform training and validation set into lgbm datasets
train_dataset = lgb.Dataset(train_X, label=train_y)
valid_dataset = lgb.Dataset(valid_X, label=valid_y)

print("Building model with first 3 quarter pieces and evaluating the model on the last quarter:")
# lgb_model = lgb.train(updated_params,
#                               train_set=train_dataset,
#                               num_boost_round=1000,
#                               valid_sets=[train_dataset, valid_dataset],
#                               verbose_eval=300,
#                               early_stopping_rounds=300)

lgbm_regressor.fit(X_final_train,Y_final_train, eval_metric='rmse',
eval_set = [(valid_X, valid_y)],verbose = True)
y_pred   = lgbm_regressor.predict(X_final_test)

final_test['meter_reading'] = np.expm1(y_pred)

print('feature_important: \n', lgbm_regressor.feature_importances_)

submission = final_test[['row_id', 'meter_reading']]
submission['row_id'] = submission['row_id'].astype('int')
submission.sort_values(by='row_id').to_csv('submissionsQF_lgb.csv', index=False)