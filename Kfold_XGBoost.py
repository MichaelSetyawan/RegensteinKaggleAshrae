#!/bin/bash
# XGboost

import numpy as np
import pandas as pd
from numpy import loadtxt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime, timedelta
import math
from sklearn.metrics import mean_squared_error


complete_test= pd.read_csv('complete_testV1.csv')
complete_train= pd.read_csv('complete_trainV1.csv')

def reduce_memory(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

complete_test= reduce_memory(complete_test)
complete_train= reduce_memory(complete_train)

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

# convert timestamp to datetime format
complete_train['timestamp'] = pd.to_datetime(complete_train['timestamp'])
complete_test['timestamp']  = pd.to_datetime(complete_test['timestamp'])

print('converted timestamp')

# remove bad rows
remove_dates = _date_range_parameters('2016-01-01', '2016-05-20')
complete_train_part = complete_train[complete_train['building_id'] != 1099 ]
filter1 = (complete_train_part['meter']==0)
filter2 = (complete_train_part['timestamp'].isin(remove_dates))
complete_train = complete_train_part.drop(complete_train_part[filter1 & filter2].index)

print('bad rows removed')


categorical=['weekday', 'month',
             'day_hour', 'primary_use', 'site_id','meter']

train_cats = complete_train[categorical]
test_cats  = complete_test[categorical]
num_train  = train_cats.shape[0]
cat_merged = pd.concat([train_cats, test_cats],axis=0)
print('merged')

#one-hot encoding

encode_df = pd.get_dummies(cat_merged, columns=categorical)

#Split data
X_train = encode_df.iloc[0:num_train,]
X_test  = encode_df.iloc[num_train:,]
#print (X_train.shape, X_test.shape)

drop_cols = categorical +['timestamp', 'day',
                          'week', 'floor_count',
                          'is_holiday_break', 'season']
complete_train.drop(drop_cols, axis=1, inplace=True)
complete_test.drop(drop_cols, axis=1, inplace=True)

X_test = X_test.reset_index()

final_train = pd.concat([X_train,complete_train],axis=1)
final_test  = pd.concat([X_test,complete_test],axis=1)

final_train_x     = final_train[final_train['log_meter_reading']!=0]
X_final_train     = final_train_x
final_test_wrowid = final_test.drop('row_id',axis=1)
X_final_test      = final_test_wrowid

print('Finished with one hot encoding')

X = X_final_train.drop(['log_meter_reading'], axis=1)
Y = X_final_train[['log_meter_reading']]

#X_test = X_final_test
X_final_test=X_final_test.drop(['index'], axis=1)

# kfolds
folds=2
kf = StratifiedKFold(n_splits=folds, shuffle=False)

def Kfoldscores(train):
    models = []
    rmsescores = []
    for train_index, valid_index in kf.split(train, train['building_id']):
        # splitting train and validation using building id to ensure good representation of data
        print('kfold starting')
        kfold_train = train.iloc[train_index].drop('building_id', axis=1)
        kfold_train_x = kfold_train.drop('log_meter_reading', axis=1)
        kfold_train_y = kfold_train['log_meter_reading']
        kfold_valid = train.iloc[valid_index].drop('building_id', axis=1)
        kfold_valid_x = kfold_valid.drop('log_meter_reading', axis=1)
        kfold_valid_y = kfold_valid['log_meter_reading']

        # run cross validation on XGBRegressor
        reg = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bytree=0.8, learning_rate=0.07, max_depth=7, min_child_weight=4,
                    n_estimators=250, n_jobs=1, nthread=4, objective='reg:linear',
                    reg_lambda=1, scale_pos_weight=1, silent=True, subsample=0.8)
        print("Model run complete")
        reg.fit(kfold_train_x, kfold_train_y, verbose=False)
        print("Model fit complete")
        rmsescores.append(rmse(kfold_valid_y, reg.predict(kfold_valid_x)))
        models.append(reg)

        del kfold_train, kfold_valid

    return models, rmsescores

def rmse(y, y_pred):
    return mean_squared_error(y, y_pred.clip(0))

# place final train as train, nestimators (no of trees) =30
# print out the rmse scores
models, rmsescores = Kfoldscores(X_final_train)

print("RMSE", rmsescores)

# function to predict meter reading using cross validation model ensemble

def prediction(test_X):
    preds = np.expm1(sum([model.predict(X_final_test.drop('building_id',axis=1)) for model in models])/folds)
    return preds.tolist()

# just to generate results
results = prediction(X_final_test)
print("prediction is done")

final_test['meter_reading'] = results

submission = final_test[['row_id', 'meter_reading']]
submission['row_id'] = submission['row_id'].astype('int')
submission.sort_values(by='row_id').to_csv('submissionsKfoldXGB.csv', index=False)