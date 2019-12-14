# Kfold cross validation ensemble on random forest
# Kaggle competition: Ashrae
# Team: QFollowers

# Note
# to run in midway2, you need to pip install tqdm --USER and pip install datetime --USER

import numpy as np
import pandas as pd
import os
import datetime
from datetime import datetime, timedelta
from building_fill import *
from weather_fill import *
from reduce_memory import *
from scipy import stats
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor

# define date parameters

def _date_range_parameters(start, end):
    DATE_FORMAT = '%Y-%m-%d'
    start = datetime.datetime.strptime(start, DATE_FORMAT)
    end   = datetime.datetime.strptime(end, DATE_FORMAT)
    delta = end - start
    day_period = []
    for i in range(delta.days + 1):
        day = start + timedelta(days=i)
        day_period.append(day)
    return day_period

# read files
complete_train = pd.read_csv('complete_trainV1.csv')
complete_test = pd.read_csv('complete_testV1.csv')

# reduce file memory
reduce_memory(complete_train)
reduce_memory(complete_test)

# convert timestamp to datetime format
complete_train['timestamp'] = pd.to_datetime(complete_train['timestamp'])
complete_test['timestamp']  = pd.to_datetime(complete_test['timestamp'])

# remove bad rows
remove_dates = _date_range_parameters('2016-01-01', '2016-05-20')
complete_train_part = complete_train[complete_train['building_id'] != 1099 ]
filter1 = (complete_train_part['meter']==0)
filter2 = (complete_train_part['timestamp'].isin(remove_dates))
complete_train = complete_train_part.drop(complete_train_part[filter1 & filter2].index)

# merge complete train and test files
categorical=['site_id', 'primary_use', 'meter', 'weekday', 'month', 'is_holiday_break',
             'day_hour', 'season']

train_cats = complete_train[categorical]
test_cats = complete_test[categorical]
num_train = train_cats.shape[0]
cat_merged = pd.concat([train_cats,test_cats],axis=0)

# one-hot encoding
encode_df = pd.get_dummies(cat_merged, columns=categorical)
print('Finished with one hot encoding')

# Split data
X_train = encode_df.iloc[0:num_train, ]
X_test = encode_df.iloc[num_train:, ]
print(X_train.shape, X_test.shape)

drop_cols = categorical +['timestamp', 'day']
complete_train2 = complete_train.drop(drop_cols, axis=1, inplace=True)
complete_test2 = complete_test.drop(drop_cols, axis=1, inplace=True)

X_test = X_test.reset_index()
final_train = pd.concat([X_train, complete_train], axis=1)
final_test = pd.concat([X_test, complete_test], axis=1)

final_test2 = final_test.drop('index', axis=1)
final_test_wrowid = final_test2.drop('row_id', axis=1)
X_final_test = final_test_wrowid

print('Finished with final train and test')

# kfolds
folds=5  # general is 5
kf = StratifiedKFold(n_splits=folds, shuffle=False, random_state=50)


# cross validation ensemble for random forest

def Kfoldscores(train, n_estimators, max_features, max_depth, random_state):
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

        # run cross validation on random forest
        reg = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,
                                    max_depth=max_depth, random_state=random_state)
        reg.fit(kfold_train_x, kfold_train_y)
        rmsescores.append(rmse(kfold_valid_y, reg.predict(kfold_valid_x)))
        models.append(reg)

        del kfold_train, kfold_valid

    return models, rmsescores

def rmse(y, y_pred):
    return mean_squared_error(y, y_pred.clip(0))

# place final train as train, nestimators (no of trees) =10, maxfeatures =50, max_depth 10
# print out the rmse scores
models, rmsescores = Kfoldscores(final_train, n_estimators= 10, max_features=50, max_depth= 10, random_state=50)

print("RMSE", rmsescores)

# function to predict meter reading using cross validation model ensemble

def prediction(test_X):
    preds = np.expm1(sum([model.predict(X_final_test.drop('building_id',axis=1)) for model in models])/folds)
    return preds.tolist()

# just to generate results
results = prediction(X_final_test)
print("prediction is done")

# save results as per sample submission
final_test2['meter_reading'] = results
submission = final_test2[['row_id', 'meter_reading']]
submission['row_id'] = submission['row_id'].astype('int')
submission.sort_values(by='row_id').to_csv('submissionsQF_kfoldrf.csv',
                                           index=False)


