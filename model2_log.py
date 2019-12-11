import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from building_fill import *
from weather_fill import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import datetime

complete_train = pd.read_csv('complete_train.csv')
complete_test  = pd.read_csv('complete_test.csv')
print (cal_missing_val(complete_train))
# complete_train['timestamp']  = pd.to_datetime(complete_train['timestamp'])
# complete_test['timestamp']   = pd.to_datetime(complete_test['timestamp'])
# complete_train['weekday']    = complete_train['timestamp'].dt.weekday
# complete_test['weekday']   = complete_test['timestamp'].dt.weekday
#complete_train['is_weekday']=complete_train['weekday'].apply(lambda x: 1 if x in [0,1,2,3,4] else 0)
#complete_test['is_weekday']=complete_test['weekday'].apply(lambda x: 1 if x in [0,1,2,3,4] else 0)

categorical=['site_id', 'primary_use', 'meter', 'weekday', 'month']

train_cats = complete_train[categorical]
test_cats  = complete_test[categorical]
num_train  = train_cats.shape[0]
cat_merged=pd.concat([train_cats,test_cats],axis=0)

#one-hot encoding
# encode_df = pd.DataFrame()

# for col in categorical:
#         enc = OneHotEncoder(handle_unknown='ignore')
#         X = cat_merged[[col]]
#         enc.fit(X)
#         df_cols = [col + '_' + str(cat_n) for cat_n in enc.categories_[0]]
#         print (df_cols)
#         label_code = pd.DataFrame(enc.transform(X).toarray(), columns=df_cols)
#         encode_df = pd.concat([encode_df, label_code], axis=1)
encode_df = pd.get_dummies(cat_merged, columns=categorical)
X_train=encode_df.iloc[0:num_train,]
X_test=encode_df.iloc[num_train:,]

drop_cols = categorical +['building_id','timestamp' ]
complete_train.drop(drop_cols, axis=1, inplace=True)
complete_test.drop(drop_cols, axis=1, inplace=True)

final_train = pd.concat([X_train,complete_train],axis=1,ignore_index=False)
final_test  = pd.concat([X_test,complete_test],axis=1,ignore_index=False)

final_test_norowid =final_test.drop('row_id',axis=1)

print('Finished with one hot encoding')

#modeling
final_train_x = final_train[final_train['meter_reading']!=0]
X_final_train = final_train_x.drop(['meter_reading'], axis=1)
X_final_train = X_final_train.values
Y_final_train = final_train_x['meter_reading'].values
Y_final_train_log = np.log10(Y_final_train)
X_final_test  = final_test_norowid.values

# m_train, m_test, y_train, y_test = train_test_split(X_final_train, Y_final_train,shuffle=False)
model         = LinearRegression()
model_train_2 = model.fit(X_final_train, Y_final_train_log)
y_pred_log = model_train_2.predict(X_final_test)
y_pred_antilog = 10 ** y_pred_log

# print(cross_val_score(model_train,))
final_test['meter_reading'] = y_pred_antilog

# regression coefficients
print('Coefficients: \n', model_train_2.coef_)

submission = final_test[['row_id', 'meter_reading']]
submission['row_id'] = submission['row_id'].astype('int')
submission.sort_values(by='row_id').to_csv('submissionsQF.csv')