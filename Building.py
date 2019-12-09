"""
DATA PREPROCESSING FOR KAGGLE BUILDING DATA

Authors:
    Qinzi Cao
    Yang Xiang
    Reshma Kelkar
    Vincent Lee
    Michael Setyawan
    Manaswi Mishra
"""

#Libraries needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer

print('datapath=',os.getcwd())
datapath="/Users/michaelsetyawan/Desktop/UCHICAGO/PYTHON/Kaggle"
building=pd.read_csv(datapath+'/building_metadata.csv')

"""
Function for missing values
"""
def cal_missing_val(df):
    data_dict = {}
    for col in df.columns:
        data_dict[col] = df[col].isnull().sum()/df.shape[0]*100
    return pd.DataFrame.from_dict(data_dict, orient='index', columns=['MissingValue'])

missing_building=cal_missing_val(building)
print('Before imputation')
print(missing_building,'\n')

"""
Imputation of floor_count column using Linear Regression
"""
building_part = building[['building_id', 'site_id', 'primary_use', 'square_feet', 'floor_count']]
cols = ['primary_use', 'site_id']
encode_df = pd.DataFrame()
for col in cols:
    lb_style = LabelBinarizer()
    lb_results = lb_style.fit_transform(building_part[col])
    label_code = pd.DataFrame(lb_results, columns=lb_style.classes_)
    encode_df = pd.concat([encode_df, label_code], axis=1)

building_encode = pd.concat([encode_df, building_part[['building_id', 'square_feet', 'floor_count']]], axis=1)
building_encode = building_encode.set_index('building_id')
test_ = building_encode[building_encode['floor_count'].isnull()]
train_ = building_encode[building_encode['floor_count'].notnull()]
model = LinearRegression()
X_test = test_.drop(['floor_count'], axis=1)
X_train = train_.iloc[:, :-1].values
Y_train = train_['floor_count'].values
building_model = model.fit(X_train, Y_train)
y_pred = building_model.predict(X_test)
X_test['floor_count'] = np.around(y_pred).astype(int)
combine_data = pd.concat([train_, X_test], axis=0)
building_noNA = pd.merge( \
    building[['site_id', 'building_id', 'primary_use', 'square_feet', 'year_built']],
    combine_data[['floor_count']], how='inner', left_index=True, right_index=True)

print('After imputation \n')
missing_building=cal_missing_val(building_noNA)
print(missing_building,'\n')

building=building_noNA

plt.scatter(building['site_id'],building['year_built'])
plt.show()
building=building.drop(['year_built'],axis=1)
#drop year_built because it has more than 50% missing values and some site_ids year_built are completely blank
print(building.head())
#Uncomment code below to save into csv file
#building.to_csv('building_done.csv',index=False,header=True)