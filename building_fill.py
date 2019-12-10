"""
Imputation for Building Missing Values in ASHRAE Data

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

"""
Function to calculate missing values by column
"""
def cal_missing_val(df):
    data_dict = {}
    for col in df.columns:
        data_dict[col] = df[col].isnull().sum()/df.shape[0]*100
    return pd.DataFrame.from_dict(data_dict, orient='index', columns=['MissingValue'])

"""
Function to impute missing values in building_metadata dataframe
"""
def sort_fill_building(building_df):
    building_part = building_df[['building_id', 'site_id', 'primary_use', 'square_feet', 'floor_count']]
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
    building_model = model.fit(X_train, Y_train)
    y_pred = building_model.predict(X_test)
    X_test['floor_count'] = np.around(y_pred).astype(int)
    combine_data = pd.concat([train_, X_test], axis=0)
    building_noNA = pd.merge( \
        building_df[['site_id', 'building_id', 'primary_use', 'square_feet', 'year_built']],\
        combine_data[['floor_count']], how='inner', left_index=True, right_index=True)
    building = building_noNA.drop(['year_built'], axis=1)
    return building
