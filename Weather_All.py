'''
DATA PREPROCESSING FOR KAGGLE WEATHER DATA

Authors:
    Qinzi Cao
    Yang Xiang
    Reshma Kelkar
    Vincent Lee
    Michael Setyawan
    Manaswi Mishra
'''

#Libraries needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

"""
Code to load and combine weather train and weather test
"""
print('datapath=',os.getcwd(),'\n')
datapath='/Users/michaelsetyawan/Desktop/UCHICAGO/PYTHON/Kaggle/'
train=pd.read_csv(datapath+'/weather_train.csv') #139773 rows
test=pd.read_csv(datapath+'/weather_test.csv') #277243 rows
all=pd.concat([train,test]).reset_index() #417016 rows
#print(train.shape)
#print(test.shape)
#print(all.shape)
#Commented because it takes time, uncomment this to save csv file
#all.to_csv('weather_all.csv',index=False,header=True)
#weather_all=all

"""
Part of code to check missing values
Included function to check missing values
"""
def cal_missing_val(df):
    data_dict = {}
    for col in df.columns:
        data_dict[col] = df[col].isnull().sum()/df.shape[0]*100
    return pd.DataFrame.from_dict(data_dict, orient='index', columns=['MissingValue'])

missing_weather  = cal_missing_val(train)
print('Missing weather columns before imputations: \n')
print(missing_weather,'\n')


"""
Change date for weather_all
Included function to process the data
"""
def change_datetime(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].map(lambda x: x.strftime('%Y')).astype(int)
    df['month'] = df['timestamp'].map(lambda x: x.strftime('%m')).astype(int)
    df['weekday'] = df['timestamp'].map(lambda x: x.weekday()).astype(int)
    df['hour'] = df['timestamp'].map(lambda x: x.hour).astype(int)
    return df
train=change_datetime(train)

missing_values=train.set_index('site_id').isna().sum(level=0)
print('Missing values of weather_all columns grouped by site_id \n')
print(missing_values,'\n')

#Might want to remove site_id 5, because site_id 5 has no sea level pressure data
#weather_all=weather_all[weather_all['site_id']!=5]


"""
Imputing rolling mean weather 
"""
def rolling_mean_weather(df, cols, num_of_hours):
    # assumes that functions will be mean (can include median, max, and min too)
    for col in cols:
        for hour in num_of_hours:
            print("processing missing values for", col, hour)
            df[col + '_' + 'R' + str(hour) + '_' + 'mean'] = \
                df.groupby('site_id')[col].apply(lambda x: x.fillna (x.rolling(hour,center=True, min_periods=1).mean()))
    return df
cols_to_impute = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
                  'wind_direction', 'wind_speed']
train= rolling_mean_weather(train, cols_to_impute, [72])
missing_weather= cal_missing_val(train)
print(missing_weather)


def rev_rolling_mean_weather(df,cols,num_of_hours):
    # assumes that functions will be mean (can include median, max, and min too)
    df[::-1]
    for col in cols:
        for hour in num_of_hours:
            print("processing missing values for", col, hour)
            df[col + '_' + 'R' + str(hour) + '_' + 'mean_reversed'] = \
                df.groupby('site_id')[col].apply(lambda x: x.fillna(x.rolling(hour, center=True, min_periods=1).mean()))
    return df
cols_to_impute = ['cloud_coverage_R72_mean', 'precip_depth_1_hr_R72_mean', 'sea_level_pressure_R72_mean']
weather_all= rev_rolling_mean_weather(train, cols_to_impute, [72])
missing_weather= cal_missing_val(train)
print(missing_weather)

missing_values=train.set_index('site_id').isna().sum(level=0)
print(missing_values.head())
#missing_values2=missing_values[['cloud_coverage_R72_mean','precip_depth_1_hr_R72_mean','sea_level_pressure_R72_mean']]
#missing_values2