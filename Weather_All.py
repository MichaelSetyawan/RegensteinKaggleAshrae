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
import datetime

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
Sort and fill weather
"""
def sort_fill_weather(weather_df):
    #Code similar to Bojan Tunguz's
    #0. find missing hours and reset
    #1. sort and reindex by site_id/day/month
    #2. fill the missing NA values by day means
    #3. fill the missing NA values by ffill
    #4. fill the missing NA values by bfill

    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(), time_format)
    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(), time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = [] #list for missing hours
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list, site_hours), columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df, new_rows])

        weather_df = weather_df.reset_index(drop=True)

    # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month
    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id', 'day', 'month'])

    # Checking with columns are empty
    empty_cols = []
    for i in weather_df.columns:
        if weather_df[i].isnull().any() == True:
            empty_cols.append(i)

    # Fill using day mean
    for col in empty_cols:
        col_filler=pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])[col].mean(), columns=[col])
        weather_df.update(col_filler,overwrite=False)

    # Checking again which columns are still empty
    empty_cols = []
    for i in weather_df.columns:
        if weather_df[i].isnull().any() == True:
            empty_cols.append(i)

    # Fill using ffill day mean
    for col in empty_cols:
        col_filler = weather_df.groupby(['site_id', 'day', 'month'])[col].mean()
        col_filler = pd.DataFrame(col_filler.fillna(method='ffill'), columns=[col])
        weather_df.update(col_filler, overwrite=False)

    # Checking again which columns are still empty
    empty_cols = []
    for i in weather_df.columns:
        if weather_df[i].isnull().any() == True:
            empty_cols.append(i)

    # Fill using bfill day mean
    for col in empty_cols:
        col_filler = weather_df.groupby(['site_id', 'day', 'month'])[col].mean()
        col_filler = pd.DataFrame(col_filler.fillna(method='bfill'), columns=[col])
        weather_df.update(col_filler, overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime', 'day', 'week', 'month'], axis=1)

    return weather_df

train=sort_fill_weather(train)
missing_weather  = cal_missing_val(train)
print('i')









