'''
Imputation for Weather Missing Values in ASHRAE Data

Authors:
    Qinzi Cao
    Yang Xiang
    Reshma Kelkar
    Vincent Lee
    Michael Setyawan
    Manaswi Mishra
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime

"""
Function to calculate missing values by column
"""
def cal_missing_val(df):
    data_dict = {}
    for col in df.columns:
        data_dict[col] = df[col].isnull().sum()/df.shape[0]*100
    return pd.DataFrame.from_dict(data_dict, orient='index', columns=['MissingValue'])

"""
Function to get empty columns list
"""
def get_empty_cols(df):
    empty_cols = []
    for i in df.columns:
        if df[i].isnull().any() == True:
            empty_cols.append(i)
    return empty_cols

"""
Function to fill site id, used as in intermediate step in the next function
"""
def fill_site_id(val, col, df):
    df_1 = df[df["timestamp"] == val['timestamp']]
    return df_1[col].mean()

"""
Function to impute missing values in weather dataframe
"""
def sort_fill_weather(weather_df):
    # Add new Features regarding
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["timestamp"].dt.day
    weather_df["week"] = weather_df["timestamp"].dt.week
    weather_df["month"] = weather_df["timestamp"].dt.month

    empty_cols = get_empty_cols(weather_df)
    # Method 1 - fill NAs using rolling forward
    if empty_cols:
        for col in empty_cols:
            weather_df[col] = weather_df.groupby(['site_id'])[col].apply(
                lambda x: x.fillna(x.rolling(72, min_periods=1).mean()))

    empty_cols = get_empty_cols(weather_df)
    #Method 2 - fill NAs using rolling backward
    if empty_cols:
        for col in empty_cols:
            weather_df[col] = weather_df.groupby(['site_id'])[col].apply(
                lambda x: x.fillna(x.rolling(72, min_periods=1).mean().shift(-71)))

    empty_cols = get_empty_cols(weather_df)
    # Method 3 - when whole site_id is missing - timestamp-avg column imputation
    if empty_cols:
        for col in empty_cols:
            weather_df[col] = weather_df[["timestamp", col]].apply(
                lambda x: fill_site_id(x, col, weather_df) if pd.isnull(x[col]) else x[col], axis=1)

    return weather_df


