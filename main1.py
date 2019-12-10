'''
ASHRAE Kaggle Merging, Function Engineering and Model 1
Use 2 other script: weather_fill.py and building_fill.py
* Make sure you also have those two other scripts within the same folder with this script!*

Authors:
    Qinzi Cao
    Yang Xiang
    Reshma Kelkar
    Vincent Lee
    Michael Setyawan
    Manaswi Mishra
'''
import pandas as pd
from weather_fill import *
from building_fill import *

##########################################################################
########################### DATA PREP ####################################
#Import weather train
weather_train=pd.read_csv('weather_train.csv') #dim:139773 rows
#Impute missing values in weather train
#weather=sort_fill_weather(weather_train)
#Uncomment code below to save into csv file
#weather.to_csv('weather.csv',index=False,header=True)

#Import building data
building_metadata=pd.read_csv('building_metadata.csv') #dim:
#Impute missing values in building_metadata
#building=sort_fill_building(building_metadata)
#Uncomment code below to save into csv file
#building.to_csv('building.csv',index=False,header=True)

#Import train data
train=pd.read_csv('train.csv')#dim:

#Merging dataframes
#train_building_df = pd.merge(train, building,  how='inner', on='building_id')
#train_building_df['timestamp']=pd.to_datetime(train_building_df['timestamp'])
#train_building_weather_df = pd.merge(train_building_df, weather,  how='inner', left_on=['timestamp','site_id'],\
#                  right_on = ['timestamp','site_id'])
#train_building_weather_df.to_csv('complete_train.csv',index=False,header=True)

#Import_complete_train_data
complete_train = pd.read_csv('complete_train.csv')



#########################################################################
########################### MODELING ####################################
"""
Model using Linear Regression
"""
from sklearn import linear_model


