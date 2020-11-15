#Import required Libraries
%matplotlib inline
import sys
from dateutil.parser import parse
import json
from random import shuffle
import random
import datetime
import os
import boto3
import s3fs
import sagemaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Lib import utils

#Create the connection to s3 to access the trainig and testing data
s3 = boto3.client('s3')
#get_execution_role gives the IAM role
role = sagemaker.get_execution_role() 
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name

#Read the Data
df = pd.read_csv('raw_data.csv')
#Keep only numbers and convert sku_id column to integers
df['sku_id'] = [int(i[3:]) for i in df['sku_id']]
#All string features need to be encoded to integer and value should start from 0. 
#Hence, “sku_id”, “model”, “storage_col”, “colour”, “customer” and “country” features should be transformed
transform_json={
"model":{"model1":0,"model2":1,"model3":2,"model4":3,"model5":4,"model6":5},
"storage_vol":{"64 gb":3,"256 gb":1,"512 gb":2,"128 gb":0},
"colour":{"black":0,"blue":1,"coral":2,"gold":3,"green":4,"silver":5,"space grey":6},
 "customer":{"retailers1":0,"retailers2":1,"retailers3":2,"retailers4":3},
"country":{"country1":0,"country2":1,"country3":2}
}

df_cat = df.copy() #Creating the data copy to make sure the no changes in raw data
#Convert all column string values to lower case
str_cols=['model','storage_vol','colour','customer','country']
for col in str_cols:
    df_cat[col]=df_cat[col].str.lower()
#Repalcing the string with corresponding encoded values
df_cat.replace(transform_json, inplace=True)
df_cat.head()

# create mapping table between sku_id and categorical data 
dict_mapping = df_cat[['sku_id', 'model', 'storage_vol', 'colour']].drop_duplicates()
dict_mapping = dict_mapping.sort_values('sku_id')
dict_mapping.head()

#Selecting the required columns
df1 = df_cat[['timestamp', 'sku_id', 'price', 'quantity','country', 'customer']]

# time series columns naming: sku_id + country + customer_group (Merging the columns as our predictions will be in this dimension)
df1['group_id'] = df1.sku_id.astype(str) + '_' + df1.country.astype(str) + '_' + df1.customer.astype(str)

#Creating Target Time series

#Converting all the timeseries in to respective columns with date index
columns = df1['group_id'].unique()
columns.sort()
df1['timestamp'] = pd.to_datetime(df1['timestamp'], errors='coerce',dayfirst=True)
df1 = df1.set_index('timestamp')
df_target = pd.DataFrame(columns = columns)
df_target['timestamp'] = pd.date_range(start='2018-09-20', end='2020-01-31', freq = 'D')
df_target = df_target.set_index('timestamp')
df_target = df_target.asfreq(freq='D') #in df_target each column is nothing but one combination time series

#Updating all the time series with target price
num_columns = len(columns)
for i in range(num_columns):
    columns_split = df_target.columns[i].replace('_', ' ').split(' ')
    temp = df1.loc[(df1['sku_id'] == int(columns_split[0])) 
               & (df1['country'] == int(columns_split[1])) 
               & (df1['customer'] == int(columns_split[2]))].resample('D').mean() #taking the mean price if we have multiple records for the same day
    df_target.iloc[:, i] = temp['price']

#Adding all the time series data to target_price list
target_price = []
num_ts = df_target.shape[1]
for i in range(num_ts):   
    target_price.append(np.trim_zeros(df_target.iloc[:,i], trim='f'))  

#Now creating related time series

#Data transformation to create realted time series data
df_dynamic = pd.DataFrame(columns = columns)
df_dynamic['timestamp'] = pd.date_range(start='2018-09-20', end='2020-01-31', freq = 'D')
df_dynamic = df_dynamic.set_index('timestamp')
df_dynamic = df_dynamic.asfreq(freq='D

num_columns = len(columns)
for i in range(num_columns):
    columns_split = df_dynamic.columns[i].replace('_', ' ').split(' ')
    temp = df1.loc[(df1['sku_id'] == int(columns_split[0])) 
               & (df1['country'] == int(columns_split[1])) 
               & (df1['customer'] == int(columns_split[2]))].resample('D').sum() #taking the sum of quantity if we ave multiple records for the same day
    df_dynamic.iloc[:, i] = temp['quantity']

df_dynamic = df_dynamic.fillna(0) #Filling all missing values with 0

dynamic_quantity = []
num_ts = df_dynamic.shape[1]

for i in range(num_ts):   
    dynamic_quantity.append(df_dynamic.iloc[:,i])   #appending all the related time series data to dynamic_quantity list

#Split data to train and testing

freq = '1D'
prediction_length = 30
context_length = 30

start_dataset = pd.Timestamp("2018-09-20", freq=freq)
end_training = pd.Timestamp("2020-01-31", freq=freq)

# cat structure : [model, storage, colour, country, customer_group]
# use dict_mapping, find the cat data for specific sku_id 
target_cat = {}

for i in range(len(target_price)):
    column_name = target_price[i].name.replace('_', ' ').split(' ')
    cat_name = dict_mapping.loc[dict_mapping['sku_id'] == int(column_name[0])][['model', 'storage_vol', 'colour']].to_numpy()
    target_cat[i] = []
    target_cat[i].append(int(cat_name[0][0]))
    target_cat[i].append(int(cat_name[0][1]))
    target_cat[i].append(int(cat_name[0][2]))
    target_cat[i].append(int(column_name[1]))
    target_cat[i].append(int(column_name[2]))
    
FREQ = 'D'
training_data = [
    {
        "start": str(start_dataset),
        "target": [i for i in ts[start_dataset:end_training - pd.Timedelta(1, unit=FREQ)].tolist()], # We use -1, because pandas indexing includes the upper bound
        "cat": target_cat[index],
        "dynamic_feat": [[j for j in dynamic_quantity[index][start_dataset:end_training - pd.Timedelta(1, unit=FREQ)].tolist()]]
    }
    for index, ts in enumerate(target_price)
]
print(len(training_data))

for i in range(len(training_data)):
    training_data[i]['target'] = [x if np.isfinite(x) else "NaN" for x in training_data[i]['target']]

FREQ = 'D'
num_test_windows = 4

test_data = [
    {
        "start": str(start_dataset),
        "target": [i for i in ts[start_dataset:end_training + pd.Timedelta(k * prediction_length, unit=FREQ)].tolist()],
        "cat": target_cat[index], 
        "dynamic_feat": [[j for j in dynamic_quantity[index][start_dataset:end_training + pd.Timedelta(k * prediction_length, unit=FREQ)].tolist()]]
    }
    for k in range(1, num_test_windows +1) 
    for index, ts in enumerate(target_price)
]
print(len(test_data))

for i in range(len(test_data)):
    test_data[i]['target'] = [x if np.isfinite(x) else "NaN" for x in test_data[i]['target']]
    

#Convert training and testing data to jsonlines
%%time
utils.write_dicts_to_file("train.json", training_data)
utils.write_dicts_to_file("test.json", test_data)

#transfer the data to s3
utils.copy_to_s3("train.json", input_data + "/train/train.json")
utils.copy_to_s3("test.json", input_data + "/test/test.json")

#Training a Model

#Deep AR Docker image
image_name = sagemaker.amazon.amazon_estimator.get_image_uri(region, "forecasting-deepar", "latest")

#To intilaize the estimaor instance
estimator = sagemaker.estimator.Estimator(
    sagemaker_session=sagemaker_session,
    image_name=image_name,
    role=role,
    train_instance_count=1,
    train_instance_type='ml.c4.2xlarge',
    base_job_name='deep-ar-testing-price-latest',
    output_path=output_data
)

#Hyperparameters to tune the model
hyperparameters = {
    "time_freq": freq,
    "epochs": "400",
    "early_stopping_patience": "40",
    "mini_batch_size": "64",
    "learning_rate": "5E-4",
    "likelihood" : "gaussian",
    "context_length": str(context_length),
    "prediction_length": str(prediction_length)
}

estimator.set_hyperparameters(**hyperparameters)

#To train the model ans save the model artifact to s3
data_channels = {
    "train": "{}/train/".format(input_data),
    "test": "{}/test/".format(input_data)
}

estimator.fit(inputs=data_channels, wait=True)

#to generate real time predictions endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge')
    
