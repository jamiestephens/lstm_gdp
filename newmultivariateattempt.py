# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 18:28:25 2021

@author: Administrator
"""
from datetime import date, timedelta
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.layers import Activation
from sklearn.metrics import mean_squared_error

def preprocessing(outputpath,inputpath, lagtime):
    outputs = pd.read_csv(str('./data/'+outputpath))
    inputs = pd.read_csv(str('./data/'+inputpath))
    
    outputs['DATE']= pd.to_datetime(outputs['DATE'])

    outputs['DATE'] = outputs['DATE'] - timedelta(17)

    outputs['DATE'] = outputs['DATE'].dt.strftime('%Y-%m')
    
    inputs.set_index("Date", inplace = True)  
    
    inputs['Final'] = inputs.values.tolist()
    
    #inputs['Final'] = '[' + inputs['oneyr-threemo'] + ", " + inputs['fiveyr-twoyr'] + ", " + inputs['twoyr-threemo'] + ", " + inputs['tenyr-threemo'] + ", " + inputs['tenyr-twoyr'] + ']'

    del inputs['tenyr-threemo']
    del inputs['fiveyr-twoyr']
    del inputs['oneyr-threemo']
    del inputs['twoyr-threemo']
    del inputs['tenyr-twoyr']
    
    print(inputs.head())
     
  #  np.vstack(inputs.Final)
  #  np.concatenate(inputs.Final).reshape(inputs.shape[0],-1)
    input_values = inputs.loc[:,'Final'].values
    print(input_values)
#    return processed_df


sequenced_df = preprocessing('real_gdp_growth_monthly.csv','daily_treasury_spreads_reformatted_2008_2020.csv',12)

# tf.keras.backend.clear_session()
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
# model.add(Dense(1))

# model.compile(optimizer='adam', loss='mse')

# wandb.init(entity='ayush-thakur', project='dl-question-bank')

# history = model.fit(X, Y, epochs=1000, validation_split=0.2, verbose=1, callbacks=[WandbCallback()])