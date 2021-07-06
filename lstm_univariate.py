# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 19:21:17 2021

@author: Administrator
"""
from numpy import array
import pandas as pd
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def lstm():
    df = pd.read_csv('./data/daily_treasury_yield_curve_rates.csv')
    df = df[['Date','10 yr']]
    df.rename(columns={'10 yr':'Tenyr'},inplace=True)
    
    df = df[df.Tenyr != "N/A "]
    
    df[['Tenyr']] = df[['Tenyr']].apply(pd.to_numeric)
    
    df1 = df[['Tenyr']]
    
    X = df1.Tenyr.diff()
    
    X = X.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    scaled_X = scaler.transform(X)
    
    scaled_series = Series(scaled_X[:, 0])
    
    inverted_X = scaler.inverse_transform(scaled_X)
    inverted_series = Series(inverted_X[:, 0])
    print(inverted_series.head())
    



lstm()