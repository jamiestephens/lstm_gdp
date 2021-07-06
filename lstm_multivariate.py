# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:48:56 2021

@author: Administrator
"""


import pandas as pd
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder


def lstm():
    df = pd.read_csv('./data/daily_intrates.csv', skiprows=1, names=['Date','Onemo','Twomo','Threemo','Sixmo',
                                                'Oneyr','Twoyr','Threeyr','Fiveyr','Sevenyr',
                                                    'Tenyr','Twentyyr','Thirtyyr'])
    del df['Twomo']
    
    df.index.name = 'Date'
    
   # print("Number of N/A values: " ,len(df[df['Tenyr'] == 'N/A ']))
    
    groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    
   # df.replace(to_replace ="N/A ",
   #              value ="NaN")
    
    #for group in groups:
        #df[group] =  df[group].fillna((df[group].shift() + df[group].shift(-1))/2)
        
    df = df[~(df == 'N/A ').any(axis=1)]
    
    df['Date'] = df['Date'].astype('datetime64[ns]')
    
    convert_dict = {'Onemo':float,
                    'Threemo':float,
                    'Sixmo':float,
                    'Oneyr':float,
                    'Twoyr':float,
                    'Threeyr':float,
                    'Fiveyr':float,
                    'Sevenyr':float,
                    'Tenyr':float,
                    'Twentyyr':float,
                    'Thirtyyr':float
               }
    df = df.astype(convert_dict)
    del df['Date']
    values = df.values
    
    pyplot.rcParams.update({'font.size': 8})
    
    i = 1
    
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(df.columns[group], y=0.5, loc='right')
        i += 1
    
    pyplot.show()
    
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    reframed = series_to_supervised(scaled, 1, 1)

    print(reframed)
    
    values = reframed.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
    
	return agg

    
lstm()