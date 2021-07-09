# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:07:43 2021

@author: Administrator
"""

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import preprocessing
df = preprocessing.df_2
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

def custom_activation(x, beta = 1):
        return (K.sigmoid(beta * x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
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
	agg = concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg

bondchoices = ['Onemo','Threemo','Sixmo','Oneyr','Fiveyr','Sevenyr',"Tenyr",'Twentyyr','Thirtyyr']

for bondchoice in bondchoices:
    lastvalue = df[bondchoice].iloc[-1]
    
    x = int(len(df)*0.20)
    
    col_no = df.columns.get_loc(bondchoice)
    
    col_count = len(df.columns)
    
    arr = []
    for i in range(col_count,col_count*2):
        arr.append(i)
     
    output_ind = arr.index(col_count+col_no)
    
    del arr[output_ind]
        
    values = df.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)
    
    reframed.drop(reframed.columns[arr], axis=1, inplace=True)
    
    train = values[x:]
    test = values[:x]
    
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    
    model = Sequential()
    
    model.add(Activation(custom_activation,name = "Swish"))
    
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    history = model.fit(train_X, train_y, epochs=50, batch_size=15, validation_data=(test_X, test_y), verbose=0, shuffle=False)

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.ylabel("Loss")
    pyplot.xlabel("Epoch Number")
    pyplot.title(bondchoice)
    pyplot.show()
    
    yhat = model.predict(test_X)
    
    a = col_count-1
    
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_yhat = concatenate((yhat, test_X[:, -a:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -a:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    percentwrong = rmse/lastvalue
    print(bondchoice)
    print('Test RMSE: %.3f' % rmse)
    print("Proportion from most recent data point: ",percentwrong)
        

