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
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

df = pd.read_csv('./data/daily_treasury_yield_curve_rates.csv', skiprows=1, names=['Date','Onemo','Twomo','Threemo','Sixmo',
                                    'Oneyr','Twoyr','Threeyr','Fiveyr','Sevenyr',
                                    'Tenyr','Twentyyr','Thirtyyr'])

#df = df[df.Tenyr != "N/A "]

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
                    'Thirtyyr':float}

df = df.astype(convert_dict)    

#df1 = df[['Tenyr']]
    
#X = df1.Tenyr.diff()

X_col = df.loc[:,'Tenyr']

x_values = X_col.values

print(x_values)

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


n_steps = 3
# split into samples
X, y = split_sequence(x_values, n_steps)
# summarize the data
for i in range(len(x_values)):
	print(X[i], y[i])

