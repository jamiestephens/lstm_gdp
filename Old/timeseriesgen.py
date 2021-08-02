# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 21:26:54 2021

@author: Administrator
"""

import pandas as pd
import preprocessing
from numpy import array
from numpy import hstack
from keras.preprocessing.sequence import TimeseriesGenerator

df  = preprocessing.df_2

Onemo_a = df['Onemo'].to_numpy()
Threemo_a = df['Threemo'].to_numpy()
Sixmo_a = df['Sixmo'].to_numpy()
Oneyr_a = df['Oneyr'].to_numpy()
Twoyr_a = df['Twoyr'].to_numpy()
Threeyr_a = df['Threeyr'].to_numpy()
Fiveyr_a = df['Fiveyr'].to_numpy()
Sevenyr_a = df['Sevenyr'].to_numpy()
Tenyr_a = df['Tenyr'].to_numpy()
Twentyyr_a = df['Twentyyr'].to_numpy()
Thirtyyr_a = df['Thirtyyr'].to_numpy()

# inputs: 
Onemo_a = Onemo_a.reshape((len(Onemo_a), 1))
Threemo_a = Threemo_a.reshape((len(Threemo_a), 1))
Sixmo_a = Sixmo_a.reshape((len(Sixmo_a), 1))
Oneyr_a = Oneyr_a.reshape((len(Oneyr_a), 1))
Twoyr_a = Twoyr_a.reshape((len(Twoyr_a), 1))
Threeyr_a = Threeyr_a.reshape((len(Threeyr_a), 1))
Fiveyr_a = Fiveyr_a.reshape((len(Fiveyr_a), 1))
Sevenyr_a = Sevenyr_a.reshape((len(Sevenyr_a), 1))
Twentyyr_a = Twentyyr_a.reshape((len(Twentyyr_a), 1))
Thirtyyr_a = Thirtyyr_a.reshape((len(Thirtyyr_a), 1))

# output:
Tenyr_a = Tenyr_a.reshape((len(Tenyr_a), 1))

# horizontally stack columns
dataset = hstack((Onemo_a, Threemo_a, Sixmo_a, Oneyr_a, Twoyr_a, Threeyr_a, Fiveyr_a, Sevenyr_a, Twentyyr_a, Thirtyyr_a))

print(dataset)

# define generator
n_input = 1
generator = TimeseriesGenerator(dataset, Tenyr_a, length=n_input, batch_size=1)
# print each sample
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))