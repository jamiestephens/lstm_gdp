# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 23:45:44 2021

@author: Administrator
"""
import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

df = preprocessing.df_2
date_l = preprocessing.datelist


plot_graph = pd.DataFrame(date_l,columns=['Date'])
plot_graph = plot_graph.loc[(plot_graph['Date'] >= '2010-01-01')]


Tenyr_a = df['Tenyr'].to_numpy()

plot_graph['actual'] = Tenyr_a

print(plot_graph)

Tenyr_a = Tenyr_a.reshape((len(Tenyr_a), 1))

print(Tenyr_a)
train=Tenyr_a[500:]
test=Tenyr_a[:500]

scaler=MinMaxScaler()
scaled_train=scaler.fit_transform(train)
scaled_test=scaler.transform(test)

n_input=20
n_features=1

train_generator=TimeseriesGenerator(scaled_train,
                                     scaled_train,
                                      n_input,
                                      batch_size=1)

model=Sequential()
model.add(LSTM(100,activation='relu',input_shape=(n_input,n_features),return_sequences=True))
model.add(LSTM(50,activation='relu',return_sequences=True))
model.add(LSTM(10,activation='relu'))
model.add(Dense(1))


model.compile(optimizer='adam',loss='mse')
model.summary()

model.fit(train_generator,epochs=20)

test_predictions = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    pred = model.predict(current_batch)[0]
    test_predictions.append(pred)
    current_batch = np.append(current_batch[:,1:,:],[[pred]],axis=1)
    
print(test_predictions)


actual_predictions = scaler.inverse_transform(test_predictions)
print("No of actual predictions: ",len(actual_predictions))

#test['Predictions'] = actual_predictions
#test.plot(figsize=(12,8));

#pyplot.plot(test)
#pyplot.plot(actual_predictions)
#pyplot.show()

plot_graph.plot(x ='Date', y='actual', kind = 'scatter')
