# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 18:58:22 2021

@author: Administrator
"""

import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from datetime import datetime
import numpy as np

df = pd.read_csv('./data/daily_treasury_yield_curve_rates.csv', na_values= ['N/A '],skiprows=1, names=['Date','Onemo','Twomo','Threemo','Sixmo',
                                    'Oneyr','Twoyr','Threeyr','Fiveyr','Sevenyr',
                                    'Tenyr','Twentyyr','Thirtyyr'])

del df['Twomo']

df['Date'] = df['Date'].astype('datetime64[ns]')

df.Onemo = df.Onemo.fillna((df.Onemo.shift() + df.Onemo.shift(-1))/2)
df.Threemo = df.Threemo.fillna((df.Threemo.shift() + df.Threemo.shift(-1))/2)
df.Sixmo = df.Onemo.fillna((df.Sixmo.shift() + df.Sixmo.shift(-1))/2)
df.Oneyr = df.Oneyr.fillna((df.Oneyr.shift() + df.Oneyr.shift(-1))/2)
df.Twoyr = df.Twoyr.fillna((df.Twoyr.shift() + df.Twoyr.shift(-1))/2)
df.Threeyr = df.Threeyr.fillna((df.Threeyr.shift() + df.Threeyr.shift(-1))/2)
df.Fiveyr = df.Fiveyr.fillna((df.Fiveyr.shift() + df.Fiveyr.shift(-1))/2)
df.Sevenyr = df.Sevenyr.fillna((df.Sevenyr.shift() + df.Sevenyr.shift(-1))/2)
df.Tenyr = df.Tenyr.fillna((df.Tenyr.shift() + df.Tenyr.shift(-1))/2)
df.Twentyyr = df.Twentyyr.fillna((df.Twentyyr.shift() + df.Twentyyr.shift(-1))/2)
df.Thirtyyr = df.Thirtyyr.fillna((df.Thirtyyr.shift() + df.Thirtyyr.shift(-1))/2)

r = pd.date_range(start=df.Date.min(), end=df.Date.max())

df.set_index('Date').reindex(r).fillna(0.0).rename_axis('Date').reset_index()

datelist = pd.date_range('2000-01-01', periods=7840).tolist()

df_1 = pd.DataFrame(datelist,columns=['Date'])

df_2 = pd.merge(df,df_1,on='Date',how='right')

df_2['DayWeek'] = df_2['Date'].dt.day_name()
print(df_2.dtypes)


m = df_2['DayWeek'].ffill().str.contains('Saturday')
cols = ['Onemo']

df_2[cols] = np.where(m,df_2[cols].ffill(),df_2[cols])

#df_2.update(df_2.loc[df_2['DayWeek'].str.contains('Saturday').ffill(), ['Onemo','Threemo']].ffill())

#groups = [3,4,5]
#for group in groups:
    #df_2.update(df_2.loc[df_2['DayWeek'].str.contains('Saturday').ffill(), [group]].ffill())



print(df_2)

df_2.to_csv('test.csv')

