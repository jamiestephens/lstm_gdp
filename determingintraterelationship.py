# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:46:26 2021

@author: Administrator
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


def howareintratesconnected():
    df = pd.read_csv('./data/daily_intrates.csv', skiprows=1, names=['Date','Onemo','Twomo','Threemo','Sixmo',
                                                        'Oneyr','Twoyr','Threeyr','Fiveyr','Sevenyr',
                                                        'Tenyr','Twentyyr','Thirtyyr'])
    
    del df['Twomo']
    
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
    
    print(df.dtypes)
    
    y = df    
    
       
    calc_vif(y)
    
    y['Onemo'] = y['Onemo'].diff(1)
    y['Threemo'] = y['Threemo'].diff(1)
    y['Sixmo'] = y['Sixmo'].diff(1)
    y['Oneyr'] = y['Oneyr'].diff(1)    
    y['Twoyr'] = y['Twoyr'].diff(1)
    y['Threeyr'] = y['Threeyr'].diff(1)
    y['Fiveyr'] = y['Fiveyr'].diff(1)
    y['Sevenyr'] = y['Sevenyr'].diff(1)
    y['Tenyr'] = y['Tenyr'].diff(1)
    y['Twentyyr'] = y['Twentyyr'].diff(1)
    y['Thirtyyr'] = y['Thirtyyr'].diff(1)    
    
    print(y)
    
    plt.plot(y['Date'], y['Onemo'])
    plt.plot(y['Date'], y['Threemo'])
    plt.plot(y['Date'], y['Sixmo'])
    plt.plot(y['Date'], y['Oneyr'])
    plt.plot(y['Date'], y['Twoyr'])
    plt.plot(y['Date'], y['Threeyr'])
    plt.plot(y['Date'], y['Fiveyr'])
    plt.plot(y['Date'], y['Sevenyr'])
    plt.plot(y['Date'], y['Tenyr'])
    plt.plot(y['Date'], y['Twentyyr'])
    plt.plot(y['Date'], y['Thirtyyr'])
    plt.title('Interest Rates over Time', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Int Rate', fontsize=14)
    plt.grid(True)
    plt.show()
    
 
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(vif)

    

howareintratesconnected()