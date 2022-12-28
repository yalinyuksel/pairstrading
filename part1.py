#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 01:22:07 2022

@author: yalinyuksel
"""

import yfinance as yf #import yfinance
import pandas as pd #import pandas
import numpy as np #import numpy
import matplotlib.pyplot as plt #import plot library


#Enter tickers you want to be downloaded to the list
assets = ["BZ=F","CL=F"]

#Function that downloads Daily OHLC data for ticker starting from 15-11-2020
#Please refer to https://pypi.org/project/yfinance/ for more details and options
def dataDownloader(ticker):
    df = yf.download(ticker,start="2000-11-15",interval="1d",progress=False)["Close"]
    df.index = pd.to_datetime(df.index, format = '%Y/%m/%d').strftime('%Y-%m-%d')
    return df

#Data Plot function
def pricePlot(dataframe,colname):  
    fig, ax = plt.subplots(figsize=(10,8))
    dataframe.plot.line(y=colname,color='crimson', ax=ax)
    plt.ylabel(colname)
    plt.show()
   
def sortData(dataframe):
    #Checks if index is monotonically increasing
    isSorted = dataframe.index.is_monotonic 
    if not isSorted:
        dataframe.sort_index(inplace=True, ascending=True) 
        #ascending=False for descending data
    return dataframe

def detectNull(dataframe,colname): #detect if there are null values
    
    isnull = dataframe[colname].isnull().values.any()
    if isnull:
        dataframe[colname].interpolate(method = 'linear', inplace = True)
    return dataframe

def detectOutliers(dataframe,colname):
    thres = 3 #threshold which eliminates the outlier data
    mean = np.mean(dataframe[colname]) #find average price 
    std = np.std(dataframe[colname]) #find standard deviation
    for i in dataframe[colname]:
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            dataframe[colname].interpolate(method = 'linear', inplace = True)

    return dataframe


#Dataframe that will hold Daily OHLC Data
pairsData  = dataDownloader(assets)

#Plot time-series
pricePlot(pairsData,assets[0])
pricePlot(pairsData,assets[1])

#inplace changes the dataframe completely, no need to assign again
sortData(pairsData)
detectNull(pairsData,assets[0])
detectNull(pairsData,assets[1])

#Find the first index for both data where data is not nan or null
first_index1 = pairsData[assets[0]].first_valid_index()
first_index2 = pairsData[assets[1]].first_valid_index()

#If indices are not equal, delete every row that contains nan, or null value
if first_index1 != first_index2:
    pairsData.dropna(inplace=True)

#Detect Outliers, remove and interpolate
detectOutliers(pairsData,assets[0])
detectOutliers(pairsData,assets[1])
