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
assets = []

while True:
    if len(assets) != 2:
        asset = input("Enter the assets: ")
        assets.append(asset)
    else:
        break
        
#Function that downloads Daily OHLC data for ticker starting from 15-11-2020
#Please refer to https://pypi.org/project/yfinance/ for more details and options
def dataDownloader(ticker):
    df = yf.download(ticker,start="2020-11-15",interval="1d",progress=False)["Close"]
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
    isSorted = dataframe.index.is_monotonic_decreasing
    if not isSorted:
        #ascending=False for descending data
        dataframe.sort_index(inplace=True, ascending=False)
        dataframe.drop(index=dataframe.index[-1],axis=0,inplace=True)
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

#Check if any null value, then interpolate
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

##########################
###########################
########### PART II ########
############################

def describeData(data,colname):
    
    stat = data[colname].describe()
    stat.loc['var'] = data[colname].var()
    stat.loc['skew'] = data[colname].skew()
    stat.loc['kurt'] = data[colname].kurtosis()
    
    return stat

#Statistical Summary of Timeseries Data for a final check
statData = pd.DataFrame(describeData(pairsData,assets[0]))
statData.insert(1,assets[1],describeData(pairsData,assets[1]))

#Calculate Returns 
pairsData[assets[0]+"_%Return"] = round(pairsData[assets[0]].pct_change(),4)*100
pairsData[assets[1]+"_%Return"] = round(pairsData[assets[1]].pct_change(),4)*100

#Sort Dataframe according to Date index in descending format
sortData(pairsData)

#ADF Test
#Import library for ADF Test
from statsmodels.tsa.stattools import adfuller

def ADFTest(data,colname):
    
    #Variable that holds statistical results of ADF
    global adfStats
    adfStats = adfuller(data[colname],maxlag=0)

    # Test statistics for the given dataset
    print('Augmented Dickey_fuller Statistic: %f' % adfStats[0])
    # p-value 
    print('p-value: %f' % adfStats[1])
 
    # printing the critical values at different alpha levels.
    print('critical values at different levels:')
    for k, v in adfStats[4].items():
        print('\t%s: %.3f' % (k, v))
    print("\n")
    return adfStats[1]

print('ADF Test for ', assets[0])
pValue1 = ADFTest(pairsData, assets[0]+"_%Return")

print('ADF Test for ', assets[1])
pValue2 = ADFTest(pairsData, assets[1]+"_%Return")

if round(pValue1,2) == 0.00 and round(pValue2,2) == 0.00:

    #If returns are individually stationary, then calculate the correlation between two returns
    print('Pearson\'s: %f' % pairsData[assets[0]+"_%Return"].corr(pairsData[assets[1]+"_%Return"],method='pearson'))
    print('Spearman\'s: %f' % pairsData[assets[0]+"_%Return"].corr(pairsData[assets[1]+"_%Return"],method='spearman'))
    print('Kendall\'s: %f' % pairsData[assets[0]+"_%Return"].corr(pairsData[assets[1]+"_%Return"],method='kendall'),end="\n\n")
    
    
#Build Cointegration Model
import statsmodels.api as sm
model = sm.OLS(pairsData[assets[0]+"_%Return"],pairsData[assets[1]+"_%Return"])
model = model.fit()
hedgeRatio = round(model.params[0],2)


pairsData['Spread'] = pairsData[assets[1]] - model.params[0] * pairsData[assets[0]]

#Plot Spread 
import matplotlib.pyplot as plt
pairsData.Spread.plot(figsize=(8,4))
plt.ylabel('Spread')
plt.show()
print('ADF Test for Spread',end="\n\n")
pValueResidual = ADFTest(pairsData,"Spread")

if round(pValueResidual,2) == 0.00:
    print(assets[0],"and", assets[1], "are suitable for Pairs Trading System")
else:
    print("Check another pair, or change time interval")
    

#Calculate theoretical price
pairsData[assets[0] + "_tPrice"]  = pairsData[assets[0]] * hedgeRatio

pairsData[assets[0]].plot(figsize=(30,15),label='Market Price '+assets[0])
pairsData[assets[0] + "_tPrice"].plot(figsize=(30,15),label='Theoretical Price ' + assets[0])
#mergedData.LogPriceA.plot(figsize=(30,15),label='Market Price A')
plt.legend(loc='upper right',prop={'size':30})
plt.ylabel('Price')
pairsData.Spread.plot(figsize=(8,4))
plt.ylabel('Spread')
plt.show()







