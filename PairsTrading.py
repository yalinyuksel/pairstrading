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
    df = yf.download(ticker,start="2015-01-01",interval="1d",progress=False)["Close"]
    df.index = pd.to_datetime(df.index, format = '%Y/%m/%d').strftime('%Y-%m-%d')
    return df

#Data Plot function
def pricePlot(dataframe,colname):  
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(dataframe.index,dataframe[colname],color='navy',label=colname, ls='-')
    ax.legend(loc=1)
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
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
model = sm.OLS(pairsData[assets[0]],pairsData[assets[1]])
model = model.fit()
hedgeRatio = round(model.params[0],2)

#Calculate theoretical price of assets_0
pairsData[assets[0] + "_tPrice"]  = pairsData[assets[1]] * hedgeRatio

#Calculate spread between theo price of assets_0 and real price
pairsData['Spread'] = pairsData[assets[0]] - pairsData[assets[0] + "_tPrice"]

#Plot Spread 
plt.ylabel('Prices')
pairsData[assets[0]].sort_index(ascending=True).plot(figsize=(30,15),label='Market Price '+assets[0])
pairsData[assets[0] + "_tPrice"].sort_index(ascending=True).plot(figsize=(30,15),label='Theoretical Price ' + assets[0])
pairsData.Spread.sort_index(ascending=True).plot(figsize=(8,4))
plt.legend(loc='upper right',prop={'size':8})
plt.show()


plt.ylabel('Spread')
pairsData.Spread.sort_index(ascending=True).plot(figsize=(8,4),label='Spread')
plt.legend(loc='upper right',prop={'size':8})
plt.show()

pValueResidual = ADFTest(pairsData,"Spread")

if round(pValueResidual,2) < 0.05:
    print("Asset A and Asset B are suitable for Pairs Trading System")
    #print(assets[0],"and", assets[1], "B are suitable for Pairs Trading System")
else:
    print("Check another pair, or change time interval")



########### PART III ################
            
def getTradeBands(prices, rate=50):
    # Calculate the simple moving average (SMA) using the specified rate
    sma = prices.rolling(rate).mean()

    # Calculate the standard deviation (std) using the specified rate
    std = prices.rolling(rate).std()

    # Calculate the upper band by adding 1.5 times the std to the SMA
    bandUp = sma + std * 1.5

    # Calculate the lower band by subtracting 1.5 times the std from the SMA
    bandDown = sma - std * 1.5

    # Return the upper and lower bands
    return bandUp, bandDown

#Sort Spread Data historically, and export spread 
spreadPrices = pairsData['Spread'].sort_index(ascending=True)

bandUp, bandDown = getTradeBands(spreadPrices)

def pairsTradeStrategy(data, bandDown, bandUp):
    buyPrice = []        # List to store buy prices
    sellPrice = []       # List to store sell prices
    spreadSignal = []    # List to store spread signals
    signal = 0           # Variable to track current signal
    
    for i in range(0, len(data)):
        # Check for buy signal
        if data[i-1] > bandDown[i-1] and data[i] < bandDown[i]:
            if signal != 1:
                buyPrice.append(data[i])
                sellPrice.append(np.nan)
                signal = 1
                spreadSignal.append(signal)
            else:
                buyPrice.append(np.nan)
                sellPrice.append(np.nan)
                spreadSignal.append(0)
        # Check for sell signal
        elif data[i-1] < bandUp[i-1] and data[i] > bandUp[i]:
            if signal != -1:
                buyPrice.append(np.nan)
                sellPrice.append(data[i])
                signal = -1
                spreadSignal.append(signal)
            else:
                buyPrice.append(np.nan)
                sellPrice.append(np.nan)
                spreadSignal.append(0)
        else:
            # No signal
            buyPrice.append(np.nan)
            sellPrice.append(np.nan)
            spreadSignal.append(0)
            
    return buyPrice, sellPrice, spreadSignal

# Call the pairsTradeStrategy function with the necessary inputs
buyPrice, sellPrice, spreadSignal = pairsTradeStrategy(spreadPrices, bandDown, bandUp)

# Create a copy of pairsData and sort it in ascending order
tradeFrame = pairsData[[assets[0], assets[1]]].copy().sort_index(ascending=True)

# Add the spreadSignal column to the tradeFrame DataFrame
tradeFrame['Signal'] = spreadSignal


#Signal 1 implies long in Downband -> Short B, Long A
#Signal -1 implies short in UpBand -> Long B , Short A
position = 0
long = 0
short = 0
pnl = []
marginReq = []

def openLong(data,index,position,long):
    
    entryLongA = data[assets[0]].iloc[index]
    entryShortB = data[assets[1]].iloc[index]
    position = 1
    long = 1

    return entryLongA,entryShortB,long,position

def closeLong(data,index,position,long):
    
    exitLongA = data[assets[0]].iloc[index]
    exitShortB = data[assets[1]].iloc[index]
    position = 0
    long = 0
    
    return exitLongA,exitShortB,long,position

def openShort(data,index,position,short):
    
    entryShortA = data[assets[0]].iloc[index]
    entryShortA = data[assets[1]].iloc[index]
    position = 1
    short = 1
    
    return entryShortA,entryShortA,short,position

def closeShort(data,index,position,short):
    
    exitShortA = data[assets[0]].iloc[index]
    exitLongB = data[assets[1]].iloc[index]
    position = 0
    short = 0
    
    return exitShortA,exitLongB,short,position
    
for i in range(len(tradeFrame['Signal'])):
    
    if position == 0: #Open position loop
        if tradeFrame['Signal'].iloc[i] == 1: #Portfolio Long -> Short B, Long A
            entryLongA,entryShortB, position,long = openLong(tradeFrame,i,position,long)
            marginReq.append(entryLongA + entryShortB*1.5)
            continue
        elif tradeFrame['Signal'].iloc[i] == -1 : #Portfolio Short -> Long B, Short A
            entryShortA,entryLongB,position,short = openShort(tradeFrame,i,position,short)
            marginReq.append(entryShortA *1.5 + entryLongB)
            continue
    elif position == 1: #Close position loop
        if tradeFrame['Signal'].iloc[i] == -1 and long: #close long if exist
            exitLongA,exitShortB,position,long = closeLong(tradeFrame, i, position, long)
            profit = (exitLongA - entryLongA) + (entryShortB - exitShortB)
            pnl.append(round(profit,5))
            entryShortA,entryLongB,position,short = openShort(tradeFrame,i,position,short)
            marginReq.append(entryShortA *1.5 + entryLongB)
            continue
        elif tradeFrame['Signal'].iloc[i] == 1 and short: #close short if exist
            exitShortA,exitLongB,position,short = closeShort(tradeFrame,i,position,short)
            profit = (entryShortA - entryShortA) + (exitLongB - entryLongB)
            pnl.append(round(profit,5))
            entryLongA,entryShortB, position,long = openLong(tradeFrame,i,position,long)
            marginReq.append(entryLongA + entryShortB*1.5)
            continue
            

spreadPrices.plot(figsize=(30,15),label='Spread B',c='b')
plt.plot(bandUp,label='Up Spread', c='g')
plt.plot(bandDown,label='Down Spread', c='r')
plt.scatter(pairsData.sort_index(ascending=True).iloc[0:].index, buyPrice, marker = '^', color = 'green', label = 'BUY', s = 200)
plt.scatter(pairsData.sort_index(ascending=True).iloc[0:].index, sellPrice, marker = 'v', color = 'red', label = 'SELL', s = 200)
plt.legend(loc="upper left")
plt.show()
marginReq.pop(1)
totalsum = 0
totalcumsum = []

for i in pnl:
    
    totalsum = totalsum + i
    totalcumsum.append(totalsum)
    

tradeResults = pd.DataFrame(list(zip(pnl,totalcumsum,marginReq)),columns=['PnL','Cumulative','Margin'])
tradeResults['Returns'] = tradeResults['PnL']/tradeResults['Margin']*100
tradeResults['Equity'] = (1 + tradeResults['Returns']/100).cumprod() * 100
"""equityCurve = 100
for i in tradeResults['Returns']:
    tradeResults['EquityCurve'].iloc[i] = equityCurve * (1+i)
    equityCurve = tradeResults['EquityCurve'].iloc(i)"""

tradeResults.Equity.plot(figsize=(30,15),label='PnL Chart',c='b')
plt.bar(tradeResults.index, tradeResults.PnL, color ='maroon',width = 0.4)
plt.ylabel('Profit in $')
y_min = tradeResults['Equity'].min()
y_max = tradeResults['Equity'].max()
y_margin = (y_max - y_min) * 0.1  # Add a margin for better visualization
plt.ylim(y_min - y_margin, y_max + y_margin)
plt.show()

fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(1, 1, 1)
tradeResults['Returns'].hist(bins=20, ax=ax1)
ax1.set_xlabel('Return')
ax1.set_ylabel('Sample')
ax1.set_title('Return distribution')
plt.show()

# Assuming your DataFrame is called 'df' and contains the 'equity' column
equity = tradeResults['Equity']

# Calculate the total percent return
total_return = ((equity.iloc[-1] / equity.iloc[0]) - 1) * 100

# Calculate the drawdown as the difference between equity and the cumulative maximum
drawdown = equity - equity.cummax()

# Calculate the maximum drawdown as a percentage of the peak value
max_drawdown = (drawdown / equity.cummax()).min() * 100

# Calculate the Calmar ratio
calmar_ratio = total_return / abs(max_drawdown)

# Print the results
print("Total Percent Return (%):", round(total_return,2))
print("Maximum Drawdown (%):", abs(round(max_drawdown,2)))
print("Calmar Ratio:", round(calmar_ratio,2))