
import pandas as pd
import numpy as np
import math as mth
import AnalysisFunctions as af
import datetime as dt
import os
from os import path, remove
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler,scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


"""
modularized code for getting stock prices
that are below $80 this
will improve our use of time in analysis

"""

"""
dates to grab the data from

"""

ticker_start_date = dt.date(2019,9,13)
ticker_start_date_string = ticker_start_date.strftime('%Y-%m-%d')
ticker_end_date = dt.date(2021,9,15)
ticker_end_date_string = ticker_end_date.strftime('%Y-%m-%d')
df = af.getPortfolio(portfolio_file =r'c:\users\cosmi\onedrive\desktop\portfolio.csv'
                     ,prices_file = r'c:\users\cosmi\onedrive\desktop\get_data_all.csv'
                     ,index_file = r'c:\users\cosmi\onedrive\desktop\spy500.csv' 
                     ,ticker_start_date = ticker_start_date_string
                    ,ticker_end_date = ticker_end_date_string)

ticker_start_date = dt.date(2019,9,13)
ticker_start_date_string = ticker_start_date.strftime('%Y-%m-%d')
ticker_end_date = dt.date(2021,9,15)
ticker_end_date_string = ticker_end_date.strftime('%Y-%m-%d')

"""
decide to refresh the indexes and or the jobs
"""

dia = af.get_index_data(ticker = 'DIA',start_date = ticker_start_date,outfile=r'c:\users\cosmi\onedrive\desktop\dia.csv', refreshFileOutput=True)
dia_df = af.get_ticker_jobs(refresh_index = False,refresh_data=True, index='DOW', njobs=4, start_date=ticker_start_date)
dia_df['index_Value'] = 'DOW'
dia_df = dia_df.merge(right=dia,how='inner', on='date')
dia_df.rename(columns={'close_x':'close', 'open_x':'open', 'high_x':'high', 'low_x':'low', 'volume_x':'volume', 'ticker_x':'ticker', 'adjclose':'market_close'}, inplace=True)
dia_df.drop(labels=['open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'ticker_y'],axis = 1, inplace=True)

spy500 = af.get_index_data(ticker = 'SPY',start_date = ticker_start_date, outfile=r'c:\users\cosmi\onedrive\desktop\spy500.csv', refreshFileOutput=True)
sp_500_df = af.get_ticker_jobs(refresh_index = False,refresh_data=True, index='SP500', njobs=2, start_date=ticker_start_date)
sp_500_df['index_Value'] = "SPY"
sp_500_df = sp_500_df.merge(right=spy500,how='inner', on='date')
sp_500_df.rename(columns={'close_x':'close', 'open_x':'open', 'high_x':'high', 'low_x':'low', 'volume_x':'volume', 'ticker_x':'ticker', 'adjclose':'market_close'}, inplace=True)
sp_500_df.drop(labels=['open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'ticker_y'],axis = 1, inplace=True)

ndx100 = af.get_index_data(ticker = 'NDX',start_date = ticker_start_date ,outfile=r'c:\users\cosmi\onedrive\desktop\ndx.csv', refreshFileOutput=True)
nasdaq_df = af.get_ticker_jobs(refresh_index = False,refresh_data=True, index='NASDAQ', njobs=8, start_date=ticker_start_date)
nasdaq_df['index_Value'] = "NDX"
nasdaq_df = nasdaq_df.merge(right=ndx100,how='inner', on='date')
nasdaq_df.rename(columns={'close_x':'close', 'open_x':'open', 'high_x':'high', 'low_x':'low', 'volume_x':'volume', 'ticker_x':'ticker', 'adjclose':'market_close'}, inplace=True)
nasdaq_df.drop(labels=['open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'ticker_y'],axis = 1, inplace=True)

mcx250 = af.get_index_data(ticker = 'MCX',start_date = ticker_start_date,outfile=r'c:\users\cosmi\onedrive\desktop\mcx.csv', refreshFileOutput=True)
ftse_250_df = af.get_ticker_jobs(refresh_index = False,refresh_data=True, index='FTSE250', njobs=2, start_date=ticker_start_date)
ftse_250_df['index_Value'] = "MCX"
ftse_250_df = ftse_250_df.merge(right=mcx250,how='inner', on='date')
ftse_250_df.rename(columns={'close_x':'close', 'open_x':'open', 'high_x':'high', 'low_x':'low', 'volume_x':'volume', 'ticker_x':'ticker', 'adjclose':'market_close'}, inplace=True)
ftse_250_df.drop(labels=['open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'ticker_y'],axis = 1, inplace=True)

etf_df = af.get_ticker_jobs(refresh_index=False,refresh_data=True,index='ETF',njobs=1,start_date=ticker_start_date)
etf_df['index_Value'] = "SPY"
etf_df = etf_df.merge(right=spy500,how='inner', on='date')
etf_df.rename(columns={'close_x':'close', 'open_x':'open', 'high_x':'high', 'low_x':'low', 'volume_x':'volume', 'ticker_x':'ticker', 'adjclose':'market_close'}, inplace=True)
etf_df.drop(labels=['open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'ticker_y'],axis = 1, inplace=True)
                            
list_of_index_values = [sp_500_df,nasdaq_df,ftse_250_df,dia_df,etf_df]
all_tickers_df = pd.concat([pd.DataFrame(data=x)for x in list_of_index_values])
"""
drop duplicates where the equity is listed under two different indexes
"""
all_tickers_df.drop_duplicates(['ticker','date'],inplace=True)
all_tickers_df.reset_index(drop=True,inplace=True)

"""
create the various averages and standard deviations

"""

all_tickers_df.sort_values(by=['ticker','date'], inplace=True)
all_tickers_df['previous_close'] = all_tickers_df.groupby(by=['ticker'])['close'].shift(periods=1)
all_tickers_df['pctChange'] = all_tickers_df['close'] / all_tickers_df['previous_close']
all_tickers_df['naturalLog'] = np.log(all_tickers_df['pctChange'])
rolling_std_column = all_tickers_df.groupby(by=['ticker'], as_index=False)['naturalLog'].rolling(20).std()
all_tickers_df['rollingSTD'] = rolling_std_column.reset_index(level=0, drop=True)
one_std_column = (all_tickers_df.groupby(by=['ticker'], as_index=False)['close'].rolling(1).sum() * all_tickers_df.groupby(by=['ticker'], as_index=False)['rollingSTD'].rolling(1).sum() * (np.sqrt(5))) / (np.sqrt(365))
all_tickers_df['onestandarddeviationmove'] = one_std_column.reset_index(level=0, drop=True)
max_close_column = all_tickers_df.groupby(by=['ticker'])['close'].rolling(200).max()
max_close_df = pd.DataFrame(data=max_close_column)
max_close_df.reset_index(level=0, inplace=True)
all_tickers_df['maxPrice'] = max_close_df['close'].round(decimals=4)
min_close_column = all_tickers_df.groupby(by=['ticker'])['close'].rolling(200).min()
min_close_df = pd.DataFrame(data=min_close_column)
min_close_df.reset_index(level=0, inplace=True)
all_tickers_df['minPrice'] = min_close_df['close'].round(decimals=4)
all_tickers_df['highandLowPriceDiff'] = all_tickers_df['maxPrice'] - all_tickers_df['minPrice'] 
all_tickers_df['twentythreeRetracement'] = np.round(all_tickers_df['maxPrice'] - (all_tickers_df['highandLowPriceDiff'] *.236),decimals=4)
all_tickers_df['thirtyeightRetracement'] = np.round(all_tickers_df['maxPrice'] - (all_tickers_df['highandLowPriceDiff'] *.38) ,decimals=4)
all_tickers_df['sixtytwoRetracement'] = np.round(all_tickers_df['maxPrice'] - (all_tickers_df['highandLowPriceDiff'] *.618),decimals=4) 
all_tickers_df['twentythreeRetracementDiff'] = all_tickers_df['close'] - all_tickers_df['twentythreeRetracement']
all_tickers_df['thirtyeightRetracementDiff'] = all_tickers_df['close'] - all_tickers_df['thirtyeightRetracement']
all_tickers_df['sixtytwoRetracementDiff'] = all_tickers_df['close'] - all_tickers_df['sixtytwoRetracement']
all_tickers_df.reset_index(drop=True,inplace=True)
price_target = [all_tickers_df['sixtytwoRetracement'][x] if all_tickers_df['sixtytwoRetracementDiff'][x] > 0 else all_tickers_df['thirtyeightRetracement'][x] if all_tickers_df['thirtyeightRetracementDiff'][x] else all_tickers_df['twentythreeRetracement'][x] for x in all_tickers_df.index ]
all_tickers_df['priceTarget'] = price_target
all_tickers_df.drop(labels=['rollingSTD','naturalLog', 'pctChange'],axis=1,inplace=True)
thirtyDayMVA = all_tickers_df.groupby(by=['ticker'])['close'].rolling(20).mean()
fortyFiveDayMVA = all_tickers_df.groupby(by=['ticker'])['close'].rolling(30).mean()
fortyFiveDaySTD = all_tickers_df.groupby(by=['ticker'])['close'].rolling(30).std()
thirtyDayMVA_df = pd.DataFrame(data=thirtyDayMVA)
fortyFiveDayMVA_df = pd.DataFrame(data=fortyFiveDayMVA)
fortyFiveDaySTD_df = pd.DataFrame(data=fortyFiveDaySTD)
thirtyDayMVA_df.rename(columns={"close": "thirtyDayMVA"}, inplace = True)
fortyFiveDayMVA_df.rename(columns={"close": "fortyFiveDayMVA"}, inplace = True)
fortyFiveDaySTD_df.rename(columns={"close": "fortyFiveDaySTD"}, inplace = True)
thirtyDayMVA_df.reset_index(level = 0,inplace = True)
fortyFiveDayMVA_df.reset_index(level = 0,inplace = True)
fortyFiveDaySTD_df.reset_index(level = 0,inplace = True)
thirtyDayMVA_df.sort_index(inplace=True)
thirtyDayMVA_df.reset_index(inplace=True)
fortyFiveDayMVA_df.sort_index(inplace=True)
fortyFiveDayMVA_df.reset_index(inplace=True)
fortyFiveDaySTD_df.sort_index(inplace=True)
fortyFiveDaySTD_df.reset_index(inplace=True)
all_tickers_df.sort_index(inplace=True)
all_tickers_df['thirtyDayMVA'] = thirtyDayMVA_df['thirtyDayMVA']
all_tickers_df['fortyFiveDayMVA'] = fortyFiveDayMVA_df['fortyFiveDayMVA']
all_tickers_df['fortyFiveDaySTD'] = fortyFiveDaySTD_df['fortyFiveDaySTD']
all_tickers_df['BollingerBandLow'] = all_tickers_df['fortyFiveDayMVA'] - (all_tickers_df['fortyFiveDaySTD']*2)
all_tickers_df['BollingerBandHigh'] = all_tickers_df['fortyFiveDayMVA'] + (all_tickers_df['fortyFiveDaySTD']*2)
all_tickers_df['aboveThirtyDayMVA'] = all_tickers_df['close'] > all_tickers_df['thirtyDayMVA']
all_tickers_df['belowBollingerBandLow'] =  all_tickers_df['BollingerBandLow'] > all_tickers_df['close']
all_tickers_df['diffBollingBandLowClose'] =  (all_tickers_df['BollingerBandLow'] - all_tickers_df['close']).abs() / all_tickers_df['close']
all_tickers_df['belowBollingerBandHigh'] = all_tickers_df['close'] < all_tickers_df['BollingerBandHigh']
all_tickers_df['aboveBollingerBandLow'] =  all_tickers_df['BollingerBandLow'] < all_tickers_df['close']
all_tickers_df['aboveBollingerBandHigh'] = all_tickers_df['close'] > all_tickers_df['BollingerBandHigh']
all_tickers_df['diffBollingerBandHighClose'] = (all_tickers_df['close'] - all_tickers_df['BollingerBandHigh']).abs() / all_tickers_df['close']
all_tickers_df.loc[all_tickers_df['aboveThirtyDayMVA'] == True, 'distanceAboveThirtyDayMVA'] = (all_tickers_df['close'] - all_tickers_df['BollingerBandHigh'])**2
all_tickers_df.loc[all_tickers_df['aboveThirtyDayMVA'] == False,'distanceBelowThirtyDayMVA'] = (all_tickers_df['close'] - all_tickers_df['BollingerBandLow'])**2

"""
keep all the dates in the get data all csv

"""
all_tickers_df_max = all_tickers_df.loc[(all_tickers_df['date'] == ticker_end_date_string) & (all_tickers_df['close'] <= 80) & (all_tickers_df['volume'].astype('int64') >= 1000000)]
all_tickers_df_max.to_csv(r'c:\users\cosmi\onedrive\desktop\get_data_max_date.csv')
list_of_tickers = pd.DataFrame(data=all_tickers_df_max['ticker'].drop_duplicates(inplace=False))
all_tickers_to_analyze_df = pd.concat([pd.DataFrame(data=all_tickers_df.loc[all_tickers_df['ticker'] == x]) for x in list_of_tickers['ticker']])

"""
keep the max date in the max date csv

"""

"""
 
the scalar normalizeds the distance above or below the thirty day average from 0 to 1

"""
scaler = MinMaxScaler()
scaler.fit(all_tickers_df_max[['distanceAboveThirtyDayMVA', 'distanceBelowThirtyDayMVA']])
adjustedValues = scaler.transform(all_tickers_df_max[['distanceAboveThirtyDayMVA','distanceBelowThirtyDayMVA']])
all_tickers_df_max['distanceAboveThirtyDayMVAAdjusted'] = adjustedValues[:,0]
all_tickers_df_max['distanceBelowThirtyDayMVAAdjusted'] = adjustedValues[:,1]
all_tickers_df_max.to_csv(r'c:\users\cosmi\onedrive\desktop\get_data_max_date.csv')


"""

linear regression 
modulairized code

"""
dateTuple = (30,45,60,120,180,240)
param_list = [{'number_of_days':x} for x in dateTuple]
#with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = [executor.submit(af.getSecurityLinearModels , df=all_tickers_to_analyze_df,number_of_days=param.get('number_of_days')) for param in param_list]
#     return_value = [f.result() for f in futures]

with concurrent.futures.ThreadPoolExecutor() as executor:
     futures = [executor.submit(af.getSecurityLinearModels2 , df=all_tickers_to_analyze_df,number_of_days=param.get('number_of_days'), ticker=ticker) for ticker in list_of_tickers['ticker'] for param in param_list]
     return_value = [f.result() for f in futures]

model_DF = pd.concat([pd.DataFrame(data=x)for x in return_value])
model_DF['y1'] = model_DF['intercept_'] + (model_DF['coef_0'] * 0) + (model_DF['coef_1'] * 0) + (model_DF['coef_2'] * 0) + (model_DF['coef_3'] * 0) 
model_DF['y2'] = model_DF['intercept_'] + (model_DF['coef_0'] * 1) + (model_DF['coef_1'] * 1) + (model_DF['coef_2'] * 1) + (model_DF['coef_3'] * 1) 
model_DF['y1-y2'] = model_DF['y1'] - model_DF['y2']
model_DF['x1'] = 0
model_DF['x2'] = 1
model_DF['x1-x2'] = model_DF['x1'] - model_DF['x2']
model_DF['slope'] = (model_DF['y1-y2'] / model_DF['x1-x2']) 
model_DF['min_date_time'] = pd.to_datetime(model_DF['min_date_time'])
all_tickers_df['date'] = pd.to_datetime(all_tickers_df['date'])
af.to_csv_bulk(data=model_DF,df_size=1000000, chunk_count = 100000, refreshOutput=True,outputfile=r'c:\users\cosmi\onedrive\desktop\model_Data.csv')
all_tickers_df = all_tickers_df.merge(right=model_DF,how='left',left_on=['date','ticker'],right_on=['min_date_time','security'])
all_tickers_df.drop(labels='security', axis=1, inplace=True)
all_tickers_df.drop_duplicates(subset=['date','ticker'], inplace=True)
af.to_csv_bulk(data=all_tickers_df,df_size=1000000, chunk_count = 100000, refreshOutput=True,outputfile=r'c:\users\cosmi\onedrive\desktop\get_data_all.csv')




