import pandas as pd
import numpy as np
import math as mth
import StockFunctions as sf
import datetime as dt
import os
from os import path, remove
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler,scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
load data

"""
ticker_start_date = dt.date(2019,9,13)
#ticker_start_date_string = ticker_start_date.strftime('%Y-%m-%d')
ticker_end_date = dt.date(2021,9,20)
#ticker_end_date_string = ticker_end_date.strftime('%Y-%m-%d')

index_list =['DIA','SPY','NDX']
index_prices_file_list = [r'c:\users\cosmi\onedrive\desktop\dia_test.csv',
                          r'c:\users\cosmi\onedrive\desktop\spy500_test.csv',
                          r'c:\users\cosmi\onedrive\desktop\ndx_test.csv']

list_of_index = zip(index_list,index_prices_file_list)
list_of_index = list(list_of_index)

index_tickers_file_list = [r'c:\users\cosmi\onedrive\desktop\dow_tickers_test.csv',
                           r'c:\users\cosmi\onedrive\desktop\sp500_tickers_test.csv',
                           r'c:\users\cosmi\onedrive\desktop\nasdaq_tickers_test.csv']

list_of_index_tickers = zip(index_list,index_tickers_file_list)
list_of_index_tickers = list(list_of_index_tickers)

stock_prices_files_list = [r'c:\users\cosmi\onedrive\desktop\dow_test.csv',
                           r'c:\users\cosmi\onedrive\desktop\sp500_test.csv',
                           r'c:\users\cosmi\onedrive\desktop\nasdaq_test.csv']

list_of_stock_prices = zip(index_list,stock_prices_files_list)
list_of_stock_prices = list(list_of_stock_prices)


index_data_lists =[ sf.get_index_data
                       (
                         ticker = x[0],
                         start_date = ticker_start_date,
                        #end_date = ticker_end_date,
                         outfile= x[1], 
                         refreshFileOutput=False
                        )
                    for x in list_of_index
                  ]
index_data_df = pd.concat([pd.DataFrame(data=x) for x in index_data_lists])
index_data_df = index_data_df[['date', 'adjclose', 'ticker', 'index_rolling_std']]
index_data_df.rename(columns={'date':'index_date', 'adjclose':'index_close'},inplace=True)
index_data_df.reset_index(inplace=True)

stock_data_lists = [ sf.get_ticker_jobs
                    (
                        refresh_index = False,
                        refresh_data=False,
                        index=x[0],
                        outputfile=x[1],
                        njobs=4, 
                        start_date=ticker_start_date
                    )   
                 for x in list_of_stock_prices
               ]

stock_data_df = pd.concat([pd.DataFrame(data=x) for x in stock_data_lists])
etf_data_df = sf.get_ticker_jobs(
                     refresh_index=False,
                     refresh_data=False,
                     index='ETF',
                     outputfile=r'c:\users\cosmi\onedrive\desktop\etf_test.csv',
                     njobs=1,
                     start_date=ticker_start_date
                 )

gld_data_df = sf.get_index_data(
                     ticker = 'GLD',
                     start_date = ticker_start_date,
                     #end_date = ticker_end_date,
                     outfile=r'c:\users\cosmi\onedrive\desktop\gld.csv', 
                     refreshFileOutput=False
                   )
gld_data_df.rename(columns={'ticker':'gold_index', 'close':'gold_close', 'date':'gold_date'}, inplace=True)
gld_data_df =gld_data_df[['gold_index', 'gold_close', 'gold_date']]

sptl_data_df = sf.get_index_data(
                     ticker = 'SPTL',
                     start_date = ticker_start_date,
                     #end_date = ticker_end_date,
                     outfile=r'c:\users\cosmi\onedrive\desktop\sptl.csv',
                     refreshFileOutput=False
                   )
sptl_data_df.rename(columns={'ticker':'10_year_note_index', 'close':'10_year_note_close', 'date':'10_year_note_date'},inplace=True)
sptl_data_df =sptl_data_df[['10_year_note_index', '10_year_note_close', '10_year_note_date']]
sptl_data_df.sort_values(by=['10_year_note_date'],ascending=['ascending'],inplace=True)

stock_data_df = pd.concat(pd.DataFrame(data=x)for x in [stock_data_df,etf_data_df])
stock_data_df.drop_duplicates(['ticker','date'],inplace=True)
stock_data_df.reset_index(drop=True,inplace=True)
index = ['DOW' if index_data_df['ticker'][x] == 'DIA' else 'NASDAQ' if index_data_df['ticker'][x] == 'NDX' else 'SPY500' for x in index_data_df.index]
index_data_df['ticker'] = index
stock_data_df = stock_data_df.merge(right=index_data_df,how='inner', left_on=['index', 'date'], right_on=['ticker','index_date'])
stock_data_df.drop(columns=['ticker_y', 'index_date','index_y'], inplace=True)
stock_data_df.rename(columns={'ticker_x':'ticker','index_x':'index'}, inplace=True)
stock_data_df = stock_data_df.merge(right=gld_data_df, how='inner', left_on='date', right_on='gold_date')
stock_data_df.drop(columns='gold_date',inplace=True)
stock_data_df = stock_data_df.merge(right=sptl_data_df, how='inner', left_on='date', right_on='10_year_note_date')
stock_data_df.drop(columns='10_year_note_date',inplace=True)
companies_info_df = sf.get_companies_info(
                        stock_data_df['ticker'].drop_duplicates(keep='first',inplace=False),
                        outfile=r'c:\users\cosmi\onedrive\desktop\portfolio_analysis_test.csv',
                        refreshFileOutput=False
                        )
stock_data_df = stock_data_df.merge(right=companies_info_df,how='left',on='ticker')

"""
add additional measures

"""
stock_age_df = pd.DataFrame(data=stock_data_df.groupby(by=['ticker'])['close'].count())
stock_age_df.reset_index(inplace=True)
stock_age_df.rename(columns={'close':'record_count'},inplace=True)
"""
add the open position for real analysis

"""
#stock_data_df = stock_data_df.loc[(stock_data_df['ticker']=='FCX') | (stock_data_df['ticker']=='DOW')]
stock_data_df = stock_data_df.merge(right=stock_age_df,how='inner',on='ticker')
stock_data_df = stock_data_df.loc[(stock_data_df['record_count']>= 240)]
stock_data_df.sort_values(by=['ticker','date'], inplace=True)
stock_data_df.reset_index(drop=True,inplace=True)
stock_data_df['previous_close'] = stock_data_df.groupby(by=['ticker'])['close'].shift(periods=1)
stock_data_df['pctChange'] = stock_data_df['close'] / stock_data_df['previous_close']
stock_data_df['naturalLog'] = np.log(stock_data_df['pctChange'])
rolling_std_column = stock_data_df.groupby(by=['ticker'], as_index=False)['naturalLog'].rolling(20).std()
stock_data_df['rollingSTD'] = rolling_std_column.reset_index(level=0, drop=True)
one_std_column = (stock_data_df.groupby(by=['ticker'], as_index=False)['close'].rolling(1).sum() * stock_data_df.groupby(by=['ticker'], as_index=False)['rollingSTD'].rolling(1).sum() * (np.sqrt(5))) / (np.sqrt(365))
stock_data_df['onestandarddeviationmove'] = one_std_column.reset_index(level=0, drop=True)
max_close_column = stock_data_df.groupby(by=['ticker'])['close'].rolling(200).max()
max_close_df = pd.DataFrame(data=max_close_column)
max_close_df.reset_index(level=0, inplace=True)
stock_data_df['maxPrice'] = max_close_df['close'].round(decimals=4)
min_close_column = stock_data_df.groupby(by=['ticker'])['close'].rolling(200).min()
min_close_df = pd.DataFrame(data=min_close_column)
min_close_df.reset_index(level=0, inplace=True)
stock_data_df['minPrice'] = min_close_df['close'].round(decimals=4)
stock_data_df['highandLowPriceDiff'] = stock_data_df['maxPrice'] - stock_data_df['minPrice'] 
stock_data_df['twentythreeRetracement'] = np.round(stock_data_df['maxPrice'] - (stock_data_df['highandLowPriceDiff'] *.236),decimals=4)
stock_data_df['thirtyeightRetracement'] = np.round(stock_data_df['maxPrice'] - (stock_data_df['highandLowPriceDiff'] *.38) ,decimals=4)
stock_data_df['sixtytwoRetracement'] = np.round(stock_data_df['maxPrice'] - (stock_data_df['highandLowPriceDiff'] *.618),decimals=4) 
stock_data_df['twentythreeRetracementDiff'] = stock_data_df['close'] - stock_data_df['twentythreeRetracement']
stock_data_df['thirtyeightRetracementDiff'] = stock_data_df['close'] - stock_data_df['thirtyeightRetracement']
stock_data_df['sixtytwoRetracementDiff'] = stock_data_df['close'] - stock_data_df['sixtytwoRetracement']
stock_data_df.reset_index(drop=True,inplace=True)
price_target = [stock_data_df['close'][x] if stock_data_df['sixtytwoRetracement'][x] > stock_data_df['close'][x] else stock_data_df['twentythreeRetracement'][x] if stock_data_df['twentythreeRetracementDiff'][x] > 0 else stock_data_df['thirtyeightRetracement'][x] if stock_data_df['thirtyeightRetracementDiff'][x] > 0 else stock_data_df['sixtytwoRetracement'][x] for x in stock_data_df.index]
stock_data_df['priceTarget'] = price_target
stock_data_df.drop(labels=['naturalLog', 'pctChange'],axis=1,inplace=True)
thirtyDayMVA = stock_data_df.groupby(by=['ticker'])['close'].rolling(20).mean()
fortyFiveDayMVA = stock_data_df.groupby(by=['ticker'])['close'].rolling(30).mean()
fortyFiveDaySTD = stock_data_df.groupby(by=['ticker'])['close'].rolling(30).std()
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
stock_data_df.sort_index(inplace=True)
stock_data_df['thirtyDayMVA'] = thirtyDayMVA_df['thirtyDayMVA']
stock_data_df['fortyFiveDayMVA'] = fortyFiveDayMVA_df['fortyFiveDayMVA']
stock_data_df['fortyFiveDaySTD'] = fortyFiveDaySTD_df['fortyFiveDaySTD']
stock_data_df['BollingerBandLow'] = stock_data_df['fortyFiveDayMVA'] - (stock_data_df['fortyFiveDaySTD']*2)
stock_data_df['BollingerBandHigh'] = stock_data_df['fortyFiveDayMVA'] + (stock_data_df['fortyFiveDaySTD']*2)
stock_data_df['aboveThirtyDayMVA'] = stock_data_df['close'] > stock_data_df['thirtyDayMVA']
stock_data_df['belowBollingerBandLow'] =  stock_data_df['BollingerBandLow'] > stock_data_df['close']
stock_data_df['diffBollingBandLowClose'] =  (stock_data_df['BollingerBandLow'] - stock_data_df['close']).abs() / stock_data_df['close']
stock_data_df['belowBollingerBandHigh'] = stock_data_df['close'] < stock_data_df['BollingerBandHigh']
stock_data_df['aboveBollingerBandLow'] =  stock_data_df['BollingerBandLow'] < stock_data_df['close']
stock_data_df['aboveBollingerBandHigh'] = stock_data_df['close'] > stock_data_df['BollingerBandHigh']
stock_data_df['diffBollingerBandHighClose'] = (stock_data_df['close'] - stock_data_df['BollingerBandHigh']).abs() / stock_data_df['close']
stock_data_df.loc[stock_data_df['aboveThirtyDayMVA'] == True, 'distanceAboveThirtyDayMVA'] = (stock_data_df['close'] - stock_data_df['BollingerBandHigh'])**2
stock_data_df.loc[stock_data_df['aboveThirtyDayMVA'] == False,'distanceBelowThirtyDayMVA'] = (stock_data_df['close'] - stock_data_df['BollingerBandLow'])**2

"""
keep all the dates in the get data all csv

"""
stock_data_df_max = stock_data_df.loc[(stock_data_df['date'] == ticker_end_date.strftime('%Y-%m-%d')) & (stock_data_df['close'] <= 80) & (stock_data_df['volume'].astype('int64') >= 1000000)]
stock_data_df_max.to_csv(r'c:\users\cosmi\onedrive\desktop\get_data_max_date_test.csv')
list_of_tickers = stock_data_df_max['ticker'].drop_duplicates(keep='first',inplace=False)
stock_data_df = pd.concat([pd.DataFrame(data=stock_data_df.loc[stock_data_df['ticker'] == x]) for x in list_of_tickers._values])
stock_data_df = sf.get_min_max_scaler(df=stock_data_df)
portfolio_df = sf.getPortfolio(df=stock_data_df[['date','ticker','close','index','index_close', 'priceTarget', 'rollingSTD', 'index_rolling_std', 'onestandarddeviationmove', 'sector', 'industry']])
portfolio_df = portfolio_df[['exit_price','entry_price', 'market_corr', 'beta','Pct_To_Entry_Price']]
stock_data_df.reset_index(drop=True, inplace=True)
stock_data_df['Pct_To_Entry_Price'] = portfolio_df['Pct_To_Entry_Price'] 
stock_data_df['market_corr'] = portfolio_df['market_corr'] 
stock_data_df['entry_price'] = portfolio_df['entry_price'] 
stock_data_df['exit_price'] = portfolio_df['exit_price'] 
stock_data_df['beta'] = portfolio_df['beta']
print(stock_data_df.loc[stock_data_df['ticker']=='FCX'].tail(50))
sf.to_csv_bulk(data=stock_data_df,df_size = 1000000,chunk_count=100000,refreshOutput=True,outputfile = r'c:\users\cosmi\onedrive\desktop\get_data_all_test.csv')


                