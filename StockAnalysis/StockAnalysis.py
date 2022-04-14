from operator import le
import pandas as pd
import numpy as np
import math as mth
import StockFunctions as sf
import datetime as dt
import os
from os import path, remove
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler,StandardScaler,scale, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree


"""
load data

"""

ticker_start_date = dt.date(2019,9,16)
ticker_end_date = dt.date(2022,4,13)

#df = sf.read_csv_bulk(input_file = r'c:\investment_data\get_data_all_test.csv', file_size = 1000000000,chunk_count = 100000)
#df = sf.backtestDecisionTree(data=df)
#print(df.shape)


index_list =['^DJI','^GSPC','^IXIC']
index_prices_file_list = [r'c:\users\cosmi\onedrive\desktop\dia.csv',
                          r'c:\users\cosmi\onedrive\desktop\spy500.csv',
                          r'c:\users\cosmi\onedrive\desktop\nasdaq.csv']

list_of_index = zip(index_list,index_prices_file_list)
list_of_index = list(list_of_index)

index_tickers_file_list = [r'c:\users\cosmi\onedrive\desktop\dow_tickers_test.csv',
                           r'c:\users\cosmi\onedrive\desktop\sp500_tickers_test.csv',
                           r'c:\users\cosmi\onedrive\desktop\nasdaq_tickers_test.csv']

list_of_index_tickers = zip(index_list,index_tickers_file_list)
list_of_index_tickers = list(list_of_index_tickers)

stock_prices_files_list = [r'c:\investment_data\dow.csv',
                           r'c:\investment_data\sp500.csv',
                           r'c:\investment_data\nasdaq.csv']

list_of_stock_prices = zip(index_list,stock_prices_files_list)
list_of_stock_prices = list(list_of_stock_prices)


index_data_lists =[ sf.get_index_data
                       (
                         ticker = x[0],
                         start_date = ticker_start_date,
                        #end_date = ticker_end_date,
                         outfile= x[1], 
                         refreshFileOutput=True
                        )
                    for x in list_of_index
                  ]
index_data_df = pd.concat([pd.DataFrame(data=x) for x in index_data_lists])
export_index_df = index_data_df
index_data_df = index_data_df[['date', 'adjclose', 'ticker', 'index_rolling_std']]
index_data_df.rename(columns={'date':'index_date', 'adjclose':'index_close'},inplace=True)
index_data_df.reset_index(inplace=True)
print(index_data_df.loc[index_data_df['ticker']=='^IXIC'].tail(50))
stock_data_lists = [ sf.get_ticker_jobs
                    (
                        refresh_index = False,
                        refresh_data=True,
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
                     refresh_data=True,
                     index='ETF',
                     outputfile=r'c:\investment_data\etf.csv',
                     njobs=1,
                     start_date=ticker_start_date
                 )
other_data_df = sf.get_ticker_jobs(
                     refresh_index=False,
                     refresh_data=True,
                     index='Other',
                     outputfile=r'c:\investment_data\other.csv',
                     njobs=1,
                     start_date=ticker_start_date
                 )
oil_data_df = sf.get_index_data(
                     ticker = 'CL=F',
                     start_date = ticker_start_date,
                     #end_date = ticker_end_date,
                     outfile=r'c:\investment_data\oil.csv', 
                     refreshFileOutput=True
                   )
oil_data_df.dropna(subset = ['open'], inplace=True)
export_index_df = pd.concat([export_index_df,oil_data_df],ignore_index=True)
bit_data_df = sf.get_index_data(
                     ticker = 'BTC-USD',
                     start_date = ticker_start_date,
                     #end_date = ticker_end_date,
                     outfile=r'c:\investment_data\bitcoin.csv', 
                     refreshFileOutput=True
                   )
bit_data_df.dropna(subset = ['open'], inplace=True)
export_index_df = pd.concat([export_index_df,bit_data_df],ignore_index=True)
gld_data_df = sf.get_index_data(
                     ticker = 'GLD',#'GC=F',
                     start_date = ticker_start_date,
                     #end_date = ticker_end_date,
                     outfile=r'c:\investment_data\gld.csv', 
                     refreshFileOutput=True
                   )
gld_data_df.dropna(subset = ['open'], inplace=True)
export_index_df = pd.concat([export_index_df, gld_data_df], ignore_index=True)
gld_data_df.rename(columns={'ticker':'gold_index', 'close':'gold_close', 'date':'gold_date'}, inplace=True)
gld_data_df =gld_data_df[['gold_index', 'gold_close', 'gold_date']]
sptl_data_df = sf.get_index_data(
                     ticker = '^TNX',
                     start_date = ticker_start_date,
                     #end_date = ticker_end_date,
                     outfile=r'c:\investment_data\ten_year_bond.csv',
                     refreshFileOutput=True
                   )
sptl_data_df.dropna(subset = ['open'], inplace=True)
export_index_df = pd.concat([export_index_df, sptl_data_df], ignore_index=True)
sf.to_csv_bulk(data=export_index_df,df_size = 1000000,chunk_count=100000,refreshOutput=True,outputfile = r'c:\investment_data\all_indexes.csv')
sptl_data_df.rename(columns={'ticker':'10_year_note_index', 'close':'10_year_note_close', 'date':'10_year_note_date'},inplace=True)
sptl_data_df =sptl_data_df[['10_year_note_index', '10_year_note_close', '10_year_note_date']]
sptl_data_df.sort_values(by=['10_year_note_date'],ascending=['ascending'],inplace=True)
stock_data_df = pd.concat(pd.DataFrame(data=x)for x in [stock_data_df,etf_data_df,other_data_df])
stock_data_df.drop_duplicates(['ticker','date'],inplace=True)
stock_data_df.reset_index(drop=True,inplace=True)
index = ['DOW' if index_data_df['ticker'][x] == '^DJI' else 'NASDAQ' if index_data_df['ticker'][x] == '^IXIC' else 'SPY500' for x in index_data_df.index]
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
                        outfile=r'c:\investment_data\companies_info.csv',
                        refreshFileOutput=True
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

#stock_data_df = stock_data_df.loc[(stock_data_df['ticker']=='HAL') | (stock_data_df['ticker']=='DOW') ]
stock_data_df = stock_data_df.merge(right=stock_age_df,how='inner',on='ticker')
stock_data_df = stock_data_df.loc[(stock_data_df['record_count']>= 240)]
stock_data_df.sort_values(by=['ticker','date'], inplace=True)
stock_data_df.reset_index(drop=True,inplace=True)
stock_data_df['previous_close'] = stock_data_df.groupby(by=['ticker'])['close'].shift(periods=1)
stock_data_df['pctChange'] = stock_data_df['close'] / stock_data_df['previous_close']
stock_data_df['naturalLog'] = np.log(stock_data_df['pctChange'])
stock_data_df['rollingSTD'] = stock_data_df.groupby(by=['ticker'])['naturalLog'].transform(lambda x: x.rolling(20).std())
stock_data_df['oneyearrollingSTD'] = stock_data_df.groupby(by=['ticker'])['naturalLog'].transform(lambda x: x.rolling(252).std())
stock_data_df['oneyearrollingESTSTD'] = stock_data_df['rollingSTD'] * np.sqrt(252)
stock_data_df['oneweekstandarddeviationmove'] = (stock_data_df['rollingSTD'] * np.sqrt(5)) * stock_data_df['close'] 
stock_data_df['twoweekstandarddeviationmove'] =  (stock_data_df['rollingSTD'] * np.sqrt(10)) * stock_data_df['close']
stock_data_df['maxPrice'] = stock_data_df.groupby(by=['ticker'])['close'].transform(lambda x: x.rolling(200).max()).round(decimals=2)
stock_data_df['minPrice'] = stock_data_df.groupby(by=['ticker'])['close'].transform(lambda x: x.rolling(200).min()).round(decimals=2)
stock_data_df['highandLowPriceDiff'] = stock_data_df['maxPrice'] - stock_data_df['minPrice'] 
stock_data_df['twentythreeRetracement'] = np.round(stock_data_df['maxPrice'] - (stock_data_df['highandLowPriceDiff'] *.236),decimals=4)
stock_data_df['thirtyeightRetracement'] = np.round(stock_data_df['maxPrice'] - (stock_data_df['highandLowPriceDiff'] *.38) ,decimals=4)
stock_data_df['sixtytwoRetracement'] = np.round(stock_data_df['maxPrice'] - (stock_data_df['highandLowPriceDiff'] *.618),decimals=4) 
stock_data_df['twentythreeRetracementDiff'] = stock_data_df['close'] - stock_data_df['twentythreeRetracement']
stock_data_df['thirtyeightRetracementDiff'] = stock_data_df['close'] - stock_data_df['thirtyeightRetracement']
stock_data_df['sixtytwoRetracementDiff'] = stock_data_df['close'] - stock_data_df['sixtytwoRetracement']
stock_data_df['twentythreeRetracementUp'] = np.round(stock_data_df['minPrice'] + (stock_data_df['highandLowPriceDiff'] *.236),decimals=4)
stock_data_df['thirtyeightRetracementUp'] = np.round(stock_data_df['minPrice'] + (stock_data_df['highandLowPriceDiff'] *.38) ,decimals=4)
stock_data_df['sixtytwoRetracementUp'] = np.round(stock_data_df['minPrice'] + (stock_data_df['highandLowPriceDiff'] *.618),decimals=4) 
stock_data_df.reset_index(drop=True,inplace=True)
stock_data_df['priceTarget'] = [stock_data_df['close'][x] if stock_data_df['sixtytwoRetracement'][x] > stock_data_df['close'][x] else stock_data_df['twentythreeRetracement'][x] if stock_data_df['twentythreeRetracementDiff'][x] > 0 else stock_data_df['thirtyeightRetracement'][x] if stock_data_df['thirtyeightRetracementDiff'][x] > 0 else stock_data_df['sixtytwoRetracement'][x] for x in stock_data_df.index]
stock_data_df.drop(labels=['naturalLog', 'pctChange'],axis=1,inplace=True)
stock_data_df.sort_index(inplace=True)
stock_data_df['twentyDayMVA'] = stock_data_df.groupby(by=['ticker'])['close'].transform(lambda x: x.rolling(20).mean())
stock_data_df['twentyDaySTD'] = stock_data_df.groupby(by=['ticker'])['close'].transform(lambda x: x.rolling(20).std())
stock_data_df['fiftyDayMVA'] = stock_data_df.groupby(by=['ticker'])['close'].transform(lambda x: x.rolling(50).mean())
stock_data_df['twohundredDayMVA'] = stock_data_df.groupby(by=['ticker'])['close'].transform(lambda x: x.rolling(200).mean())
stock_data_df['BollingerBandLow'] = stock_data_df['twentyDayMVA'] - (stock_data_df['twentyDaySTD']*2.5)
stock_data_df['BollingerBandHigh'] = stock_data_df['twentyDayMVA'] + (stock_data_df['twentyDaySTD']*2.5)
stock_data_df['aboveTwentyDayMVA'] = stock_data_df['close'] > stock_data_df['twentyDayMVA']
stock_data_df['aboveFiftyDayMVA'] = stock_data_df['close'] > stock_data_df['fiftyDayMVA']
stock_data_df['belowBollingerBandLow'] =  stock_data_df['BollingerBandLow'] > stock_data_df['close']
stock_data_df['diffBollingBandLowClose'] =  (stock_data_df['BollingerBandLow'] - stock_data_df['close']).abs() / stock_data_df['close']
stock_data_df['belowBollingerBandHigh'] = stock_data_df['close'] < stock_data_df['BollingerBandHigh']
stock_data_df['aboveBollingerBandLow'] =  stock_data_df['BollingerBandLow'] < stock_data_df['close']
stock_data_df['aboveBollingerBandHigh'] = stock_data_df['close'] > stock_data_df['BollingerBandHigh']
stock_data_df['diffBollingerBandHighClose'] = (stock_data_df['close'] - stock_data_df['BollingerBandHigh']).abs() / stock_data_df['close']
stock_data_df.loc[stock_data_df['aboveTwentyDayMVA'] == True, 'distanceAboveTwentyDayMVA'] = (stock_data_df['close'] - stock_data_df['BollingerBandHigh'])**2
stock_data_df.loc[stock_data_df['aboveTwentyDayMVA'] == False,'distanceBelowTwentyDayMVA'] = (stock_data_df['close'] - stock_data_df['BollingerBandLow'])**2

"""
keep all the dates in the get data all csv
2021-12-22 updated the code to always return the stocks
we currently have in our portfolio

"""
open_positions_df = sf.getOpenPositions(file = r'c:\users\cosmi\onedrive\desktop\portfolio.csv')
open_positions_df1 = sf.getOpenPositions(file = r'c:\users\cosmi\onedrive\desktop\portfolio_1.csv')
open_positions_df = open_positions_df.append(other = open_positions_df1, ignore_index = True)
open_positions_df.drop_duplicates(subset=['ticker'], inplace=True)
stock_data_df = stock_data_df.merge(right=open_positions_df,how='left',left_on=['ticker','date'],right_on=['ticker', 'date'])
stock_data_df.drop(labels=['Unnamed: 0','entry_price', 'current_price', 'exit_price', 'quantity'], axis = 1, inplace=True)
stock_data_df_max = stock_data_df.loc[(stock_data_df['date'] == ticker_end_date.strftime('%Y-%m-%d')) & (stock_data_df['close'] <= 120) & (stock_data_df['volume'].astype('int64') >= 1000000) | (stock_data_df['open_position']==1)]
stock_data_df.drop(labels=['open_position'],axis = 1, inplace = True)
stock_data_df_max.drop_duplicates(subset=['ticker'], inplace=True)
stock_data_df_max.to_csv(r'c:\investment_data\get_data_max_date.csv')
list_of_tickers = stock_data_df_max['ticker'].drop_duplicates(keep='first',inplace=False)
stock_data_df = pd.concat([pd.DataFrame(data=stock_data_df.loc[stock_data_df['ticker'] == x]) for x in list_of_tickers._values])
stock_data_df = sf.get_min_max_scaler(df=stock_data_df)
portfolio_df = sf.getPortfolio(df=stock_data_df[['date','ticker','close','index','index_close', 'priceTarget', 'rollingSTD', 'index_rolling_std', 'oneweekstandarddeviationmove', 'twoweekstandarddeviationmove', 'sector', 'industry']])
portfolio_df = portfolio_df[['exit_price','entry_price', 'market_corr', 'beta','Pct_To_Entry_Price']]
stock_data_df.reset_index(drop=True, inplace=True)
stock_data_df['Pct_To_Entry_Price'] = portfolio_df['Pct_To_Entry_Price'] 
stock_data_df['market_corr'] = portfolio_df['market_corr'] 
stock_data_df['entry_price'] = portfolio_df['entry_price'] 
stock_data_df['exit_price'] = portfolio_df['exit_price'] 
stock_data_df['beta'] = portfolio_df['beta']

stock_data_df['200Close'] = stock_data_df['close'].shift(periods=200)
stock_data_df['futureClose'] = stock_data_df['close'].shift(periods=-20)
stock_data_df['highlowPct'] = (stock_data_df['close'] - stock_data_df['200Close'])/(stock_data_df['200Close'])

#dateTuple = (45,)
#param_list = [{'number_of_days':x} for x in dateTuple]
#with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = [executor.submit(sf.getSecurityLinearModels , df=stock_data_df,number_of_days=param.get('number_of_days'), ticker=ticker) for ticker in list_of_tickers for param in param_list]
#     return_value = [f.result() for f in futures]

#model_DF = pd.concat([pd.DataFrame(data=x)for x in return_value])
#model_DF['y1'] = model_DF['intercept_'] + (model_DF['coef_0'] * 0) + (model_DF['coef_1'] * 0) + (model_DF['coef_2'] * 0) + (model_DF['coef_3'] * 0) + (model_DF['coef_4'] * 0)
#model_DF['y2'] = model_DF['intercept_'] + (model_DF['coef_0'] * 1) + (model_DF['coef_1'] * 1) + (model_DF['coef_2'] * 1) + (model_DF['coef_3'] * 1) + (model_DF['coef_4'] * 1)
#model_DF['y1-y2'] = model_DF['y1'] - model_DF['y2']
#model_DF['x1'] = 0
#model_DF['x2'] = 1
#model_DF['x1-x2'] = model_DF['x1'] - model_DF['x2']
#model_DF['slope'] = (model_DF['y1-y2'] / model_DF['x1-x2'])
#model_DF = stock_data_df.loc[stock_data_df['date']== ticker_end_date.strftime('%Y-%m-%d'), ['highlowPct', 'date', 'ticker']]
#model_DF.rename(columns={'date':'max_date_time', 'ticker':'security'}, inplace=True)
#model_DF = model_DF[['highlowPct', 'max_date_time', 'security']]
#model_DF = model_DF[['score', 'slope', 'max_date_time', 'security']]
#model_DF['max_date_time'] = model_DF['max_date_time'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
#stock_data_df = stock_data_df.merge(right=model_DF,how='left',left_on=['date','ticker'],right_on=['max_date_time','security'])
#stock_data_df.drop(labels='security', axis=1, inplace=True)
stock_data_df.drop_duplicates(subset=['date','ticker'], inplace=True)
stock_data_1_df = stock_data_df

open_positions_df = sf.getOpenPositions(file = r'c:\users\cosmi\onedrive\desktop\portfolio.csv')
stock_data_df = stock_data_df.merge(right=open_positions_df,how='left', left_on=['date','ticker'], right_on=['date','ticker'])
stock_data_df['entry_price'] = [stock_data_df['entry_price_y'][x] if stock_data_df['open_position'][x] == 1 else stock_data_df['entry_price_x'][x] for x in stock_data_df.index] 
stock_data_df['exit_price'] = [stock_data_df['exit_price_y'][x] if stock_data_df['open_position'][x] == 1 else stock_data_df['exit_price_x'][x] for x in stock_data_df.index] 
stock_data_df.drop(columns=['exit_price_y','exit_price_x','entry_price_y', 'entry_price_x', 'Unnamed: 0'],inplace=True)
model_DF = sf.backtestDecisionTree(data = stock_data_df)
stock_data_df = stock_data_df.merge(right=model_DF, how='left', on=['date', 'ticker'])
sf.to_csv_bulk(data=stock_data_df,df_size = 1000000,chunk_count=100000,refreshOutput=True,outputfile = r'c:\investment_data\get_data_all_test.csv')

open_positions_1_df = sf.getOpenPositions(file = r'c:\users\cosmi\onedrive\desktop\portfolio_1.csv')
stock_data_1_df = stock_data_1_df.merge(right=open_positions_1_df,how='left', left_on=['date','ticker'], right_on=['date','ticker'])
stock_data_1_df['entry_price'] = [stock_data_1_df['entry_price_y'][x] if stock_data_1_df['open_position'][x] == 1 else stock_data_1_df['entry_price_x'][x] for x in stock_data_1_df.index] 
stock_data_1_df['exit_price'] = [stock_data_1_df['exit_price_y'][x] if stock_data_1_df['open_position'][x] == 1 else stock_data_1_df['exit_price_x'][x] for x in stock_data_1_df.index] 
stock_data_1_df.drop(columns=['exit_price_y','exit_price_x','entry_price_y', 'entry_price_x', 'Unnamed: 0'],inplace=True)
model_DF = sf.backtestDecisionTree(data = stock_data_1_df)
stock_data_1_df = stock_data_1_df.merge(right=model_DF, how='left', on=['date', 'ticker'])
sf.to_csv_bulk(data=stock_data_1_df,df_size = 1000000,chunk_count=100000,refreshOutput=True,outputfile = r'c:\investment_data\get_data_all_test_1.csv')


                