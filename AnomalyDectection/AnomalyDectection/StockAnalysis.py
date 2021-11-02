import pandas as pd
import numpy as np
import math as mth
import StockFunctions as sf
import datetime as dt
import os
from os import path, remove
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler,StandardScaler,scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
load data

"""

ticker_start_date = dt.date(2019,9,13)
ticker_end_date = dt.date(2021,11,1)

oil_df  = sf.get_reports()
print(oil_df.tail(50))
sf.to_csv_bulk(data=oil_df,df_size = 1000000,chunk_count=100000,refreshOutput=True,outputfile = r'c:\users\cosmi\onedrive\desktop\oil_prices.csv')
companies_df = sf.read_csv_bulk(input_file =  r'c:\users\cosmi\onedrive\desktop\companies_info_test.csv',file_size = 1000000000,chunk_count = 100000)
companies_df = companies_df[['ticker','industry']]
vix_df = sf.read_csv_bulk(input_file =  r'c:\users\cosmi\onedrive\desktop\vix_data.csv',file_size = 1000000000,chunk_count = 100000)
vix_df = vix_df[['date', 'close']]
spy_df = sf.read_csv_bulk(input_file =  r'c:\users\cosmi\onedrive\desktop\spy500_test.csv',file_size = 1000000000,chunk_count = 100000)
etf_df = sf.read_csv_bulk(input_file =  r'c:\users\cosmi\onedrive\desktop\etf_test.csv',file_size = 1000000000,chunk_count = 100000)
#oil_df = etf_df.loc[etf_df['ticker']=='USO']
#oil_df = oil_df[['date', 'close']]
energy_df = etf_df.loc[etf_df['ticker']=='XLE']
energy_df = energy_df[['date', 'close']]
data_df = sf.read_csv_bulk(input_file =  r'c:\users\cosmi\onedrive\desktop\get_data_all_test.csv',file_size = 1000000000,chunk_count = 100000)
#data_df = data_df.loc[data_df['ticker']=='HAL']
data_df = data_df[['close','ticker','date', 'distanceAboveTwentyDayMVA','distanceBelowTwentyDayMVA','twentyDayMVA', 'fiftyDayMVA', 'index_close', 'volume']]
data_df = data_df.merge(right=vix_df,how='inner', on='date')
data_df.rename(columns={"close_x": "close", "close_y": "vix_close"}, inplace = True)
data_df = data_df.merge(right=oil_df,how='left', on='date')
data_df.rename(columns={"close_x": "close"}, inplace = True)
data_df = data_df.merge(right=energy_df,how='inner', on='date')
data_df.rename(columns={"close_x": "close", "close_y": "energy_close"}, inplace = True)
data_df = data_df.merge(right=companies_df,how='inner', on='ticker')
data_df = data_df.loc[data_df['industry']== 'Oil & Gas Equipment & Services']
distance_from_20_day = [1 if data_df['close'][record] > data_df['twentyDayMVA'][record] else 0 for record in data_df.index]
data_df.loc[:, 'distance'] = distance_from_20_day
#print(data_df.loc[data_df['ticker']=='BKR',['date','ticker', 'close', 'twentyDayMVA', 'distanceAboveTwentyDayMVA', 'distanceBelowTwentyDayMVA', 'distance']].tail(50))
#variable_df = sf.featureSelection(df = data_df, number_of_days = 45, ticker = 'BKR')
#model_df = sf.getLinearModel(df = data_df, number_of_days = 45, ticker = 'HAL')
#model_df = sf.getSecurityLinearModels(df = data, number_of_days = 45, ticker = 'DOW')
list_of_tickers = data_df['ticker'].drop_duplicates(keep='first',inplace=False)
dateTuple = (45,)
param_list = [{'number_of_days':x} for x in dateTuple]
with concurrent.futures.ThreadPoolExecutor() as executor:
     futures = [executor.submit(sf.getSecurityLinearModels , df=data_df.loc[data_df['ticker']==ticker],number_of_days=param.get('number_of_days'), ticker=ticker) for ticker in list_of_tickers for param in param_list]
     return_value = [f.result() for f in futures]

model_DF = pd.concat([pd.DataFrame(data=x)for x in return_value])
sf.to_csv_bulk(data=model_DF,df_size = 1000000,chunk_count=100000,refreshOutput=True,outputfile = r'c:\users\cosmi\onedrive\desktop\model_test.csv')
#data = sf.get_ticker_jobs(
 #                       refresh_index = False,
 #                       refresh_data=False,
 #                       index='SPY',
 #                       outputfile=r'c:\users\cosmi\onedrive\desktop\sp500_test.csv',
 #                       njobs=4, 
 #                       start_date=ticker_start_date
 #                   )  

#data = data.loc[data['ticker']=='CMCSA', ['close', 'twentyDayMVA','open', 'high', 'low']]
#print(data.tail(45))
#scaler = StandardScaler()
#Scale
#print(scaler.fit(data))
#print(scaler.mean_)
#print(scaler.transform([[20, 20,20,20]]))

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
other_data_df = sf.get_ticker_jobs(
                     refresh_index=False,
                     refresh_data=False,
                     index='SPY',
                     outputfile=r'c:\users\cosmi\onedrive\desktop\other_test.csv',
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
stock_data_df = pd.concat(pd.DataFrame(data=x)for x in [stock_data_df,etf_data_df,other_data_df])
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
                        outfile=r'c:\users\cosmi\onedrive\desktop\companies_info_test.csv',
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
one_week_std_column = (stock_data_df.groupby(by=['ticker'], as_index=False)['close'].rolling(1).sum() * stock_data_df.groupby(by=['ticker'], as_index=False)['rollingSTD'].rolling(1).sum() * (np.sqrt(5))) / (np.sqrt(253))
stock_data_df['oneweekstandarddeviationmove'] = one_week_std_column.reset_index(level=0, drop=True)
two_week_std_column = (stock_data_df.groupby(by=['ticker'], as_index=False)['close'].rolling(1).sum() * stock_data_df.groupby(by=['ticker'], as_index=False)['rollingSTD'].rolling(1).sum() * (np.sqrt(10))) / (np.sqrt(253))
stock_data_df['twoweekstandarddeviationmove'] = two_week_std_column.reset_index(level=0, drop=True)
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
twentyDayMVA = stock_data_df.groupby(by=['ticker'])['close'].rolling(20).mean()
fiftyDayMVA = stock_data_df.groupby(by=['ticker'])['close'].rolling(50).mean()
twentyDaySTD = stock_data_df.groupby(by=['ticker'])['close'].rolling(20).std()
twentyDayMVA_df = pd.DataFrame(data=twentyDayMVA)
fiftyDayMVA_df = pd.DataFrame(data=fiftyDayMVA)
twentyDaySTD_df = pd.DataFrame(data=twentyDaySTD)
twentyDayMVA_df.rename(columns={"close": "twentyDayMVA"}, inplace = True)
fiftyDayMVA_df.rename(columns={"close": "fiftyDayMVA"}, inplace = True)
twentyDaySTD_df.rename(columns={"close": "twentyDaySTD"}, inplace = True)
twentyDayMVA_df.reset_index(level = 0,inplace = True)
fiftyDayMVA_df.reset_index(level = 0,inplace = True)
twentyDaySTD_df.reset_index(level = 0,inplace = True)
twentyDayMVA_df.sort_index(inplace=True)
twentyDayMVA_df.reset_index(inplace=True)
fiftyDayMVA_df.sort_index(inplace=True)
fiftyDayMVA_df.reset_index(inplace=True)
#fortyFiveDaySTD_df.sort_index(inplace=True)
#fortyFiveDaySTD_df.reset_index(inplace=True)
stock_data_df.sort_index(inplace=True)
stock_data_df['twentyDayMVA'] = twentyDayMVA_df['twentyDayMVA']
stock_data_df['fiftyDayMVA'] = fiftyDayMVA_df['fiftyDayMVA']
stock_data_df['twentyDaySTD'] = twentyDaySTD_df['twentyDaySTD']
stock_data_df['BollingerBandLow'] = stock_data_df['twentyDayMVA'] - (stock_data_df['twentyDaySTD']*2.5)
stock_data_df['BollingerBandHigh'] = stock_data_df['twentyDayMVA'] + (stock_data_df['twentyDaySTD']*2.5)
stock_data_df['aboveTwentyDayMVA'] = stock_data_df['close'] > stock_data_df['twentyDayMVA']
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

"""
stock_data_df_max = stock_data_df.loc[(stock_data_df['date'] == ticker_end_date.strftime('%Y-%m-%d')) & (stock_data_df['close'] <= 120) & (stock_data_df['volume'].astype('int64') >= 1000000)]
stock_data_df_max.to_csv(r'c:\users\cosmi\onedrive\desktop\get_data_max_date_test.csv')
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

dateTuple = (45,)
param_list = [{'number_of_days':x} for x in dateTuple]
with concurrent.futures.ThreadPoolExecutor() as executor:
     futures = [executor.submit(sf.getSecurityLinearModels , df=stock_data_df,number_of_days=param.get('number_of_days'), ticker=ticker) for ticker in list_of_tickers for param in param_list]
     return_value = [f.result() for f in futures]

model_DF = pd.concat([pd.DataFrame(data=x)for x in return_value])
model_DF['y1'] = model_DF['intercept_'] + (model_DF['coef_0'] * 0) + (model_DF['coef_1'] * 0) + (model_DF['coef_2'] * 0) + (model_DF['coef_3'] * 0) + (model_DF['coef_4'] * 0) + (model_DF['coef_5'] * 0) + (model_DF['coef_6'] * 0) + (model_DF['coef_7'] * 0)
model_DF['y2'] = model_DF['intercept_'] + (model_DF['coef_0'] * 1) + (model_DF['coef_1'] * 1) + (model_DF['coef_2'] * 1) + (model_DF['coef_3'] * 1) + (model_DF['coef_4'] * 1) + (model_DF['coef_5'] * 1) + (model_DF['coef_6'] * 1) + (model_DF['coef_7'] * 1) 
model_DF['y1-y2'] = model_DF['y1'] - model_DF['y2']
model_DF['x1'] = 0
model_DF['x2'] = 1
model_DF['x1-x2'] = model_DF['x1'] - model_DF['x2']
model_DF['slope'] = (model_DF['y1-y2'] / model_DF['x1-x2']) 
model_DF = model_DF[['score', 'slope', 'max_date_time', 'security']]
model_DF['max_date_time'] = model_DF['max_date_time'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
stock_data_df = stock_data_df.merge(right=model_DF,how='left',left_on=['date','ticker'],right_on=['max_date_time','security'])
stock_data_df.drop(labels='security', axis=1, inplace=True)
stock_data_df.drop_duplicates(subset=['date','ticker'], inplace=True)

open_positions_df = sf.getOpenPositions(file = r'c:\users\cosmi\onedrive\desktop\portfolio.csv')
stock_data_df = stock_data_df.merge(right=open_positions_df,how='left', left_on=['date','ticker'], right_on=['date','ticker'])
stock_data_df['entry_price'] = [stock_data_df['entry_price_y'][x] if stock_data_df['open_position'][x] == 1 else stock_data_df['entry_price_x'][x] for x in stock_data_df.index] 
stock_data_df['exit_price'] = [stock_data_df['exit_price_y'][x] if stock_data_df['open_position'][x] == 1 else stock_data_df['exit_price_x'][x] for x in stock_data_df.index] 
stock_data_df.drop(columns=['exit_price_y','exit_price_x','entry_price_y', 'entry_price_x', 'Unnamed: 0'],inplace=True)
#print(stock_data_df.loc[stock_data_df['ticker']=='DOW', ['date', 'ticker', 'close', 'record_count']].head(20))
sf.to_csv_bulk(data=stock_data_df,df_size = 1000000,chunk_count=100000,refreshOutput=True,outputfile = r'c:\users\cosmi\onedrive\desktop\get_data_all_test.csv')


                