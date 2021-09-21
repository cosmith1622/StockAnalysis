import pandas as pd
import numpy as np
import math as mth
import datetime as dt
import sys
import concurrent.futures
import os
from os import path, remove
import requests_html as request
import html5lib as html
import PandasExtra as pe
from yahoo_fin.stock_info import get_data, get_company_info, tickers_nasdaq,tickers_sp500,tickers_dow,tickers_ftse250
from yahoo_finance_api2 import share
from sklearn.preprocessing import MinMaxScaler,scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def get_index_data(ticker: str,start_date: str, end_date: str=None,outfile: str=None, index_as_date: bool=False,refreshFileOutput: bool=False) ->pd.DataFrame:

    if path.exists(outfile) and refreshFileOutput==False:

        return pd.read_csv(outfile)

    else:
    
        index_table = get_data(ticker=ticker, start_date = start_date, end_date = end_date)
        index_table['previous_close'] = index_table['close'].shift(periods=1)
        index_table['pct_change'] = index_table['close'] / index_table['previous_close']
        index_table['natural_log'] = np.log(index_table['pct_change'])
        rolling_std = index_table['natural_log'].rolling(20).std()
        index_table['index_rolling_std'] = rolling_std
        index_table.reset_index(inplace=True)
        index_table.rename(columns={'index':'date'},inplace=True)
        index_table.drop(labels=['previous_close', 'pct_change', 'natural_log'], axis=1, inplace=True)
        if index_table.dtypes['date'] == 'datetime64[ns]':
            index_table['date'] = index_table['date'].dt.strftime('%Y-%m-%d')
        index_table.to_csv(outfile)
        return index_table

def get_ticker_jobs(refresh_index: bool, refresh_data:bool, outputfile:str, index:str, njobs:int, start_date:dt.datetime) ->pd.DataFrame:

    #update index and data
    if refresh_index == True:

        if index == "SPY":
        
            sp500_tickers = get_tickers_sp500(outfile =r'c:\users\cosmi\onedrive\desktop\sp500_tickers.csv',refreshFileOutput=True)
            list_of_arrays = np.array_split(sp500_tickers,njobs)
            param_list = [{'ticker':list(x['Symbol']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\sp500.csv')


        elif index == "NDX":

            nasdaq_tickers = get_tickers_nasdaq(outfile =r'c:\users\cosmi\onedrive\desktop\nasdaq_tickers.csv',refreshFileOutput=True)
            list_of_arrays = np.array_split(nasdaq_tickers,njobs)
            param_list = [{'ticker':list(x['Symbol']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000, refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\nasdaq.csv')
      
        elif index == "DIA": 
        
            dow_tickers = get_tickers_dow(outfile =r'c:\users\cosmi\onedrive\desktop\dow_tickers.csv',refreshFileOutput=True)
            dow_tickers.rename(columns={"Symbol":"Ticker"}, inplace=True)
            list_of_arrays = np.array_split(dow_tickers,njobs)
            param_list = [{'ticker':list(x['Ticker']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\dow.csv')

        elif index == 'ETF':

            dow_tickers = pd.read_csv(filepath_or_buffer =r'c:\users\cosmi\onedrive\desktop\etf_tickers.csv')
            dow_tickers.rename(columns={"ticker":"Ticker"},inplace=True)
            list_of_arrays = np.array_split(dow_tickers,njobs)
            param_list = [{'ticker':list(x['Ticker']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\etf.csv')

    #update data
    
    elif refresh_data == True:

        if index == "SPY":  
            
            sp500_tickers = pd.read_csv(filepath_or_buffer =r'c:\users\cosmi\onedrive\desktop\sp500_tickers.csv')
            sp500_tickers.rename(columns={'Symbol':'ticker'},inplace=True)
            sp500_tickers = list(sp500_tickers['ticker'])
            #list_of_arrays = np.array_split(sp500_tickers,njobs)
            #param_list = [{'ticker':list(x['Symbol']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data_yahoo_finance_yahoo_2, symbol = ticker) for ticker in sp500_tickers]
                #futures = [executor.submit(tickers_data, ticker=ticker,start_date=start_date) for ticker in sp500_tickers]
                #futures = [executor.submit(get_tickers_data, ticker=sp500_tickers,start_date=start_date)]
                #futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            df['index'] = 'SPY500'
            del return_value
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=outputfile)

        elif index == "NDX":

            nasdaq_tickers = pd.read_csv(filepath_or_buffer =r'c:\users\cosmi\onedrive\desktop\nasdaq_tickers.csv')
            nasdaq_tickers.rename(columns={'Symbol':'ticker'},inplace=True)
            nasdaq_tickers = list(nasdaq_tickers['ticker'])
            #list_of_arrays = np.array_split(nasdaq_tickers,njobs)
            #param_list = [{'ticker':list(x['Symbol']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data_yahoo_finance_yahoo_2, symbol = ticker) for ticker in nasdaq_tickers]
                #futures = [executor.submit(get_tickers_data, ticker=nasdaq_tickers,start_date=start_date)]
                #futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            df['index'] = 'NASDAQ'
            del return_value
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000, refreshOutput=refresh_data,outputfile=outputfile)

        elif index == "DIA":

            dow_tickers = pd.read_csv(filepath_or_buffer =r'c:\users\cosmi\onedrive\desktop\dow_tickers.csv')
            dow_tickers.rename(columns={"Symbol":"ticker"},inplace=True)
            dow_tickers = list(dow_tickers['ticker'])
            #list_of_arrays = np.array_split(dow_tickers,njobs)
            #param_list = [{'ticker':list(x['Ticker']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data_yahoo_finance_yahoo_2, symbol = ticker) for ticker in dow_tickers]
                #futures = [executor.submit(get_tickers_data, ticker=dow_tickers,start_date=start_date)]
                #futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            df['index'] = 'DOW'
            del return_value
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=outputfile)

        elif index == 'ETF':

            etf_tickers = pd.read_csv(filepath_or_buffer =r'c:\users\cosmi\onedrive\desktop\etf_tickers.csv')
            etf_tickers.rename(columns={"Ticker":"ticker"},inplace=True)
            etf_tickers = list(etf_tickers['ticker'])
            #list_of_arrays = np.array_split(dow_tickers,njobs)
            #param_list = [{'ticker':list(x['Ticker']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data_yahoo_finance_yahoo_2, symbol = ticker) for ticker in etf_tickers]
                #futures = [executor.submit(get_tickers_data, ticker=etf_tickers,start_date=start_date)]
                #futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            df['index'] = 'SPY500'
            del return_value
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=outputfile)

    else:

            df = read_csv_bulk(input_file = outputfile,file_size = 1000000000,chunk_count = 100000)

    return df

def get_min_max_scaler(df:pd.DataFrame):

    ticker_start_date = dt.date(2019,10,31)
    ticker_start_date_string = ticker_start_date.strftime('%Y-%m-%d')
    ticker_end_date = dt.date(2019,11,30)
    ticker_end_date_string = ticker_end_date.strftime('%Y-%m-%d')
    scaler = MinMaxScaler()
    df_fit = df.loc[(df['date']>=ticker_start_date_string) & (df['date']<=ticker_end_date_string)]
    scaler.fit(df_fit[['distanceAboveThirtyDayMVA', 'distanceBelowThirtyDayMVA']])
    adjusted_values = scaler.transform(df[['distanceAboveThirtyDayMVA', 'distanceBelowThirtyDayMVA']])
    df['distanceAboveThirtyDayMVAAdjusted'] = adjusted_values[:,0]#return_list[0][0]._values 
    df['distanceBelowThirtyDayMVAAdjusted'] = adjusted_values[:,1]#return_list[0][1]._values
    return df

#return the current quote and append it to the price history list
def get_tickers_data_yahoo_finance_yahoo_2(symbol:str):

    try:

        my_share = share.Share(symbol)
        symbol_data = None
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_YEAR,
                                              2,
                                              share.FREQUENCY_TYPE_DAY,
                                               1)
        close = [x for x in symbol_data.get('close')]
        high = [x for x in symbol_data.get('high')]
        low = [x for x in symbol_data.get('low')]
        open = [x for x in symbol_data.get('open')]
        volume = [x for x in symbol_data.get('volume')]
        date = [x for x in symbol_data.get('timestamp')]
        results = pd.DataFrame(data = {'close':close,'volume':volume,'date':date, 'high':high, 'low':low,'open':open})
        results['ticker'] = symbol 
        results['date'] = pd.to_datetime(results['date'] / 1000,unit='s').dt.date
        return results

    except:
        return pd.DataFrame()

def get_companies_info(tickers: list,outfile: str=None,refreshFileOutput: bool=False) ->pd.DataFrame:

        if path.exists(outfile) and refreshFileOutput==False:

            companies_info = pd.read_csv(outfile)
            companies_info = companies_info[['ticker', 'sector', 'industry']]
            return  companies_info

        else:

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_company_info_with_exception, ticker=x) for x in tickers]
            return_value = [f.result() for f in futures]
            companies_info = pd.concat([pd.DataFrame(data=pd.DataFrame(data=x).transpose())for x in return_value])
            companies_info['ticker'] = tickers._values
            companies_info = companies_info[['ticker', 'sector', 'industry']]
            return companies_info

def get_company_info_with_exception(ticker:str)->pd.DataFrame:

    try:
        return get_company_info(ticker=ticker)

    except:

        print('Unable to find company information for ' + ticker + '.')
        return pd.DataFrame(data={'ticker':ticker},index=[0])

def getPortfolio(df:pd.DataFrame):

    print(df.loc[df['ticker']=='PAA'].tail)
    #print(df.shape)
    #fcx_df = df.loc[df['ticker']=='FCX']
    df_corr = df[['date','ticker','close','index','index_close']]
    df_corr.sort_values(by=['ticker','date'],ascending=['ascending','ascending'],inplace=True)
    df_corr['ticker_pct_change'] = df_corr.groupby(['ticker']).close.pct_change()
    df_corr['index_pct_change'] = df_corr.groupby(['ticker']).index_close.pct_change()
    analysis_df = df_corr.groupby(['ticker'])[['ticker_pct_change', 'index_pct_change']].rolling(50).corr().reset_index([0,1])
    #print(analysis_df.shape)
    #print(analysis_df.head(51))
    #print(df.head(51))
    #print(df.shape)
    #fcx_df = analysis_df.loc[analysis_df['ticker']=='FCX']
    #print(fcx_df.loc[fcx_df['ticker']=='FCX'].tail(50))
    #analysis_df = analysis_df.loc[analysis_df['index_pct_change'].round(1) != 1]
    #analysis_df.reset_index(inplace=True)
    #fcx_df = analysis_df.loc[analysis_df['ticker']=='FCX']
    #print(fcx_df.loc[fcx_df['ticker']=='FCX'].tail(50))
    #print(analysis_df.tail(50))
    analysis_df = analysis_df.loc[['ticker_pct_change'],['index_pct_change', 'ticker']]
    #print(analysis_df.shape)
    #print(analysis_df.head(51))
    analysis_df.reset_index(inplace=True)
    #print(analysis_df.shape)
    #print(analysis_df.head(51))
    #print(analysis_df.loc[analysis_df['ticker']=='FCX'].head(50))
    analysis_df = analysis_df.loc[analysis_df['index']=='ticker_pct_change']
    #print(analysis_df.shape)
    #print(analysis_df.head(51))
    analysis_df.drop(columns=['index'],inplace=True)
    #print(analysis_df.shape)
    #print(analysis_df.head(51))
    #print(analysis_df.loc[analysis_df['ticker']=='FCX'].tail(50))
    #fcx_df = analysis_df.loc[analysis_df['ticker']=='FCX']
    #print(fcx_df.loc[fcx_df['ticker']=='FCX'].tail(50))
    df.reset_index(drop=True, inplace=True)
    analysis_df.reset_index(drop=True, inplace=True)
    #print(df.tail(50))
    #print(analysis_df.tail(50))
    df['index_pct_change'] = analysis_df['index_pct_change']
    #print(df.shape)
    #print(df.head(51))
    #fcx_df = df[['ticker','index_pct_change', 'rollingSTD', 'index_rolling_std']]
    #fcx_df = fcx_df.loc[fcx_df['ticker']=='FCX']
    #print(fcx_df.loc[fcx_df['ticker']=='FCX'].tail(50))
    df['exit_price'] = df['priceTarget'] * 1.05
    df = df[['ticker','priceTarget','exit_price','sector','industry','close', 'index_pct_change', 'onestandarddeviationmove', 'rollingSTD','index_rolling_std']]
    df.rename(columns={"priceTarget": "entry_price", "index_pct_change": "market_corr"}, inplace = True)
    df['Pct_To_Entry_Price'] = (df['close'] - df['entry_price'] ) / df['entry_price']
    df = getStockBeta(df=df)
    print(df.loc[df['ticker']=='PAA'].tail(50))
    #df.drop_duplicates(inplace=True)
    return df

def getStockBeta(df:pd.DataFrame): 

    df['beta'] = df['market_corr'] * (df['rollingSTD']/df['index_rolling_std'])
    return df

def to_csv_bulk(data: pd.DataFrame,  df_size:int, chunk_count:int, refreshOutput:bool=True, outputfile:str=None):

    """
    Keyword arguments:
    data -- the dataframe to write to a csv
    df_size -- the max number of records to read into memory at a single time
    chunk_count --the amount of records to read at a time into memory if the number of records in the df is larger than the df_size
    refreshOutput -- Y/N to refresh the outputfile
    outputfile --the location to output the df

    """
    try:

        df = read_csv_bulk(input_file = outputfile, file_size = 1000000000,chunk_count = 100000)
     
        if df.empty:

            raise OSError
        
        if data.shape[0] <= df_size: 

            if refreshOutput == True:

                data.to_csv(path_or_buf=outputfile,index=False)

            else:

                data.to_csv(path_or_buf=outputfile, header=False,mode='a', index=False)        

        elif data.shape[0] > df_size:

            if refreshOutput == True:

                data.to_csv(path_or_buf=outputfile, chunksize=chunk_count, index=False)

            else:
        
                data.to_csv(path_or_buf=outputfile, chunksize=chunk_count,header=False,mode='a', index=False)

    except OSError as err:

        data.to_csv(path_or_buf=outputfile, chunksize=chunk_count, index=False)


def read_csv_bulk(input_file: str, file_size:int, chunk_count:int):

    """
    Keyword arguments:
    input_file -- the file to read into a dataframe
    file_size -- the max file size to read in bytes at a single time into memory
    chunk_count --the amount of records to read at a time into memory if the size of the input file is larger than the file_size

    """
    df = pd.DataFrame()
    try:

        if os.stat(input_file).st_size <= file_size and os.stat(input_file).st_size > 0 :
   
            df = pd.read_csv(filepath_or_buffer=input_file)

        elif os.stat(input_file).st_size == 0:

            return df

        else:

            reader = pd.read_csv(filepath_or_buffer=input_file,chunksize=chunk_count)
            for chunk in reader:
                df = df.append(other=chunk)

    except Exception as err:

        print(err)
        return df

    return df
