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
from yahoo_fin.stock_info import get_data, get_company_info, tickers_nasdaq,tickers_sp500,tickers_dow,tickers_ftse250, tickers_other
from yahoo_finance_api2 import share
from sklearn.preprocessing import MinMaxScaler,StandardScaler,scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

def get_index_data(ticker: str,start_date: str, end_date: str=None,outfile: str=None, index_as_date: bool=False,refreshFileOutput: bool=False) ->pd.DataFrame:

    if path.exists(outfile) and refreshFileOutput==False:

        return pd.read_csv(outfile)

    else:
    
        index_table = get_data(ticker=ticker, start_date = start_date, end_date = end_date)
        if ticker =='NDX':
            
            index_table.dropna(subset=['close'], inplace=True)
            index_table = index_table.append(other=pd.DataFrame(data={'open': np.array(15955.50),
                                                        'high':np.array(16017.39 ),
                                                        'low':np.array(15815.95),
                                                        'close':np.array(15905.10),
                                                        'adjclose':np.array(15905.10),
                                                        'volume':np.array(699526181),
                                                        'ticker':np.array('NDX')
                                                       },index= ['2022-1-12']
                                                  )
                               )
                                                                       
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

        elif index == 'Other':

            other_tickers = pd.read_csv(filepath_or_buffer =r'c:\users\cosmi\onedrive\desktop\other_tickers.csv')
            other_tickers.rename(columns={"Symbol":"ticker"},inplace=True)
            other_tickers = list(other_tickers['ticker'])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data_yahoo_finance_yahoo_2, symbol = ticker) for ticker in other_tickers]
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

    ticker_start_date = dt.date(2021,1,1)
    ticker_start_date_string = ticker_start_date.strftime('%Y-%m-%d')
    ticker_end_date = dt.date(2021,1,31)
    ticker_end_date_string = ticker_end_date.strftime('%Y-%m-%d')
    df_fit = df.loc[(df['date']>=ticker_start_date_string) & (df['date']<=ticker_end_date_string)]
    scaler = MinMaxScaler()
    scaler.fit(df_fit[['distanceAboveTwentyDayMVA', 'distanceBelowTwentyDayMVA']])
    adjusted_values = scaler.transform(df[['distanceAboveTwentyDayMVA', 'distanceBelowTwentyDayMVA']])
    df['distanceAboveTwentyDayMVAAdjusted'] = adjusted_values[:,0]
    df['distanceBelowTwentyDayMVAAdjusted'] = adjusted_values[:,1]
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
            to_csv_bulk(data=companies_info,df_size=1000000,chunk_count=1000000,refreshOutput=refreshFileOutput,outputfile=outfile)
            return companies_info

def get_company_info_with_exception(ticker:str)->pd.DataFrame:

    try:
        return get_company_info(ticker=ticker)

    except:

        print('Unable to find company information for ' + ticker + '.')
        return pd.DataFrame(data={'ticker':np.array(ticker)},index=[0])

def getPortfolio(df:pd.DataFrame):

    df_corr = df[['date','ticker','close','index','index_close']]
    df_corr.sort_values(by=['ticker','date'],ascending=['ascending','ascending'],inplace=True)
    df_corr['ticker_pct_change'] = df_corr.groupby(['ticker']).close.pct_change()
    df_corr['index_pct_change'] = df_corr.groupby(['ticker']).index_close.pct_change()
    analysis_df = df_corr.groupby(['ticker'])[['ticker_pct_change', 'index_pct_change']].rolling(50).corr().reset_index([0,1])
    analysis_df = analysis_df.loc[['ticker_pct_change'],['index_pct_change', 'ticker']]
    analysis_df.reset_index(inplace=True)
    analysis_df = analysis_df.loc[analysis_df['index']=='ticker_pct_change']
    analysis_df.drop(columns=['index'],inplace=True)
    df.reset_index(drop=True, inplace=True)
    analysis_df.reset_index(drop=True, inplace=True)
    df['index_pct_change'] = analysis_df['index_pct_change']
    df['exit_price'] = df['priceTarget'] * 1.05
    df = df[['ticker','priceTarget','exit_price','sector','industry','close', 'index_pct_change', 'oneweekstandarddeviationmove', 'twoweekstandarddeviationmove', 'rollingSTD','index_rolling_std']]
    df.rename(columns={"priceTarget": "entry_price", "index_pct_change": "market_corr"}, inplace = True)
    df['Pct_To_Entry_Price'] = (df['close'] - df['entry_price'] ) / df['entry_price']
    df = getStockBeta(df=df)
    return df

def getStockBeta(df:pd.DataFrame): 

    df.loc[:,'beta'] = df['market_corr'] * (df['rollingSTD'] / df['index_rolling_std'])
    print(df.tail(50))
    #df['beta'] = df['market_corr'].copy() * (df['rollingSTD'].copy()/df['index_rolling_std'].copy())
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

def getSecurityLinearModels(df: pd.DataFrame, number_of_days: int, ticker: str) -> pd.DataFrame: 
    
    """
    Keyword arguments:
    df -- the data frame that is providing the x and y data
    number_of_days -- the last n days to create the model against
    resultsDF --the data frame will hold the linear regression model for each security (optional)
    list_of_Equities -- the list of securities in the df (optional)
    list_of_Equity_position -- the position in the recurison

    """
    try:

        best_fit, score, max_date,min_date,equity_DF,days = getLinearModel(df = df, number_of_days = number_of_days, ticker = ticker)
        #linearmodeldf = pd.DataFrame(data =dict(security=np.array(ticker)),index = [0])
        linearmodeldf = pd.DataFrame(data =dict(
                                                  security = np.array(ticker),
                                                  max_date_time = np.array(max_date),                           
                                                  score = np.array(score),
                                                  intercept_ = np.array(best_fit.intercept_),
                                                  coef_0 = np.array(best_fit.coef_[0]),
                                                  coef_1 = np.array(best_fit.coef_[1]),
                                                  coef_2 = np.array(best_fit.coef_[2]),
                                                  coef_3 = np.array(best_fit.coef_[3]),
                                                  coef_4 = np.array(best_fit.coef_[4]),
                                                  coef_5 = np.array(best_fit.coef_[5]),
                                                  coef_6 = np.array(best_fit.coef_[6]),
                                                  coef_7 = np.array(best_fit.coef_[7]),
                                                  coef_8 = np.array(best_fit.coef_[8]),
                                                                    
                                                 )
                                                 ,index = [0]
                                     )
                            
    except Exception as e:

          print(e)
          return

    return linearmodeldf 

def featureSelection(df : pd.DataFrame, ticker : str):

    print('cole')
    estimator = LinearRegression()
    selector = RFE(estimator,n_features_to_select = 5, step = 1 )
    df = df.loc[df['ticker']==ticker].tail(50)
    print(df.columns)
    X = df[['high', 'low', 'open', 'volume', 
       'index_close', 'gold_close',
       '10_year_note_close',
       'twentyDayMVA', 'fiftyDayMVA',
     ]]
    X.fillna(-99999,inplace=True)
    y = np.array(df[['close']])
    selector.fit(X,y)
    print(X.columns)
    print(selector.support_)

def getLinearModel(df: pd.DataFrame, number_of_days: int, ticker: str):

    try:
 
        df = df[['date','ticker', 'close', 'twentyDayMVA','fiftyDayMVA', 'open', 'high', 'low', 'volume','10_year_note_close', 'index_close', 'gold_close', 'beta']].copy()
        #df = df[['date','ticker', 'close', 'open', 'high', 'low']].copy()
        df['date'] =  pd.to_datetime(arg=df['date']).dt.date 
        df = df.loc[df['ticker'] == ticker]
        df = df.tail(number_of_days)
        max_date = df['date'].max()
        min_date = df['date'].min()
        df.fillna(-99999,inplace=True)
        df.dropna(inplace=True)
        X = np.array(df.drop(['close','ticker', 'index_close', 'date'],1))
        y = np.array(df['close'])
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, train_size=.8)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        best_fit = LinearRegression()
        best_fit.fit(X_train, y_train)
        return best_fit, best_fit.score(X_test,y_test), max_date, min_date, df, number_of_days

    except Exception as err:

        print(err)

def getOpenPositions(file:str):

     df = read_csv_bulk(input_file = file, file_size = 1000000000,chunk_count = 100000)
     return df

def get_tickers_dow(include_company_data: bool=True,outfile: str=r'c:\users\cosmi\onedrive\desktop\dow_tickers.csv', refreshFileOutput: bool=False) ->pd.DataFrame:

    if path.exists(outfile) and refreshFileOutput==False:

        return pd.read_csv(outfile)

    else:
    
        dow_table = tickers_dow(include_company_data)
        dow_table.to_csv(outfile)
        return dow_table

def get_tickers_sp500(include_company_data: bool=True, outfile: str=r'c:\users\cosmi\onedrive\desktop\sp500_tickers.csv', refreshFileOutput: bool=False) ->pd.DataFrame:

    if path.exists(outfile) and refreshFileOutput==False:

        return pd.read_csv(outfile)

    else:

        sp500_table = tickers_sp500(include_company_data)
        sp500_table.to_csv(outfile)
        return sp500_table

def get_tickers_nasdaq(include_company_data: bool=True,outfile: str=r'c:\users\cosmi\onedrive\desktop\nasdaq_tickers.csv', refreshFileOutput: bool=False) ->pd.DataFrame:

    if path.exists(outfile) and refreshFileOutput==False:

        return pd.read_csv(outfile)

    else:
    
        nasdaq_table = tickers_nasdaq(include_company_data)
        nasdaq_table.to_csv(outfile)
        return nasdaq_table

def get_tickers_other(include_company_data: bool=True,outfile: str=r'c:\users\cosmi\onedrive\desktop\other_tickers.csv', refreshFileOutput: bool=False) ->pd.DataFrame:

    if path.exists(outfile) and refreshFileOutput==False:

        return pd.read_csv(outfile)

    else:
    
        other_table = tickers_other(include_company_data)
        other_table.to_csv(outfile)
        return other_table

