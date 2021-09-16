import pandas as pd
import numpy as np
import math as mth
import datetime as dt
import sys
import os
import concurrent.futures
from os import path, remove
import requests_html as request
import html5lib as html
import PandasExtra as pe
from yahoo_fin.stock_info import get_data, get_company_info, tickers_nasdaq,tickers_sp500,tickers_dow,tickers_ftse250
from yahoo_finance_api2 import share
from sklearn.preprocessing import MinMaxScaler,scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def tickers_data(ticker: str,start_date: str, index_as_date: bool=False, iter=None) ->pd.DataFrame:

    try:
        if (ticker == 'TWTR') | (ticker == 'AAPL'):
            df = get_data(ticker=ticker,start_date=start_date, index_as_date=index_as_date)
            test = df.tail(10)
            print(df.tail(10))
            return df
    except:
        return pd.DataFrame()
    #df = pd.DataFrame()
    #df = df.append(other=pd.DataFrame(data=(get_data(ticker=ticker,start_date=start_date) for ticker in tickers)))
    #return df

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
 


          

def get_index_data(ticker: str,start_date: str,outfile: str=None, index_as_date: bool=False,refreshFileOutput: bool=False) ->pd.DataFrame:

    if path.exists(outfile) and refreshFileOutput==False:

        return pd.read_csv(outfile)

    else:
    
        index_table = get_data(ticker=ticker, start_date = start_date)
        index_table.reset_index(inplace=True)
        index_table.rename(columns={'index':'date'},inplace=True)
        if index_table.dtypes['date'] == 'datetime64[ns]':
            index_table['date'] = index_table['date'].dt.strftime('%Y-%m-%d')
        index_table.to_csv(outfile)
        return index_table

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

def get_tickers_ftse250(include_company_data: bool=True,outfile: str=r'c:\users\cosmi\onedrive\desktop\ftse250_tickers.csv', refreshFileOutput: bool=False) ->pd.DataFrame:

    if path.exists(outfile) and refreshFileOutput==False:

        return pd.read_csv(outfile)

    else:
    
        ftse250_table = tickers_ftse250(include_company_data)
        ftse250_table.to_csv(outfile)
        return ftse250_table

def get_ticker_jobs(refresh_index: bool, refresh_data:bool, index:str, njobs:int, start_date:dt.datetime) ->pd.DataFrame:

    #update index and data
    if refresh_index == True:

        if index == "SP500":
        
            sp500_tickers = get_tickers_sp500(outfile =r'c:\users\cosmi\onedrive\desktop\sp500_tickers.csv',refreshFileOutput=True)
            list_of_arrays = np.array_split(sp500_tickers,njobs)
            param_list = [{'ticker':list(x['Symbol']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\sp500.csv')


        elif index == "NASDAQ":

            nasdaq_tickers = get_tickers_nasdaq(outfile =r'c:\users\cosmi\onedrive\desktop\nasdaq_tickers.csv',refreshFileOutput=True)
            list_of_arrays = np.array_split(nasdaq_tickers,njobs)
            param_list = [{'ticker':list(x['Symbol']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000, refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\nasdaq.csv')

        elif index == "FTSE250":

            ftse250_tickers = get_tickers_ftse250(outfile =r'c:\users\cosmi\onedrive\desktop\ftse250_tickers.csv',refreshFileOutput=True)
            list_of_arrays = np.array_split(ftse250_tickers,njobs)
            param_list = [{'ticker':list(x['Ticker']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\ftse250.csv')
      
        elif index == "DOW": 
        
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

        if index == "SP500":  
            
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
            del return_value
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\sp500.csv')

        elif index == "NASDAQ":

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
            del return_value
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000, refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\nasdaq.csv')

        elif index == "FTSE250":

            ftse250_tickers = pd.read_csv(filepath_or_buffer =r'c:\users\cosmi\onedrive\desktop\ftse250_tickers.csv')
            ftse250_tickers.rename(columns={'Ticker':'ticker'},inplace=True)
            ftse250_tickers = list(ftse250_tickers['ticker'])
            #list_of_arrays = np.array_split(ftse250_tickers,njobs)
            #param_list = [{'ticker':list(x['Ticker']),'start_date':start_date} for x in list_of_arrays]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(get_tickers_data_yahoo_finance_yahoo_2, symbol = ticker) for ticker in ftse250_tickers]
                #futures = [executor.submit(get_tickers_data, ticker=ftse250_tickers,start_date=start_date)]
                #futures = [executor.submit(get_tickers_data, ticker=param.get('ticker'),start_date=param.get('start_date')) for param in param_list]
                return_value = [f.result() for f in futures]
            df = pd.concat([pd.DataFrame(data=x)for x in return_value],sort=True)
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            del return_value
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\ftse250.csv')

        elif index == "DOW":

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
            del return_value
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\dow.csv')

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
            del return_value
            df = df.dropna()
            to_csv_bulk(data=df,df_size=1000000,chunk_count=1000000,refreshOutput=refresh_data,outputfile=r'c:\users\cosmi\onedrive\desktop\etf.csv')

    else:

        if index == "SP500":

            df = read_csv_bulk(input_file = r'c:\users\cosmi\onedrive\desktop\sp500.csv',file_size = 1000000000,chunk_count = 100000)
            #df['volume'] = df['volume'].apply('{:.0f}'.format)

        elif index == "NASDAQ":

            df = read_csv_bulk(input_file = r'c:\users\cosmi\onedrive\desktop\nasdaq.csv',file_size = 1000000000,chunk_count = 100000)
            #df['volume'] = df['volume'].apply('{:.0f}'.format)        

        elif index == "FTSE250":

            df = read_csv_bulk(input_file = r'c:\users\cosmi\onedrive\desktop\ftse250.csv',file_size = 1000000000,chunk_count = 100000)
            #df['volume'] = df['volume'].apply('{:.0f}'.format) 

        elif index == "DOW":

            df = read_csv_bulk(input_file = r'c:\users\cosmi\onedrive\desktop\dow.csv',file_size = 1000000000,chunk_count = 100000)
            #df['volume'] = df['volume'].apply('{:.0f}'.format) 

        elif index == "ETF":

            df = read_csv_bulk(input_file = r'c:\users\cosmi\onedrive\desktop\etf.csv',file_size = 1000000000,chunk_count = 100000)
            #df['volume'] = df['volume'].apply('{:.0f}'.format) 

    return df

def get_tickers_data(ticker: list,start_date: str, index_as_date: bool=False, iter=None) ->pd.DataFrame:

    if iter == None:

        iter = 0
        try:
            df = get_data(ticker=ticker[iter], start_date = start_date,index_as_date=index_as_date)
            df['volume'] = df['volume'].astype('str')
            #df['volume'] = df['volume'].apply('{:.0f}'.format)
            df.drop(columns=['close'],inplace=True)
            df.rename(columns={'adjclose':'close'},inplace=True)

            if len(ticker) == 1:

                return df

            else:

                return df.append(get_tickers_data(ticker=ticker, start_date=start_date,index_as_date=index_as_date,iter=iter))

        except:

            if len(ticker) == 1:

                return pd.DataFrame()

            else:

                iter += 1
                df = pd.DataFrame()
                return df.append(get_tickers_data(ticker=ticker, start_date=start_date,index_as_date=index_as_date,iter=iter))
    
    else:

        if iter == (len(ticker)-2):

            iter += 1
            try:

                df = get_data(ticker=ticker[iter], start_date = start_date,index_as_date=index_as_date)
                df['volume'] = df['volume'].astype('str')
                #df['volume'] = df['volume'].apply('{:.0f}'.format)
                df.drop(columns=['close'],inplace=True)
                df.rename(columns={'adjclose':'close'},inplace=True)
                return df

            except:

                 return pd.DataFrame()

        else:

            iter += 1
            try:

                df = get_data(ticker=ticker[iter], start_date = start_date, index_as_date=index_as_date)
                #if ticker[iter]=='TWTR':
                #    print(df.dtypes['volume'])
                #    print(df.tail(10))
                df['volume'] = df['volume'].astype('str')
                #df['volume'] = df['volume'].apply('{:.0f}'.format)
                df.drop(columns=['close'],inplace=True)
                df.rename(columns={'adjclose':'close'},inplace=True)
                return df.append(get_tickers_data(ticker=ticker, start_date=start_date,index_as_date=index_as_date,iter=iter))

            except:

                df = pd.DataFrame()
                return df.append(get_tickers_data(ticker=ticker, start_date=start_date,index_as_date=index_as_date,iter=iter))
    
  
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
  

def getLinearModel(df: pd.DataFrame, number_of_days: int, list_of_Equities: list=False,list_of_Equity_position: float=0):

    columnList = ['date','ticker', 'close', 'thirtyDayMVA', 'fortyFiveDayMVA', 'fortyFiveDaySTD', 'open', 'high', 'low', 'volume','index_Value']
    yColumn = 'Close'
    equitiesDF = df[columnList]
    marketEquity = df['index_Value'].iloc[list_of_Equity_position]
    if list_of_Equity_position == 0:

        linearModelDF = pd.DataFrame()
        #equitiesDF['datetime'] =  pd.to_datetime(arg=equitiesDF['datetime']).dt.strftime(date_format='%d-%b-%Y')
        equitiesDF['date'] =  pd.to_datetime(arg=equitiesDF['date']).dt.date
        list_of_Equities = equitiesDF[equitiesDF['ticker'] != marketEquity]
        list_of_Equities.drop_duplicates(subset=['ticker'],inplace=True)
        list_of_Equities = list_of_Equities['ticker']
        
    lengthofEquities = list_of_Equities.size - 1
    marketDF = equitiesDF.loc[equitiesDF['index_Value'] == marketEquity]
    securityDF = equitiesDF.loc[equitiesDF['ticker']==list_of_Equities.values[list_of_Equity_position]]
    securityDF = securityDF.tail(number_of_days)
    max_date = securityDF['date'].max()
    min_date = securityDF['date'].min()
    securityDF = securityDF.merge(right=marketDF,how='inner',on='date') #problem pas 5M
    securityDF.drop(labels=['ticker_y','index_Value_y','thirtyDayMVA_y','fortyFiveDayMVA_y','fortyFiveDaySTD_y','open_y', 'high_y', 'low_y', 'volume_y'],axis = 1, inplace=True)
    securityDF.rename(columns={'ticker_x':'ticker','index_Value_x':'index_Value','thirtyDayMVA_x':'thirtyDayMVA','fortyFiveDayMVA_x':'fortyFiveDayMVA','fortyFiveDaySTD_x':'fortyFiveDaySTD', 'close_y':'marketClose', 'close_x':'close', 'open_x':'open','high_x':'high', 'low_x':'low', 'volume_x':'volume'},inplace=True)
    securityDF.fillna(-99999,inplace=True)
    securityDF.dropna(inplace=True)
    #print(securityDF.columns)
    X = np.array(securityDF.drop(['close','ticker', 'date','index_Value'],1))
    y = np.array(securityDF['close'])
    X = scale(X)
    y = np.array(securityDF['close'])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, train_size=.8)
    best_fit = LinearRegression()
    best_fit.fit(X_train, y_train)
    return best_fit, best_fit.score(X_test,y_test), max_date, min_date, securityDF, list_of_Equities, list_of_Equity_position, lengthofEquities, equitiesDF, number_of_days

def getSecurityLinearModels(df: pd.DataFrame, number_of_days: int, list_of_Equities: list=False, list_of_Equity_position: float=0,num1: int=0) -> pd.DataFrame: 
    
    """
    Keyword arguments:
    df -- the data frame that is providing the x and y data
    number_of_days -- the last n days to create the model against
    resultsDF --the data frame will hold the linear regression model for each security (optional)
    list_of_Equities -- the list of securities in the df (optional)
    list_of_Equity_position -- the position in the recurison

    """
    best_fit, score, max_date,min_date,security_data_frame, equity_list,equity_position,len_of_Equities,equity_DF,days = getLinearModel(df = df, number_of_days=number_of_days, list_of_Equities=list_of_Equities, list_of_Equity_position = list_of_Equity_position)
    try:

       linearmodeldf = read_csv_bulk(input_file = r'c:\users\cosmi\onedrive\desktop\results' + str(days) + '.csv', file_size = 1000000000,chunk_count = 100000)
    
    except Exception as e:

       #print(e)
       linearmodeldf = pd.DataFrame()

    linearmodeldf = linearmodeldf.append(other=pd.DataFrame(dict(
                                                                   security = equity_list.values[equity_position],
                                                                    max_date_time = max_date,
                                                                    min_date_time = min_date,
                                                                    num_days = days,
                                                                    max_date_close = pd.DataFrame(data=security_data_frame.loc[security_data_frame['date'] == max_date,['close']]).iloc[0,0],
                                                                    min_date_close = pd.DataFrame(data=security_data_frame.loc[security_data_frame['date'] == min_date,['close']]).iloc[0,0],
                                                                    score = score,
                                                                    intercept_ = best_fit.intercept_,
                                                                    coef_0 = best_fit.coef_[0],
                                                                    coef_1 = best_fit.coef_[1],
                                                                    coef_2 = best_fit.coef_[2],
                                                                    coef_3 = best_fit.coef_[3],
                                                                    coef_4 = best_fit.coef_[4],
                                                                    coef_5 = best_fit.coef_[5],
                                                                    coef_6 = best_fit.coef_[6],
                                                                    coef_7 = best_fit.coef_[7]
                                                                    
                                                                ),
                                                                index=[equity_position]
                                                            )
                                      )

    to_csv_bulk(data=linearmodeldf,df_size = 1000000,chunk_count=100000,refreshOutput=True,outputfile = r'c:\users\cosmi\onedrive\desktop\results' + str(days) +'.csv')
    if list_of_Equity_position == len_of_Equities:
      
         return  linearmodeldf

    else:

         equity_position += 1
         return getSecurityLinearModels(df=equity_DF,number_of_days = days,list_of_Equities=equity_list,list_of_Equity_position=equity_position)


def get_company_info_with_exception(ticker:str)->pd.DataFrame:

    try:
        return get_company_info(ticker=ticker)
    except:

        return pd.DataFrame(data={'ticker':ticker},index=[0])

def get_companies_info(tickers: list) ->pd.DataFrame:

    
        if tickers.size == 0:

            return None

        else:
    
            #companies_info = pd.concat([pd.DataFrame(data=pd.DataFrame(data=get_company_info(ticker=x))).loc[['sector','industry']].transpose() for x in tickers])
            companies_info = pd.concat([pd.DataFrame(data=pd.DataFrame(data=get_company_info_with_exception(ticker=x))).transpose() for x in tickers])
            companies_info['ticker'] = tickers._values
            companies_info = companies_info[['ticker', 'sector', 'industry']]
            return companies_info
  


def getPortfolio(portfolio_file:str, prices_file:str, index_file:str,ticker_start_date:str, ticker_end_date:str):

    portfolio_df = read_csv_bulk(input_file = portfolio_file, file_size = 1000000000,chunk_count = 100000)
    #companies_info = get_companies_info(tickers=portfolio_df['ticker'])
    all_stocks = read_csv_bulk(input_file = prices_file, file_size = 1000000000,chunk_count = 100000)
    list_of_stocks = all_stocks.loc[(all_stocks['date']>=ticker_end_date) & (all_stocks['close'] <= 80) & (all_stocks['volume'].astype('int64') >= 1000000)]
    list_of_stocks = list_of_stocks.loc[list_of_stocks['ticker']=='GM']
    all_stocks = all_stocks.merge(right=list_of_stocks,how='inner',on='ticker')
    all_stocks.rename(columns={"close_x" : "close", "date_x": "date", "ticker_x" : "ticker", "priceTarget_x" : "priceTarget", "onestandarddeviationmove_x" : "onestandarddeviationmove"},inplace=True)
    companies_info = get_companies_info(tickers=all_stocks['ticker'])
    all_stocks = all_stocks.merge(right=companies_info, how='left',left_on=['ticker'],right_on=['ticker'])
    all_stocks = all_stocks.merge(right=portfolio_df, how='left',left_on=['ticker'],right_on=['ticker'])
    all_stocks = all_stocks[['close','date','ticker', 'priceTarget', 'open_position', 'industry', 'sector', 'onestandarddeviationmove']]
    all_stocks.reset_index(inplace=True)
    open_position = [0 if all_stocks['open_position'][x] != 1 else all_stocks['open_position'][x] for x in all_stocks.index]
    all_stocks['open_position'] = open_position
    #company_df = portfolio_df.merge(right=companies_info, how='inner',left_on=['ticker'],right_on=['ticker'])
    #portfolio_df = company_df.merge(right=all_stocks,how='inner',left_on=['ticker'],right_on=['ticker'],suffixes=('_left', '_right'))
    #portfolio_df = company_df.merge(right=all_stocks,how='right',left_on=['ticker'],right_on=['ticker'],suffixes=('_left', '_right'))
    analysis_df = all_stocks[['ticker', 'close', 'date']]
    analysis_df['date'] = pd.to_datetime(all_stocks['date'])
    spy_index_df = read_csv_bulk(input_file = index_file, file_size = 1000000000,chunk_count = 100000)
    spy_index_df['date'] = pd.to_datetime(spy_index_df['date'])
    analysis_df = analysis_df.merge(right=spy_index_df,how='inner',left_on=['date'],right_on=['date'])
    analysis_df.rename(columns={"ticker_x": "ticker", "close_x": "ticker_close", "ticker_y": "index", "close_y": "index_close"}, inplace = True)
    analysis_df = analysis_df[['date','ticker','ticker_close','index','index_close']]
    analysis_df.sort_values(by=['ticker','date'],ascending=['ascending','ascending'],inplace=True)
    analysis_df['ticker_pct_change'] = analysis_df.groupby(['ticker']).ticker_close.pct_change()
    analysis_df['index_pct_change'] = analysis_df.groupby(['ticker']).index_close.pct_change()
    #fcx_portfolio = analysis_df.tail(50)
    #print(fcx_portfolio)
    #fcx_portfolio = fcx_portfolio[['ticker_pct_change', 'index_pct_change']]
    #fcx_portfolio = fcx_portfolio.corr()
    #fcx_corr = fcx_portfolio.loc['ticker_pct_change','index_pct_change']
    #print(fcx_corr)
    analysis_corr = analysis_df.groupby(['ticker'])[['ticker_pct_change', 'index_pct_change']].rolling(50).corr().reset_index([0,1])
    analysis_corr = analysis_corr.loc[analysis_corr['index_pct_change'].round(1) != 1]
    analysis_corr_group = analysis_corr.groupby(['ticker'])[['level_1']].max()
    analysis_corr = analysis_corr.reset_index()
    analysis_corr = analysis_corr.merge(right = analysis_corr_group,how='inner',left_on=['ticker', 'level_1'],right_on=['ticker', 'level_1'])
    #analysis_corr = analysis_corr.merge(right=company_df,how='inner',left_on=['ticker'],right_on=['ticker'])
    analysis_corr.drop(labels=['level_1'], axis=1, inplace=True)
    #analysis_corr.drop(labels=['level_1', 'Unnamed: 0'], axis=1, inplace=True)
    max_all_stocks = all_stocks.loc[(all_stocks['date']==ticker_end_date)]
    analysis_corr = analysis_corr.merge(right=max_all_stocks,how='inner',left_on=['ticker'],right_on=['ticker'])
    #analysis_corr = analysis_corr.merge(right=all_stocks,how='right',left_on=['ticker'],right_on=['ticker'])
    #analysis_corr = analysis_corr.merge(right=max_all_stocks,how='right',left_on=['ticker'],right_on=['ticker'])
    analysis_corr['exit_price'] = analysis_corr['priceTarget'] * 1.05
    analysis_corr = analysis_corr[['ticker','priceTarget','exit_price','open_position','sector','industry','close', 'index_pct_change', 'onestandarddeviationmove']]
    analysis_corr.rename(columns={"priceTarget": "entry_price"}, inplace = True)
    analysis_corr['Pct_To_Entry_Price'] = (analysis_corr['close'] - analysis_corr['entry_price'] ) / analysis_corr['entry_price']
    analysis_corr.drop_duplicates(inplace=True)
    to_csv_bulk(data=analysis_corr,df_size = 1000000,chunk_count=100000,refreshOutput=True,outputfile = r'c:\users\cosmi\onedrive\desktop\portfolio_analysis.csv')
    return portfolio_df

def getSecurityLinearModels2(df: pd.DataFrame, number_of_days: int, ticker: str) -> pd.DataFrame: 
    
    """
    Keyword arguments:
    df -- the data frame that is providing the x and y data
    number_of_days -- the last n days to create the model against
    resultsDF --the data frame will hold the linear regression model for each security (optional)
    list_of_Equities -- the list of securities in the df (optional)
    list_of_Equity_position -- the position in the recurison

    """
    try:

        best_fit, score, max_date,min_date,equity_DF,days = getLinearModel2(df = df, number_of_days = number_of_days, ticker = ticker)
        linearmodeldf = pd.DataFrame(data =dict(
                                                  security = ticker,
                                                  max_date_time = max_date,
                                                  min_date_time = min_date,
                                                  num_days = days,
                                                  max_date_close = pd.DataFrame(data=equity_DF.loc[equity_DF['date'] == max_date,['close']]).iloc[0,0],
                                                  min_date_close = pd.DataFrame(data=equity_DF.loc[equity_DF['date'] == min_date,['close']]).iloc[0,0],
                                                  score = score,
                                                  intercept_ = best_fit.intercept_,
                                                  coef_0 = best_fit.coef_[0],
                                                  coef_1 = best_fit.coef_[1],
                                                  coef_2 = best_fit.coef_[2],
                                                  coef_3 = best_fit.coef_[3],
                                                                    
                                                 )
                                                 ,index = [0]
                                     )
                            
    except Exception as e:

          print(e)
          return

    return linearmodeldf 

def getLinearModel2(df: pd.DataFrame, number_of_days: int, ticker: str):

    try:

        df = df[['date','ticker', 'close', 'thirtyDayMVA', 'fortyFiveDayMVA', 'open', 'high', 'low', 'volume','index_Value', 'market_close']]
        marketEquity = df['index_Value'].iloc[0]
        df['date'] =  pd.to_datetime(arg=df['date']).dt.date 
        df = df.loc[df['ticker'] == ticker]
        df = df.tail(number_of_days)
        max_date = df['date'].max()
        min_date = df['date'].min()
        df.fillna(-99999,inplace=True)
        df.dropna(inplace=True)
        X = np.array(df.drop(['close','ticker', 'date','index_Value', 'low', 'high', 'open'],1))
        y = np.array(df['close'])
        X = scale(X)
        y = np.array(df['close'])
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, train_size=.8)
        best_fit = LinearRegression()
        best_fit.fit(X_train, y_train)
        return best_fit, best_fit.score(X_test,y_test), max_date, min_date, df, number_of_days

    except Exception as err:

        print(err)