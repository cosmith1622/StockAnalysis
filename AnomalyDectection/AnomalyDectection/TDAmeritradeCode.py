from TDAmeriTradeClass import TDAmeriTradeClass
import pandas as pd
import datetime
import time
import json
import numpy as np

file = open("C:\\Users\\cosmi\\OneDrive\\Desktop\\securityPrices.csv","r+")
file.truncate(0)
file.close()

file = open("C:\\Users\\cosmi\\OneDrive\\Desktop\\tdmultilegcsv.csv","r+")
file.truncate(0)
file.close()

file = open("C:\\Users\\cosmi\\OneDrive\\Desktop\\tdindividualcsv.csv","r+")
file.truncate(0)
file.close()

file = open("C:\\Users\\cosmi\\OneDrive\\Desktop\\tdoptionscsv.csv","r+")
file.truncate(0)
file.close()

#global variable to hold the list of securities
securities = '' 

#function get the securities and the strategy to return records for
def get_securities(filepath):
      
      return pd.read_csv(filepath)


#get the list of securities and strategies
securities = get_securities(r'C:\Users\cosmi\OneDrive\Desktop\list_of_securities_multi.csv')

#securityTuple to capture the securities that we found price history data on
securityTuple = []

#fill in the nan with blank values
securities.fillna(value='', inplace = True)
headerCounter = 0
isTdDeleted = False
matchingSecurity = pd.DataFrame()

#loop through the list of securities and strategies
for row in range(0,securities.shape[0]):
    
    #check to see if the security already exist in the securityPrices csv
    #if so we don't need to print the data to the spreadsheet a second time
    if row != 0:

        securityPrices_CSV = pd.read_csv(r'C:\Users\cosmi\OneDrive\Desktop\securityPrices.csv', encoding='utf-8')
        matchingSecurity = securityPrices_CSV[securityPrices_CSV['security'] == securities['Security'][row]]

    if not matchingSecurity.empty and row != 0:

        is_Match = True

    else:

        is_Match = False

    #if isTdDeleted == True:

    #print('I am true')
    #create the td object on the first iteration this is our first
    #security that we will grab data for 
    if row == 0 or isTdDeleted == True:
    
        td = TDAmeriTradeClass();
        isTdDeleted = False
    #print(td.fortyfivedayVolatility)
    #grab data about the secuirty spreadsheet
    print(securities['Security'][row])
    securityTuple.append(securities['Security'][row])
    
    if not td.price_history or row == 0:

        #get always use a date that gets us atleast 60 days of data so we
        #can get the historical volatility
        history_date = int((datetime.datetime(2018,7,1,0,0).timestamp()) * 1000)
        #history_date = time.mktime(history_date.timetuple())
        td.get_price_history(symbol=securities['Security'][row],
                             periodType = 'month',
                             frequencyType = 'daily',
                             startDate = history_date,
                             apikey= 'CEVREV5DD'
                                                                          
                          );

        td.get_quote_price_history(symbol=securities['Security'][row],
                        
                                  );

        #grab the historical volatility for the security
        td.get_historical_volatility(td.price_history)
       
        if row == 0:
          
           securityPrices = json.dumps(td.price_history[0]['candles'])
           df_securityPrices = pd.read_json(securityPrices)
           #df_securityPrices.sort_values('datetime', ascending=False, inplace = True)
           #df_securityPrices.set_index(keys='datetime', inplace = True)
           df_securityPrices['security'] = securities['Security'][row]
           df_securityPrices['twentyDayMVA'] = df_securityPrices['close'].rolling(20).mean()
           df_securityPrices['fiftyDayMVA'] = df_securityPrices['close'].rolling(50).mean()
           df_securityPrices['twohundredDayMVA'] = df_securityPrices['close'].rolling(200).mean()
           df_securityPrices['closeDelta']= df_securityPrices[['close']].diff()
           df_securityPrices['positiveGain'] =  df_securityPrices['closeDelta'].apply(lambda x: 0 if x <= 0 else  x)
           df_securityPrices['negativeGain'] =  df_securityPrices['closeDelta'].apply(lambda x: 0 if x >= 0 else  abs(x))
           df_securityPrices['averagePositiveGain'] = df_securityPrices['positiveGain'].rolling(14).mean()
           df_securityPrices['averageNegativeGain'] = df_securityPrices['negativeGain'].rolling(14).mean()
           df_securityPrices['relativeStrength'] = df_securityPrices['averagePositiveGain']/df_securityPrices['averageNegativeGain']
           df_securityPrices['rsi'] = df_securityPrices.apply(lambda x: 0 if x.averageNegativeGain == 0 else 100 - (100 / (1 + x.relativeStrength)),axis = 1)
           df_securityPrices['obv'] = ""
           for idx in range(len(df_securityPrices.index)- 1):
            
               if idx == 0:

                   df_securityPrices.at[idx, 'obv'] = df_securityPrices.iloc[idx, df_securityPrices.columns.get_loc('volume')] 

               elif idx > 0:

                   previousIdx = idx - 1
                   #nextIdx = idx + 1

                   if df_securityPrices.at[idx,'close'] > df_securityPrices.at[previousIdx,'close']:

                        df_securityPrices.at[idx, 'obv'] = (df_securityPrices.iloc[previousIdx, df_securityPrices.columns.get_loc('obv')] + df_securityPrices.iloc[idx, df_securityPrices.columns.get_loc('volume')] )

                   elif df_securityPrices.at[idx,'close'] == df_securityPrices.at[previousIdx,'close']:

                        df_securityPrices.at[idx, 'obv'] = df_securityPrices.iloc[previousIdx, df_securityPrices.columns.get_loc('obv')] 

                   else:

                        df_securityPrices.at[idx, 'obv'] = (df_securityPrices.iloc[previousIdx, df_securityPrices.columns.get_loc('obv')]  - df_securityPrices.iloc[idx, df_securityPrices.columns.get_loc('volume')])

         
           df_securityPrices['thirtyDaySTD'] = df_securityPrices['close'].rolling(20).std()
           df_securityPrices['thirtyDayVolumeMVA'] = df_securityPrices['volume'].rolling(20).mean()
           df_securityPrices['fortyFiveDayMVA'] = df_securityPrices['close'].rolling(30).mean()
           df_securityPrices['fortyFiveDaySTD'] = df_securityPrices['close'].rolling(30).std()
           df_securityPrices['fortyFiveDayVolumeMVA'] = df_securityPrices['volume'].rolling(30).mean()
           df_securityPrices['sixtyDayMVA'] = df_securityPrices['close'].rolling(40).mean()
           df_securityPrices['sixtyDaySTD'] = df_securityPrices['close'].rolling(40).std()
           df_securityPrices['sixtyDayVolumeMVA'] = df_securityPrices['volume'].rolling(40).mean()
           df_securityPrices['rollingSixtyDayVolatility']= df_securityPrices[['close']].pct_change().apply(lambda x:np.log(1+x)).rolling(39).std().apply(lambda x: x*np.sqrt(252))
           df_securityPrices['rollingSixtyDayVolatilityRank'] = df_securityPrices['rollingSixtyDayVolatility'].rank(pct=True)
           df_securityPrices['rollingFortyFiveDayVolatility']= df_securityPrices[['close']].pct_change().apply(lambda x:np.log(1+x)).rolling(29).std().apply(lambda x: x*np.sqrt(252))           
           df_securityPrices['rollingFortyFiveDayVolatilityRank'] = df_securityPrices['rollingFortyFiveDayVolatility'].rank(pct=True)
           df_securityPrices['rollingThirtyDayVolatility']= df_securityPrices[['close']].pct_change().apply(lambda x:np.log(1+x)).rolling(19).std().apply(lambda x: x*np.sqrt(252)) 
           df_securityPrices['rollingThirtyDayVolatilityRank'] = df_securityPrices['rollingThirtyDayVolatility'].rank(pct=True)
           df_securityPrices.to_csv(r'C:\Users\cosmi\OneDrive\Desktop\securityPrices.csv')

        elif is_Match == False:
           securityPrices = json.dumps(td.price_history[0]['candles'])
           df_securityPrices = pd.read_json(securityPrices)
           df_securityPrices['security'] = securities['Security'][row]
           df_securityPrices['twentyDayMVA'] = df_securityPrices['close'].rolling(20).mean()
           df_securityPrices['fiftyDayMVA'] = df_securityPrices['close'].rolling(50).mean()
           df_securityPrices['twohundredDayMVA'] = df_securityPrices['close'].rolling(200).mean()
           df_securityPrices['closeDelta']= df_securityPrices[['close']].diff()
           df_securityPrices['positiveGain'] =  df_securityPrices['closeDelta'].apply(lambda x: 0 if x <= 0 else  x)
           df_securityPrices['negativeGain'] =  df_securityPrices['closeDelta'].apply(lambda x: 0 if x >= 0 else  abs(x))
           df_securityPrices['averagePositiveGain'] = df_securityPrices['positiveGain'].rolling(14).mean()
           df_securityPrices['averageNegativeGain'] = df_securityPrices['negativeGain'].rolling(14).mean()
           df_securityPrices['relativeStrength'] = df_securityPrices['averagePositiveGain']/df_securityPrices['averageNegativeGain']
           df_securityPrices['rsi'] = df_securityPrices.apply(lambda x: 0 if x.averageNegativeGain == 0 else 100 - (100 / (1 + x.relativeStrength)),axis = 1)
           df_securityPrices['obv'] = ""
           for idx in range(len(df_securityPrices.index)- 1):
            
               if idx == 0:

                   df_securityPrices.at[idx, 'obv'] = df_securityPrices.iloc[idx, df_securityPrices.columns.get_loc('volume')] 

               elif idx > 0:

                   previousIdx = idx - 1
                   #nextIdx = idx + 1

                   if df_securityPrices.at[idx,'close'] > df_securityPrices.at[previousIdx,'close']:

                        df_securityPrices.at[idx, 'obv'] = (df_securityPrices.iloc[previousIdx, df_securityPrices.columns.get_loc('obv')] + df_securityPrices.iloc[idx, df_securityPrices.columns.get_loc('volume')] )

                   elif df_securityPrices.at[idx,'close'] == df_securityPrices.at[previousIdx,'close']:

                        df_securityPrices.at[idx, 'obv'] = df_securityPrices.iloc[previousIdx, df_securityPrices.columns.get_loc('obv')] 

                   else:

                        df_securityPrices.at[idx, 'obv'] = (df_securityPrices.iloc[previousIdx, df_securityPrices.columns.get_loc('obv')]  - df_securityPrices.iloc[idx, df_securityPrices.columns.get_loc('volume')])     
           df_securityPrices['thirtyDaySTD'] = df_securityPrices['close'].rolling(20).std()
           df_securityPrices['thirtyDayVolumeMVA'] = df_securityPrices['volume'].rolling(20).mean()
           df_securityPrices['fortyFiveDayMVA'] = df_securityPrices['close'].rolling(30).mean()
           df_securityPrices['fortyFiveDaySTD'] = df_securityPrices['close'].rolling(30).std()
           df_securityPrices['fortyFiveDayVolumeMVA'] = df_securityPrices['volume'].rolling(30).mean()
           df_securityPrices['sixtyDayMVA'] = df_securityPrices['close'].rolling(40).mean()
           df_securityPrices['sixtyDaySTD'] = df_securityPrices['close'].rolling(40).std()
           df_securityPrices['sixtyDayVolumeMVA'] = df_securityPrices['volume'].rolling(40).mean()
           df_securityPrices['rollingSixtyDayVolatility']= df_securityPrices[['close']].pct_change().apply(lambda x:np.log(1+x)).rolling(39).std().apply(lambda x: x*np.sqrt(252))
           df_securityPrices['rollingSixtyDayVolatilityRank'] = df_securityPrices['rollingSixtyDayVolatility'].rank(pct=True)
           df_securityPrices['rollingFortyFiveDayVolatility']= df_securityPrices[['close']].pct_change().apply(lambda x:np.log(1+x)).rolling(29).std().apply(lambda x: x*np.sqrt(252))           
           df_securityPrices['rollingFortyFiveDayVolatilityRank'] = df_securityPrices['rollingFortyFiveDayVolatility'].rank(pct=True)
           df_securityPrices['rollingThirtyDayVolatility']= df_securityPrices[['close']].pct_change().apply(lambda x:np.log(1+x)).rolling(19).std().apply(lambda x: x*np.sqrt(252)) 
           df_securityPrices['rollingThirtyDayVolatilityRank'] = df_securityPrices['rollingThirtyDayVolatility'].rank(pct=True)
           df_securityPrices.to_csv(r'C:\Users\cosmi\OneDrive\Desktop\securityPrices.csv', mode='a', header = False)


   # print(securities['includeQuotes'][row])
    #store the option chains as a json object for a given security and strategy
    td.get_option_chain(symbol=securities['Security'][row], 
                        apikey= 'CEVREV5DD',
                        contractType=securities['ContractType'][row],
                        strikeCount = '',
                        includeQuotes = securities['includeQuotes'][row],
                        strategy = securities['STRATEGY'][row],
                        interval = securities['INTERVAL'][row],
                        strike = '',
                        range = '',
                        fromDate = '',
                        toDate='',
                        volatility = '',
                        underlyingPrice ='',
                        interestRate = '',
                        daysToExpiration = '',
                        expMonth = '',
                        optionType = ''
                        
                     );

    #print(len(td.options_chain))
    #verify if the strategy has a single leg
    if securities['STRATEGY'][row] == "SINGLE":

        tdindividualjson = td.json_Individual_Option_Chain(securities['ContractType'][row], securities['STRATEGY'][row])

        #if we can't find any options trades
        #delete the object and move on to the next security
        if tdindividualjson is None:

            del td
            isTdDeleted = True

        else:

            df_individualjson = pd.read_json(tdindividualjson)

            df_individualjson.rename(columns = {'bidAskSize':'bidasksizeprimaryleg',                                              
                                                'deliverableNote':'deliverablenotprimaryleg',
                                                'description': 'descriptionprimaryleg',
                                                'exchangeName':'exchangenameprimaryleg',
                                                'expirationDate': 'expirationdateprimaryleg',
                                                'expirationType': 'expirationtypeprimaryleg',
                                                'inTheMoney': 'inthemoneyprimaryleg',
                                                'isIndexOption': 'isindexoptionprimaryleg',
                                                'lastTradingDay': 'lasttradingdayprimaryleg',
                                                'markChange': 'markchangeprimaryleg',
                                                'markPercentChange': 'markpercentchangeprimaryleg',
                                                'mini': 'miniprimaryleg',
                                                'nonStandard': 'nonstandardprimaryleg',
                                                'optionDeliverablesList':'optiondeliverableslistprimaryleg',
                                                'putCall': 'putcallprimaryleg',
                                                'quoteTimeInLong': 'quotetimeinlongprimaryleg',
                                                'settlementType': 'settlementtypeprimaryleg',
                                                'strategy':'strategyprimaryleg',
                                                'symbol':'symbolprimaryleg',
                                                'tradeDate': 'tradedateprimaryleg',
                                                'underlyingPrice':'underlyingprimaryleg',
                                                'ask':'askprimaryleg',
                                                'askSize':'asksizeprimaryleg',
                                                'close':'closeprimaryleg',
                                                'closePrice':'closepriceprimaryleg',
                                                'daysToExpiration':'daystoexpirationprimaryleg',
                                                'delta':'deltaprimaryleg',
                                                'bid':'bidprimaryleg',                          
                                                'bidSize':'bidsizeprimaryleg',
                                                'gamma': 'gammaprimaryleg',
                                                'highPrice': 'highpriceprimaryleg',
                                                'last': 'lastprimaryleg',
                                                'lastSize': 'lastsizeprimaryleg',
                                                'lowPrice': 'lowpriceprimaryleg',
                                                'mark': 'markprimaryleg',
                                                'multiplier': 'multiplierprimaryleg',
                                                'netChange': 'netchangeprimaryleg',
                                                'oepnInterest': 'oepninterestprimaryleg',
                                                'openPrice': 'openpriceprimaryleg',
                                                'percentChange': 'percentchangeprimaryleg',
                                                'rho': 'rhoprimaryleg',
                                                'strikePrice': 'strikepriceprimaryleg',
                                                'theoreticalOptionValue': 'theoreticaloptionvalueprimaryleg',
                                                'theoreticalVolatility': 'theoreticalvolatilityprimaryleg',
                                                'theta': 'thetaprimaryleg',
                                                'timeValue': 'timevalueprimaryleg',
                                                'totalVolume': 'totalvolumeprimaryleg',
                                                'tradeTimeInLong': 'tradetimeinlongprimaryleg',
                                                'vega': 'vegaprimaryleg',
                                                'volatility': 'volatilityprimaryleg'

                                           }, inplace = True)

            #add in the extra columns for the second leg
            #even though we don't have one so we can use the same data model
            df_individualjson['bidasksizesecondleg'] = None
            df_individualjson['deliverablenotsecondleg'] = None
            df_individualjson['exchangenamesecondleg'] = None
            df_individualjson['expirationdatesecondleg'] = None
            df_individualjson['expirationtypesecondleg'] = None
            df_individualjson['inthemoneysecondleg'] = None
            df_individualjson['isindexoptionsecondleg'] = None
            df_individualjson['lasttradingdaysecondleg'] = None
            df_individualjson['markchangesecondleg'] = None
            df_individualjson['markpercentchangesecondleg'] = None
            df_individualjson['minisecondleg'] = None
            df_individualjson['nonstandardsecondleg'] = None
            df_individualjson['optiondeliverableslistsecondleg'] = None
            df_individualjson['putcallsecondleg'] = None
            df_individualjson['quotetimeinlongsecondleg'] = None
            df_individualjson['settlementtypeprimaryleg'] = None
            df_individualjson['strategysecondleg'] = None
            df_individualjson['symbolsecondleg'] = None
            df_individualjson['tradedatesecondleg'] = None
            df_individualjson['underlyingsecondleg'] = None
            df_individualjson['askprimaryleg'] = None
            df_individualjson['asksizesecondleg'] = None
            df_individualjson['closepriceprimaryleg'] = None
            df_individualjson['closesecondleg'] = None
            df_individualjson['closepricesecondleg'] = None
            df_individualjson['daystoexpirationsecondleg'] = None
            df_individualjson['deltasecondleg'] = None
            df_individualjson['bidsecondleg'] = None                          
            df_individualjson['bidsizesecondleg'] = None
            df_individualjson['gammasecondleg'] = None
            df_individualjson['highpricesecondleg'] = None
            df_individualjson['lastsecondleg'] = None
            df_individualjson['lastsizesecondleg'] = None
            df_individualjson['lastsizesecondleg'] = None
            df_individualjson['lowpricesecondleg'] = None
            df_individualjson['marksecondleg'] = None
            df_individualjson['multipliersecondleg'] = None
            df_individualjson['netchangesecondleg'] = None
            df_individualjson['oepninterestsecondleg'] = None
            df_individualjson['openpricesecondleg'] = None
            df_individualjson['percentchangesecondleg'] = None
            df_individualjson['rhosecondleg'] = None
            df_individualjson['strikepricesecondleg'] = None
            df_individualjson['theoreticaloptionvaluesecondleg'] = None
            df_individualjson['theoreticalvolatilitysecondleg'] = None
            df_individualjson['thetaprimaryleg'] = None
            df_individualjson['timevalueprimaryleg'] = None
            df_individualjson['totalvolumesecondleg'] = None
            df_individualjson['tradetimeinlongsecondleg'] = None
            df_individualjson['vegasecondleg'] = None
            df_individualjson['volatilitysecondleg'] = None

                                                

            #if this is the first securitity/record then we will include header and insert the records for the 
            #individualleg json otherwise we will append the records
            #we send the individualjson to the tdoptionscsv file.  this is the same file we send the analytical trades too
            if headerCounter == 0:
           
                df_individualjson.to_csv(r'c:\users\cosmi\onedrive\desktop\tdoptionscsv.csv')
                headerCounter += 1
                del td

            else:
            
                df_individualjson.to_csv(r'c:\users\cosmi\onedrive\desktop\tdoptionscsv.csv', mode = 'a', header=False)
                headerCounter += 1
                del td

    else:

        tdmultilegjson = td.json_MultiLeg_Option_Chain(0)
        #check that we returned a json object with a list of multi legged trades
        #if we don't then we need to move onto the next security
        if tdmultilegjson is None:

            del td
            isTdDeleted = True

        else:

            #grab the indivdual trades that correspond back to the strategy that and put them in a json object
            #was used to get the multi leg json
            td.get_option_chain(symbol=securities['Security'][row], 
                                apikey= 'CEVREV5DD',
                                contracttype=securities['ContractType'][row],
                                strikecount = '',
                                includequotes = securities['includeQuotes'][row],
                                strategy = 'SINGLE',
                                interval = '',
                                strike = '',
                                range = '',
                                fromdate = '',
                                todate='',
                                volatility = '',
                                underlyingprice ='',
                                interestrate = '',
                                daystoexpiration = '',
                                expmonth = '',
                                optiontype = ''

                        
                                      );
            #based on the indivdual objecy json contract type
            #we will grab the json that matches to that partcual type i.e. call = callexpdate amp
            tdindividualjson = td.json_Individual_Option_Chain(securities['ContractType'][row], securities['STRATEGY'][row])

            #read the multilegjson and put it a variable
            df_multileg = pd.read_json(tdmultilegjson)

            #if this is the securitity/record then we will include header and insert the records for the 
            #multileg json otherwise we will append the records
            if headerCounter == 0:

                df_multileg.to_csv(r'c:\users\cosmi\onedrive\desktop\tdmultilegcsv.csv')

            else:

                df_multileg.to_csv(r'c:\users\cosmi\onedrive\desktop\tdmultilegcsv.csv', mode = 'a', header=False)


            df_individualjson = pd.read_json(tdindividualjson)

            #if this is the securitity/record then we will include header and insert the records for the 
            #individualleg json otherwise we will append the records
            if headerCounter == 0:
                df_individualjson.to_csv(r'c:\users\cosmi\onedrive\desktop\tdindividualcsv.csv')
            else:
                df_individualjson.to_csv(r'c:\users\cosmi\onedrive\desktop\tdindividualcsv.csv', mode = 'a', header=False)

            #merge the multileg and individual json on the primaryLeg = description
            #do the same on the secondleg and return the df_analyticaltrade
            df_primarytrade = pd.merge(df_multileg, df_individualjson,left_on = 'primaryLegDescription', right_on = 'description')
            df_analyticaltrade = pd.merge(df_primarytrade, df_individualjson, left_on = 'secondLegDescription', right_on = 'description')
            #rename the columns to distinguish between the primary and secondary legs
            df_analyticaltrade.rename(columns = {'bidAskSize_x':'bidasksizeprimaryleg',
                                            'bidAskSize_y':'bidasksizesecondleg',
                                            'deliverableNote_x':'deliverablenotprimaryleg',
                                            'deliverableNote_y':'deliverablenotsecondleg',
                                            'description_x': 'descriptionprimaryleg',
                                            'description_y': 'descriptionsecondleg',
                                            'exchangename_x':'exchangenameprimaryleg',
                                            'exchangename_y':'exchangenamesecondleg',
                                            'expirationDate_x': 'expirationdateprimaryleg',
                                            'expirationDate_y': 'expirationdatesecondleg',
                                            'expirationType_x': 'expirationtypeprimaryleg',
                                            'expirationType_y': 'expirationtypesecondleg',
                                            'inTheMoney_x': 'inthemoneyprimaryleg',
                                            'inTheMoney_y': 'inthemoneysecondleg',
                                            'isIndexOption_x': 'isindexoptionprimaryleg',
                                            'isIndexOption_y': 'isindexoptionsecondleg',
                                            'lastTradingDay_x': 'lasttradingdayprimaryleg',
                                            'lastTradingDay_y': 'lasttradingdaysecondleg',
                                            'mini_x': 'miniprimaryleg',
                                            'mini_y': 'minisecondleg',
                                            'nonStandard_x': 'nonstandardprimaryleg',
                                            'nonStandard_y': 'nonstandardsecondleg',
                                            'optionDeliverablesList_x':'optiondeliverableslistprimaryleg',
                                            'optionDeliverablesList_y':'optiondeliverableslistsecondleg',
                                            'putCall_x': 'putcallprimaryleg',
                                            'putCall_y': 'putcallsecondleg',
                                            'quoteTimeInLong_x': 'quotetimeinlongprimaryleg',
                                            'quoteTimeInLong_y': 'quotetimeinlongsecondleg',
                                            'settlementType_x': 'settlementtypeprimaryleg',
                                            'settlementType_y': 'settlementtypesecondleg',
                                            'strategy_x':'strategyprimaryleg',
                                            'strategy_y':'strategysecondleg',
                                            'symbol_x':'symbolprimaryleg',
                                            'symbol_y':'symbolsecondleg',
                                            'tradeDate_x': 'tradedateprimaryleg',
                                            'tradeDate_y': 'tradedatesecondleg',
                                            'underlying_x':'underlyingprimaryleg',
                                            'underlying_y':'underlyingsecondleg',
                                            'ask_x':'askprimaryleg',
                                            'ask_y':'asksecondleg',
                                            'askSize_x':'asksizeprimaryleg',
                                            'askSize_y':'asksizesecondleg',
                                            'close_x':'closeprimaryleg',
                                            'close_y':'closesecondleg',
                                            'closePrice_x':'closepriceprimaryleg',
                                            'closePrice_y':'closepricesecondleg',
                                            'daysToExpiration_x':'daystoexpirationprimaryleg',
                                            'daysToExpiration_y':'daystoexpirationsecondleg',
                                            'delta_x':'deltaprimaryleg',
                                            'delta_y':'deltasecondleg',
                                            'bid_x':'bidprimaryleg',
                                            'bid_y':'bidsecondleg',
                                            'bidSize_x':'bidsizeprimaryleg',
                                            'bidSize_y':'bidsizesecondleg',
                                            'gamma_x': 'gammaprimaryleg',
                                            'gamma_y': 'gammasecondleg',
                                            'highPrice_x': 'highpriceprimaryleg',
                                            'highPrice_y': 'highpricesecondleg',
                                            'last_x': 'lastprimaryleg',
                                            'last_y': 'lastsecondleg',
                                            'lastSize_x': 'lastsizeprimaryleg',
                                            'lastSize_y': 'lastsizesecondleg',
                                            'lowPrice_x': 'lowpriceprimaryleg',
                                            'lowPrice_y': 'lowpricesecondleg',
                                            'mark_x': 'markprimaryleg',
                                            'mark_y': 'marksecondleg',
                                            'multiplier_x': 'multiplierprimaryleg',
                                            'multiplier_y': 'multipliersecondleg',
                                            'netChange_x': 'netchangeprimaryleg',
                                            'netChange_y': 'netchangesecondleg',
                                            'oepnInterest_x': 'oepninterestprimaryleg',
                                            'oepnInterest_y': 'oepninterestsecondleg',
                                            'openPrice_x': 'openpriceprimaryleg',
                                            'openPrice_y': 'openpricesecondleg',
                                            'percentChange_x': 'percentchangeprimaryleg',
                                            'percentChange_y': 'percentchangesecondleg',
                                            'rho_x': 'rhoprimaryleg',
                                            'rho_y': 'rhosecondleg',
                                            'strikePrice_x': 'strikepriceprimaryleg',
                                            'strikePrice_y': 'strikepricesecondleg',
                                            'theoreticalOptionValue_x': 'theoreticaloptionvalueprimaryleg',
                                            'theoreticalOptionValue_y': 'theoreticaloptionvaluesecondleg',
                                            'theoreticalVolatility_x': 'theoreticalvolatilityprimaryleg',
                                            'theoreticalVolatility_y': 'theoreticalvolatilitysecondleg',
                                            'theta_x': 'thetaprimaryleg',
                                            'theta_y': 'thetasecondleg',
                                            'timeValue_x': 'timevalueprimaryleg',
                                            'timeValue_y': 'timevaluesecondleg',
                                            'totalVolume_x': 'totalvolumeprimaryleg',
                                            'totalVolume_y': 'totalvolumesecondleg',
                                            'tradeTimeInLong_x': 'tradetimeinlongprimaryleg',
                                            'tradeTimeInLong_y': 'tradetimeinlongsecondleg',
                                            'vega_x': 'vegaprimaryleg',
                                            'vega_y': 'vegasecondleg',
                                            'volatility_x': 'volatilityprimaryleg',
                                            'volatility_y': 'volatilitysecondleg'}, inplace = True)
        
            #if this is the first securitity/record then we will include header and insert the records for the 
            #individualleg json otherwise we will append the records
    
            if headerCounter == 0:

                df_analyticaltrade.to_csv(r'c:\users\cosmi\onedrive\desktop\tdoptionscsv.csv')
            else:

                df_analyticaltrade.to_csv(r'c:\users\cosmi\onedrive\desktop\tdoptionscsv.csv', mode='a', header = False)

            headerCounter += 1
            #print('I am ' + securities['Security'][row])
            #print('I am too ' + securities['Security'][row + 1])
            securityPrices_CSV = ''
            if row + 1 < securities.shape[0]:
                
                #remove the options chain from the last multi leg call
                td.options_chain =[]
                if securities['Security'][row] != securities['Security'][row + 1]:

                    del td
                    isTdDeleted = True
 



