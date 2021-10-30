import json
import datetime
import requests
import math
import numpy as np
import datetime as dt
import pandas as pd
from yahoo_finance_api2 import share
class TDAmeriTradeClass(object):
    """description of class"""
    
    def __init__(self):
       
        self.options_chain = [] 
        self.price_history = []
        self.strategyType = ''
        self.fortyfivedayVolatility = ''
        self.sixtydayVolatility = ''
        self.thirtydayVolatility = ''
    
    #return the price history and append it to the price history list
    def get_price_history(self, **arguments):

         url = 'https://api.tdameritrade.com/v1/marketdata/' + arguments.get('symbol') + '/pricehistory?'
         params ={}
         for arg in arguments:
             if arg != 'symbol':
                parameter = {arg: arguments.get(arg)}
                params.update(parameter)

         self.price_history.append(requests.get(url, params=params).json())

    #return the current quote and append it to the price history list
    def get_quote_price_history(self, **arguments):

         symbol = arguments.get('symbol')
         my_share = share.Share(symbol)
         symbol_data = None
         symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                          1,
                                          share.FREQUENCY_TYPE_DAY,
                                          1)
         results = {'close':round(symbol_data['close'][0],2), 'datetime':symbol_data['timestamp'][0], 'high':round(symbol_data['high'][0],2), 'low':round(symbol_data['low'][0],2), 'open':round(symbol_data['open'][0],2), 'volume':round(symbol_data['volume'][0],2)}
         self.price_history[0]['candles'].append(results)

    #get HV for 30,45 and 60 days
    def get_historical_volatility(self, prices):

        json_candles = json.dumps(self.price_history[0]['candles'])        
        df_candles = pd.read_json(json_candles)
        #df_candles.sort_values('datetime', ascending=False, inplace = True)
        #df_candles.set_index(keys='datetime', inplace = True)
        #get the volatility for 30,45,60 days
        self.sixtydayVolatility = df_candles[['close']].tail(40).pct_change().apply(lambda x:np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))._values[0]
        self.fortyfivedayVolatility = df_candles[['close']].tail(30).pct_change().apply(lambda x:np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))._values[0]
        self.thirtydayVolatility = df_candles[['close']].tail(20).pct_change().apply(lambda x:np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))._values[0]

    #get the options chain returns for a given strategy and security
    #and append it to the options chain list    
    def get_option_chain(self,**arguments):

        url = 'https://api.tdameritrade.com/v1/marketdata/chains?'
    
        params ={}
        spreadTuples = ('COVERED', 'VERTICAL', 'CALENDAR', 'STRANGLE', 'STRADDLE',
                        'BUTTERFLY', 'CONDOR', 'DIAGONAL', 'COLLAR', 'ROLL' );
        rangeTuples = ('ITM','NTM','OTM','SAK''SBK','SNK','ALL');

        #populte the objects into the paramas object to pass to the url as the paramters
        #the loop below updates the paramter with a default value if we didn't pass a value to it 
        #the code also set global variables so we can indentify additional objects for the objects returned
        for arg in arguments:
            if arg == 'contractType' and arguments[arg] == '':
                parameter = {arg: 'ALL'}
            elif arg == 'includeQuotes' and arguments[arg] == '':
                 parameter = {arg: 'FALSE'}
            elif arg == 'strategy' and arguments[arg] == '':
                 parameter = {arg: 'SINGLE'}
            elif arg == 'interval' and arguments[arg] == '' and params.get('contractType') in spreadTuples:
                 parameter = {arg: 1}
            elif arg == 'range' and arguments[arg] =='':
                 parameter = {arg:'ALL'}
            elif arg == 'strategy':
                 self.strategyType = arguments[arg]
                 parameter = {arg: arguments.get(arg)}
                 params.update(parameter)
            else:
                parameter = {arg: arguments.get(arg)}
            params.update(parameter)
            
        self.options_chain.append(requests.get(url, params=params).json())
        


    def json_MultiLeg_Option_Chain(self,underlying_position): 
        
        optionChains = []
        
        #check to verify that we have multlegged trades coming in,
        #if we don't then want to exit the function 
        try:
            if self.options_chain[underlying_position]['status'] == "FAILED" :

                return 
             
            else:        
                monthlyStrategyList =  self.options_chain[underlying_position]['monthlyStrategyList']

                for option in monthlyStrategyList:       
            
                    optionStrategyList =  option['optionStrategyList']
                    for optionList in optionStrategyList:

                        data = {}
                        #information about the underlying asset
                        data['symbol'] = self.options_chain[underlying_position]['symbol']
                        data['underlying'] = self.options_chain[underlying_position]['underlying']['description']
                        data['close'] = self.options_chain[underlying_position]['underlying']['close']
                        data['last'] = self.options_chain[underlying_position]['underlying']['last']
                        data['strategy'] = self.options_chain[underlying_position]['strategy']
                        data['interest_rate'] = self.options_chain[underlying_position]['interestRate']
                        data['underlying_price'] = self.options_chain[underlying_position]['underlyingPrice']
                        data['volatility'] = self.options_chain[underlying_position]['volatility']
                        data['days_to_expiration'] = self.options_chain[underlying_position]['daysToExpiration']
                        data['number_of_contracts'] = self.options_chain[underlying_position]['numberOfContracts']

                        #data about all the options for a given month, in this case the detail is about the month
                        if option['month'] == 'Jan':
                            data['month'] = 1
                        elif option['month'] == 'Feb':
                            data['month'] = 2
                        elif option['month'] == 'Mar':
                            data['month'] = 3
                        elif option['month'] == 'Apr':
                            data['month'] = 4
                        elif option['month'] == 'May':
                            data['month'] = 5
                        elif option['month'] == 'Jun':
                            data['month'] = 6
                        elif option['month'] == 'Jul':
                            data['month'] = 7
                        elif option['month'] == 'Aug':
                            data['month'] = 8
                        elif option['month'] == 'Sep':
                            data['month'] = 9
                        elif option['month'] == 'Oct':
                            data['month'] = 10
                        elif option['month'] == 'Nov':
                            data['month'] = 11
                        elif option['month'] == 'Dec':
                            data['month'] = 12

                        data['year'] = option['year']
                        data['day'] = option['day']
                        data['daysToExp'] = option['daysToExp']
                        data['secondaryMonth'] = option['month']
                        data['secondaryYear'] = option['year']
                        data['secondaryDay'] = option['day']
                        data['secondaryDaysToExp'] = option['daysToExp']
                        data['type'] = option['type']
                        data['secondaryType'] = option['secondaryType']
                        data['leap'] = option['leap']
                        option['legType'] = 'primaryLeg'
                
                        primaryLeg = optionList['primaryLeg'] 
                        secondLeg = optionList['secondaryLeg']

                        #grab data about indvidual trade legs, this is the lowest level of granularlity
                        data['primaryLegSymbol'] = primaryLeg['symbol']
                        data['primaryLegCallInd'] = primaryLeg['putCallInd']
                        data['primaryLegDescription'] = primaryLeg['description']
                        data['primaryLegbid'] = primaryLeg['bid']
                        data['primaryLegask'] = primaryLeg['ask']
                        data['primaryLegrange'] = primaryLeg['range']
                        data['primaryLegStrikePrice'] = primaryLeg['strikePrice']
                        data['primaryLegTotalVolume'] = primaryLeg['totalVolume']
                        data['secondLegSymbol'] = secondLeg['symbol']
                        data['secondLegCallInd'] = secondLeg['putCallInd']
                        data['secondLegDescription'] = secondLeg['description']
                        data['secondLegbid'] = secondLeg['bid']
                        data['secondLegask'] = secondLeg['ask']
                        data['secondLegrange'] = secondLeg['range']
                        data['secondLegStrikePrice'] = secondLeg['strikePrice']
                        data['secondLegTotalVolume'] = secondLeg['totalVolume']
                        data['strategyStrike'] = optionList['strategyStrike']
                        data['strategyBid'] = optionList['strategyBid']
                        data['strategyAsk'] = optionList['strategyAsk']
                        data['sixtyDayHV'] = self.sixtydayVolatility
                        data['fortyfiveDayHV'] = self.fortyfivedayVolatility
                        data['thirtyDayHV'] = self.thirtydayVolatility

                        if self.strategyType == 'VERTICAL':

                            data['debitSpreadCost'] = optionList['strategyBid']
                            data['creditSpreadCost'] = optionList['strategyAsk']

                        elif self.strategyType == 'STRANGLE':

                            data['debitStrangleCost'] = optionList['strategyBid']
                            data['creditStrangleCost'] = optionList['strategyAsk']

                        elif self.strategyType == 'STRADDLE':

                            data['debitStrangleCost'] = optionList['strategyBid']
                            data['creditStrangleCost'] = optionList['strategyAsk']

                        #add the option to the chain to the list
                        optionChains.append(data)
            
                    #return the options chain list to the caller
                    optionsChain = json.dumps(optionChains)
                return optionsChain
        except:
              return

    #return the invidual options chain to the caller
    def json_Individual_Option_Chain(self,contractType, isSingle): 
        
        optionChains = []
        
        #determine if the indvidual options chain is a list of calls, puts or both
        if contractType.upper() == 'CALL':

            contractType = 'callExpDateMap'
            contractTypes = ['callExpDateMap']

        elif contractType.upper() == 'PUT':

            contractType = 'putExpDateMap'
            contractTypes = ['putExpDateMap']

        else:

            contractType = 'callExpDateMap'
            contractTypes = ['callExpDateMap','putExpDateMap']


        for contractType in contractTypes:
            
            if isSingle == "SINGLE":
                
                underlying_position = 0

            else:

                underlying_position = 1

            callExpDataMapList =  self.options_chain[underlying_position][contractType]
            i = 0
            dateKeys = callExpDataMapList.keys()
            while i < len(callExpDataMapList):  
            
                dateKey = list(dateKeys)[i]
                x = 0    
                priceObject = callExpDataMapList[dateKey]
                priceKeys = priceObject.keys()

                while x < len(priceKeys):
 
                    priceKey = list(priceObject)[x]
                    data = {}
                    #information about the underlying asset
                    data['symbol'] = self.options_chain[underlying_position]['symbol']
     #              data['underlying'] = self.options_chain[underlying_position]['underyling']['description']
      #             data['close'] = self.options_chain[underlying_position]['underlying']['close']
     #               data['strategy'] = priceObject[priceKey][0]['strategy']
                    data['strategy'] = self.options_chain[underlying_position]['strategy']
                    data['interest_rate'] = self.options_chain[underlying_position]['interestRate']
                    data['underlying_price'] = self.options_chain[underlying_position]['underlyingPrice']
                    data['volatility'] = self.options_chain[underlying_position]['volatility']
                    data['days_to_expiration'] = self.options_chain[underlying_position]['daysToExpiration']
                    data['number_of_contracts'] = self.options_chain[underlying_position]['numberOfContracts']

                    #grab data about indvidual trade  this is the lowest level of granularlity
                    data['putCall'] = priceObject[priceKey][0]['putCall']
                    data['symbol'] = priceObject[priceKey][0]['symbol']
                    data['description'] = priceObject[priceKey][0]['description']
                    data['exchangeName'] = priceObject[priceKey][0]['exchangeName']
                    data['bid'] = priceObject[priceKey][0]['bid']
                    data['ask'] = priceObject[priceKey][0]['ask']
                    data['last'] = priceObject[priceKey][0]['last']
                    data['mark'] = priceObject[priceKey][0]['mark']
                    data['bidSize'] = priceObject[priceKey][0]['bidSize']
                    data['askSize'] = priceObject[priceKey][0]['askSize']
                    data['bidAskSize'] = priceObject[priceKey][0]['bidAskSize']
                    data['lastSize'] = priceObject[priceKey][0]['lastSize']
                    data['highPrice'] = priceObject[priceKey][0]['highPrice']
                    data['lowPrice'] = priceObject[priceKey][0]['lowPrice']
                    data['openPrice'] = priceObject[priceKey][0]['openPrice']
                    data['closePrice'] = priceObject[priceKey][0]['closePrice']
                    data['totalVolume'] = priceObject[priceKey][0]['totalVolume']
                    data['tradeDate'] = priceObject[priceKey][0]['tradeDate']
                    data['tradeTimeInLong'] = priceObject[priceKey][0]['tradeTimeInLong']
                    quote_date = dt.datetime.fromtimestamp(int(priceObject[priceKey][0]['quoteTimeInLong'])/1000)
                    data['quoteTimeInLong'] = quote_date.strftime("%d-%b-%y" )
                    data['netChange'] = priceObject[priceKey][0]['netChange']
                    data['volatility'] = priceObject[priceKey][0]['volatility']
                    data['delta'] = priceObject[priceKey][0]['delta']
                    data['gamma'] = priceObject[priceKey][0]['gamma']
                    data['theta'] = priceObject[priceKey][0]['theta']
                    data['vega'] = priceObject[priceKey][0]['vega']
                    data['rho'] = priceObject[priceKey][0]['rho']
                    data['oepnInterest'] = priceObject[priceKey][0]['openInterest']
                    data['timeValue'] = priceObject[priceKey][0]['timeValue']
                    data['theoreticalOptionValue'] = priceObject[priceKey][0]['theoreticalOptionValue']
                    data['theoreticalVolatility'] = priceObject[priceKey][0]['theoreticalVolatility']
                    data['optionDeliverablesList'] = priceObject[priceKey][0]['optionDeliverablesList']
                    data['strikePrice'] = priceObject[priceKey][0]['strikePrice']
                    expiration_date = dt.datetime.fromtimestamp(int(priceObject[priceKey][0]['expirationDate'])/1000)
                    data['expirationDate'] = expiration_date.strftime("%d-%b-%y" )
                    data['daysToExpiration'] = priceObject[priceKey][0]['daysToExpiration']
                    data['expirationType'] = priceObject[priceKey][0]['expirationType']
                    last_trading_date = dt.datetime.fromtimestamp(int(priceObject[priceKey][0]['lastTradingDay'])/1000)
                    data['lastTradingDay'] = last_trading_date.strftime("%d-%b-%y" )
                    data['multiplier'] = priceObject[priceKey][0]['multiplier']
                    data['settlementType'] = priceObject[priceKey][0]['settlementType']
                    data['deliverableNote'] = priceObject[priceKey][0]['deliverableNote']
                    data['isIndexOption'] = priceObject[priceKey][0]['isIndexOption']
                    data['percentChange'] = priceObject[priceKey][0]['percentChange']
                    data['markChange'] = priceObject[priceKey][0]['markChange']
                    data['markPercentChange'] = priceObject[priceKey][0]['markPercentChange']
                    data['mini'] = priceObject[priceKey][0]['mini']
                    data['inTheMoney'] = priceObject[priceKey][0]['inTheMoney']
                    data['nonStandard'] = priceObject[priceKey][0]['nonStandard']
                    data['sixtyDayHV'] = self.sixtydayVolatility
                    data['fortyfiveDayHV'] = self.fortyfivedayVolatility 
                    data['thirtyDayHV'] = self.thirtydayVolatility 
                    optionChains.append(data)
                    x += 1
                i += 1
            optionsChain = json.dumps(optionChains)
        return optionsChain


       

