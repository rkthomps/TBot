import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import os

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta


class Alpaca_Scraper:

    def __init__(self):
        self.api = tradeapi.REST()
        self.storage_path = '../data_files/stock_data/'
    
    '''
    Return the data associated with a single ticker.
    We're just gonna rock with Days for now
    '''
    def get_ticker_data(self, ticker, f_date, t_date):
        barset = self.api.get_bars(ticker, TimeFrame.Day,  f_date, t_date, adjustment='raw').df 
        #barset = self.api.get_barset(ticker, freq, limit=limit)
        #return barset[ticker].df.reset_index()
        return barset.reset_index()


    '''
    Return a dictionary with tickers as keys and their 
    historical barsets as values
    '''
    def get_tickers_data(self, tickers, f_date, t_date):
        ret = {}
        for i, ticker in enumerate(tickers):
            print('Getting data for', i, ticker)
            ret[ticker] = self.get_ticker_data(ticker, f_date, t_date)
        return ret


    '''
    Updates the dictionary of company stock data to a local set of files for 
    quick reading. Finds the last day the stock was updated if it exists, and updates
    past that date.

    Assumptions of this function: 
        - granularity is day. 
        - If a file exists for a certain stock, we only need to write data
          since the last write for that stock. We assume that we have data from
          limit until the last write. I don't want to deal with business days.
        - If we want a fresh run, use the function "write_tickers_data"
    '''
    def update_tickers_data(self, tickers, f_date, t_date):
        for ticker in tickers:
            t_storage_name = self.storage_path + ticker + '.csv'
            if os.path.exists(t_storage_name):
                current_data = pd.read_csv(t_storage_name)
                current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])
                date_col = current_data['timestamp']
                max_date = date_col.max()
                min_date = date_col.min()
                str_min_date = str(min_date.date())
                str_max_date = str(max_date.date())

                to_concat = []
                if str_min_date > f_date:
                    df1 = self.get_ticker_data(ticker, f_date, str_min_date)
                    df1_dates = df1['timestamp'].dt.date < min_date.date()
                    df1 = df1.loc[df1_dates].copy()
                    to_concat.append(df1)

                to_concat.append(current_data)
                if str_max_date < t_date:
                    df2 =  self.get_ticker_data(ticker, str_max_date, t_date)
                    df2_dates = df2['timestamp'].dt.date > max_date.date()
                    df2 = df2.loc[df2_dates].copy()
                    to_concat.append(df2)

                if len(to_concat) > 1:
                    write_df = pd.concat(to_concat)
                else:
                    write_df = current_data

                write_df.to_csv(t_storage_name, index=False)

            else:
                new_data = self.get_ticker_data(ticker, f_date, t_date)
                new_data.to_csv(t_storage_name, index=False)

