import os
import sys
import yfinance as yf
import requests
import json
import time 
import glob
import re

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from pandas.io.json import json_normalize

from Company_Lister import Company_Lister


class Scraper:
    def __init__(self):
        self.storage_path = '../data_files/stock_data/'
        self.quarter_path = '../data_files/quarterly_data/'
        self.sector_path = '../data_files/sectors.csv'
        self.q4_path = '../data_files/attr_dict.json'
        self.default_from = '1970-01-01'
        self.default_to = str(datetime.datetime.now().date())
    
    '''
    Return the data associated with a single ticker.
    We're just gonna rock with Days for now
    '''
    def get_ticker_data(self, ticker, f_date=None, t_date=None):
        if f_date is None:
            f_date = self.default_from
        if t_date is None:
            t_date = self.default_to

        t = yf.Ticker(ticker)
        df = t.history(start=f_date, end=t_date, interval='1d')
        return df.reset_index() 

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
            print("Getting Data For", ticker)
            t_storage_name = self.storage_path + ticker + '.csv'
            if os.path.exists(t_storage_name):
                current_data = pd.read_csv(t_storage_name)
                current_data['Date'] = pd.to_datetime(current_data['timestamp'])
                date_col = current_data['Date']
                max_date = date_col.max()
                min_date = date_col.min()
                str_min_date = str(min_date.date())
                str_max_date = str(max_date.date())

                to_concat = []
                if str_min_date > f_date:
                    df1 = self.get_ticker_data(ticker, f_date, str_min_date)
                    df1_dates = df1['Date'].dt.date < min_date.date()
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

    '''
    At this point in the project, there is no need to specify a date to scrape to for
    this function. I suppose it is inefficient to scrape over data we have already scraped
    through. However, detecting last scrape dates is a little messy. We can start doing that
    once we actually need to go live with this project
    '''
    def get_quarterly_data(self, ticker):
        tokenized_url = 'https://api.polygon.io/vX/reference/financials?ticker=<ticker>&filing_date.lt=<c_date>&limit=100&timeframe=<tf>&apiKey=ESm_UpusESwUVkzhU8V41cpSNVuVFjlk'
        tokenized_url = tokenized_url.replace('<ticker>', ticker)
        cur_url = tokenized_url.replace('<c_date>', str(datetime.datetime.now().date()))
        urls = [cur_url.replace('<tf>', 'quarterly'), cur_url.replace('<tf>', 'annual')]
        dfs = []
        for url in urls:
            cur_url = url
            rows = []
            while True:
                result = requests.get(cur_url)
                result_dict = json.loads(result.content)
                while result_dict['status'] != 'OK':
                    if 'error' in result_dict and 'maximum requests' in result_dict['error']:
                        print('sleeping')
                        time.sleep(5)
                        result = requests.get(cur_url)
                        result_dict = json.loads(result.content)
                    else:
                        print(result_dict)
                        break
                for fin_dict in result_dict['results']:
                    next_row = {}
                    for measure_type, mt_dict in fin_dict['financials'].items():
                        for measure, m_dict in mt_dict.items():
                            next_row[measure] = m_dict['value']
                    next_row['Year'] = fin_dict['fiscal_year']
                    next_row['Quarter'] = fin_dict['fiscal_period']
                    next_row['Start'] = fin_dict['start_date']
                    next_row['End'] = fin_dict['end_date']
                    rows.append(next_row)
                try:
                    print('next_page')
                    cur_url = result_dict['next_url'] + '&apiKey=ESm_UpusESwUVkzhU8V41cpSNVuVFjlk'
                except KeyError as e:
                    break
            ret_df = pd.DataFrame(rows)
            ret_df['Company'] = ticker
            dfs.append(ret_df)
        return pd.concat(dfs) 


    '''
    Uses previous analysis done on the quarterly attributes to select
    only commonly reported attributes.
        - Analysis done in the financials jupyter notebook
        - Eventually I should make a util function or something that can do all of the analysis
            - Such notebooks should be run on a set interval depending on when their dependancies
              change
            - We're not at production yet
        - Right now I should just focus on making the notebooks reproducable
    Uses the same prior analysis to determine whether or not a transformation
    must be applied for the stock's q4 value (if we have to subtract q1-q3)

    ## I COULD WRITE THIS MUCH FASTER BUT IT WOULD TAKE MORE WORK 
    '''
    def clean_final_df(self, final_df):
        with open(self.q4_path, 'r') as fin:
            attr_dict = json.load(fin)
        attrs = [] 
        for attr_type in attr_dict:
            attrs.extend(attr_dict[attr_type]) ## Only will include numeric attributes
        attrs.extend(['Company', 'Quarter', 'Start', 'End'])

        ## Get only the commonly reported attributes
        ## Drop Q4 values (they really exist in FY)
        clean_df = final_df.loc[final_df['Quarter'] != 'Q4', attrs].dropna()
        clean_df['TimeFrame'] = clean_df['Quarter'].apply(lambda x: x[0])
        grouped = clean_df.groupby(['Company', 'Year', 'TimeFrame'])[attr_dict['Additive']].sum()
        grouped = grouped.reset_index(level=2)
        q_cols = grouped['TimeFrame'] == 'Q'
        grouped = grouped.drop(columns = ['TimeFrame'], axis=1)
        grouped.loc[q_cols] = grouped.loc[q_cols] * -1
        q4_vals = grouped.reset_index().groupby(['Company', 'Year']).sum()
        q4_vals['Quarter'] = 'FY'
        q4_vals = q4_vals.set_index(['Quarter'], append=True)
        for i, attrs in q4_vals.iterrows():
            clean_df.loc[(clean_df['Company'] == i[0]) & (clean_df['Year'] == i[1]) & (clean_df['Quarter'] == 'FY'), attrs.index.values] = attrs.values
        return clean_df 
        

    '''
    Updates all of the quarterly data for all the given tickers
    '''
    def get_all_quarterly(self, tickers, overwrite=False):
        paths = glob.glob(self.quarter_path + '*.csv')
        saved = []
        for p in paths:
            result = re.search(r'\/([A-z]+)\.csv', p)
            if result is not None:
                saved.append(result.groups()[0])
        for ticker in tickers:
            print('Getting Quarterly Data For', ticker)
            if (not overwrite) and ticker in saved:
                continue
            self.get_quarterly_data(ticker).to_csv(self.quarter_path + ticker + '.csv', index=False)

        dfs = []
        paths = glob.glob(self.quarter_path + '*.csv')
        for path in paths:
            if 'quarters' in path:
                continue
            df = pd.read_csv(path)
            dfs.append(df)
        
        final_df = pd.concat(dfs)
        final_df = self.clean_final_df(final_df)
        final_df.to_csv(self.quarter_path + 'quarters.csv', index=False)
        return final_df

    '''
    Gets the quarterly data from factset
    '''
    def get_factset_quarterly(self, tickers):
        authorization = (os.getenv('FACT_USER'), os.getenv('FACT_KEY'))
        print(authorization)
        fundamentals_endpoint = 'https://api.factset.com/content/factset-fundamentals/v1/fundamentals'
        request = {
            'ids': tickers[:3],
            'periodicity': 'QTR',
            'fiscalPeriodStart': '2018-01-01',
            'fiscalPeriodEnd': '2018-03-01',
            'metrics': '\metrics',
            'currency': 'USD',
            'restated': 'RF'
            }
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        post = json.dumps(request)
        response = requests.post(
                url=fundamentals_endpoint, 
                data=post, 
                auth=authorization, 
                headers=headers, 
                verify=False)
        print(response)
        print(response.text)
        data = json.loads(response.text)
        print(data)

            


    '''
    Gets the sector of each company in the input list.
    '''
    def get_sector(self, tickers, overwrite=False):
        measures = ['sector', 'industry']
        if os.path.exists(self.sector_path) and (not overwrite):
            return
        c_table = []
        c_columns = ['company'] + measures
        for ticker in tickers:
            print("Retrieving Data for", ticker)
            t = yf.Ticker(ticker)
            next_row = []
            next_row.append(ticker)
            for m in measures:
                try:
                    next_row.append(t.info[m])
                except KeyError as e:
                    print(e)
            c_table.append(next_row)
        ret_df = pd.DataFrame(c_table, columns=c_columns)
        ret_df.to_csv(self.sector_path, index=False)
        return ret_df



        
'''
The main method of this module scrapes all SNP 500 data from 1970 to the day
- This method has a couple of options
    - The default action is to update 
    - On the '-q' flag it will scrape quarterly data
'''
def main():
    snp = sorted(Company_Lister().get_snp())
    scrape = Scraper()
    now = datetime.datetime.now()
    if '-q' in sys.argv:
        if '-o' in sys.argv:
            scrape.get_all_quarterly(snp, overwrite=True)
            scrape.get_sector(snp, overwrite=True)
        else:
            scrape.get_quarterly(snp)
            scrape.get_sector(snp)
    if '-qf' in sys.argv:
        scrape.get_factset_quarterly(snp)
    else:
        scrape.update_tickers_data(snp, '1970-01-01', str(now.date()))

def temp_main():
    snp = sorted(Company_Lister().get_snp())
    scrape = Scraper()
    scrape.get_factset_quarterly(snp)
    

if __name__ == '__main__':
    temp_main()


