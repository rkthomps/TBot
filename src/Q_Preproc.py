

import json, glob, re, os
import numpy as np
import pandas as pd


class Quarterly_Preprocessor:
    def __init__(self):
        ## Paths for data
        self.quarter_loc = '../data_files/quarterly_data/quarters.csv'
        self.rates_loc = '../data_files/PRIME.csv'
        self.sector_loc = '../data_files/sectors.csv'
        self.redundant_loc = '../data_files/redundant.json'
        self.tech_loc = '../data_files/stock_data/'

        ## Load to dataframes
        self.load_data()

        ## Normalize quarterly dates
        self.normalize_dates()
        
        ## Merge the financials, technicals, and interest rates 
        self.merge_data()



    '''
    Returns a 7-tuple with the following values
        - examples_df: x values
        - y
        - buy_price
        - buy_date
        - sell_date
        - company 
        - start_week: 
        
    '''
    def produce_ind_and_response(self, n_quarters, start_year=None, end_year=None):
        if start_year is None:
            start_year = self.quarter_df['Year'].min()
        if end_year is None:
            end_year = self.quarter_df['Year'].max()
         
        x, y, price, b_date, s_date = self.create_examples(n_quarters, start_year, end_year)
        examples_df = pd.merge(x, self.sectors, left_on='Company', right_on='company', how='left')
        example_industry = pd.get_dummies(examples_df['industry'])
        example_sector = pd.get_dummies(examples_df['sector'])
        examples_df = pd.concat((examples_df, example_sector), axis=1)
        companies = x['Company']
        start_qs = x['start_q']

        to_drop = [
            'company', 
            'TimeFrame',
            'Quarter',
            'Year',
            'Date',
            'Start',
            'End',
            'q_index',
            'q_num',
            'P_Date',
            'industry',
            'sector',
            'Company',
            'start_q'
        ]

        # Add columns that were found to be redundant through extreme correlation 
        to_drop.extend(self.redundant)

        examples_df = self.drop_columns(examples_df, to_drop)
        return (examples_df.values, y.values, list(examples_df.columns), 
            price.values, b_date.values, s_date.values, companies.values, start_qs.values)




    '''
    Produce a dataframe where each row represents the information available
    from n_quarters quarters, and the target variable represents the closing
    price for one quarter in advance
    '''
    def create_examples(self, n_quarters, start_year, end_year):
        quarter_map = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'FY': 3}
        self.quarter_df = self.quarter_df.loc[(self.quarter_df['Year'] >= start_year) & (self.quarter_df['Year'] <= end_year)].copy()
        self.quarter_df['q_num'] = self.quarter_df['Quarter'].map(quarter_map)
        self.quarter_df['q_index'] = (4 * (self.quarter_df['Year'] - start_year) + self.quarter_df['q_num']).astype(int)
        print(np.sort(self.quarter_df['q_index'].unique()))
        self.quarter_df['yq'] = self.quarter_df.apply(lambda x: str(x['Year']) + str(x['q_num']), axis=1)
        self.quarter_df = self.quarter_df.groupby(['Company', 'q_index']).first().reset_index()


        results = []
        for i in range(int(self.quarter_df['q_index'].max() - n_quarters)):
            valid_comp = self.quarter_df.loc[(self.quarter_df['q_index'] >= i) & (self.quarter_df['q_index'] <= i + n_quarters)]
            counts = valid_comp['Company'].value_counts()
            use_companies = counts[counts == n_quarters + 1].index.values
            if len(use_companies) == 0:
                continue
            comp_data = valid_comp.loc[valid_comp['Company'].isin(use_companies)].set_index('Company')
            ind_ex = []
            for j in range(i, i + n_quarters):
                cur_df = comp_data.loc[comp_data['q_index'] == j]
                cur_df.columns = [c + '_' + str(j - i) for c in cur_df.columns]
                ind_ex.append(cur_df)

            ind_ex.append(comp_data.loc[comp_data['q_index'] == i + n_quarters, ['Close', 'P_Date']])
            result = pd.concat(ind_ex, axis=1).reset_index()
            result['start_q'] = i
            results.append(result)

        ret_df = pd.concat(results)
        buy_price = ret_df['Close_' + str(n_quarters - 1)]
        buy_date = ret_df['P_Date_' + str(n_quarters - 1)]
        sell_date = ret_df['P_Date']
        target = ret_df['Close'] / ret_df['Close_' + str(n_quarters - 1)]
        ret_df = ret_df.drop(columns=['Close', 'P_Date'], axis=1)
        return ret_df, target, buy_price, buy_date, sell_date


    ## Drops all columns from the given dataframe where any of hte
    ## given values to drop is a prefix of the column name
    def drop_columns(self, df, to_drop):
        drop_these = []
        for c in df.columns:
            for t in to_drop:
                if c.startswith(t):
                    drop_these.append(c)
        return df.drop(columns=drop_these, axis=1)



    '''
    Produces a dataframe that includes the target variable - closing
    value for a quarter in advance, and also includes interest rates
    for each quarter
    '''
    def merge_data(self):
        ## Join stock data
        closes = self.stock_df.groupby(['Company', 'Date'])['Close'].first().reset_index()
        closes = closes.set_index(['Company', 'Date']).reindex(self.quarters.set_index(['Company', 'P_Date']).index, method='pad').reset_index().drop_duplicates()
        quarter_df = pd.merge(self.quarters, closes, on=['Company', 'P_Date'], how='left')

        ## Join interest data
        ## Here we pad the interest rates to the granularity of the financials 
        c_rates = self.rates.set_index('DATE').reindex(quarter_df.set_index('End').index, method='pad').reset_index().drop_duplicates()
        self.quarter_df = pd.merge(quarter_df, c_rates, on='End', how='inner')


    '''
    Since quarterly reporting can happen at different times between
    Companies, I'll introduce a standard filing time for each quarter
    and call it 'P_Date' in the dataframe
    '''
    def normalize_dates(self):
        to_predict = {
            'Q1': '03-31',
            'Q2': '06-30',
            'Q3': '09-30',
            'FY': '12-31'
        }

        self.quarters['P_Date'] = self.quarters['Quarter'].map(to_predict)
        self.quarters['P_Date'] = self.quarters.apply(lambda x: str(int(x['Year'])) + '-' + x['P_Date'], axis=1)
        self.quarters['P_Date'] = pd.to_datetime(self.quarters['P_Date'], errors='coerce')
        self.quarters = self.quarters.dropna()


    '''
    Creates Dataframes for sectors, quarters, and rates. Also gathers
    the target variable: the closing value for a quarter in advance
    '''
    def load_data(self):
        self.sectors = pd.read_csv(self.sector_loc)
        self.sectors = self.sectors.groupby('company').first().reset_index()
        self.quarters = pd.read_csv(self.quarter_loc)
        self.quarters['Start'] = pd.to_datetime(self.quarters['Start'])
        self.quarters['End'] = pd.to_datetime(self.quarters['End'])
        self.rates = pd.read_csv(self.rates_loc)
        self.rates['DATE'] = pd.to_datetime(self.rates['DATE'])

        with open(self.redundant_loc, 'r') as fin:
            self.redundant = json.load(fin)

        min_date = self.quarters['Start'].min()

        # Get the technicals data
        stocks = []
        for path in glob.glob(self.tech_loc + '*.csv'):
            head, tail = os.path.split(path)
            name = tail.split('.')[0]
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            small_df = df.loc[df['Date'] > min_date].copy()
            small_df['Company'] = name
            stocks.append(small_df)
        self.stock_df = pd.concat(stocks)


