

import json, glob, re, os
import numpy as np
import pandas as pd
import datetime


class Weekly_Preprocessor:
    def __init__(self):
        ## Paths for data
        self.sector_loc = '../data_files/sectors.csv'
        self.tech_loc = '../data_files/stock_data/'

        ## Load to dataframes
        self.load_data()
        
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
    def produce_ind_and_response(self, n_weeks, start_year=None, end_year=None):
        if start_year is None:
            start_year = self.all_df['Date'].min().year
        if end_year is None:
            end_year = self.all_df['Date'].max().year
         
        x, y, price, x_names, b_date, s_date, companies, start_weeks = self.create_stacked_examples(n_weeks, start_year, end_year)
        
        return x, y, price, x_names, b_date, s_date, companies, start_weeks



    def create_stacked_examples(self, n_weeks, start_year, end_year):
        all_df = self.all_df.loc[(self.all_df['Date'].dt.year >= start_year) & (self.all_df['Date'].dt.year <= end_year)]
        min_date = all_df['Date'].min().date()
        max_date = all_df['Date'].max().date()
        first_monday = all_df.loc[all_df['Date'].dt.strftime('%a') == 'Mon']['Date'].min().date()
        mondays = pd.date_range(first_monday, max_date + pd.Timedelta(7, 'd'), freq='7D')
        all_df['wi'] = pd.cut(all_df['Date'], mondays, right=False, labels=np.arange(len(mondays) - 1))
        all_df['di'] = all_df['Date'].dt.strftime('%w')

        ys = []
        xs = []
        prices = []
        buy_dates = []
        sell_dates = []
        companies = []
        start_weeks = []
        dis = []
        desired_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] # Could add Dividends 'Stock Splits'
        not_x = ['Company', 'Date', 'wi', 'di']
        desired_cols = list(all_df.columns)
        for r in not_x:
            desired_cols.remove(r)

        for company, df in all_df.groupby('Company'):
            start_wi = df['wi'].min()
            end_wi = df['wi'].max()
            full_cdf = df[(df['wi'] > start_wi) & (df['wi'] < end_wi)]
            col_indices = full_cdf.columns.get_indexer(desired_cols)
            close_loc = full_cdf.columns.get_loc('Close')
            di_loc = full_cdf.columns.get_loc('di')
            date_loc = full_cdf.columns.get_loc('Date')
            wi_loc = full_cdf.columns.get_loc('wi')

            full_cdf = full_cdf.sort_values('Date')
            for i in range(0, len(full_cdf) - (5 * (n_weeks + 1)) + 1, 5):
                xs.append(full_cdf.iloc[i:(i + (5 * n_weeks)), col_indices].values)
                dis.append(full_cdf.iloc[i:(i + (5 * n_weeks)), di_loc].values)
                target_close = full_cdf.iloc[i+(5*(n_weeks+1))-1, close_loc]
                last_close = full_cdf.iloc[i+(5*n_weeks)-1, close_loc]
                ys.append(target_close / last_close)
                buy_dates.append(str(full_cdf.iloc[i+(5*n_weeks)-1, date_loc].date()))
                sell_dates.append(str(full_cdf.iloc[i+5*(n_weeks+1)-1, date_loc].date()))
                companies.append(company)
                start_weeks.append(full_cdf.iloc[i, wi_loc])
            print(company, all([(d == dis[0]).all() for d in dis]))
        return (np.array(xs), np.array(ys), np.array(prices), np.array(desired_cols), 
            np.array(buy_dates), np.array(sell_dates), np.array(companies), np.array(start_weeks))


    '''
    Produce a dataframe where each row represents the information available
    from n_quarters quarters, and the target variable represents the closing
    price for one quarter in advance
    '''
    def create_examples(self, n_quarters, start_year, end_year):
        pass # could draw inspo from quarter_df or from past implementation


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
    Merge stock_df and sectors on the company
    '''
    def merge_data(self):
        ## Dummy Df for Industry and Sector
        example_industry = pd.get_dummies(self.sectors['industry'])
        example_sector = pd.get_dummies(self.sectors['sector'])
        self.sectors = pd.concat((self.sectors, example_industry, example_sector), axis=1)
        self.sectors = self.sectors.drop(columns=['industry', 'sector'], axis=1)
        ## Join stock data
        self.all_df = pd.merge(self.stock_df, self.sectors, how='inner', on='Company')


    '''
    Creates dataframes for all of the stock data (in one dataframe), and 
    gathers sector and industry information
    '''
    def load_data(self):
        self.sectors = pd.read_csv(self.sector_loc)
        self.sectors = self.sectors.groupby('company').first().reset_index()
        self.sectors = self.sectors.rename(columns={'company': 'Company'}) 

        # Get the technicals data
        stocks = []
        for path in glob.glob(self.tech_loc + '*.csv')[:1]:
            head, tail = os.path.split(path)
            name = tail.split('.')[0]
            df = pd.read_csv(path)
            if len(df) == 0:
                continue
            df['Date'] = pd.to_datetime(df['Date'])
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            # Makes sure the dataframe has no missing value. Fills in holidays with forward padding
            df = df.set_index('Date').reindex(pd.date_range(min_date, max_date, freq='B', name='Date'), method='pad').reset_index()
            df['Company'] = name
            stocks.append(df)
        self.stock_df = pd.concat(stocks)
        self.stock_df = self.stock_df.groupby(['Company', 'Date']).first().reset_index()

if __name__ == '__main__':
    x, y, price, x_names, b_date, s_date, companies, start_weeks = Weekly_Preprocessor().produce_ind_and_response(40)
    exit()

    ## THIS STUFF OCCUPYS AN UNTENABLE AMOUNT OF MEMORY
    head = os.path.join('..', 'data_files', 'backtest_data', 'w_formatted')
    print('Writing x shape')
    np.savetxt(os.path.join(head, 'x_shape.txt'), np.array(x.shape), fmt='%d')
    print('Writing flattened x')
    np.savetxt(os.path.join(head, 'x_flat.txt'), x.flatten(), fmt='%.10e')
    print('Writing y')
    np.savetxt(os.path.join(head, 'y.txt'), y, fmt='%.7e')
    print('Writing price')
    np.savetxt(os.path.join(head, 'price.txt'), price, fmt='%.10e')
    print('Writing x names')
    np.savetxt(os.path.join(head, 'x_names.txt'), x_names, fmt='%s')
    print('Writing b_date')
    np.savetxt(os.path.join(head, 'b_date.txt'), b_date, fmt='%s')
    print('Writing s_date')
    np.savetxt(os.path.join(head, 's_date.txt'), s_date, fmt='%s')
    print('Writing companies')
    np.savetxt(os.path.join(head, 'companies.txt'), companies, fmt='%s')
    print('Writing start weeks')
    np.savetxt(os.path.join(head, 'start_weeks.txt'), start_weeks, fmt='%d')




