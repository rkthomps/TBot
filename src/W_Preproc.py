

import json, glob, re, os
import numpy as np
import pandas as pd
import datetime


class Weekly_Preprocessor:
    # Scrape is either local or 'yf' implying the data is from the yf api
    def __init__(self, n_weeks, start_year, end_year, scrape='local', binary=False):
        self.start_year = start_year
        self.end_year = end_year
        self.n_weeks = n_weeks
        self.binary=binary
        
        ## For streaming in data
        self.cur_week = 1

        ## Paths for data
        self.sector_loc = '../data_files/sectors.csv'
        self.tech_loc = '../data_files/stock_data/'

        ## Load to dataframes
        self.load_data(scrape)
        
        ## Merge the financials, technicals, and interest rates 
        self.merge_data()

        ## Bin Examples into monday-aligned week indices
        self.bin_data()

    '''
    Get the input and output examples where the first week of training
    data is aligned with self.cur_week. If there is not enough data
    to return training examples for self.cur_week, return None
    '''
    def get_next_week(self):
        ys = []
        xs = []
        prices = []
        companies = []
        buy_date = None
        sell_date = None
        dis = [] # test correctness

        #Norm cols are columns that will be normalized
        norm_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        not_x = ['Company', 'Date', 'wi', 'di', 'Dividends', 'Stock Splits']
        not_x.extend(norm_cols)
        desired_cols = list(self.all_df.columns)
        ### TODO: NORMALIZE ONLY NORMALIZE COLS (DESIRED COLS BUT CHANGE THE NAME)
        for r in not_x:
            desired_cols.remove(r)

        # For indexing
        col_indices = self.all_df.columns.get_indexer(desired_cols)
        norm_indices = self.all_df.columns.get_indexer(norm_cols)
        close_loc = self.all_df.columns.get_loc('Close')
        di_loc = self.all_df.columns.get_loc('di')
        date_loc = self.all_df.columns.get_loc('Date')
        wi_loc = self.all_df.columns.get_loc('wi')

        max_wi = self.all_df['wi'].max() - 1 # Subtract one to ensure a full week for the max week
        if max_wi < (self.cur_week + self.n_weeks):
            self.cur_week = 1 # Reset the current week in case streaming should continue (training)
            return None
        
        cur_df = self.all_df.loc[(self.all_df['wi'] >= self.cur_week) & (self.all_df['wi'] <= (self.cur_week + self.n_weeks))]
        candidates = cur_df['Company'].value_counts()
        candidates = candidates[candidates == 5 * (self.n_weeks + 1)]
        
        for candidate in candidates.index.values:
            c_data = cur_df.loc[cur_df['Company'] == candidate].sort_values('Date')
            norm_x = c_data.iloc[:(5 * self.n_weeks), norm_indices].values
            other_x = c_data.iloc[:(5 * self.n_weeks), col_indices].values
            std = norm_x.std(axis=0)
            if (std == 0).any() or (np.isnan(std).any()):
                continue
            norm_x = (norm_x - norm_x.mean(axis=0)) / std 
            xs.append(np.concatenate((norm_x, other_x), axis=1)) # Normalize within the group
            dis.append(c_data.iloc[:(5 * self.n_weeks), di_loc].values)
            target_close = c_data.iloc[5 * (self.n_weeks + 1) - 1, close_loc]
            last_close = c_data.iloc[5 * self.n_weeks - 1, close_loc]
            changes = target_close / last_close
            if not self.binary:
                ys.append(changes)
            else:
                ys.append((changes > 1).astype(float))
            prices.append(last_close)
            companies.append(candidate)
            buy_date = c_data.iloc[5 * self.n_weeks - 1, date_loc]
            sell_date = c_data.iloc[5 * (self.n_weeks + 1) - 1, date_loc]
        self.cur_week += 1

        return np.array(xs), np.array(ys), norm_cols + desired_cols, np.array(prices), np.array(companies), buy_date, sell_date, self.cur_week - 1

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
    Bin the data into monday-aligned weekly binned. This allows us to have clean data as well
    as identify which data we can trade on every week
    '''
    def bin_data(self):
        min_date = self.all_df['Date'].min().date()
        max_date = self.all_df['Date'].max().date()
        first_monday = self.all_df.loc[self.all_df['Date'].dt.strftime('%a') == 'Mon']['Date'].min().date()
        mondays = pd.date_range(first_monday, max_date + pd.Timedelta(7, 'd'), freq='7D')
        self.all_df['wi'] = pd.cut(self.all_df['Date'], mondays, right=False, labels=np.arange(len(mondays) - 1))
        self.all_df['di'] = self.all_df['Date'].dt.strftime('%w')

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
    def load_data(self, scrape):
        self.sectors = pd.read_csv(self.sector_loc)
        self.sectors = self.sectors.groupby('company').first().reset_index()
        self.sectors = self.sectors.rename(columns={'company': 'Company'}) 

        if scrape == 'local':
            # Get the technicals data
            stocks = []
            for path in glob.glob(self.tech_loc + '*.csv'):
                head, tail = os.path.split(path)
                name = tail.split('.')[0]
                df = pd.read_csv(path)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.loc[(df['Date'].dt.year >= self.start_year) & (df['Date'].dt.year <= self.end_year)]
                if len(df) == 0:
                    continue
                min_date = df['Date'].min()
                max_date = df['Date'].max()
                # Makes sure the dataframe has no missing value. Fills in holidays with forward padding
                df = df.set_index('Date').reindex(pd.date_range(min_date, max_date, freq='B', name='Date'), method='pad').reset_index()
                df['Company'] = name
                stocks.append(df)
        else:
            from Company_Lister import Company_Lister as CL
            import yfinance as yf
            start = datetime.datetime(self.start_year, 1, 1)
            end = datetime.datetime(self.end_year, 12, 31)

            comps = CL().get_snp()
            stocks = []
            for comp in comps:
                print('Getting', comp)
                t = yf.Ticker(comp)
                df = t.history(start=start, end=end, interval='1d').reset_index()
                if len(df) == 0:
                    continue
                df = df.set_index('Date').reindex(pd.date_range(start, end, freq='B', name='Date'), method='pad').reset_index()
                df['Company'] = comp
                stocks.append(df)
        self.stock_df = pd.concat(stocks)
        months = pd.get_dummies(self.stock_df['Date'].dt.strftime('%B'))
        self.stock_df = pd.concat((self.stock_df, months), axis=1)
        self.stock_df = self.stock_df.groupby(['Company', 'Date']).first().reset_index()

                



if __name__ == '__main__':
    wp = Weekly_Preprocessor(40, 2001, 2009)
    result = wp.get_next_week()
    cur_week = 0
    print("I'm running for no reason")
    while result != None:
        print(cur_week)
        cur_week += 1
        result = wp.get_next_week()
    #x, y, price, x_names, b_date, s_date, companies, start_weeks = Weekly_Preprocessor().produce_ind_and_response(40)
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




