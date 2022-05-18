import sys, os
import re

import numpy as np
import pandas as pd
import datetime

DATADIR = os.path.join('..', 'data_files')

# Columns to use for technical analysis
USE_TECH = [
    'Close',
    'Volume'
]

# Columns to use for fundamental analysis
USE_FUND = [
    'chsoq',
    'norm_rev',
    'epspxq',
    'epspiq',
    'norm_cheq',
    'norm_dlttq',
    'norm_ltq',
    'norm_oiadpq',
    'dvpspq'
]

# Columns to use for interest rate analysis
USE_INT = [
    'FF_O',
    'SL_Y20',
    'MORTG_NA',
    'PRIME_NA',
    'AAA_NA',
    'BAA_NA',
    'TCMNOM_Y1',
    'TCMNOM_Y3',
    'TCMNOM_Y5',
    'TCMNOM_Y7',
    'TCMNOM_Y10',
    'ED_M1',
    'ED_M3',
    'ED_M6',
    'TB_M3',
    'TB_M6',
]

# Width to use to create examples
X_WIDTH = max(len(USE_TECH), len(USE_FUND), len(USE_INT))

class MergedPreprocessor:
    '''
    Preprocesses and normalizes technical, fundamental, and
    interest rate data for combined modeling with a time
    series model.
    
    Args:
        start_year (int): Represents which year we should
            start prediction. 
        end_year (int): Represents which year we should end
            prediction.
        n_tech (int): Number of weeks of technical data
            to use for each prediction
        n_fund (int): Number of quarters of fundamental data
            to use for each prediction
        n_int (int): Number of months of interest rates data
            to use for each prediction
        m_advance (int): Number of months to estimate in advance
    '''

    def __init__(
            self,
            start_year,
            end_year,
            n_tech,
            n_fund,
            n_int,
            m_advance):

        self.start_year = start_year
        self.end_year = end_year
        self.n_tech = n_tech
        self.n_fund = n_fund
        self.n_int = n_int
        self.m_advance = m_advance

        self.norm_params = {}

        self.w_tech, self.m_tech, self.m_dates = self.get_technical()
        self.fund = self.get_fundamental()
        self.int = self.get_interest()

        self.w_tech_comp_loc = self.w_tech.columns.get_loc('Company')
        self.fund_comp_loc = self.fund.columns.get_loc('tic')

        self.w_tech_use_locs = self.w_tech.columns.get_indexer(USE_TECH)
        self.fund_use_locs = self.fund.columns.get_indexer(USE_FUND)
        self.int_use_locs = self.int.columns.get_indexer(USE_INT)

        self.normalize_df('tech')
        self.normalize_df('fund')
        self.normalize_df('int')

        self.cur = 0
        self.adjust_cur = self.set_adjust_cur()


    def next(self, use_cur=None):
        '''
        Get the input and output examples alligned with use_cur. This will
        use self.cur unless specified. the m_df which defines the granularity
        of examples will be able to create examples from index cur
        '''
        xs = []
        ys = []
        prices = []
        comp_names = []
        buy_date = []
        sell_date = []

        if use_cur is None:
            use_cur = self.cur + self.adjust_cur
            self.cur += 1

        date = self.m_dates.iloc[use_cur]
        try:
            companies = self.m_tech.loc[date, ['Company', 'Close', 'Sell_Price', 'Sell_Date']]
        except KeyError:
            return None

        # Retrieve Interest Rates
        int_loc = self.int['date'].searchsorted(date, side='right')
        if int_loc >= self.n_int:
            int_values = self.int.iloc[int_loc - self.n_int:int_loc, self.int_use_locs].values
            shape_diff = X_WIDTH - int_values.shape[1]
            if shape_diff > 0:
                int_values_pad = np.full((int_values.shape[0], shape_diff), 0.0)
                int_values = np.concatenate([int_values, int_values_pad], axis=1)
        else:
            return None

        # Retrieve data for each company
        for i, row in companies.iterrows():
            search_val = row['Company'] + str(i)

            # Retrieve company technicals
            tech_loc = self.w_tech['searchcol'].searchsorted(search_val, side='right')
            if (tech_loc >= self.n_tech) and \
                    (self.w_tech.iloc[tech_loc - self.n_tech, self.w_tech_comp_loc] == row['Company']):
                week_values = \
                        self.w_tech.iloc[tech_loc - self.n_tech:tech_loc, self.w_tech_use_locs].values
                shape_diff = X_WIDTH - week_values.shape[1]
                if shape_diff > 0: 
                    week_values_pad = np.full((week_values.shape[0], shape_diff), 0.0)
                    week_values = np.concatenate([week_values, week_values_pad], axis=1)
            else:
                continue

            # Retrieve company fundamentals
            fund_loc = self.fund['searchcol'].searchsorted(search_val, side='right')
            if (fund_loc >= self.n_fund) and \
                    (self.fund.iloc[fund_loc - self.n_fund, self.fund_comp_loc] == row['Company']):
                fund_values = \
                        self.fund.iloc[fund_loc - self.n_fund:fund_loc, self.fund_use_locs].values
                shape_diff = X_WIDTH - fund_values.shape[1]
                if shape_diff > 0: 
                    fund_values_pad = np.full((fund_values.shape[0], shape_diff), 0.0)
                    fund_values = np.concatenate([fund_values, fund_values_pad], axis=1)
            else:
                continue

            # Add to examples
            xs.append(np.concatenate([int_values, week_values, fund_values]))
            prices.append(row['Close'])
            comp_names.append(row['Company'])
            ys.append(row['Sell_Price'])

        buy_date = companies.index[0]
        sell_date = companies.iloc[0, companies.columns.get_loc('Sell_Date')]
        
        if (len(xs) == 0) or (len(ys) == 0):
            return None
        return np.array(xs), np.array(ys), [], np.array(prices), comp_names, buy_date, sell_date, self.cur - 1


    def set_adjust_cur(self):
        '''
        Calls next until a sensible result is achieved. Once it is,
        return the offset that allowed the sensible result
        '''
        offset = 0
        while self.next(offset) is None:
            offset += 1
        return offset


    def normalize_df(self, normalize_type):
        '''
        We won't do anything spectacular here. We just impute each
        value with its nearest neighbor in the time series, and 
        we divide by the mean and standard deviation of the column. If
        the normal parameters do not have the mean and std for the column, 
        add them.

        We also need to do some housekeeping. There are some columns we do not
        want to normalize. We need to save everything to the right place
        '''
        if normalize_type == 'tech':
            include = USE_TECH
            df = self.w_tech
        elif normalize_type == 'fund':
            include = USE_FUND
            df = self.fund
        elif normalize_type == 'int':
            include = USE_INT
            df = self.int
        else:
            raise ValueError('Illegal Argument to normalize_df: ' + normalize_type)

        if not normalize_type in self.norm_params:
            self.norm_params[normalize_type] = {}
        for col in df.columns:
            if col not in include:
                continue
            if col not in self.norm_params[normalize_type]:
                self.norm_params[normalize_type][col] = {}
            df[col] = impute_col(df[col]) 
            if not 'mean' in self.norm_params[normalize_type][col]:
                self.norm_params[normalize_type][col]['mean'] = df[col].mean()
            if not 'std' in self.norm_params[normalize_type][col]:
                self.norm_params[normalize_type][col]['std'] = df[col].std()
            df[col] -= self.norm_params[normalize_type][col]['mean']
            df[col] /= self.norm_params[normalize_type][col]['std']


    def get_technical(self):
        '''
        Retrieves technical data and returns a sorted dataframe
        on the company, then on the date. Ensures that there are
        no missing combinations of company and date by padding
        missing dates.

        For now we're just looking at every Friday
        '''
        print('Loading Technical Data...')
        tech_dir = os.path.join(DATADIR, 'stock_data')
        weekly_dfs = []
        monthly_dfs = []

        # Find all of the fridays since 1970
        business_days = pd.date_range('1960-01-01', str(datetime.datetime.now().date()), freq='B')
        fridays = business_days[business_days.strftime('%A') == 'Friday'].date.copy()

        # Find the last business_days days of the month since 1970
        month_days = pd.Series(business_days).groupby(business_days.strftime('%Y%m')).max().dt.date
        month_days.name = 'Date'
        month_purch = month_days.shift(-1 * self.m_advance)
        month_purch.name = 'Sell'
        month_data = pd.concat((month_days, month_purch), axis=1)

        for fname in os.listdir(tech_dir):
            ticker_match = re.search(r'(.*?).csv', fname)
            if not ticker_match:
                continue
            ticker_df = pd.read_csv(os.path.join(tech_dir, fname))[['Date', 'Close', 'Volume']]
            ticker_df['Date'] = pd.to_datetime(ticker_df['Date'])
            ticker_df = ticker_df.loc[(ticker_df['Date'].dt.year >= self.start_year) & (ticker_df['Date'].dt.year <= self.end_year)].copy()
            if len(ticker_df) == 0:
                continue
            ticker_df['Date'] = ticker_df['Date'].dt.date
            min_date = ticker_df['Date'].min()
            max_date = ticker_df['Date'].max()
            friday_range = fridays[(fridays >= min_date) & (fridays <= max_date)]
            month_range = month_days[(month_days >= min_date) & (month_days <= max_date)]
            weekly_df = ticker_df.set_index('Date').reindex(friday_range, method='nearest').reset_index()
            weekly_df['Company'] = ticker_match.groups()[0]
            monthly_df = ticker_df.set_index('Date').reindex(month_range, method='nearest').reset_index()
            monthly_df['Company'] = ticker_match.groups()[0]
            weekly_dfs.append(weekly_df)
            monthly_dfs.append(monthly_df)

        weekly_tech = pd.concat(weekly_dfs)
        monthly_tech = pd.concat(monthly_dfs)
        weekly_tech['searchcol'] = get_search_col(weekly_tech, 'Company', 'Date')
        weekly_tech = weekly_tech.sort_values('searchcol')

        month_with_sell = pd.merge(monthly_tech, month_data, on='Date', how='inner')
        to_merge = month_with_sell[['Date', 'Company', 'Close']]
        to_merge.columns = ['Sell_Date', 'Sell_Company', 'Sell_Price']
        month_with_sell_price = pd.merge(
                month_with_sell, 
                to_merge, 
                left_on=['Sell', 'Company'],
                right_on=['Sell_Date', 'Sell_Company'],
                how='right')
        monthly_tech = month_with_sell_price.dropna() \
                        [['Date', 'Company', 'Close', 'Sell_Date', 'Sell_Price']] \
                        .set_index('Date')
        return weekly_tech.reset_index(drop=True), monthly_tech, month_days


    def get_fundamental(self):
        '''
        Retrieves quarterly data and returns a sorted dataframe
        on company, then on date. Ensures that there are no quarters
        missing with padding. 
        '''

        print('Loading Fundamental Data...')
        # Create all combinations of quarter and year for each company
        years = pd.DataFrame({'year': np.arange(1960, datetime.datetime.now().year)})
        quarters = pd.DataFrame({'qtr': np.arange(4) + 1})
        all_qtrs = pd.merge(years, quarters, how='cross')

        qdf = pd.read_csv(os.path.join(DATADIR, 'quarterly_filtered.csv'))
        qdf['datadate'] = pd.to_datetime(qdf['datadate'])
        qdf = qdf.loc[(qdf['datadate'].dt.year >= self.start_year) & (qdf['datadate'].dt.year <= self.end_year)].copy()
        qdf['datadate'] = qdf['datadate'].dt.date
        qdf.sort_values('datadate')
        qdf = qdf.groupby('tic').apply(lambda df: process_company_fundamentals(df, all_qtrs)).reset_index(drop=True)
        qdf['searchcol'] = get_search_col(qdf, 'tic', 'datadate')
        qdf = qdf.sort_values('searchcol')


        # Normalize certain columns by number of shares
        qdf['norm_rev'] = qdf['revtq'] / qdf['cshoq']
        qdf['norm_cheq'] = qdf['cheq'] / qdf['cshoq']
        qdf['norm_dlttq'] = qdf['dlttq'] / qdf['cshoq']
        qdf['norm_ltq'] = qdf['ltq'] / qdf['cshoq']
        qdf['norm_oiadpq'] = qdf['oiadpq'] / qdf['cshoq']
        qdf = qdf.drop(columns=['revtq', 'cheq', 'dlttq', 'ltq', 'oiadpq'], axis=1)

        return qdf.reset_index(drop=True)


    def get_interest(self):
        '''
        Retrieves interest rate data and returns a dataframe sorted
        by date. Ensures there are no months of interest rate data
        missing. Fills missing values with padding. 

        However, there are no missing values to fill so we don't need
        to worry about it
        '''
        print('Loading Interest Data...')
        interest_df = pd.read_csv(os.path.join(DATADIR, 'interest_rate_filtered.csv'))
        interest_df['date'] = pd.to_datetime(interest_df['date'])
        interest_df = interest_df.loc[(interest_df['date'].dt.year >= self.start_year) & (interest_df['date'].dt.year <= self.end_year)].copy()
        interest_df['date'] = interest_df['date'].dt.date
        interest_df = interest_df.sort_values('date')
        return interest_df.reset_index(drop=True)


def process_company_fundamentals(comp_df, all_qtrs):
    '''
    Processes fundamental data for a single company

    Args:
        comp_df: Fundamental dataframe for a single company
        all_qtrs: Dataframe that will later be converted to a
            Multiindex that has year and quarter since 1960
    '''
    comp_df = comp_df.groupby(['fyearq', 'fqtr']).first().reset_index() # No duplicates

    year_col = comp_df.columns.get_loc('fyearq')
    qtr_col = comp_df.columns.get_loc('fqtr')
    min_year = comp_df.iloc[0, year_col]
    min_qtr = comp_df.iloc[0, qtr_col]
    max_year = comp_df.iloc[len(comp_df) - 1, year_col]
    max_qtr = comp_df.iloc[len(comp_df) - 1, qtr_col]

    new_index = all_qtrs.loc[((all_qtrs['year'] > min_year) | \
            ((all_qtrs['year'] == min_year) & (all_qtrs['qtr'] >= min_qtr))) & \
            ((all_qtrs['year'] < max_year) | ((all_qtrs['year'] == max_year) & \
            (all_qtrs['qtr'] <= max_qtr)))]

    new_index_series = new_index['year'] * 4 + new_index['qtr']
    comp_df_index = comp_df['fyearq'] * 4 + comp_df['fqtr']

    comp_df = comp_df.set_index(comp_df_index).reindex(new_index_series, method='nearest')
    return comp_df


def get_search_col(df, tic_col, date_col):
    '''
    Our search structure for preparing the time series data
    will be a string created by prepending the ticker to the
    date

    Args:
        tic_col (str): Name of column with the tickers 
        date_col (str): Name of column with pd.Datetime.date objects

    Returns:
        pd.Series(str): Series representing the search column
    '''
    return df[tic_col] + df[date_col].astype(str)

def impute_col(col):
    '''
    Takes a pd.Series as input and imputes the missing values with the 
    nearest full values
    '''
    new_col = col.dropna()
    return new_col.reindex(col.index, method='nearest')




if __name__ == '__main__':
    mp = MergedPreprocessor(1970, 2001, 1, 1, 1, 1)
    xs = []
    ys = []
    result = mp.next()
    while result:
        x_examples, y_examples, x_names, prices, comp_names, buy_date, sell_date, cur_month = result
        xs.append(x_examples)
        ys.append(y_examples)
        print(mp.cur)
        result = mp.next()
    final_xs = np.concatenate(xs)
    final_ys = np.concatenate(ys)
    print(len(final_xs))
    print(len(final_ys))
    


