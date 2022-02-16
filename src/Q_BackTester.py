import pandas as pd
import numpy as np
import datetime
import os, json
from Scraper import Scraper

import meta

class Q_BackTester:
    '''
    This backtester tests the performance of our model 
    on train_qs of production on the given symbols. 

    Params:
        - Symbols: stock data on which to model
        - Model: String of how to instantiate the model
        - num_weeks: Number of quarters to have the model in production
        - retrain: If retrain is false, then there should be cached data
            with models that are pre-fit and data that is pre-formatted
    '''
    def __init__(self, preprocessor, initial_value, num_qs, mod_op, stacked=False, 
            retrain=True, train_qs=8, end_year=None, train_every=None):

        if end_year is None:
            end_year = datetime.datetime.now().year

        self.end_year = end_year
        self.initial_value = initial_value
        self.current_value = initial_value
        self.x_loc = 0
        self.y_loc = 1
        self.x_names_loc = 2
        self.price_loc = 3
        self.buy_date_loc = 4
        self.sell_date_loc = 5
        self.comp_loc = 6
        self.q_loc = 7
        self.num_qs = num_qs
        self.mod_op = mod_op
        self.retrain = retrain
        self.preprocessor = preprocessor()
        self.stacked = stacked
        self.train_qs = train_qs
        self.snp_data = Scraper().get_ticker_data('^GSPC')[['Date', 'Close']].groupby('Date').first().reset_index()
        self.snp_data['Date'] = pd.to_datetime(self.snp_data['Date']).dt.date
        self.train_every = train_every

        if retrain:
            self.formatted = self.preprocessor.produce_ind_and_response(self.train_qs, end_year=self.end_year)
            self.save_formatted()
        else:
            self.formatted = self.get_formatted()

    '''
    Actually runs the backtesting for all the given parameters for the class initiation
    '''
    def backtest(self):
        # Create Segments for every symbol
        symbol_dic = {}
        num_segments = 0
        value_dfs = []
        trade_dfs = []
        self.end_date = datetime.datetime(self.end_year, 12, 31)

        all_sell = pd.to_datetime(self.formatted[self.sell_date_loc])
        x = self.formatted[self.x_loc][all_sell < self.end_date]
        y = self.formatted[self.y_loc][all_sell < self.end_date]
        price = self.formatted[self.price_loc][all_sell < self.end_date]
        buy_time = self.formatted[self.buy_date_loc][all_sell < self.end_date]
        sell_time = all_sell[all_sell < self.end_date]
        companies = self.formatted[self.comp_loc][all_sell < self.end_date]
        start_qs = self.formatted[self.q_loc][all_sell < self.end_date]

        test_start = start_qs.max() - self.num_qs

        x_train = x[start_qs < test_start]
        y_train = y[start_qs < test_start]
        segments = self.get_test_segments(x, y, price, buy_time, sell_time, companies, start_qs, test_start)

        # Iterate through Segments
        for i in range(len(segments)):

            x_in, y_in = self.mod_op.preprocess(x_train, y_train)
            self.model = self.mod_op.instantiate(x_in, y_in)
            print('Fitting model', 'segement', i + 1, 'of', len(segments))
            self.model = self.mod_op.fit(self.model, x_in, y_in, self.retrain, i) # Model class for quarters is different

            x_test, y_test = self.mod_op.process_test(segments[i]['x'], segments[i]['y'])
            pred = self.model.predict(x_test)
            ## This is for classification: Assuming highest class is the best
            if len(pred.shape) == 2:
                pred = pred[:, -1]

            basic_dex = np.arange(len(pred))
            buy_test = segments[i]['buy']
            sell_test = segments[i]['sell']
            price_test = segments[i]['price']
            company_test = segments[i]['Company']
            start_q_test = segments[i]['start_q']

            predictions = pd.Series(pred, name='pred', index=basic_dex)
            real_change = pd.Series(y_test, name='act', index=basic_dex)
            buy_date = pd.Series(buy_test, name='buy_date', index=basic_dex)
            sell_date = pd.Series(sell_test, name='sell_date', index=basic_dex)
            prices = pd.Series(price_test, name='price', index=basic_dex)
            companies = pd.Series(company_test, name='Company', index=basic_dex)
            start_q = pd.Series(start_q_test, name='start_q', index=basic_dex)

            prediction_df = pd.concat([prices, predictions, real_change, buy_date, sell_date, companies, start_q], axis=1)

            x_train = np.concatenate((x_train, segments[i]['x']), axis=0)
            y_train = np.concatenate((y_train, segments[i]['y']), axis=0)

            trade_df, value_df = self.apply_allocation_strat(prediction_df)
            trade_dfs.append(trade_df)
            value_dfs.append(value_df)

        value_df = pd.concat(value_dfs, axis=0)
        trade_df = pd.concat(trade_dfs, axis=0)

        value_df.to_csv(meta.backtest_loc + 'value.csv', index=False)
        trade_df.to_csv(meta.backtest_loc + 'trade.csv', index=False)
        print('Number of Quarters', self.num_qs)
        print('Initial value', self.initial_value)
        print('Final Value', self.current_value)
        print('Percent Change', self.current_value / self.initial_value)
        return (self.current_value / self.initial_value)

    '''
    We first have to limit the realm of predictions to those that 
    we have enough capitol to purchase. Then we will apply the following set of operations:
        1. For the largest prediction we can buy, buy shares so that the sum of shares we buy
           does not exceed 10% of our portfolio's current_value.
        2. Continue to buy stocks in this way until: 
            1. There are no longer stocks were we predict a positive result.
            2. We don't have enough capital to purchase more stocks
    '''
    def apply_allocation_strat(self, prediction_df):
        # Get all the symbols we are predicting on
        prediction_df['Date'] = pd.to_datetime(prediction_df['buy_date']).dt.date
        prediction_df = pd.merge(prediction_df, self.snp_data, how='left', on='Date')
        prediction_df['Close'] = self.fill_col(prediction_df['Close'])
        close_col = prediction_df.columns.get_loc('Close')
        trade_history = [] 
        trade_history_cols = ['ticker', 'buy_date', 'sell_date', 'num_shares', 'buy_price', 'sell_price', 
                'week_rank', 'position', 'predicted_change', 'actual_change']
        value_history = []
        value_history.append((prediction_df['buy_date'].min(), self.current_value))
        value_history_cols = ['date', 'value']

        unique_starts = sorted(list(prediction_df['start_q'].unique()))
        prediction_df = prediction_df.set_index('start_q')
        for q in unique_starts: 
            candidates = prediction_df.loc[q].copy()
            candidates = candidates.loc[(candidates['pred'] > self.mod_op.buy_cut) | (candidates['pred'] < self.mod_op.sell_cut)]
            candidates = candidates.sort_values('pred', key=lambda x: -1 * abs(1 - x))
            cur_ceiling = 0.1 * self.current_value
            value_left = self.current_value

            shorts = {}
            buys = {}
            rank = 0
            for i, row in candidates.iterrows():
                act_ceiling = min(cur_ceiling, value_left)
                num_shares = act_ceiling // row['price']
                value_in = num_shares * row['price']
                position = ''
                if row['pred'] > self.mod_op.buy_cut:
                    buys[row['Company']] = value_in
                    position = 'buy'
                else:
                    shorts[row['Company']] = value_in
                    position = 'short'
                value_left -= value_in
                trade_history.append((row['Company'], row['buy_date'], row['sell_date'], num_shares, row['price'],
                    row['price'] * row['act'], rank, position, row['pred'], row['act']))
                if value_left == 0:
                    break
                rank += 1

            for symbol, invested in buys.items():
                value_left += invested * row['act']

            for symbol, invested in shorts.items():
                value_left += (invested + invested * (1 - row['act']))

            self.current_value = value_left
            value_history.append((row['sell_date'], self.current_value))

        trade_df = pd.DataFrame(trade_history, columns=trade_history_cols) 
        value_df = pd.DataFrame(value_history, columns=value_history_cols) 
        return trade_df, value_df


    '''
    Segments the testing data into multiple week chunks according to train_every
    Train_every is in terms of quarters
    Input data can have multiple rows for the same quarter
    '''
    def get_test_segments(self, x, y, price, buy, sell, companies, start_qs, test_start):
        jump = len(x) if self.train_every is None else self.train_every
        segment_indices = np.arange(test_start, start_qs.max() + jump, jump) 
        segments = []
        for i in range(len(segment_indices) - 1):
            segment_locs = (start_qs >= segment_indices[i]) & (start_qs < segment_indices[i+1])
            segments.append({
                'x': x[segment_locs],
                'y': y[segment_locs],
                'price': price[segment_locs],
                'buy': buy[segment_locs],
                'sell': sell[segment_locs],
                'Company': companies[segment_locs],
                'start_q': start_qs[segment_locs]
            })
        return segments


    '''
    Imputes a series by using the previous value if it exists and using the next value if the previous value
    Doesn't exist
    '''
    def fill_col(self, col):
        prev = -1
        new_col = col.copy()

        for i, item in col.iteritems():
            if not pd.isna(item):
                prev = item
            elif prev >= 0:
                new_col.loc[i] = prev

        prev = -1
        for i, item in col.iloc[::-1].iteritems():
            if not pd.isna(item):
                prev = item
            elif prev >= 0:
                new_col.loc[i] = prev
        return new_col

    '''
    Returns cached formatted input data
    '''
    def get_formatted(self):
        self.x_loc = 0
        self.y_loc = 1
        self.x_names_loc = 2
        self.price_loc = 3
        self.buy_date_loc = 4
        self.sell_date_loc = 5
        self.comp_loc = 6
        self.q_loc = 7
        
        save_loc = os.join('..', 'data_files', 'backtest_data')
        with open(os.join(save_loc, 'x_names'), 'r') as fin:
            x_names = json.load(fin)

        try:
            x = np.loadtxt(os.join(save_loc, 'x'))
            y = np.loadtxt(os.join(save_loc, 'y'))
            price = np.loadtxt(os.join(save_loc, 'price'))
            buy_d = np.loadtxt(os.join(save_loc, 'buy_d'))
            sell_d = np.loadtxt(os.join(save_loc, 'sell_d'))
            comp = np.loadtxt(os.join(save_loc, 'comp'))
            q = np.loadtxt(os.join(save_loc, 'quarter'))
        except IOError as e:
            print('Retrain files do not exist')
            exit()

        return x, y, x_names, price, buy_d, sell_d, comp, q

    '''
    Saves formatted input data
    '''
    def save_formatted(self):
        save_loc = os.join('..', 'data_files', 'backtest_data')
        with open(os.join(save_loc, 'x_names'), 'w') as fout:
            fout.write(json.dumps(self.formatted[self.x_names_loc]))

        np.savetxt(os.join(save_loc, 'x'), self.formatted[self.x_loc])
        np.savetxt(os.join(save_loc, 'y'), self.formatted[self.y_loc])
        np.savetxt(os.join(save_loc, 'price'), self.formatted[self.price_loc])
        np.savetxt(os.join(save_loc, 'buy_d'), self.formatted[self.buy_date_loc])
        np.savetxt(os.join(save_loc, 'sell_d'), self.formatted[self.sell_date_loc])
        np.savetxt(os.join(save_loc, 'comp'), self.formatted[self.comp_loc])
        np.savetxt(os.join(save_loc, 'quarter'), self.formatted[self.q_loc])




