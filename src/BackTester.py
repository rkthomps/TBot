import pandas as pd
import numpy as np
import datetime
import os
from Scraper import Scraper

import meta

class BackTester:
    '''
    This backtester tests the performance of our model 
    on num_weeks of production on the given symbols. 

    Params:
        - Symbols: stock data on which to model
        - Model: String of how to instantiate the model
        - num_weeks: Number of weeks to have the model in production
    '''
    def __init__(self, symbols, preprocessor, initial_value, num_weeks, mod_op, stacked=False, 
            rescrape=False, retrain=True, train_weeks=8, end_date=None, bad_flag=False, train_every=None):

        if end_date is None:
            end_date = datetime.datetime.now()

        self.end_date = end_date
        self.formatted_dfs = {}
        self.initial_value = initial_value
        self.current_value = initial_value
        self.models = {}
        self.x_loc = 0
        self.y_loc = 1
        self.x_names_loc = 2
        self.price_loc = 3
        self.buy_date_loc = 4
        self.sell_date_loc = 5
        self.num_weeks = num_weeks
        self.mod_op = mod_op
        self.rescrape = rescrape
        self.retrain = retrain
        self.preprocessor = preprocessor
        self.stacked = stacked
        self.train_weeks = train_weeks
        self.snp_data = Scraper().get_ticker_data('^GSPC')[['Date', 'Close']]
        self.snp_data['Date'] = pd.to_datetime(self.snp_data['Date']).dt.date
        self.bad_flag = bad_flag
        self.train_every = train_every

        for symbol in symbols:
            try:
                print('Processing', symbol)
                tups = self.get_formatted(symbol)
                self.save_formatted(symbol, tups)
                if tups[self.x_loc].shape[0] < (num_weeks * 2):
                    print("Not enough data to backtest", symbol)
                    continue
                self.formatted_dfs[symbol] = tups 
            except ValueError as e:
                print(e)
                continue

    '''
    Tries to get data from a saved location. If rescrape is true, redoes the data gathering    
    '''
    def get_formatted(self, symbol):
        paths = [
            meta.form_loc + symbol + '_x',
            meta.form_loc + symbol + '_y',
            meta.form_loc + symbol + '_x_names',
            meta.form_loc + symbol + '_price',
            meta.form_loc + symbol + '_buy_date',
            meta.form_loc + symbol + '_sell_date',
        ]
        if self.rescrape or not all([os.path.exists(p + '.csv') for p in paths]):
            return self.preprocessor(symbol).produce_ind_and_response(stacked=self.stacked, n_weeks=self.train_weeks)
        ret = []
        for p in paths:
            res = pd.read_csv(p + '.csv')
            if p.endswith('date'):
                res.iloc[:, 0] = pd.to_datetime(res.iloc[:, 0], format='%Y-%m-%d')
            vals = res.values
            if vals.shape[1] == 1:
                vals = vals[:, 0]
            ret.append(vals)
        return ret
            

    '''
    Saves the formatted data for a particular stock
    '''
    def save_formatted(self, symbol, symbol_tup):
        paths = [
            meta.form_loc + symbol + '_x',
            meta.form_loc + symbol + '_y',
            meta.form_loc + symbol + '_x_names',
            meta.form_loc + symbol + '_price',
            meta.form_loc + symbol + '_buy_date',
            meta.form_loc + symbol + '_sell_date',
        ]
        if all([os.path.exists(p) for p in paths]):
            return
        for i, arr in enumerate(symbol_tup):
            to_save = arr if len(arr.shape) > 1 else arr[:, np.newaxis]
            pd.DataFrame(to_save).to_csv(paths[i] + '.csv', index=False)

    '''
    Actually runs the backtesting for all the given parameters for the class
    initiation
    '''
    def backtest(self):
        # Create Segments for every symbol
        symbol_dic = {}
        num_segments = 0
        value_dfs = []
        trade_dfs = []

        for symbol, tup in self.formatted_dfs.items():
            try:
                all_sell = pd.to_datetime(tup[self.sell_date_loc])
                x = tup[self.x_loc][all_sell < self.end_date]
                y = tup[self.y_loc][all_sell < self.end_date]
                price = tup[self.price_loc][all_sell < self.end_date]
                buy_time = tup[self.buy_date_loc][all_sell < self.end_date]
                sell_time = all_sell[all_sell < self.end_date]
            except IndexError as e:
                print(e)
                continue

            if self.num_weeks * 2 > x.shape[0]:
                print('Not enough data for', symbol)
                continue

            # Split into Train and Test data
            symbol_dic[symbol] = {}
            test_start = x.shape[0] - self.num_weeks
            symbol_dic[symbol]['x_train'] = x[:test_start]
            symbol_dic[symbol]['y_train'] = y[:test_start]
            symbol_dic[symbol]['segments'] = self.get_test_segments(x, y, price, buy_time, sell_time, test_start)
            num_segments = len(symbol_dic[symbol]['segments'])

        # Iterate through Segments
        for i in range(num_segments):
            prediction_dfs = []
            for symbol, data_dic in symbol_dic.items():
                x_in, y_in = self.mod_op.preprocess(data_dic['x_train'], data_dic['y_train'])
                self.models[symbol] = self.mod_op.instantiate(x_in, y_in)
                print('Fitting model', symbol, 'segement', i + 1, 'of', num_segments)
                self.models[symbol] = self.mod_op.fit(symbol, self.models[symbol], x_in, y_in, self.retrain)

                x_test, y_test = self.mod_op.process_test(data_dic['segments'][i]['x'], data_dic['segments'][i]['y'])
                pred = self.models[symbol].predict(x_test)
                if len(pred.shape) == 2:
                    pred = pred[:, -1]
                basic_dex = np.arange(len(pred))
                buy_test = data_dic['segments'][i]['buy']
                sell_test = data_dic['segments'][i]['sell']
                price_test = data_dic['segments'][i]['price']

                predictions = pd.Series(pred, name=symbol + '_pred', index=basic_dex)
                real_change = pd.Series(y_test, name=symbol + '_act', index=basic_dex)
                buy_date = pd.Series(buy_test, name=symbol + '_buy_date', index=basic_dex)
                sell_date = pd.Series(sell_test, name=symbol + '_sell_date', index=basic_dex)
                prices = pd.Series(price_test, name=symbol +'_price', index=basic_dex)
                prediction_dfs.append(pd.concat(
                    [prices, predictions, real_change, buy_date, sell_date], axis=1))

                symbol_dic[symbol]['x_train'] = np.concatenate((data_dic['x_train'], data_dic['segments'][i]['x']), axis=0)
                symbol_dic[symbol]['y_train'] = np.concatenate((data_dic['y_train'], data_dic['segments'][i]['y']), axis=0)

            prediction_df = pd.concat(prediction_dfs, axis=1)
            trade_df, value_df = self.apply_allocation_strat(prediction_df)
            trade_dfs.append(trade_df)
            value_dfs.append(value_df)

        value_df = pd.concat(value_dfs, axis=0)
        trade_df = pd.concat(trade_dfs, axis=0)

        value_df.to_csv(meta.backtest_loc + 'value.csv', index=False)
        trade_df.to_csv(meta.backtest_loc + 'trade.csv', index=False)
        print('Number of Weeks', self.num_weeks)
        print('Initial value', self.initial_value)
        print('Final Value', self.current_value)
        print('Percent Change', self.current_value / self.initial_value)
        return (self.current_value / self.initial_value), list(self.formatted_dfs.keys())

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
        change_cols = [c for c in prediction_df.columns if c.endswith('pred')]
        sell_date_cols = [c for c in prediction_df.columns if c.endswith('sell_date')]
        buy_date_cols = [c for c in prediction_df.columns if c.endswith('buy_date')]
        prediction_df['Date'] = pd.to_datetime(prediction_df[buy_date_cols[0]]).dt.date
        prediction_df = pd.merge(prediction_df, self.snp_data, how='left', on='Date')
        prediction_df['Close'] = self.fill_col(prediction_df['Close'])
        close_col = prediction_df.columns.get_loc('Close')
        trade_history = [] 
        trade_history_cols = ['ticker', 'buy_date', 'sell_date', 'num_shares', 'buy_price', 'sell_price', 
                'week_rank', 'position', 'predicted_change', 'actual_change']
        value_history = []
        value_history_cols = ['date', 'value']
        cur_row = 0

        for i, row in prediction_df.iterrows():

            ## I should do bad snp detection here 
            if self.bad_flag:
                compare_to = max(cur_row - 10, 0)
                if prediction_df.iloc[cur_row, close_col] < prediction_df.iloc[compare_to, close_col]:
                    value_history.append((row[sell_date_cols[0]], self.current_value))
                    cur_row += 1
                    continue
            
            cur_ceiling = 0.1 * self.current_value
            value_left = self.current_value 
            changes = row.loc[change_cols]
            changes = changes.loc[(changes > self.mod_op.buy_cut) | (changes < self.mod_op.short_cut)].sort_values(key=lambda x: -1 * abs(1 - x))
            print(changes)
            # Will contain the stocks we purchased and how much we invested
            shorts = {}
            buys = {}
            rank = 0
            for stock_change, val in changes.iteritems():
                cur_symbol = stock_change.split('_')[0]
                act_ceiling = min(cur_ceiling, value_left)
                num_shares = act_ceiling // row[cur_symbol + '_price']
                value_in = num_shares * row[cur_symbol + '_price']
                position = ''
                if val > self.mod_op.buy_cut:
                    buys[cur_symbol] = value_in
                    position = 'buy'
                else:
                    shorts[cur_symbol] = value_in
                    position = 'short'
                value_left -= value_in
                trade_history.append((
                    cur_symbol, row[cur_symbol + '_buy_date'], row[cur_symbol + '_sell_date'],
                    num_shares, row[cur_symbol + '_price'], row[cur_symbol + '_price'] * row[cur_symbol + '_act'],
                    rank, position, row[cur_symbol + '_pred'], row[cur_symbol + '_act']))
                rank += 1

            for symbol, invested in buys.items():
                value_left += invested * row[symbol + '_act']

            for symbol, invested in shorts.items():
                value_left += (invested + invested * (1 - row[symbol + '_act']))

            self.current_value = value_left
            value_history.append((row[sell_date_cols[0]], self.current_value))
            cur_row += 1

        trade_df = pd.DataFrame(trade_history, columns=trade_history_cols) 
        value_df = pd.DataFrame(value_history, columns=value_history_cols) 
        return trade_df, value_df
        


    '''
    Segments the testing data into multiple week chunks according to train_every
    Returns a dictionary of 
    '''
    def get_test_segments(self, x, y, price, buy, sell, test_start):
        jump = len(x) if self.train_every is None else self.train_every
        segment_indices = np.arange(test_start, len(x) + jump, jump) 
        segments = []
        for i in range(len(segment_indices) - 1):
            segments.append({
                'x': x[segment_indices[i]:segment_indices[i+1]], 
                'y': y[segment_indices[i]:segment_indices[i+1]], 
                'price': price[segment_indices[i]:segment_indices[i+1]], 
                'buy': buy[segment_indices[i]:segment_indices[i+1]], 
                'sell': sell[segment_indices[i]:segment_indices[i+1]]
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



