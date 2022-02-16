import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

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
    def __init__(self, symbols, preprocessor, model, initial_value, num_weeks):
        self.formatted_dfs = {}
        self.initial_value = initial_value
        self.current_value = initial_value
        self.models = {}
        self.df_loc = 0
        self.res_loc = 1
        self.price_loc = 2
        self.num_weeks = num_weeks
        for symbol in symbols:
            try:
                self.formatted_dfs[symbol] = preprocessor(symbol).produce_ind_and_response()
                self.models[symbol] = eval(model)
            except ValueError as e:
                print(e)
                continue

    '''
    Actually runs the backtesting for all the given parameters for the class
    initiation
    '''
    def backtest(self):
        # Fits all the models
        prediction_dfs = []
        for symbol, tup in self.formatted_dfs.items():
            df = tup[self.df_loc]
            res_col = tup[self.res_loc]
            price_col = tup[self.price_loc]

            if self.num_weeks / len(df) > 0.5:
                print('Not enough data for', symbol)
                continue

            # Split into Train and Test data
            test_start = len(df)-self.num_weeks 
            train_data = df.iloc[:test_start]
            test_data = df.iloc[test_start:]

            # Fit the models
            print('Fitting model', symbol)
            self.fit_model(symbol, train_data, res_col)

            # Predict on the models
            Y = test_data[res_col]
            X = test_data.drop(columns=res_col, axis=1)

            prices = X[price_col].reset_index(drop=True)
            prices.name = symbol + '_price'
            predictions = pd.Series(self.models[symbol].predict(X), name=symbol + '_pred')
            values = Y.reset_index(drop=True)
            values.name = symbol + '_act'
            prediction_dfs.append(pd.concat([prices, predictions, values], axis=1))

        prediction_df = pd.concat(prediction_dfs, axis=1)
        self.apply_allocation_strat(prediction_df)

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

        for i, row in prediction_df.iterrows():
            cur_ceiling = 0.1 * self.current_value
            value_left = self.current_value 
            changes = row[change_cols]
            changes = changes.loc[changes > 1].sort_values(ascending=False)
            # Will contain the stocks we purchased and how much we invested
            holding = {}
            for stock_change, val in changes.iteritems():
                cur_symbol = stock_change.split('_')[0]
                act_ceiling = min(cur_ceiling, value_left)
                value_in = (act_ceiling // row[cur_symbol + '_price']) * row[cur_symbol + '_price']
                holding[cur_symbol] = value_in
                value_left -= value_in

            for symbol, invested in holding.items():
                value_left += invested * row[symbol + '_act']
            self.current_value = value_left
        print('Started with', self.initial_value)
        print('Ended with', self.current_value)
        print('Change of', self.current_value / self.initial_value)
        print('Duration', self.num_weeks, 'weeks')




    '''
    Trains the models on the data available (not in production)
    '''
    def fit_model(self, symbol, train_data, res_col):
        Y = train_data[res_col]
        X = train_data.drop(columns=[res_col], axis=1) 
        self.models[symbol].fit(X, Y)
