import pandas as pd
import numpy as np
import datetime
import os, json
from Scraper import Scraper

import meta

class W_BackTester:
    '''
    This backtester tests the performance of our model 
    on train_qs of production on the given symbols. 

    Params:
        - preprocessor: Weekly_Preprocessor for almost every case
        - strategy: strategy object to implement the trading for this backtester
        - initial_value: initial value of the portfolio
        - train_qs: number of weeks to train for 
        - start_year: year to start trading: for pretrained model should be > 2011
        - end_year: year to stop trading
    '''
    def __init__(self, preprocessor, strategies, model, start_year, end_year, train_weeks=40):
        self.start_year = start_year
        self.end_year = end_year
        self.model = model
        self.strategies = strategies
        self.preprocessor = preprocessor(train_weeks, start_year, end_year)
        self.train_weeks = train_weeks

        self.num_predictions = 0
        self.squared_error = 0
        self.hl_acc = 0

    '''
    Actually runs the backtesting for all the given parameters for the class initiation
    '''
    def backtest(self):
        result = self.preprocessor.get_next_week()
        while result != None:
            x, y, x_names, prices, companies, buy_date, sell_date, week_num = result
            pred = self.model.predict(x)[:, 0]
            self.num_predictions += len(pred)
            se = ((pred - y) ** 2).sum()
            hl = (((pred > 1) & (y > 1)) | ((pred < 1) & (y < 1))).sum()
            self.squared_error += se
            self.hl_acc += hl
            print('Trading on', buy_date, 'MSE:', '%07.04f' % (se / len(pred)), 'HL:', '%07.04f' % (hl / len(pred)))

            basic_dex = np.arange(len(pred))
            predictions = pd.Series(pred, name='pred', index=basic_dex)
            real_change = pd.Series(y, name='act', index=basic_dex)
            prices = pd.Series(prices, name='price', index=basic_dex)
            companies = pd.Series(companies, name='Company', index=basic_dex)

            prediction_df = pd.concat([prices, predictions, real_change, companies], axis=1)
            for strat in self.strategies:
                strat.apply_allocation_strat(prediction_df, buy_date, sell_date, week_num)
            result = self.preprocessor.get_next_week()
        for strat in self.strategies:
            strat.cleanup()
        print('Total MSE', "%07.04f" % (self.squared_error / self.num_predictions))
        print('Total_HL', "%07.04f" % (self.hl_acc / self.num_predictions))



