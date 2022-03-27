import pandas as pd
import numpy as np
import os


# This class will allow for testing of many different allocation strategies
class Strategy:
    '''
    initital value: starting value for the portfolio
    buy_cut: minimum prediction to buy a security
    sell_cut: maximum prediction to short a security
    max_invest: proportion of portfolio algorithm is allowed to invest into
        one security
    our: Directory to write results
    '''
    def __init__(self, initial_value, buy_cut, sell_cut, max_invest, out=None):
        self.initial_value = initial_value
        self.current_value = initial_value
        self.buy_cut = buy_cut
        self.sell_cut = sell_cut
        self.max_invest = max_invest

        if out is None:
            out = os.path.join('..', 'data_files', 'backtest_data', 'results')
        if not os.path.exists(out):
            os.makedirs(out)

        
        this_id = ''
        this_id += "%05.03f" % buy_cut + '_'
        this_id += "%05.03f" % sell_cut + '_'
        this_id += "%05.03f" % max_invest 

        self.trade_out = os.path.join(out, 'trade_'+ this_id + '.csv')
        self.value_out = os.path.join(out, 'value_'+ this_id + '.csv')

        self.trade_history = [] 
        self.value_history = []
        self.value_history_cols = ['date', 'value']
        self.trade_history_cols = ['ticker', 'buy_date', 'sell_date', 'num_shares', 'buy_price', 'sell_price', 
                'week_rank', 'position', 'predicted_change', 'actual_change'] 


    '''
    We first have to limit the realm of predictions to those that 
    we have enough capitol to purchase. Then we will apply the following set of operations:
        1. For the largest prediction we can buy, buy shares so that the sum of shares we buy
           does not exceed 10% of our portfolio's current_value.
        2. Continue to buy stocks in this way until: 
            1. There are no longer stocks were we predict a positive result.
            2. We don't have enough capital to purchase more stocks
    '''
    def apply_allocation_strat(self, prediction_df, buy_date, sell_date, week_num):
        candidates = prediction_df.loc[(prediction_df['pred'] > self.buy_cut) | (prediction_df['pred'] < self.sell_cut)]
        candidates = candidates.sort_values('pred', key=lambda x: -1 * abs(1 - x))
        cur_ceiling = self.max_invest * self.current_value
        value_left = self.current_value

        shorts = {}
        buys = {}
        rank = 0
        for i, row in candidates.iterrows():
            act_ceiling = min(cur_ceiling, value_left)
            num_shares = act_ceiling // row['price']
            value_in = num_shares * row['price']
            position = ''
            if row['pred'] > self.buy_cut:
                buys[row['Company']] = value_in
                position = 'buy'
            else:
                shorts[row['Company']] = value_in
                position = 'short'
            value_left -= value_in
            self.trade_history.append((row['Company'], buy_date, sell_date, num_shares, row['price'],
                row['price'] * row['act'], rank, position, row['pred'], row['act']))
            if value_left == 0:
                break
            rank += 1

        for symbol, invested in buys.items():
            value_left += invested * row['act']

        for symbol, invested in shorts.items():
            value_left += (invested + invested * (1 - row['act']))

        self.current_value = value_left
        self.value_history.append((sell_date, self.current_value))

    '''
    Create dataframes for trade and value history and write them away
    '''
    def cleanup(self):
        trade_df = pd.DataFrame(self.trade_history, columns=self.trade_history_cols)
        value_df = pd.DataFrame(self.value_history, columns=self.value_history_cols)
        trade_df.to_csv(self.trade_out, index=False)
        value_df.to_csv(self.value_out, index=False)





