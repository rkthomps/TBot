import numpy as np
import pandas as pd
import os

import alpaca_trade_api 

from Alpaca_Scraper import Alpaca_Scraper
from Company_Lister import Company_Lister
from Tree import Tree


data_path = '../data_files/'
training_name = 'training.csv'

class Trader:
    def __init__(self, back_test=-1, train=False, starting_balance=-1):
        c_list = Company_Lister()
        a_scrape = Alpaca_Scraper()
        api = alpaca_trade_api.REST()
        account = get_account()

        if back_test < 0:
            self.snp_data = a_scrape.get_tickers_data(c_list.get_snp(), limit=365) 
        else:
            self.snp_data = a_scrape.get_tickers_data(c_list.get_snp(), limit=365 + back_test) 
        self.back_test = back_test

        if train or not os.path.exists(data_path + training_name):
            self.train()

        if starting_balance < 0:
            starting_balance = account.equity()
            print('Current Equity', starting_balance)

    def train(self):
        max_level = 5
        days_after = 4
        num_children = 7
        child_width = 0.5
        tree = Tree(self.snp_data, max_level, days_after, num_children, child_width)
        tree.traverse().to_csv(data_path + training_name)







def train

if __name__ == '__main__':

