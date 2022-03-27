
from Q_BackTester import Q_BackTester
from BackTester import BackTester
from W_BackTester import W_BackTester
from W_Preproc import Weekly_Preprocessor
from Daily_Preprocessor import Daily_Preprocessor
from Q_Preproc import Quarterly_Preprocessor
from sklearn.ensemble import RandomForestRegressor
from Company_Lister import Company_Lister
from Models import LSTM_Operator, RF_Operator, Q_RF_Operator, W_LSTM
from Strategy import Strategy
import numpy as np
import datetime, os

def test1():
    snp_since = datetime.datetime(2011, 1, 1)
    tickers = sorted(Company_Lister().get_snp())
    #tickers = ['AAPL', 'AMZN']
    modeler = RF_Operator(buy_cut = 1.005, short_cut = 0.995)
    bt = BackTester(
            tickers, 
            preprocessor=Daily_Preprocessor, 
            initial_value=100000, 
            num_weeks=780, 
            mod_op=modeler, 
            train_weeks=40, 
            end_date='2007-01-01', 
            train_every=None,
            bad_flag=False)
    bt.backtest()

def test_quarterly():
    modeler = Q_RF_Operator(buy_cut=1.005, short_cut = 0.0)
    bt = Q_BackTester(
            preprocessor=Quarterly_Preprocessor, 
            initial_value=100000, 
            num_qs=32, 
            mod_op=modeler, 
            train_qs=4, 
            end_year=2020, 
            train_every=4)
    bt.backtest()


def test_weekly():
    model = W_LSTM().load_model()
    buy_cuts = np.linspace(1, 1.1, 10)
    sell_cuts = np.linspace(0.9, 1, 10)
    max_alloc = np.linspace(0.05, 0.5, 10)

    strats = []
    for b in buy_cuts:
        for s in sell_cuts:
            for m in max_alloc:
                strats.append(Strategy(100000, b, s, m))
                
    bt = W_BackTester(
            preprocessor = Weekly_Preprocessor,
            strategies = strats,
            model=model,
            start_year = 2001,
            end_year = 2009)
    bt.backtest()

def final_test_weekly():
    model = W_LSTM().load_model()
    out_dir = os.path.join('..', 'data_files', 'backtest_data', 'test_results')
    bt = W_BackTester(
            preprocessor = Weekly_Preprocessor,
            strategies = [Strategy(100000, 1.022, 0.956, 0.35, out_dir)],
            model=model,
            start_year = 2010,
            end_year = 2021
            )
    bt.backtest()


def test():
    snp_since = datetime.datetime(2011, 1, 1)
    tickers = Company_Lister().get_snp_since(snp_since)
    #tickers = ['GOOGL', 'AAPl', 'AMZN']
    modeler = LSTM_Operator()
    bt = BackTester(tickers, Daily_Preprocessor, 100000, 52, modeler)
    bt.backtest()




if __name__ == '__main__':
    final_test_weekly()
