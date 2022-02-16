
from Q_BackTester import Q_BackTester
from BackTester import BackTester
from Daily_Preprocessor import Daily_Preprocessor
from Q_Preproc import Quarterly_Preprocessor
from sklearn.ensemble import RandomForestRegressor
from Company_Lister import Company_Lister
from Models import LSTM_Operator, RF_Operator, Q_RF_Operator
import datetime

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


def test():
    snp_since = datetime.datetime(2011, 1, 1)
    tickers = Company_Lister().get_snp_since(snp_since)
    #tickers = ['GOOGL', 'AAPl', 'AMZN']
    modeler = LSTM_Operator()
    bt = BackTester(tickers, Daily_Preprocessor, 100000, 52, modeler)
    bt.backtest()




if __name__ == '__main__':
    test_quarterly()
