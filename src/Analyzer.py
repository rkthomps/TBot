from Alpaca_Scraper import Alpaca_Scraper
from Company_Lister import Company_Lister
from Tree import Tree
import numpy as np
import pandas as pd
import datetime

data_path = '../data_files/'

class Analyzer:
    def __init__(self):
        c_list = Company_Lister()
        a_scrape = Alpaca_Scraper()
        self.snp_data = a_scrape.get_tickers_data(c_list.get_snp(), limit=365)
        self.default_levels = np.arange(3, 7, step=1)
        self.default_days_after = [4]
        self.default_num_children = np.arange(5, 15, step=1)
        self.default_child_width = np.arange(0.1, 1.7, step=0.5)

    '''
    Tests different combinations of parameters in order to maximize the number of
    patterns that meet the given thresholds 
    '''
    def analyze_count_thresh(self, thresh_cols, thresh_vals, thresh_ops, levels=None, 
            days_after=None, n_children=None, c_width=None):
        if any([tc not in ['Mean', 'Count', 'Lower', 'Upper', 'Days_After'] for tc in thresh_cols]):
            raise ValueError('Invalid Threshold Column')
        if any([to not in ['<', '>'] for to in thresh_ops]):
            raise ValueError('Invalid Threshold Op')
        thresholds = [(t_c, t_v, t_o) for t_c, t_v, t_o in zip(thresh_cols, thresh_vals, thresh_ops)]
        ret_tups = []
        ret_cols = ['M_Level', 'D_After', 'N_Children', 'C_Width', 'Count']
        if levels is None:
            levels = self.default_levels
        if days_after is None:
            days_after = self.default_days_after
        if n_children is None:
            n_children = self.default_num_children
        if c_width is None:
            c_width = self.default_child_width
        to_analyze = len(levels) * len(days_after) * len(n_children) * len(c_width) 
        num_analyzed = 0
        for level in levels:
            for d_a in days_after:
                for n_c in n_children:
                    for c_w in c_width:
                        tree = Tree(self.snp_data, level, d_a, n_c, c_w)
                        res_df = tree.traverse()
                        print('pre_filter', len(res_df))
                        for t_col, t_val, t_op in thresholds:
                            if t_op == '<':
                                res_df = res_df.loc[res_df[t_col] < t_val]
                            if t_op == '>':
                                res_df = res_df.loc[res_df[t_col] > t_val]
                        print('post_filter', len(res_df))
                        ret_tups.append((
                            level, d_a, n_c, c_w, res_df['Count'].sum()
                        ))
                        num_analyzed += 1
                        print('Analyzed', num_analyzed, 'out of', to_analyze, 'combinations')
        return pd.DataFrame(ret_tups, columns=ret_cols).sort_values('Count', ascending=False)

if __name__ == '__main__':
    start = datetime.datetime.now()
    analyzer = Analyzer()
    levels = [3]
    days = [4]
    n_c = [6]
    c_w = [0.1]
    a_df = analyzer.analyze_count_thresh(
            thresh_cols=['Count', 'Lower'],
            thresh_vals=[30, 1.005],
            thresh_ops = ['>', '>'])
#            levels=levels, 
#            days_after=days, 
#            n_children = n_c, 
#            c_width = c_w)
    end = datetime.datetime.now()
    print(a_df)
    a_df.to_csv(data_path + 'analysis.csv', index=False)
    print('Analyzed data in', (end - start).total_seconds(), 'seconds')

