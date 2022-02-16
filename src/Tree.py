import numpy as np
import pandas as pd
import scipy.stats

# This class builds the analysis tree that is used to identify 
# lucritive numerical trends

class Tree:
    def __init__(self, df_dic, max_level, days_after, num_children, child_width):
        self.max_level = max_level
        self.days_after = days_after
        self.num_children = num_children
        self.child_width = child_width
        self.tree = self.build_tree(df_dic)


    '''
    Given the dictionary of stock market data, build
    a pattern tree
    '''
    def build_tree(self, df_dic):
        root = Node(self.num_children, self.max_level, 0, '')
        count = 0
        for ticker, df in df_dic.items():
            print('Analyzing', count, ticker)
            count += 1
            df = df.reset_index()
            df['change'] = df['close'].values / np.concatenate(([np.nan], df['close'].values[:len(df) - 1]))
            df = df.loc[~pd.isna(df['change'])].reset_index(drop=True)
            df['adj_close'] = self.insert_normalized(df)
            for i, row in df.iterrows():
                if i > len(df) - (self.max_level + self.days_after):
                    break
                cur_node = root
                prev_node = None
                for j in range(self.max_level + self.days_after):
                    roi = df.iloc[i + j]
                    next_node = cur_node.add_closing(roi['adj_close'], roi['change'], prev_node)
                    prev_node = cur_node
                    cur_node = next_node
        return root


    '''
    Traverse the tree and return summary statistics for the leaves
    '''
    def traverse(self):
        cur_nodes = [self.tree]
        ret_tup = []
        ret_cols = ['Id', 'Level', 'Mean', 'Std', 'Count', 'Lower', 'Upper', 'Days_After']
        while len(cur_nodes) != 0:
            next_nodes = []
            for cur_node in cur_nodes:
                if len(cur_node.leaves) != 0:
                    for leaf, arr in cur_node.leaves.items():
                        leaf_arr = np.array(arr)
                        mean, lower, upper = mean_confidence_interval(leaf_arr)
                        ret_tup.append((
                            cur_node.id, 
                            cur_node.level, 
                            mean, 
                            leaf_arr.std(), 
                            len(leaf_arr), 
                            lower, 
                            upper,
                            leaf))
                else:
                    next_nodes.extend(
                            [c for c in cur_node.children if c is not None])
            cur_nodes = next_nodes
        return pd.DataFrame(ret_tup, columns=ret_cols)


    '''
    Given a dataframe of stock market data for a single company,
    return a corresponding series with the adjusted closing prices 
    '''
    def insert_normalized(self, df, close_col='close'):
        closings = df[close_col].copy()
        centered_closings = closings - closings.mean()
        scaled_closings = centered_closings / closings.std()
        if self.num_children % 2 == 0:
            start = -1 * self.child_width * self.num_children / 2
            stop = self.child_width * self.num_children / 2
            bins = np.arange(start, stop, step=self.child_width)
        else:
            start = -1 * self.child_width / 2 - (self.child_width * (self.num_children - 1) / 2)
            stop = self.child_width / 2 + (self.child_width * (self.num_children - 1) / 2)
            bins = np.arange(start, stop, step=self.child_width)
        bins = bins[:self.num_children]

        for i, item in scaled_closings.iteritems():
            for j in range(len(bins)):
                if j < len(bins) - 1 and item < bins[j + 1]:
                    scaled_closings.iloc[i] = j
                    break
                elif j == len(bins) - 1:
                    scaled_closings.iloc[i] = j
                    break
        return scaled_closings.apply(lambda x: int(x))


'''
Utility class for the Tree class. Defines the nodes within
the tree
'''
class Node:
    def __init__(self, num_children, max_level, level, given_id):
        self.id = given_id
        self.children = [None] * num_children
        self.level = level 
        self.max_level = max_level
        self.leaves = {} 
        self.cur_leaf = 1 

    '''
    Given the index in which to insert a child node, insert a child
    node at that index and add the given change data
    '''
    def add_closing(self, c_index, change, prev_node):
        if self is prev_node:
            self.cur_leaf += 1
        else:
            self.cur_leaf = 1 

        if self.level == self.max_level:
            if self.cur_leaf not in self.leaves:
                self.leaves[self.cur_leaf] = []
            self.leaves[self.cur_leaf].append(change)
            return self


        else:
            if self.children[c_index] is None:
                self.children[c_index] = \
                        Node(len(self.children), 
                                self.max_level, 
                                self.level + 1, 
                                self.id + str(c_index) + 'a')
            return self.children[c_index]


'''
Calculate the confidence interval for the mean given a list of data.
I got this purely from stack overflow. Not ashamed.
'''
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n <= 1:
        return np.mean(a) if len(a) > 0 else np.nan, np.nan, np.nan
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
