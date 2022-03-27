from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import joblib
import numpy as np
import util
import os
import meta

import numpy as np
import pandas as pd

class LSTM_Operator:
    def __init__(self):
        self.buy_cut = 0.5
        ## Would want to make this a regressor for shorting to work
        self.short_cut = 0.5 ## Shorting won't work with the current backtest architecture

    def instantiate(self, input_shape):
        model = Sequential()
        model.add(LSTM(20, input_shape=input_shape))
        model.add(Dropout(0.4))
        model.add(Dense(10))
        model.add(Dense(2))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def preprocess(self, x_train, y_train):
        # Transform Xs
        self.x_mins = x_train.min(axis=0)
        self.x_maxs = x_train.max(axis=0)
        x_train = (x_train - self.x_mins[np.newaxis, :]) / (self.x_maxs - self.x_mins)[np.newaxis, :]
        p_change_sd = y_train.std()

        # Transform Ys
        y_train = np.vectorize(util.assign_to_top_bin, excluded=[1])(y_train, [1 - p_change_sd, 1, 1 + p_change_sd])
        y_train = y_train[:, np.newaxis] == np.arange(2)[np.newaxis, :]
        num_good = y_train[:, 1].sum()
        bad_indices = np.where(y_train[:, 1] == 0)[0]
        good_indices = np.where(y_train[:, 1] == 1)[0]
        keep = np.concatenate((np.random.choice(bad_indices, int(num_good * 1.5)), good_indices))
        x_train = x_train[keep]
        y_train = y_train[keep]
        return x_train, y_train

    def process_test(self, x_test, y_test):
        x_test = (x_test - self.x_mins[np.newaxis, :]) / (self.x_maxs - self.x_mins)[np.newaxis, :]
        return x_test, y_test

    def fit(self, symbol, model, train_x, train_y):
        model.fit(train_x, train_y, epochs=26, batch_size=32)

class RF_Operator:
    def __init__(self, buy_cut=1.005, short_cut=0.995):
        self.buy_cut = buy_cut 
        self.short_cut = short_cut

    def instantiate(self, x_train, y_train):
        rf = RandomForestRegressor(
            n_estimators=250, # optimal 500
            min_samples_split=2,
            min_samples_leaf=4,
            max_features='sqrt',
            max_depth=175, # optimal 350
            bootstrap=True
        )
        return rf 

    def preprocess(self, x_train, y_train):
        return x_train, y_train

    def process_test(self, x_test, y_test):
        return x_test, y_test
    
    def fit(self, symbol, model, train_x, train_y, retrain):
        if (not os.path.exists(meta.model_loc + symbol)) or retrain:
            model.fit(train_x, train_y)
            joblib.dump(model, meta.model_loc + symbol)
            return model
        else:
            return joblib.load(meta.model_loc + symbol)

    '''
    Use sklearn cross-validation to select a model from randomly selecting from
    the given hyperparameters

    Credit: Will Koehrsen on TowardsDataScience 
    '''
    def select_model(self, model, train_x, train_y):
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        random_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }
        selected = RandomizedSearchCV(estimator=model, param_distributions=random_grid,
                n_iter=100, cv=3, verbose=2, n_jobs=-1)
        selected.fit(train_x, train_y)
        print(selected.best_params_)
        return selected


# We'll use this for quarterly trading
class Q_RF_Operator:
    def __init__(self, buy_cut=1.005, short_cut=0.0):
        self.buy_cut = buy_cut 
        self.sell_cut = short_cut

    def instantiate(self, x_train, y_train):
        rf = RandomForestRegressor(
            n_estimators=250, # optimal 500
            min_samples_split=14,
        )
        return rf 

    def preprocess(self, x_train, y_train):
        return x_train, y_train

    def process_test(self, x_test, y_test):
        return x_test, y_test
    
    '''
    Could take after one of the other models and save it with
    job lib but it honestly doesn't take that long to train
    '''
    def fit(self, model, train_x, train_y, retrain, segment):
        out_file = os.join('..', 'data_files', 'backtest_data', 'q_models', 'seg' + str(segment))
        if retrain:
            model.fit(train_x, train_y)
            joblib.dump(model, out_file)
            return model
        else:
            if not os.path.exists(out_file):
                raise ValueError('Retrain file does not exist')
            return joblib.load(out_file)

    '''
    Use sklearn cross-validation to select a model from randomly selecting from
    the given hyperparameters

    Credit: Will Koehrsen on TowardsDataScience 
    '''
    def select_model(self, model, train_x, train_y):
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        random_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }
        selected = RandomizedSearchCV(estimator=model, param_distributions=random_grid,
                n_iter=100, cv=3, verbose=2, n_jobs=-1)
        selected.fit(train_x, train_y)
        print(selected.best_params_)
        return selected


# This class is associated with loading in the pre-trained weekly model
class W_LSTM:
    '''
    Loads the model that was trained on dev2
    '''
    def load_model(self):
        with open(os.path.join('..', 'models', 'w_lstm', 'config.json')) as fin:
            model = tf.keras.models.model_from_json(fin.read())
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=tf.keras.losses.MeanSquaredError())
        cur_weights = 0
        cur_path = os.path.join('..', 'models', 'w_lstm', 'weights_' + str(cur_weights) + '.txt')
        weights = []
        while os.path.exists(cur_path):
            np_arr = np.loadtxt(cur_path, dtype='float32')
            if cur_weights == 15:
                np_arr = np_arr[:, None] # This is very jancky but I think there's a tensorflow version issue
            if cur_weights == 16:
                np_arr = np_arr[None]
            weights.append(np_arr)
            cur_weights += 1
            cur_path = os.path.join('..', 'models', 'w_lstm', 'weights_' + str(cur_weights) + '.txt')
        model.set_weights(weights)
        return model



        
