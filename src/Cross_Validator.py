import os, sys
import numpy as np
import pandas as pd
from W_Preproc import Weekly_Preprocessor as WP

import tensorflow as tf # temporary

class Cross_Validator:
    def __init__(
        self, 
        start_year, 
        end_year, 
        test_size, 
        model, 
        strats,
        train_batch_weeks=15,
        val_batch_weeks=4,
        batch_size=256,
        epochs=30,
        iters=50,
        out_dir=os.path.join('..', 'data_files', 'cv_results')
        ):

        self.model = model
        self.strats = strats
        self.train_batch_weeks = train_batch_weeks
        self.val_batch_weeks = val_batch_weeks
        self.batch_size = batch_size
        self.epochs = epochs
        self.iters = iters
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(out_dir)

        self.preds = []
        self.acts = []


        year_bounds = self.get_cval_list(start_year, end_year)
        self.wps = self.get_wps(year_bounds)
        x, _, _, _, _, _, _, _, _ = self.wps[0].get_next_week()
        self.wps[0].cur_week = 1
        self.model.build(x.shape)
        self.init_weights = model.get_weights()


    '''
    Returns an array of boundaries where i is the start of the interval
    and i + 1 is the end of the interval
    '''
    def get_cval_list(self, start_year, end_year, test_size):
        years_per_segment = np.floor((end_year - start_year) * test_size)
        return np.arange(start_year, end_year, years_per_segment)


    '''
    Instantiate one preprocessor for each cval segment
    '''
    def get_wps(self, year_bounds):
        WPs = []
        for i in range(len(year_bounds) - 1):
            WPs.append(WP(40, year_bounds[i], year_bounds[i + 1] - 1))
        return WPs

    '''
    Create a generator from a lists of preprocessors
    Batch size represents the number of weeks, not the number of
    examples. The number of examples is much larger than the number of
    weeks. The generator iterates sequentially over the given wps 
    '''
    def sequential_gen(self, wps, weeks_in_batch):
        for wp in wps:
            # Reset current week for each preprocessor
            wp.cur_week = 1
                                    
        which_wp = lambda x: x % len(wps)
        wp_counter = 0
        n_examples = 0
        while True:
            n_examples = 0
            xs = []
            ys = []
            while n_examples < weeks_in_batch:
                result = wps[which_wp(wp_counter)].get_next_week()
                if result is not None:
                    x, y, x_names, prices, companies, b_date, s_date, cur_week = result
                    xs.append(x)
                    ys.append(y[:, None])
                    n_examples += 1
                else:
                    wp_counter += 1
            yield np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)[:, 0], x_names

    '''
    Create a generator from a lists of preprocessors
    Batch size represents the number of weeks, not the number of
    examples. The number of examples is much larger than the number of
    weeks. The generator randomizes over the given wps in hopes of generalizing
    over different periods of time
    '''
    def stochastic_gen(self, wps, weeks_in_batch):
        num_weeks = lambda wp: (wp.end_year - wp.start_year + 1) * 52
        rand_week = lambda num_weeks: int((np.random.random() * num_weeks) + 1)
        n_examples = 0
        while True:
            n_examples = 0
            xs = []
            ys = []
            while n_examples < weeks_in_batch:
                wp_index = int(np.random.random() * len(wps))
                wps[wp_index].cur_week = rand_week(num_weeks(wps[wp_index]))
                result = wps[which_wp(wp_counter)].get_next_week()
                if result is not None:
                    x, y, x_names, prices, companies, b_date, s_date, cur_week = result
                    xs.append(x)
                    ys.append(y[:, None])
                    n_examples += 1
            yield np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)[:, 0], x_names

    '''
    Trains the given model using training and testing WPs
    '''
    def train_model(self, train_wps, test_wp):
        data_generator = self.create_gen(train_wps, self.train_batch_weeks)
        val_generator = self.create_gen([test_wp], self.val_batch_weeks)
        cur_x, cur_y, x_names = data_generator.__next__()
        val_x, val_y, _ = val_generator.__next__()

        for i in range(self.iters):
            self.model.fit(cur_x, cur_y, epochs=self.epochs, batch_size=self.batch_size, validation_data=(val_x, val_y))
            pred = model.predict(val_x)
            self.preds.append(pred)
            self.acts.append(cur_y)
            print(i, pred.std())
            del cur_x
            del cur_y
            del val_x
            del val_y

            cur_x, cur_y, _ = data_generator.__next__()
            val_x, val_y, _ = val_generator.__next__()


    '''
    Get the results by testing the performance of the model on testing set, and saving strategy
    results
    '''
    def get_results(self, test_wp):
        res_dir = os.path.join(self.out_dir, 'results_', str(int(test_wp.start_year)))
        if self.strats is None or len(self.strats) < 1:
            return

        for strat in self.strats:
            strat.set_out_dir(res_dir)

        bt = W_BackTester(
            preprocessor = test_wp,
            strategies = self.strats,
            model = self.model)

        leg_mse = bt.backtest()
        return leg_mse

    '''
    Performs cross validation for stock market prediction
    '''
    def run(self):
        for i, wp in enumerate(self.wps):
            train_wps = wps[0:i] + wps[(i+1):]
            test_wp = wps[i]
            self.model.set_weights(self.init_weights)
            self.train_model(train_wps, test_wp)
            test_wp.cur_week = 1
            self.get_results(test_wp)

        preds = np.concatenate(self.preds)
        print('preds', preds.shape)
        acts = np.concatenate(self.acts)
        print('acts', acts.shape)
        np.savetxt(os.path.join(self.out_dir, 'predicted'))
        np.savetxt(os.path.join(self.out_dir, 'actual'))


if __name__ == '__main__':
    print("Main method for Cross Validator")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    cv = Cross_Validator(1998, 2000, 0.8, model=model, strats=None)
    cv.run()



