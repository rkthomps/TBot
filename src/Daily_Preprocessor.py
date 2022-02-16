# THIS MODULE IS MAINLY BUILT TO SUPPORT PRODUCE_IND_AND_RESPONSE
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import meta
import util
import os


# This preprocessor takes in 8 weeks of market data where each week is composed
# of exactly 5 days of close and volume values. It also adds an indicator
# variable for the month. The response variable for Daily_Preprocessor is the 
# expected change over the next week
class Daily_Preprocessor:
    def __init__(self, symbol):
        if not os.path.exists(meta.stock_loc + symbol + '.csv'):
            raise ValueError('No Data loaded for', symbol)
        self.stock_df = pd.read_csv(meta.stock_loc + symbol + '.csv')
        if len(self.stock_df) < 2:
            raise ValueError('Minimal data for', symbol)

    # This function assumes that the dataframe is in ascending order in terms of date
    # and that the frequency is 'day' - each row represents one day closing price
    def produce_ind_and_response(self, stacked=False, n_weeks=8):
        input_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        ## Make sure the stock_df is clean
        stock_df = self.remove_dup_days(self.stock_df)

        ## Adding a week index to each of the rows in the dataset
        with_week_index, first_time = self.assign_week_index(stock_df)

        ## Generate all possible combinations of week index and day of week
        all_df = self.get_index_day_combs(first_time, int(with_week_index['week_index'].max()))

        ## Join the dataframes to locate missing values
        with_missing = pd.merge(all_df, stock_df, how='left', on=['week_index', 'weekday']).reset_index(drop=True)

        ## Drop the last week of data as it is often incomplete
        with_missing_truncated = with_missing.drop(with_missing.loc[with_missing['week_index'] == with_missing['week_index'].max()].index)

        ## Impute missing values with the mean of the last non-null value and the next non-null value
        ## Can raise ValueError
        imputed_df = self.impute_with_neighbors(with_missing_truncated, input_cols)


        # Creates two dimensional inputs
        if stacked:
            X, Y, X_names, buy, buy_date, sell_date= self.create_stacked_examples(imputed_df, input_cols, n_weeks)
            return X, Y, X_names, buy, buy_date, sell_date

        ## Create the actual training examples
        ## Can Raise ValueError
        formatted_df, response_col, purchase_col, buy_date, sell_date = self.create_examples(imputed_df, input_cols, n_weeks)

        return formatted_df.values, response_col.values, formatted_df.columns.values, purchase_col.values, buy_date.values, sell_date.values

    # Makes sure the stock doesn't have two entries for the same day.
    def remove_dup_days(self, stock_df):
        stock_df['timestamp'] = pd.to_datetime(stock_df['Date'])
        stock_df['date'] = stock_df['timestamp'].dt.date
        stock_df = stock_df.drop(stock_df.loc[stock_df['date'].duplicated()].index)
        stock_df = stock_df.drop(columns='date', axis=1)
        return stock_df

    # Returns the timestamp of the first monday in the dataset
    # This function assumes
    def get_first_monday(self, stock_df):
        #Add weekday (as an index starting with 0 as sunday) to the dataset
        stock_df['weekday'] = stock_df['timestamp'].apply(lambda x: int(datetime.datetime.strftime(x, '%w')))
        starting_weekday = stock_df.iloc[0, stock_df.columns.get_loc('weekday')]
        starting_timestamp = stock_df.iloc[0, stock_df.columns.get_loc('timestamp')]

        #Identify the first monday in the dataset
        if starting_weekday <= 1:
            return relativedelta(days=1 - int(starting_weekday)) + starting_timestamp
        else:
            return relativedelta(days=8 - int(starting_weekday)) + starting_timestamp


    # Assigns every row in the dataframe an index that corresponds to the week it belongs to.
    # This index is computed from the first monday in the dataframe
    def assign_week_index(self, stock_df):
        #Identify the first monday in the dataset
        first_time = self.get_first_monday(stock_df)
            
        #Find each monday to the end of the dataset. Create a dictionary matching the monday
        #with an index since the first monday
        max_date = stock_df['timestamp'].max()
        mondays = [first_time.date()]
        while True:
            cur_len = len(mondays)
            next_date = first_time + relativedelta(days = cur_len * 7)
            if next_date > max_date:
                break
            mondays.append(next_date.date())
        
        # Assign each day in the dataset to a week index. To be assigned to a week index,
        # date() >= mondays[i] & date() < mondays[i + 1]
        week_assignments = []
        cur_week = 0
        for i, row in stock_df.iterrows():
            row_date = row['timestamp'].date()
            if row_date < mondays[cur_week]:
                week_assignments.append(np.nan)
                continue
            j = 0
            while row_date >= mondays[cur_week + j]:
                j += 1
                if cur_week + j == len(mondays):
                    break
            week_assignments.append(cur_week + j - 1)

            cur_week += (j - 1)
        stock_df['week_index'] = week_assignments
        return stock_df, first_time

    ## Generates all combinations of weekday and week index. Uses the date of the first monday when the dataset
    ## starts to compute a week number for each of the values in case the week number is missing in the actual dataset
    def get_index_day_combs(self, first_time, num_mondays):
        tups = []
        cur_day = first_time
        for i in range(num_mondays):
            for j in range(1, 6):
                tups.append((i, j, datetime.datetime.strftime(cur_day, '%b')))
                cur_day = cur_day + relativedelta(days = 1)
            cur_day = cur_day + relativedelta(days = 2) # Get to the next monday
        all_df = pd.DataFrame(tups, columns=['week_index', 'weekday', 'start_time'])
        return all_df


    ## Imputes the given columns of the dataframe with the average of the most recent previous complete value
    ## and the next complete value

    # Function assumes that if one of impute_cols has a missing value, all will have missing values
    def impute_with_neighbors(self, stock_df, impute_cols):
        prev_close = len(stock_df.loc[~pd.isna(stock_df['Close'])])
        prev_non_missing = np.nan
        col_locs = [stock_df.columns.get_loc(col) for col in impute_cols]
        cur_loc = 0
        for i, row in stock_df.iterrows():
            average_of = []
            if not pd.isna(stock_df.iloc[cur_loc, col_locs[0]]):
                prev_non_missing = cur_loc
                cur_loc += 1
                continue
            if not pd.isna(prev_non_missing):
                average_of.append(prev_non_missing)
            j = 1
            while cur_loc + j < len(stock_df) and pd.isna(stock_df.iloc[cur_loc + j, col_locs[0]]):
                j += 1
            if cur_loc + j != len(stock_df):
                average_of.append(cur_loc + j)
            if len(average_of) == 0:
                raise ValueError('empty dataset')
            # Actully insert the new values
            for col_loc in col_locs:
                stock_df.iloc[cur_loc, col_loc] = stock_df.iloc[average_of, col_loc].mean()
            cur_loc += 1
        new_close = len(stock_df.loc[~pd.isna(stock_df['Close'])])
        return stock_df

    ## Create training examples with each input containig 8 weeks worth of closing price and volume,
    ## along with 12 indicator variables indicating the month of the year. The response variable
    ## will be the percent change in price over the following week. This will be measured as closing
    ## price from the last day in the data set (friday) to closing price on the following friday.
    ## A training example will be created from every week in the dataset

    def create_examples(self, imputed_df, input_cols, input_weeks):
        change_after = 1
        max_week = imputed_df['week_index'].max()
        imputed_df = imputed_df.sort_values(['week_index', 'weekday'])
        start_time_col = imputed_df.columns.get_loc('start_time')
        time_col = imputed_df.columns.get_loc('timestamp')
        
        inputs = []
        if max_week < 5 * input_weeks:
            raise ValueError("Not Enough Examples")
        
        for i in range(max_week - (input_weeks - 1 + change_after)):
            buy_time = imputed_df.loc[(imputed_df['week_index'] == i + (input_weeks - 1)) & (imputed_df['weekday'] == 5)]
            sell_time = imputed_df.loc[(imputed_df['week_index'] == i + (input_weeks - 1) + change_after) & (imputed_df['weekday'] == 5)]
            input_time = imputed_df.loc[imputed_df['week_index'] == i].iloc[0, start_time_col]
            response = sell_time.iloc[0, sell_time.columns.get_loc('Close')] / buy_time.iloc[0, buy_time.columns.get_loc('Close')]
            input_df = imputed_df.loc[(imputed_df['week_index'] >= i) & (imputed_df['week_index'] < i + input_weeks)]
            input_df = input_df.set_index(['week_index', 'weekday'])[input_cols]
            input_stack = input_df.stack(dropna=False)
            input_stack.index.names = ['week_index', 'weekday', 'metric']
            input_stack.name = 'value'
            input_df = input_stack.reset_index()
            input_df['index'] = input_df.apply(lambda x: 'wi' + str(x['week_index'] - i) + '_wd' + str(x['weekday']) + '_' + x['metric'], axis=1)
            input_df = input_df.set_index('index')
            day_series = input_df['value']
            
            final_series = pd.concat([day_series, pd.Series([
                input_time, response, buy_time.iloc[0, time_col], sell_time.iloc[0, time_col]], 
                index=['start_time', 'week_change', 'buy_time', 'sell_time'])])
            final_series.name=i
            inputs.append(final_series)
            
        ret_df = pd.DataFrame(inputs)
        ret_df, cols = util.to_one_hot(ret_df, 'start_time')
        price_col = 'wi' + str(input_weeks - 1) + '_wd5_Close'
        change_col = ret_df['week_change']
        buy_prices = ret_df[price_col]
        buy_times = ret_df['buy_time']
        sell_times = ret_df['sell_time']
        ret_input = ret_df.drop(columns=['week_change', 'buy_time', 'sell_time'], axis=1)

        return ret_input, change_col, buy_prices, buy_times, sell_times 


    ## Create examples that are suitable for input to an LSTM. Such inputs are sequential. 
    def create_stacked_examples(self, imputed_df, input_weeks):
        change_after = 1
        max_week = imputed_df['week_index'].max()
        imputed_df = imputed_df.sort_values(['week_index', 'weekday'])
        start_time_col = imputed_df.columns.get_loc('start_time')
        time_col = imputed_df.columns.get_loc('timestamp')
        close_col = imputed_df.columns.get_loc('Close')
        num_examples = max_week - (input_weeks - 1 + change_after)
        num_times = len(imputed_df.iloc[start_time_col].unique())
        
        inputs = []
        starts = []
        responses = np.full((num_examples,), 0.0)
        prices = np.full((num_examples,),0.0)
        buy_time = np.full((num_examples,), np.nan)
        sell_time = np.full((num_examples,), np.nan)


        if max_week < 100:
            raise ValueError("Not Enough Examples")
        
        for i in range(max_week - (input_weeks - 1 + change_after)):
            buy_time = imputed_df.loc[(imputed_df['week_index'] == i + (input_weeks - 1)) & (imputed_df['weekday'] == 5)].copy()
            sell_time = imputed_df.loc[(imputed_df['week_index'] == i + (input_weeks - 1) + change_after) & (imputed_df['weekday'] == 5)].copy()
            input_time = imputed_df.loc[imputed_df['week_index'] == i].iloc[0, start_time_col]
            response = sell_time.iloc[0, sell_time.columns.get_loc('Close')] / buy_time.iloc[0, buy_time.columns.get_loc('Close')]
            input_df = imputed_df.loc[(imputed_df['week_index'] >= i) & (imputed_df['week_index'] < i + input_weeks)]
            input_df = input_df.set_index(['week_index', 'weekday'])[input_cols]
            input_df['start_time'] = input_time
            inputs.append(input_df)
            responses[i] = response
            prices[i] = buy_time.iloc[0, close_col]
            buy_time[i] = buy_time.iloc[0, time_col]
            sell_time[i] = sell_time.iloc[0, time_col]


        all_df = pd.concat(inputs, ignore_index=True)
        all_df, cols = util.to_one_hot(all_df, 'start_time')
        X = all_df.values.reshape((num_examples, 5 * input_weeks, -1))
            
        return X, responses, all_df.columns, prices, buy_time, sell_time






