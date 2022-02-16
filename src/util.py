# UTILITY FUNCTIONS FOR THE PROJECT
import pandas as pd
import numpy as np



## Assigns the given value to a bin by returning the largest index
## where the value is less than the separator at that index - 1
def assign_to_bins(val, separators):
    i = 0
    while i < len(separators) and val > separators[i]:
        i += 1
    return i

## Same as above but is a binary problem. Returns 0 unless val is greater than
## the biggest separator
def assign_to_top_bin(val, separators):
    top_sep = separators[len(separators) - 1]
    if val > top_sep:
        return 1
    return 0

## Converts one of the variables in the dataframe to one hot encoding
## Returns a tuple containing the dataframe and the columns of the one hot encodingj
def to_one_hot(df, var, dummy=False, drop=True):
    var_vals = dict([(v, i) for i, v in enumerate(list(df[var].unique()))])
    rev_var = dict([(i, v) for v, i in var_vals.items()])
    var_ind = df[var].apply(lambda x: var_vals[x])
    new_mat = var_ind.values[:, np.newaxis] == np.arange(len(rev_var))[np.newaxis, :]
    columns = [rev_var[i] for i in range(len(rev_var))]
    new_df = pd.DataFrame(new_mat, columns=columns, index=var_ind.index).astype(int)
    if dummy:
        new_df = new_df.drop(columns=rev_var[len(rev_var) - 1], axis=1)
    ret_df = pd.concat((df, new_df), axis=1)
    if drop:
        ret_df = ret_df.drop(columns=var, axis=1)

    return ret_df, columns


