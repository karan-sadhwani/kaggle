# ======================================================================================================
# imports
# ======================================================================================================

import sys
import os
import time
import pandas as pd

import numpy as np

def read_data(project, path, fnames):

    train = [x for x in fnames if "train" in x]
    test = [x for x in fnames if "test" in x]
    full = [x for x in fnames if "full" in x]
    submission = [x for x in fnames if "submission" in x]

    if len(train) == 1:
        train_df = pd.read_csv('{}/{}/data/raw/{}'.format(path, project, train[0]))
        print('Training file has {} rows and {} columns'.format(train_df.shape[0], train_df.shape[1]))
    else:
        train_df = pd.DataFrame()

    if len(test) == 1:
        test_df = pd.read_csv('{}/{}/data/raw/{}'.format(path, project, test[0]))
        print('Test file has {} rows and {} columns'.format(test_df.shape[0], test_df.shape[1]))
    else:
        test_df = pd.DataFrame()

    if len(full) == 1:
        full_df = pd.read_csv('{}/{}/data/raw/{}'.format(path, project, full[0]))
        print('File has {} rows and {} columns'.format(full_df.shape[0], full_df.shape[1]))
    else:
        full_df = pd.DataFrame()

    if len(submission) == 1:
        submission = pd.read_csv('{}/{}/data/raw/{}'.format(path, project, submission[0]))
        print('Submission file has {} rows and {} columns'.format(submission.shape[0], submission.shape[1]))
    
    return train_df, test_df, full_df
    
def reduce_mem_usage(df, verbose=True):
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df


