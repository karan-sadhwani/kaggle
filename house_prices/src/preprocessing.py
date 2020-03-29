# ======================================================================================================
# imports
# ======================================================================================================

import sys
import os
import time
import pandas as pd
from preprocessing_functions import reduce_mem_usage

# ======================================================================================================
# settings
# ======================================================================================================

path = '/Users/ksadhwani001/Documents/github/kaggle'
project = 'house_prices'
fnames = ['train.csv', 'test.csv', 'sample_submission.csv']

# ======================================================================================================
# main
# ======================================================================================================

def main():

    train = [x for x in fnames if "train" in x]
    test = [x for x in fnames if "test" in x]
    submission = [x for x in fnames if "submission" in x]

    if len(train) == 1:
        train_df = pd.read_csv('{}/{}/data/raw/{}'.format(path, project, train[0]))
        print('Training file has {} rows and {} columns'.format(train_df.shape[0], train_df.shape[1]))
    if len(test) == 1:
        test_df = pd.read_csv('{}/{}/data/raw/{}'.format(path, project, test[0]))
        print('Test file has {} rows and {} columns'.format(test_df.shape[0], test_df.shape[1]))
    if len(submission) == 1:
        submission = pd.read_csv('{}/{}/data/raw/{}'.format(path, project, submission[0]))
        print('Submission file has {} rows and {} columns'.format(submission.shape[0], submission.shape[1]))
    
    for df in [train_df, test_df]:
        df = reduce_mem_usage(df)

    train_df.to_csv('{}/{}/data/preprocessed/{}'.format(path, project, train[0]), index=False)
    test_df.to_csv('{}/{}/data/preprocessed/{}'.format(path, project, test[0]), index=False)
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"----- Finished in {time.time() - start_time} seconds ----- \n")




