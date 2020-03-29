# ======================================================================================================
# imports
# ======================================================================================================

import sys
import os
import time
import pandas as pd
from fe_functions import impute_mean

# ======================================================================================================
# settings
# ======================================================================================================

TARGET = 'SalePrice'
remove_features = [TARGET]
path = '/Users/ksadhwani001/Documents/github/kaggle'
project = 'house_prices'


skew_threshold = 0.5



#features = [x for x in features if x not in features_exlcude

train_df = pd.read_csv('{}/{}/data/preprocessed/train.csv'.format(path, project))
test_df = pd.read_csv('{}/{}/data/preprocessed/test.csv'.format(path, project))










