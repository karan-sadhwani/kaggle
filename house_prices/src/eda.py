import pandas as pd
from eda_functions import missing_vals, get_numeric_stats, split_cat_num, plot
import seaborn as sns

# ======================================================================================================
# settings
# ======================================================================================================

path = '/Users/ksadhwani001/Documents/github/kaggle'
project = 'house_prices'



train_df = pd.read_csv('{}/{}/data/preprocessed/train.csv'.format(path, project))
test_df = pd.read_csv('{}/{}/data/preprocessed/test.csv'.format(path, project))


num_df, cat_df = split_cat_num(train_df)

# General
stats_all = missing_vals(train_df)
print(stats_all)

# Numerical
df_desc = train_df.describe().transpose()
print(df_desc)
num_stats = get_numeric_stats(train_df)





