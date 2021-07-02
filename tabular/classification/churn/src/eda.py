import pandas as pd
from eda_functions import missing_vals, get_numeric_stats, split_cat_num, plot
# import seaborn as sns

# ======================================================================================================
# settings
# ======================================================================================================

path = '/Users/ksadhwani001/Documents/github/kaggle'
project = 'grupo-bimbo-inventory-demand'



train_df = pd.read_csv('{}/{}/data/preprocessed/train.csv'.format(path, project))
test_df = pd.read_csv('{}/{}/data/preprocessed/test.csv'.format(path, project))

train_df = train_df.sample(n = 500000) 
test_df = test_df.sample(n= 100000)

train_df.to_csv('{}/{}/data/preprocessed/train_sample.csv'.format(path, project))
test_df.to_csv('{}/{}/data/preprocessed/test_sample.csv'.format(path, project))

# num_df, cat_df = split_cat_num(train_df)

# # General
# # stats_all = missing_vals(train_df)
# # print(stats_all)

# # Numerical
# df_desc = train_df.describe().transpose()
# print(df_desc)
# # num_stats = get_numeric_stats(train_df)





