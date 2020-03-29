import pandas as pd
import time
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score, mean_squared_error, auc
from sklearn.model_selection import KFold, StratifiedKFold
from ml_functions import Lgb_Model

# ======================================================================================================
# settings
# ======================================================================================================

TARGET = 'SalePrice'
remove_features = [TARGET]
path = '/Users/ksadhwani001/Documents/github/kaggle'
project = 'house_prices'
train_df = pd.read_csv('{}/{}/data/preprocessed/train.csv'.format(path, project))
test_df = pd.read_csv('{}/{}/data/preprocessed/test.csv'.format(path, project))
features = [col for col in list(train_df) if col not in remove_features]


def main():

    lgb_model = Lgb_Model(train_df, test_df, features, categoricals=categoricals)

    if __name__ == "__main__":
        start_time = time.time()
        main()
        print(f"----- Finished in {time.time() - start_time} seconds ----- \n")

