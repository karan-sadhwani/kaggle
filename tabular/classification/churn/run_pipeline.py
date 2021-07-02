import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing import read_data
from src.fe_functions import one_hot_encode, impute_mean
from src.ml_functions import Lgb_Model, Xgb_Model, Catb_Model, Logreg_Model, mean_squared_error

# ======================================================================================================
# Settings
# ======================================================================================================

preprocessing = True
feature_engineering = True
modelling = False

# path to project
path = '/Users/ksadhwani001/Documents/github/kaggle/tabular/classification'
# project name
project = 'churn'
# list of raw input files
fnames = ['full.csv']

# target name
TARGET = 'Churn'
# columns not to be used as features
remove_features = [TARGET, 'customerID']
tracking_cols = []


# train / test split
SPLIT = 0.8
RANDOM_STATE = 42
# valid / train split. Enter None if you want to set your own split via a 'split' column
N_SPLITS = 4

def main(args=None):

# ======================================================================================================
# preprocessing
# ======================================================================================================

    if preprocessing:
        train_df, test_df, full_df = read_data(project, path, fnames)
        
        # type changes
        full_df['TotalCharges'] = pd.to_numeric(full_df['TotalCharges'], errors='coerce')
        full_df['MonthlyCharges'] = pd.to_numeric(full_df['TotalCharges'], errors='coerce')
        full_df['tenure'] = full_df['tenure'].astype('int')

        # train/test split
        if not full_df.empty:
            train_df = full_df.sample(frac=SPLIT, random_state=RANDOM_STATE)
            test_df = full_df.drop(train_df.index)
        train_df.to_csv('./data/preprocessed/train.csv', index=False)
        test_df.to_csv('./data/preprocessed/test.csv', index=False)

# ======================================================================================================
# feature engineering
# ======================================================================================================

    if not preprocessing and feature_engineering:
        
        # read in data
        print("reading in data")
        train_df = pd.read_csv('./data/preprocessed/train.csv')
        test_df = pd.read_csv('./data/preprocessed/test.csv')

    if feature_engineering:
        print("feature engineering")
        
        # variable transformation


        # categorical encodings
        cat_cols = [i for i in train_df.columns if train_df.dtypes[i]=='object']
        cat_cols = [i for i in cat_cols if i not in remove_features]

        # make target numeric
        train_df[TARGET] = train_df[TARGET].map({'Yes': 1, 'No': 0})
        test_df[TARGET] = test_df[TARGET].map({'Yes': 1, 'No': 0})

        # train_df['Churn'] = train_df['Churn'].replace(('Yes', 'No'), (1, 0), inplace=True)
        # test_df['Churn'] = test_df['Churn'].replace(('Yes', 'No'), (1, 0), inplace=True)

        print(train_df[cat_cols])
        train_df = one_hot_encode(train_df, cat_cols)
        test_df = one_hot_encode(test_df, cat_cols)

        # create list of final features
        features = [col for col in list(train_df) if col not in remove_features]
        train_df.info()
        train_df.to_csv('./data/debug/debug_file.csv')

# ======================================================================================================
# modelling
# ======================================================================================================

    # train lgb model
    print("training model")
    lgb_model = Xgb_Model(train_df, test_df, features, TARGET, categoricals=[], n_splits=N_SPLITS)
    try:
        sample_submission = pd.read_csv('./data/raw/sample_submission.csv')
        sample_submission[TARGET] = lgb_model.y_pred
        sample_submission.to_csv('./data/model_output/predictions.csv', index=False)
    except:
        test_df['pred_prob'] = lgb_model.y_pred
        test_df['pred'] = np.where(test_df['pred_prob'] >= 0.5, 1, 0)
        loss_score = mean_squared_error(test_df[['pred']], test_df[['Churn']] , squared=False)
        print("Test loss score: {}".format(loss_score))
        accuracy = accuracy_score(test_df[['pred']], test_df[['Churn']])
        print("Test accuracy score: {}".format(accuracy))
        f1 = f1_score(test_df[['pred']], test_df[['Churn']])
        print("Test f1 score: {}".format(f1))
        test_df.to_csv('./data/model_output/predictions.csv', index=False)

    # train xgb model
    # xgb_model = Xgb_Model(train_df, test_df, features, TARGET, categoricals=[], n_splits=N_SPLITS)
    # train_df['valid_pred'] = xgb_model.oof_pred
    # train_df.to_csv('./data/model_output/xgb_oof_v2.csv', index=False)
    # sample_submission['Demanda_uni_equil'] = xgb_model.y_pred
    # sample_submission.to_csv('./data/model_output/xgb_v2.csv', index=False)
    

    # # use global mean as initial prediction
    # valid_df['prediction'] = np.mean(train_df['log_demand']) 
    # valid_df = target_encode(train_df, valid_df, interactions_dict)
    # for col in ['prediction', 'log_prediction', 'log_target', 'diff', 'log_diff']:
    #     valid_df[col] = valid_df[col].round(decimals=3)

    # valid_df = valid_df.sort_values(['log_diff'], ascending=False)
    # top_10_products = valid_df['Producto_ID'].unique().tolist()[:10]
    # print(top_10_products)
    # train_df_top_ten = train_df[train_df['Producto_ID'].isin(top_10_products)]
    # valid_df_top_ten = valid_df[valid_df['Producto_ID'].isin(top_10_products)]

    # valid_df_top_ten.to_csv('./data/model_output/valid_df_top_ten.csv', index=False)
    # train_df_top_ten.to_csv('./data/model_output/train_df_top_ten.csv', index=False)

    # # # # Round to 4 decimal places
   

    # # # # Relabel columns ready for creating submission
    # # # submit.rename(columns={'Pred': 'Demanda_uni_equil'}, inplace=True)
    # # # submit_use = submit[['id', 'Demanda_uni_equil']]

    # # print("Creating Submission...")
    # # # Saving the submission into csv
    # # valid_df.to_csv('./data/valid_df.csv', index=False)


    # # rmsle = ((np.log1p(valid_df.Demanda_uni_equil) - np.log1p(valid_df.Pred_MP)) ** 2).mean() ** .5
    # # print(rmsle)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"----- Finished in {time.time() - start_time} seconds ----- \n")

    
