""" ML functions. """
import numpy as np

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score, f1_score

import datetime
# from datetime import datetime as dt
from dateutil.relativedelta import *

class Base_Model:
        """ Irrespective of which type of model we use, supervised modelling 
        will always require certain components: 
        - a train and test set, with specified features and a target
        - a validation technique
        - a set of model parameters 
        - an accuracy measure

        method - function defined within a class
        instance method - functions which can be called on objects
        self - the first argument to all instance methods """

        # instance method for initialising new objects 
        def __init__(self, train_df, test_df, features, target, categoricals=[], n_splits=3, verbose=True):
            self.train_df = train_df
            self.test_df = test_df
            self.features = features
            self.n_splits = n_splits
            self.categoricals = categoricals
            self.target = target
            self.cv = self.get_cv() 
            self.verbose = verbose
            self.params = self.get_params()
            self.y_pred, self.score, self.model, self.oof_pred = self.fit()
            
        def train_model(self, train_set, val_set):
            raise NotImplementedError
            
        def get_cv(self):
            if self.n_splits is None:
                valid_row_idx = self.train_df[self.train_df['split'] == 'valid'].index.values
                train_row_idx = self.train_df[self.train_df['split'] == 'train'].index.values
                cv = zip([train_row_idx], [valid_row_idx])
            else:
                cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
                cv = cv.split(self.train_df, self.train_df[self.target])

                # cv = TimeBasedCV(train_period=3, test_period=1, freq='weeks')
                # cv = cv.split(self.train_df, validation_split_date=6, date_column='Semana', gap=0)
            return cv
        
        def get_params(self):
            raise NotImplementedError

        def convert_dataset(self, x_train, y_train, x_val, y_val):
            raise NotImplementedError
            
        def convert_x(self, x):
            return x
            
        def fit(self):
            oof_pred = np.zeros((len(self.train_df), ))
            y_pred = np.zeros((len(self.test_df), ))
            
            for fold, (train_idx, val_idx) in enumerate(self.cv):
                print(fold)
                # split train data into train / val components and predictors / target
                x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
                y_train, y_val = self.train_df[self.target].iloc[train_idx], self.train_df[self.target].iloc[val_idx]
                # y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]
                # convert data sets into model compliant format
                train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)

      
                # train model
                model = self.train_model(train_set, val_set)
                # prediction on validation set
                conv_x_val = self.convert_x(x_val)
                oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)
                # convert data sets into model compliant format for testing
                x_test = self.convert_x(self.test_df[self.features])
                # predict on test set
                if self.n_splits is None:
                    y_pred = model.predict(x_test).reshape(y_pred.shape) 
                else:
                    y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits
                    print('Partial validation loss score of fold {} is: {}'.format(fold, mean_squared_error(y_val, oof_pred[val_idx],squared=False)))
            
            if self.n_splits is None:
                loss_score = mean_squared_error(y_val, oof_pred[val_idx] , squared=False)
                print("Overall validation loss score: {}".format(loss_score))
            else:
                loss_score = mean_squared_error(self.train_df[self.target], oof_pred, squared=False)
                print("Overall validation loss score: {}".format(loss_score))
                oof_pred = np.where(oof_pred >= 0.5, 1, 0)
                accuracy = accuracy_score(self.train_df[self.target], oof_pred)
                print("Overall validation accuracy score: {}".format(accuracy))
              
                
                

                

            return y_pred, loss_score, model, oof_pred

class Lgb_Model(Base_Model):
    
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        return train_set, val_set

    def get_params(self):
        params = {'n_estimators':5000,
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.9,
                    'max_depth': 15,
                    'lambda_l1': 1,  
                    'lambda_l2': 1,
                    'early_stopping_rounds': 100,
                    'verbose': -1
                    }
        return params


class Xgb_Model(Base_Model):
    
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return xgb.train(self.params, train_set, 
                         num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')], 
                         verbose_eval=verbosity, early_stopping_rounds=100)
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = xgb.DMatrix(x_train, y_train)
        val_set = xgb.DMatrix(x_val, y_val)
        return train_set, val_set
    
    def convert_x(self, x):
        return xgb.DMatrix(x)
        
    def get_params(self):
        params = {'colsample_bytree': 0.8,                 
            'learning_rate': 0.01,
            'max_depth': 10,
            'subsample': 1,
            'objective':'reg:squarederror',
            'eval_metric':'rmse',
            'min_child_weight':3,
            'gamma':0.25,
            'n_estimators':5000}

        return params


class Catb_Model(Base_Model):
    
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        clf = CatBoostClassifier(**self.params)
        clf.fit(train_set['X'], 
                train_set['y'], 
                eval_set=(val_set['X'], val_set['y']),
                verbose=verbosity, 
                cat_features=self.categoricals)
        return clf
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train, 'y': y_train}
        val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set
        
    def get_params(self):
        params = {'loss_function': 'RMSE',
                   'task_type': "CPU",
                   'iterations': 5000,
                   'od_type': "Iter",
                    'depth': 10,
                  'colsample_bylevel': 0.5, 
                   'early_stopping_rounds': 300,
                    'l2_leaf_reg': 18,
                   'random_seed': 42,
                    'use_best_model': True
                    }
        return params

class Logreg_Model(Base_Model):
    
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        lr = LogisticRegression()
        train_set['X'].to_csv('./data/debug/train_set.csv')
        x_array  = train_set['X'].to_numpy()
        print(x_array)
        y_list = train_set['y'].to_list()
        print(y_list)
        lr.fit(x_array, y_list)
              #eval_set=(val_set['X'], val_set['y']),
              # verbose=verbosity)
        return lr
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train, 'y': y_train}
        val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set
        
    def get_params(self):
        params = { }
        return params


class TimeBasedCV():
    '''
    Parameters 
    ----------
    train_period: int
        number of time units to include in each train set
        default is 30
    test_period: int
        number of time units to include in each test set
        default is 7
    freq: string
        frequency of input parameters. possible values are: days, months, years, weeks, hours, minutes, seconds
        possible values designed to be used by dateutil.relativedelta class
        deafault is days
    '''
    
    def __init__(self, train_period=30, test_period=7, freq='days'):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq

    def split(self, data, validation_split_date=None, date_column='record_date', gap=0):
        '''
        Generate indices to split data into training and test set
        
        Parameters 
        ----------
        data: pandas DataFrame
            your data, contain one column for the record date 
        validation_split_date: datetime.date()
            first date to perform the splitting on.
            if not provided will set to be the minimum date in the data after the first training set
        date_column: string, deafult='record_date'
            date of each record
        gap: int, default=0
            for cases the test set does not come right after the train set,
            *gap* days are left between train and test sets
        
        Returns 
        -------
        train_index ,test_index: 
            list of tuples (train index, test index) similar to sklearn model selection
        '''
        
        # check that date_column exist in the data:
        try:
            data[date_column]
        except:
            raise KeyError(date_column)
                    
        train_indices_list = []
        test_indices_list = []

        if validation_split_date==None:
            validation_split_date = data[date_column].min().date() + eval('relativedelta('+self.freq+'=self.train_period)')
        
        start_train = validation_split_date - self.train_period
        print("start_train")
        print(start_train)
        # start_train = validation_split_date - eval('relativedelta('+self.freq+'=self.train_period)')
        end_train = start_train + self.train_period
        print("end_train")
        print(end_train)
        # end_train = start_train + eval('relativedelta('+self.freq+'=self.train_period)')
        start_test = end_train + gap
        print("start_test")
        print(start_test)
        # start_test = end_train + eval('relativedelta('+self.freq+'=gap)')
        end_test = start_test + self.test_period
        print("end_test")
        print(end_test)
        # end_test = start_test + eval('relativedelta('+self.freq+'=self.test_period)')

        # while end_test < data[date_column].max().date():
        while end_test - 1 <= data[date_column].max():
            # train indices:
            # cur_train_indices = list(data[(data[date_column].dt.date>=start_train) & 
            #                          (data[date_column].dt.date<end_train)].index)
            cur_train_indices = list(data[(data[date_column]>=start_train) & 
                                     (data[date_column]<end_train)].index)

            # test indices:
            cur_test_indices = list(data[(data[date_column]>=start_test) &
                                    (data[date_column]<end_test)].index)
            # cur_test_indices = list(data[(data[date_column].dt.date>=start_test) &
            #                         (data[date_column].dt.date<end_test)].index)
            
            print("Train period:",start_train,"-" , end_train, ", Test period", start_test, "-", end_test,
                  "# train records", len(cur_train_indices), ", # test records", len(cur_test_indices))
            train_indices_list.append(cur_train_indices)
            test_indices_list.append(cur_test_indices)

            # update dates:
            start_train = start_train + self.test_period
            print("start_train")
            print(start_train)
            # start_train = start_train + eval('relativedelta('+self.freq+'=self.test_period)')
            end_train = start_train + self.train_period
            print("end_train")
            print(end_train)
            # end_train = start_train + eval('relativedelta('+self.freq+'=self.train_period)')
            start_test = end_train + gap
            print("start_test")
            print(start_test)
            # start_test = end_train + eval('relativedelta('+self.freq+'=gap)')
            end_test = start_test + self.test_period
            print("end_test")
            print(end_test)
            # end_test = start_test + eval('relativedelta('+self.freq+'=self.test_period)')

        print("lengths")
        print(len(train_indices_list))
        print(len(test_indices_list))
        # mimic sklearn output  
        index_output = [(train,test) for train,test in zip(train_indices_list,test_indices_list)]
        # print("index_output")
        # print(index_output)
        self.n_splits = len(index_output)
        return index_output
    
    
    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits 









