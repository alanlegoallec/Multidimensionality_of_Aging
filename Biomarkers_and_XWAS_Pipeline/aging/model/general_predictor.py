#!/usr/bin/env python3

import functools
import numpy as np
import pandas as pd
import copy

from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import copy
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict, RandomizedSearchCV, PredefinedSplit, ParameterSampler, cross_validate
import numpy as np
import scipy.stats as stat
from sklearn.metrics import r2_score, f1_score
from hyperopt import fmin, tpe, space_eval, Trials, hp, STATUS_OK
from sklearn.pipeline import Pipeline

path_eid_split = '/n/groups/patel/Alan/Aging/Medical_Images/eids_split/'



MODELS = {'ElasticNet', 'RandomForest', 'GradientBoosting', 'Xgboost', 'LightGbm', 'NeuralNetwork', 'Correlation', 'CoxPh', 'CoxGbm', 'CoxRf', 'CoxXgboost', 'AftXgboost'}




class BaseModel():
    def __init__(self, model, outer_splits, inner_splits, n_iter, model_validate = 'HyperOpt'):
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.n_iter = n_iter
        self.model_validate = model_validate
        if model not in MODELS:
            raise ValueError(f'{model} model unrecognized')
        else :
            self.model_name = model

    def get_model(self):
        return self.model

    def get_hyper_distribution(self):
        if self.model_validate == 'HyperOpt':
            if self.model_name == 'ElasticNet':
                return {
                        'alpha':  hp.loguniform('alpha', low = np.log(0.01), high = np.log(10)),
                        'l1_ratio': hp.uniform('l1_ratio', low = 0.01, high = 0.99)
                        }
            elif self.model_name == 'RandomForest':
                return {
                        'n_estimators': hp.randint('n_estimators', upper = 300) + 150,
                        'max_features': hp.choice('max_features', ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
                        'max_depth': hp.choice('max_depth', [None, 10, 8, 6])
                        }
            elif self.model_name == 'GradientBoosting':
                return {
                        'n_estimators': hp.randint('n_estimators', upper = 300) + 150,
                        'max_features': hp.choice('max_features', ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
                        'learning_rate': hp.uniform('learning_rate', low = 0.01, high = 0.3),
                        'max_depth': hp.randint('max_depth', 10) + 5
                        }
            elif self.model_name == 'Xgboost' or self.model_name == 'CoxXgboost' or self.model_name == 'AftXgboost':
                return {
                        'colsample_bytree': hp.uniform('colsample_bytree', low = 0.2, high = 0.7),
                        'gamma': hp.uniform('gamma', low = 0.1, high = 0.5),
                        'learning_rate': hp.uniform('learning_rate', low = 0.02, high = 0.2),
                        'max_depth': hp.randint('max_depth', 10) + 5,
                        'n_estimators': hp.randint('n_estimators', 300) + 150,
                        'subsample': hp.uniform('subsample', 0.2, 0.8)
                }
            elif self.model_name == 'LightGbm':
                return {
                         'num_leaves': hp.randint('num_leaves', 40) + 5,
                         'min_child_samples': hp.randint('min_child_samples', 400) + 100,
                         'min_child_weight': hp.loguniform('min_child_weight', -5, 4),
                         'subsample': hp.uniform('subsample', low=0.2, high=0.8),
                         'colsample_bytree': hp.uniform('colsample_bytree', low=0.4, high=0.6),
                         'reg_alpha': hp.loguniform('reg_alpha', -2,2),
                         'reg_lambda': hp.loguniform('reg_lambda', -2, 2),
                         'n_estimators': hp.randint('n_estimators', 300) +  150
                    }
            elif self.model_name == 'NeuralNetwork':
                return {
                         'learning_rate_init': hp.loguniform('learning_rate_init', low=-5, high=-1),
                         'alpha': hp.loguniform('alpha', low=-6, high=3)
                }
            elif self.model_name == 'CoxPh':
                return {
                    'alpha' : hp.loguniform('alpha', low = np.log(0.01), high = np.log(10))
                }
            elif self.model_name == 'CoxRf' :
                return {
                    'n_estimators': hp.randint('n_estimators', 1) + 9,
                    'max_features': hp.choice('max_features', ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
                    'max_depth': hp.choice('max_depth', [None, 10, 8, 6])
                }
            elif self.model_name == 'CoxGbm':
                return {
                    'n_estimators': hp.randint('n_estimators', 1) + 9,
                    'max_features': hp.choice('max_features', ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
                    'learning_rate': hp.uniform('learning_rate', low = 0.01, high = 0.3),
                    'max_depth': hp.randint('max_depth', 10) + 5
                }
        elif self.model_validate == 'RandomizedSearch':
            if self.model_name == 'ElasticNet':
                return {
                        'alpha': np.geomspace(0.01, 10, 30),
                        'l1_ratio': stat.uniform(loc = 0.01, scale = 0.99)
                        }
            elif self.model_name == 'RandomForest':
                return {
                        'n_estimators': stat.randint(low = 50, high = 300),
                        'max_features': ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                        'max_depth': [None, 10, 8, 6]
                        }
            elif self.model_name == 'GradientBoosting':
                return {
                        'n_estimators': stat.randint(low = 250, high = 500),
                        'max_features': ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                        'learning_rate': stat.uniform(0.01, 0.3),
                        'max_depth': stat.randint(3, 6)
                        }
            elif self.model_name == 'Xgboost':
                return {
                        'colsample_bytree': stat.uniform(loc = 0.2, scale = 0.7),
                        'gamma': stat.uniform(loc = 0.1, scale = 0.5),
                        'learning_rate': stat.uniform(0.02, 0.2),
                        'max_depth': stat.randint(3, 6),
                        'n_estimators': stat.randint(low = 200, high = 400),
                        'subsample': stat.uniform(0.6, 0.4)
                }
            elif self.model_name == 'LightGbm':
                return {
                        'num_leaves': stat.randint(6, 50),
                        'min_child_samples': stat.randint(100, 500),
                        'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                        'subsample': stat.uniform(loc=0.2, scale=0.8),
                        'colsample_bytree': stat.uniform(loc=0.4, scale=0.6),
                        'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                        'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
                    }
            elif self.model_name == 'NeuralNetwork':
                return {
                        'learning_rate_init': np.geomspace(5e-5, 2e-2, 30),
                        'alpha': np.geomspace(1e-5, 1e3, 30)
                }

    def create_list_test_train_folds(self, X, y, fold, organ, view, transformation):
        X_eid = X.drop_duplicates('eid')
        y_eid = y.drop_duplicates('eid')
        eids = X_eid.eid
        if fold is None :
            splits = self.inner_splits
        else :
            splits = self.outer_splits
        #if organ in ['Eyes', 'Heart', 'Brain', 'ECG', 'Carotids', 'Vascular', 'Anthropometry', 'Urine', 'BloodB', 'BloodC', 'Lungs', 'Hand', 'Heel', 'BloodPressure', 'Hearing', 'Cognitive']:
            ### READ EIDS
            ## Compute list_test_folds_eid, and list_train_folds_eid
            # if organ in ['Brain', 'Carotids', 'Heart'] or view in ['TrailMaking', 'MatrixPatternCompletion', 'TowerRearranging', 'SymbolDigitSubstitution', 'PairedAssociativeLearning', 'AllBiomarkers']:
            #     list_test_folds = [pd.read_csv(path_eid_split + 'instances23_eids_%s.csv' % fold).columns.astype(int) for fold in range(splits)]
            # elif organ in ['Eyes', 'ECG', 'Vascular', 'Anthropometry', 'Urine', 'BloodB', 'BloodC', 'Lungs', 'Hand', 'Heel', 'BloodPressure', 'Hearing'] :
            #     list_test_folds = [pd.read_csv(path_eid_split + '%s_eids_%s.csv' % (organ, fold)).columns.astype(int) for fold in range(splits)]
            # elif organ in ['Cognitive'] and view in ['ReactionTime', 'PairsMatching', 'ProspectiveMemory', 'NumericMemory']:
            #     list_test_folds = [pd.read_csv(path_eid_split + '%s_%s_eids_%s.csv' % (organ, view, fold)).columns.astype(int) for fold in range(splits)]
            #

        #list_test_folds = [pd.read_csv(path_eid_split + '/All_eids_%s.csv' % fold).columns.astype(int) for fold in range(splits)]
        #list_test_folds_eid = [elem[elem.isin(eids)].values for elem in list_test_folds]
        list_test_folds = pd.read_csv(path_eid_split + '/All_eids.csv').astype(int).set_index('eid')
        list_test_folds_eid_ = list_test_folds.loc[eids]
        list_test_folds_eid = [list_test_folds_eid_[list_test_folds_eid_['fold'] == fold_].index for fold_ in range(splits)]
        if fold is not None:
            list_train_folds_eid = np.concatenate(list_test_folds_eid[:fold] + list_test_folds_eid[fold + 1:])
            list_train_fold_id = X.index[X.eid.isin(list_train_folds_eid)]
        else :
            list_train_fold_id = None


        list_test_folds_id = [X.index[X.eid.isin(list_test_folds_eid[elem])].values for elem in range(len(list_test_folds_eid))]
        print('list_test_folds_id', list_test_folds_id)
        return list_train_fold_id, list_test_folds_id

    def optimize_hyperparameters_fold_(self, X, y, scoring, fold, organ, view, transformation):
        """
        input X  : dataframe with features + eid
        input y : dataframe with target + eid
        """

        if self.inner_splits != self.outer_splits - 1:
            raise ValueError('n_inner_splits should be equal to n_outer_splits - 1 ! ')

        list_train_fold_id, list_test_folds_id = self.create_list_test_train_folds(X, y, fold, organ, view, transformation)
        print(list_train_fold_id, list_test_folds_id)
        #
        test_fold = (fold + 1)%self.outer_splits
        val_fold = fold
        index_train, index_test, index_val = list_train_fold_id, list_test_folds_id[test_fold], list_test_folds_id[val_fold]
        ## Create train train indexes ie 8 fold dataset
        if test_fold < val_fold :
            index_train_train = list_test_folds_id[test_fold + 1 : val_fold]
        else :
            if test_fold == self.outer_splits - 1 :
                index_train_train = list_test_folds_id[ : val_fold]
            else :
                index_train_train = list_test_folds_id[ : val_fold] + list_test_folds_id[test_fold + 1 : ]
        index_train_train = np.concatenate(index_train_train)
        list_test_folds_id = list_test_folds_id[:fold] + list_test_folds_id[fold + 1 :]
        X = X.drop(columns = ['eid'])
        y = y.drop(columns =['eid'])

        ## Create Datasets :
        X_train, X_test, X_val, y_train, y_test, y_val = X.loc[index_train], X.loc[index_test], X.loc[index_val], y.loc[index_train], y.loc[index_test], y.loc[index_val]
        X_train_train, y_train_train = X.loc[index_train_train], y.loc[index_train_train]
        print("X_train", X_train)
        print("X_test", X_test)
        print("X_val", X_val)

        ## Create custom Splits
        list_test_folds_id_index = [np.array([X_train.index.get_loc(elem) for elem in list_test_folds_id[fold_num]]) for fold_num in range(len(list_test_folds_id))]
        test_folds = np.zeros(len(X_train), dtype = 'int')
        for fold_count in range(len(list_test_folds_id)):
            print(list_test_folds_id_index[fold_count])
            test_folds[list_test_folds_id_index[fold_count]] = fold_count
        inner_cv = PredefinedSplit(test_fold = test_folds)
        y_train = y_train.values
        y_train_train = y_train_train.values
        y_train = y_train.reshape(y_train.shape[0], )
        y_train_train = y_train_train.reshape(y_train_train.shape[0], )
        if y_train.dtype == 'O':
            y_train = y_train.astype('?, <f8')
            y_train_train = y_train_train.astype('?, <f8')
            print(y_train)
        ## RandomizedSearch :
        if self.model_validate == 'RandomizedSearch':
            clf = RandomizedSearchCV(estimator = self.get_model(), param_distributions = self.get_hyper_distribution(), cv = inner_cv, n_jobs = -1, scoring = scoring, verbose = 10, n_iter = self.n_iter, return_train_score = False)
            clf.fit(X_train.values, y_train.values)
            best_estim = copy.deepcopy(clf.best_estimator_)
            best_params = copy.deepcopy(clf.best_params_)
            results = clf.cv_results_
            results = pd.DataFrame(data = results)

            params_per_fold_opt = results.params[results[['split%s_test_score' % elem for elem in range(self.inner_splits)]].idxmax()]
            params_per_fold_opt = dict(params_per_fold_opt.reset_index(drop = True))

        ## HyperOpt :
        elif self.model_validate == 'HyperOpt':
            def objective(hyperparameters):
                estimator_ = self.get_model()
                ## Set hyperparameters to the model :
                for key, value in hyperparameters.items():
                    if hasattr(estimator_, key):
                        setattr(estimator_, key, value)
                    else :
                        continue
                pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', estimator_)])
                scores = cross_validate(pipeline, X_train.values, y_train, scoring = scoring, cv = inner_cv, verbose = 10, n_jobs = -1 )
                return {'status' : STATUS_OK, 'loss' : -scores['test_score'].mean(), 'attachments' :  {'split_test_scores_and_params' :(scores['test_score'], hyperparameters)}}
            space = self.get_hyper_distribution()
            trials = Trials()
            best = fmin(objective, space, algo = tpe.suggest, max_evals=self.n_iter, trials = trials)
            best_params = space_eval(space, best)
            ## Recreate best estim :
            estim = self.get_model()
            estim_train = self.get_model()
            for key, value in best_params.items():
                if hasattr(estim, key):
                    setattr(estim, key, value)
                elif hasattr(estim_train, key):
                    setattr(estim_train, key, value)
                else :
                    continue
            pipeline_best_on_train_and_val = Pipeline([('scaler', StandardScaler()), ('estimator', estim)])
            pipeline_best_on_train = Pipeline([('scaler', StandardScaler()), ('estimator', estim_train)])
            pipeline_best_on_train_and_val.fit(X_train.values, y_train)
            pipeline_best_on_train.fit(X_train_train.values, y_train_train)



        y_predict_val = pipeline_best_on_train.predict(X_val)
        y_predict_test = pipeline_best_on_train_and_val.predict(X_test)
        y_predict_train = pipeline_best_on_train.predict(X_train)
        df_test = pd.DataFrame(data = {'id' : index_test, 'outer_fold' : fold, 'pred' : y_predict_test} )
        df_train = pd.DataFrame(data = {'id' : index_train, 'outer_fold' : fold, 'pred' : y_predict_train })
        df_val = pd.DataFrame(data = {'id' : index_val, 'outer_fold' : fold, 'pred' : y_predict_val} )


        best_params_flat = []
        for elem in best_params.values():
    	    if type(elem) == tuple:
                for sub_elem in elem:
            	    best_params_flat.append(sub_elem)
    	    else:
                best_params_flat.append(elem)
        self.best_params = best_params_flat

        return df_test, df_val, df_train


    def Create_feature_imps_for_estimator(self, best_estim, X, y, scoring, columns):
        print(best_estim)
        print(best_estim['estimator'])
        if self.model_name == 'ElasticNet':
            features_imp = best_estim['estimator'].coef_
        elif self.model_name == 'RandomForest':
            features_imp = best_estim['estimator'].feature_importances_
        elif self.model_name == 'GradientBoosting':
            features_imp = best_estim['estimator'].feature_importances_
        elif self.model_name == 'Xgboost':
            features_imp = best_estim['estimator'].feature_importances_
        elif self.model_name == 'LightGbm':
            features_imp = best_estim['estimator'].feature_importances_ / np.sum(best_estim['estimator'].feature_importances_)
        elif self.model_name == 'NeuralNetwork'  or self.model_name == 'CoxRf' or self.model_name == 'CoxGbm':
            list_scores = []
            if scoring == 'r2':
                score_max = r2_score(y, best_estim.predict(X.values))
            elif scoring == 'f1' :
                score_max = f1_score(y, best_estim.predict(X.values))
            else :
                score_max = best_estim.score(X.values, y)
            for column in columns :
                X_copy = copy.deepcopy(X)
                X_copy[column] = np.random.permutation(X_copy[column])
                if scoring == 'r2':
                    score = r2_score(y, best_estim.predict(X_copy.values))
                elif scoring == 'f1' :
                    score = f1_score(y, best_estim.predict(X_copy.values))
                else :
                    score = best_estim.score(X.values, y)
                list_scores.append(score_max - score)
            features_imp = list_scores
        elif self.model_name == 'CoxPh':
            features_imp = best_estim['estimator'].coef_
        else :
            raise ValueError('Wrong model name')
        return features_imp

    def features_importance_(self, X, y, scoring, organ, view, transformation):
        list_train_fold_id, list_test_folds_id = self.create_list_test_train_folds(X = X, y = y, fold = None, organ = organ, view = view, transformation = transformation)
        X = X.drop(columns = ['eid'])
        y = y.drop(columns =['eid'])
        columns = X.columns
        y = np.ravel(y.values)

        list_test_folds_id_index = [np.array([X.index.get_loc(elem) for elem in list_test_folds_id[fold_num]]) for fold_num in range(len(list_test_folds_id))]
        test_folds = np.zeros(len(X), dtype = 'int')
        for fold_count in range(len(list_test_folds_id)):
            test_folds[list_test_folds_id_index[fold_count]] = fold_count

        cv = PredefinedSplit(test_fold = test_folds)

        if self.model_name == 'Correlation':
            matrix = np.zeros((self.inner_splits, len(columns)))
            for fold, indexes in enumerate(list(cv.split(X.values, y))):
                train_index, test_index = indexes
                X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y[train_index], y[test_index]
                column_ = columns[0]
                list_corr = [stat.pearsonr(X_test[column].values, y_test)[0] for column in columns]
                matrix[fold] = list_corr
            self.features_imp_sd = np.std(matrix, axis = 0)
            self.features_imp_mean = np.mean(matrix, axis = 0)
            self.features_imp = [stat.pearsonr(X[column].values, y)[0] for column in columns]
        ## Else More Complex Models :
        else :
            if self.model_validate == 'RandomizedSearch':
                clf = RandomizedSearchCV(estimator = self.get_model(), param_distributions = self.get_hyper_distribution(), cv = cv, n_jobs = -1, scoring = scoring, n_iter = self.n_iter)
                clf.fit(X.values, y)
                best_estim = clf.best_estimator_
            elif self.model_validate == 'HyperOpt':
                ## Objective which saves last best estimators accross all folds
                trials = Trials()
                def objective(hyperparameters):
                    estimator_ = self.get_model()
                    ## Set hyperparameters to the model :
                    for key, value in hyperparameters.items():
                        if hasattr(estimator_, key):
                            setattr(estimator_, key, value)
                        else :
                            continue
                    pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', estimator_)])
                    scores = cross_validate(pipeline, X.values, y, scoring = scoring, cv = cv, verbose = 10, return_estimator = True)
                    if hasattr(trials, 'attachments') and 'ATTACH::0::best_score' in trials.attachments.keys():
                        old_best_score = trials.attachments['ATTACH::0::best_score']
                        if scores['test_score'].mean() > old_best_score:
                            trials.attachments['ATTACH::0::best_models'] = scores['estimator']
                            trials.attachments['ATTACH::0::best_score'] = scores['test_score'].mean()
                        return {'status' : STATUS_OK, 'loss' : -scores['test_score'].mean()}
                    else :
                        return {'status' : STATUS_OK, 'loss' : -scores['test_score'].mean(), 'attachments' :  {'best_models' : scores['estimator'], 'best_score' : scores['test_score'].mean()}}
                ## Create search space
                space = self.get_hyper_distribution()

                ## Optimize and recover best_params and best estimators ( for sd )
                best = fmin(objective, space, algo = tpe.suggest, max_evals=self.n_iter, trials = trials)
                best_params = space_eval(space, best)
                print(trials.attachments)
                best_estimators = trials.attachments['ATTACH::0::best_models']

                matrix_std = np.zeros((self.inner_splits, len(columns)))
                ## Recreate best estim :
                estim = self.get_model()
                for key, value in best_params.items():
                    if hasattr(estim, key):
                        setattr(estim, key, value)
                    else :
                        continue
                pipeline_best = Pipeline([('scaler', StandardScaler()), ('estimator', estim)])
                pipeline_best.fit(X.values, y)

            self.features_imp = self.Create_feature_imps_for_estimator(best_estim = pipeline_best, X = X, y = y, scoring = scoring, columns = columns)
            matrix = np.zeros((self.inner_splits, len(columns)))
            for fold, indexes in enumerate(list(cv.split(X.values, y))):
                train_index, test_index = indexes
                X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y[train_index], y[test_index]
                list_corr =  self.Create_feature_imps_for_estimator(best_estim = best_estimators[fold], X = X_test, y = y_test, scoring = scoring, columns = columns)
                matrix[fold] = list_corr
            self.features_imp_sd = np.std(matrix, axis = 0)
            self.features_imp_mean = np.mean(matrix, axis = 0)
