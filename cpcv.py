
from itertools import combinations
import pandas as pd
import lightgbm as lgb
import csv
import numpy as np
from ast import literal_eval
import random
from collections import Counter, defaultdict
import os
import ast
import matplotlib.pyplot as plt
from util.data import read_training_data
from loguru import logger
import optuna

NUMBER_OF_GROUPS = 6
path_to_numerai_training_data = ''
TARGET_NAME = "target"
PREDICTION_NAME = "prediction"
MAX_EVALS = 2
out_file = 'cpcv_results.csv'
SEED = 999


def get_cat_model(df, hyperparameters, feature_columns, train_eras):
    hyperparameters['verbose'] = -1
    train = df[df['G'].isin(train_eras)]
    dtrain = lgb.Dataset(get_X_array(train, feature_columns), get_Y_array(train), feature_name=feature_columns, free_raw_data=False)
    np.random.seed(SEED)
    model = lgb.train(hyperparameters, dtrain, num_boost_round=3)
    return model

# Submissions are scored by spearman correlation
def score(df):
    # method="first" breaks ties based on order in array
    return np.corrcoef(
        df[TARGET_NAME],
        df[PREDICTION_NAME].rank(pct=True, method="first")
    )[0,1]

def get_features(dataframe, unimportant=set()):
    feature_columns = dataframe.columns
    columns_to_remove = []
    for feature in feature_columns:
        if feature.startswith('target') or feature in ['G', 'prediction', 'id', 'era', 'data_type', 'prediction_kazutsugi', 'erano', 'target_custom', 'preds_neutralized'] or feature in unimportant:
            columns_to_remove.append(feature)
    feature_columns = feature_columns.drop(columns_to_remove)
    return sorted(feature_columns)

def get_X_array(df, feature_columns):
    return df.loc[:, feature_columns].values

def get_Y_array(df):
    return df.loc[:, 'target'].values

def generate_cv_combinations():
    a = list(range(1, NUMBER_OF_GROUPS+1, 1))
    cv_custom_comb = []
    test_groups = list(combinations(a, 2))
    path_dict = dict()
    for i in range(1, NUMBER_OF_GROUPS+1, 1):
        path_dict[i] = 1
    for g1, g2 in test_groups:
        out = (g1, path_dict[g1]), (g2, path_dict[g2])
        path_dict[g1] += 1
        path_dict[g2] += 1
        cv_custom_comb.append(out)
    return cv_custom_comb


def cross_validation(df, hyperparameters, iteration, best_score, feature_columns=False):
    logger.debug('Hyperparameters: {}', hyperparameters)
    if not feature_columns:
        feature_columns = get_features(df)
    path_dict = defaultdict(list)
    num_paths = 5
    num_groups = 6
    checked_path = 0
    shr_v_list = [0] * num_paths
    mn_v_list = [0] * num_paths
    min_v_list = [0] * num_paths
    threshold_value = best_score.mean() - np.std(best_score) * 3
    logger.debug('Threshold_value {}', threshold_value)
    combinations = generate_cv_combinations()
    for combs, zsp in enumerate(combinations):
        test_eras = set()
        train_eras = set()
        for i in zsp:
            test_eras.add(i[0])
        for i in range(1, num_groups+1, 1):
            if i not in test_eras:
                train_eras.add(i)
        logger.debug('Training Model for zsp {}, {} of {}', zsp, combs, len(combinations))
        model = get_cat_model(df, hyperparameters, feature_columns, train_eras)
        for i in zsp:
            idx_test = df['G'] == i[0]
            df.loc[idx_test, PREDICTION_NAME] = model.predict(get_X_array(df.loc[idx_test, feature_columns], feature_columns))
            path_dict[i[1]].append(df.loc[idx_test, [TARGET_NAME, PREDICTION_NAME, 'era']].copy())
        for path_num in range(checked_path+1, num_paths+1, 1):
            if len(path_dict[path_num]) >= num_groups:
                test_df6 = pd.concat(path_dict[path_num])
                validation_correlations = test_df6.groupby("era").apply(score)
                mn_v = validation_correlations.mean()
                std_v = validation_correlations.std()
                min_v = validation_correlations.min()
                shr_v = mn_v / std_v
                shr_v_list[path_num-1] = shr_v
                mn_v_list[path_num-1] = mn_v
                min_v_list[path_num-1] = min_v
                del path_dict[path_num]
                checked_path = path_num
        if checked_path > 0 and any(z1 < threshold_value for z1 in mn_v_list[:checked_path]):
            logger.debug('Pruning because path checked and validation corr mean lower than threshold')
            return [shr_v_list, mn_v_list, min_v_list, hyperparameters, iteration]
    logger.debug('Finished all combinations')
    return [shr_v_list, mn_v_list, min_v_list, hyperparameters, iteration]

class COCVObjective(object):

    def __init__(self, out_file, df):
        self.out_file = out_file
        self.df = df
       
    def __call__(self, trial):
        best_score = np.array([0] * 5)
        random_results = pd.read_csv(self.out_file)
        if len(random_results) > 0:
            random_results['score_list'] = random_results['score_list'].apply(lambda x: np.array(literal_eval(x)))
            random_results['score_mean'] = random_results['score_list'].apply(lambda x: np.mean(x))
            best_score = random_results.sort_values(by='score_mean',ascending=False)['score_list'].values[0]
        param_grid = {
            'boosting_type': ['gbdt', ],
            'learning_rate': trial.suggest_float('learning_rate',0.001,0.3, log=True), #list(np.logspace(np.log10(0.001), np.log10(0.3), base = 10, num = 1000)),
            'metric': ['rmse', ],
        }
        feature_columns = get_features(self.df)
        path_dict = defaultdict(list)
        num_paths = 5
        num_groups = 6
        checked_path = 0
        shr_v_list = [0] * num_paths
        mn_v_list = [0] * num_paths
        min_v_list = [0] * num_paths
        threshold_value = best_score.mean() - np.std(best_score) * 3
        combinations = generate_cv_combinations()
        for combs, zsp in enumerate(combinations):
            test_eras = set()
            train_eras = set()
            for i in zsp:
                test_eras.add(i[0])
            for i in range(1, num_groups+1, 1):
                if i not in test_eras:
                    train_eras.add(i)
            logger.debug('Training Model for zsp {}, {} of {}', zsp, combs, len(combinations))
            model = get_cat_model(self.df, param_grid, feature_columns, train_eras)
            for i in zsp:
                idx_test = self.df['G'] == i[0]
                self.df.loc[idx_test, PREDICTION_NAME] = model.predict(get_X_array(self.df.loc[idx_test, feature_columns], feature_columns))
                path_dict[i[1]].append(self.df.loc[idx_test, [TARGET_NAME, PREDICTION_NAME, 'era']].copy())
            for path_num in range(checked_path+1, num_paths+1, 1):
                if len(path_dict[path_num]) >= num_groups:
                    test_df6 = pd.concat(path_dict[path_num])
                    validation_correlations = test_df6.groupby("era").apply(score)
                    mn_v = validation_correlations.mean()
                    std_v = validation_correlations.std()
                    min_v = validation_correlations.min()
                    shr_v = mn_v / std_v
                    shr_v_list[path_num-1] = shr_v
                    mn_v_list[path_num-1] = mn_v
                    min_v_list[path_num-1] = min_v
                    del path_dict[path_num]
                    checked_path = path_num
            if checked_path > 0 and any(z1 < threshold_value for z1 in mn_v_list[:checked_path]):
                logger.debug('Pruning because path checked and validation corr mean lower than threshold')
                raise optuna.TrialPruned()
                #return [shr_v_list, mn_v_list, min_v_list, hyperparameters, iteration]
        logger.debug('Finished all combinations')
        eval_results = [shr_v_list, mn_v_list, min_v_list, param_grid, trial.number]
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow(eval_results)
        of_connection.close()
        return sum(mn_v_list) / len(mn_v_list)



#df1 = read_csv_custom(path_to_numerai_training_data)
df1 = read_training_data()
df1['erano'] = df1['era'].apply(lambda x: int(x[3:]))
df1['G'] = df1['erano'].apply(lambda x: ((x-1) // 20) + 1)
if not os.path.isfile(out_file):
    of_connection = open(out_file, 'w')
    writer = csv.writer(of_connection)
    # Write column names
    headers = ['sharpe_list', 'score_list', 'minval_list', 'params', 'iteration']
    writer.writerow(headers)
    of_connection.close()

study_name = 'CPCV_Example'
study = optuna.create_study(
    study_name=study_name, 
    storage="sqlite:///optuna/{}.db".format(study_name), 
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10, n_startup_trials=10), # this should have nothing to do 
    direction="maximize",
    load_if_exists=True)
study.optimize(COCVObjective(out_file,df1),
    n_trials=1000,
    n_jobs=1
    )

"""
random_results = random_search(df1, param_grid, out_file, MAX_EVALS)
# read in results to evaluate
random_results = pd.read_csv(out_file)
for cc in ['sharpe_list', 'score_list', 'minval_list']:
    lbl = cc.split('_')[0]
    random_results[cc] = random_results[cc].apply(lambda x: np.array(literal_eval(x)))
    random_results['%s_mean' % (lbl, )] = random_results['%s_list' % (lbl, )].apply(lambda x: np.mean(x))
    random_results['%s_std' % (lbl, )] = random_results['%s_list' % (lbl, )].apply(lambda x: np.std(x))
    random_results['%s_max' % (lbl, )] = random_results['%s_list' % (lbl, )].apply(lambda x: np.max(x))
    random_results['%s_min' % (lbl, )] = random_results['%s_list' % (lbl, )].apply(lambda x: np.min(x))
random_results['score_count'] = random_results['score_list'].apply(lambda x: sum(np.array(x) > 0))
random_results = random_results.sort_values(by='score_mean',ascending=False)
bestparams = ast.literal_eval(random_results['params'].values[0])
"""