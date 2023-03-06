import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, get_scorer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
import pickle

from xgboost import XGBClassifier, XGBRegressor

import itertools
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn import tree, metrics
from joblib import Parallel, delayed
from category_encoders import OneHotEncoder
from sklearn.base import clone

model_label_key = {'edited_rf': 'Random Forest',
                   'edited_svc': 'Support Vector Machine',
                   'edited_xgb': 'XGBoost',
                   'viewed_rf': 'Random Forest Classifier',
                   'viewed_svc': 'Support Vector Machine',
                   'viewed_xgb': 'XGBoost',
                   'views_rf': 'Random Forest Classifier',
                   'views_svc': 'Support Vector Machine',
                   'views_xgb': 'XGBoost',
                   'edits_rf': 'Random Forest',
                   'edits_svc': 'Support Vector Machine',
                   'edits_xgb': 'XGBoost'}

cont_cols_def = ['country_articles_log', 'population_log', 'cat_articles_log', 'GDP_pc_log', 'view_country_article_log',
                 'views_baseline_log']
fact_cols_def = ['gni_region', 'code', 'cat']


class ModelEvaluator:
    def __init__(self, model, df, target_col, metric, params=None, split_data_start=pd.to_datetime('2016-01-01'),
                 months_train=35, months_val=12, months_full_train=46, months_test=12,
                 cont_cols=None, factor_cols=None):
        self.model = model
        self.param_grid = params
        self.metric = metric
        self.factor_cols = factor_cols if factor_cols or len(factor_cols) == 0 else fact_cols_def
        self.cont_cols = cont_cols if cont_cols else cont_cols_def
        self.df_train, self.df_val = split_train_test(df, split_data_start, months_train, months_val)
        self.df_train, self.df_val, fact_cols_all, encoder = encode_train_test(self.df_train, self.df_val,
                                                                               self.factor_cols)
        self.df_cv = pd.concat([self.df_train, self.df_val])
        self.cv_split = PredefinedSplit(test_fold=[-1 if i_x in self.df_train.index else 0 for i_x in self.df_cv.index])
        self.df_train_full, self.df_test = split_train_test(df, split_data_start, months_full_train, months_test)
        self.df_train_full, self.df_test, fact_cols_all, self.encoder = encode_train_test(
            self.df_train_full, self.df_test, self.factor_cols)
        self.encoded_columns = self.cont_cols + fact_cols_all
        self.target_col = target_col
        self.gs_res = None
        self.cached_retrain = None
        self.rs_res = None

    def grid_search(self, params=None, metric=None, n_jobs=-1, verbose=0):
        self.gs_res = GridSearchCV(self.model, param_grid=params if params else self.param_grid,
                                   scoring=metric if metric else self.metric,
                                   verbose=verbose, n_jobs=n_jobs, cv=self.cv_split)
        return self.gs_res.fit(self.df_cv[self.encoded_columns], self.df_cv[self.target_col])

    def random_search(self, params=None, metric=None, n_iter=100, n_jobs=-1, verbose=0):
        self.rs_res = RandomizedSearchCV(self.model,
                                         param_distributions=params if params else self.param_grid,
                                         scoring=metric if metric else self.metric,
                                         verbose=verbose, n_jobs=n_jobs,
                                         n_iter=n_iter,
                                         cv=self.cv_split)
        return self.rs_res.fit(self.df_cv[self.encoded_columns], self.df_cv[self.target_col])

    def val_set_performance(self, df_train_full=None, metric=None, use_trained=True):
        if not isinstance(df_train_full, type(None)):
            df_eval = df_train_full
        else:
            df_eval = self.df_train_full

        scorer = get_scorer(metric if metric else self.metric)

        if use_trained:
            return scorer._score_func(df_eval[self.target_col],
                                      self.gs_res.best_estimator_.predict(df_eval[self.encoded_columns]))
        else:
            if not self.cached_retrain:
                cloned_est = clone(self.gs_res.best_estimator_)
                self.cached_retrain = cloned_est.fit(self.df_train[self.encoded_columns],
                                                     self.df_train[self.target_col])

            return scorer._score_func(self.df_val[self.target_col],
                                      self.cached_retrain.predict(self.df_val[self.encoded_columns]))

    def test_set_performance(self, df_test=None, metric=None):
        if not isinstance(df_test, type(None)):
            df_eval = df_test
        else:
            df_eval = self.df_test
        predictions = self.gs_res.best_estimator_.predict(df_eval[self.encoded_columns])
        scorer = get_scorer(metric if metric else self.metric)
        return scorer._score_func(df_eval[self.target_col], predictions)

    def test_set_performance_bs(self, df_test=None, metric=None, alpha=0.05, n=1000, n_jobs=144):
        if not isinstance(df_test, type(None)):
            df_eval = df_test
        else:
            df_eval = self.df_test

        def score_bs_sample(model, sampled_values, true_values, metric):
            predictions = model.predict(sampled_values)
            scorer = get_scorer(metric if metric else self.metric)
            score = scorer._score_func(true_values, predictions)
            return score

        # n_jobs = 1 if 'XGB' in str(type(self.gs_res.best_estimator_)) else n_jobs
        return bootstrap_test_performance(df_eval, self.gs_res.best_estimator_, self.encoded_columns, self.target_col,
                                          score_bs_sample, metric, alpha, n)

    def retrain_full(self, metric=None, n_jobs=None):
        self.df_full = self.get_full_dataset(decoded=False)
        # self.df_full, _, fact_cols_all, encoder = encode_train_test(self.df_train, self.df_val, self.factor_cols)
        self.full_model = clone(self.gs_res.best_estimator_)
        if n_jobs:
            self.full_model.n_jobs = n_jobs
        self.full_model.fit(self.df_full[self.encoded_columns], self.df_full[self.target_col])
        scorer = get_scorer(metric if metric else self.metric)
        score = scorer._score_func(self.df_full[self.target_col],
                                   self.full_model.predict(self.df_full[self.encoded_columns]))
        print(f'{metric if metric else self.metric}: {score:.4f}')
        return score

    def load_grid_search_result(self, filename, path='crossval'):
        with open(f'{path}/{filename}.pkl', 'rb') as f:
            self.gs_res = pickle.load(f)

    def get_full_dataset(self, decoded=True, all_cols=False):
        df_full = pd.concat([self.df_train_full, self.df_test])
        cols = self.encoded_columns + [self.target_col] if not all_cols else df_full.columns
        return df_full[cols] if not decoded else self.encoder.inverse_transform(df_full)


def standardize_var(df, col):
    return (df[col] - np.mean(df[col])) / np.std(df[col], ddof=1)


def transform_vars_for_regression(df_reg):
    df_reg['GDP_pc_z'] = standardize_var(df_reg, 'GDP_pc')

    df_reg['gdp_z'] = standardize_var(df_reg, 'GDP')
    df_reg['pop_z'] = standardize_var(df_reg, 'population')
    df_reg['views_baseline_z'] = standardize_var(df_reg, 'views_baseline')
    df_reg['view_country_article_z'] = standardize_var(df_reg, 'view_country_article')
    df_reg['bing_hits_z'] = standardize_var(df_reg, 'bing_hits')
    df_reg['worldwide'] = df_reg.code.apply(lambda c: (c == 'en') or (c == 'es'))
    df_reg['view_country_article_log'] = np.log1p(df_reg.view_country_article)
    df_reg['views_baseline_log'] = np.log1p(df_reg.views_baseline)
    # df_reg['views_before_sum_log'] = np.log1p(df_reg.views_before_sum)
    df_reg['bing_hits_log'] = np.log1p(df_reg.bing_hits)
    df_reg['GDP_pc_log'] = np.log1p(df_reg.GDP_pc)
    df_reg['GDP_log'] = np.log1p(df_reg.GDP)
    df_reg['GDP_pc_log_z'] = standardize_var(df_reg, 'GDP_log')

    df_reg['population_log'] = np.log1p(df_reg.population)
    df_reg['population_z'] = standardize_var(df_reg, 'population')
    # df_reg['views_before_log'] = np.log1p(df_reg.views_before_sum)
    # df_reg['views_before_z'] = standardize_var(df_reg, 'views_before_sum')
    # df_reg['planned'] = df_reg.planed == 'planed'
    # df_reg['breaking'] = df_reg.surprising == 'surprising'
    # page_creation = datetime.strptime('2020-10-31T23:59:59','%Y-%m-%dT%H:%M:%S')
    # event_date = datetime.strptime('2020-11-T00:00:01','%Y-%m-%dT%H:%M:%S')
    return df_reg


def prepare_experiment_dfs(path_data):
    df_crawled = pd.read_csv(path_data).drop_duplicates()
    df_crawled.event_date = pd.to_datetime(df_crawled.event_date)
    df_crawled['noticed'] = (df_crawled.views_7_sum > 10).astype(int)
    df_crawled['edited'] = (df_crawled.edits_7_sum > 0).astype(int)
    df_crawled = transform_vars_for_regression(df_crawled)

    mask_events = (df_crawled.year > 2015) & ~((df_crawled.event_date.dt.month == 1) &
                                               (df_crawled.event_date.dt.day == 1)) & (df_crawled.cat != 'undefined')
    df_reg = df_crawled[mask_events & df_crawled.noticed].copy()
    df_reg = transform_vars_for_regression(df_reg)
    df_reg['views_7_sum_log'] = np.log1p(df_reg.views_7_sum)

    df_editreg = df_crawled[mask_events & df_crawled.edited].copy()
    df_editreg = transform_vars_for_regression(df_editreg)
    df_editreg['edits_7_sum_log'] = np.log1p(df_editreg.edits_7_sum)

    df_class = df_crawled[mask_events].copy()
    df_class = transform_vars_for_regression(df_class)
    return df_crawled, df_reg, df_editreg, df_class


def bootstrap_test_performance(test_set: pd.DataFrame, model, ind_columns, dep_column, scorer_func, metric,
                               alpha=0.05, n=1000):
    # pivotal/empirical bootstrap method
    boot_samples = [test_set.sample(len(test_set), replace=True) for _ in range(n)]
    metric_test_set = scorer_func(model, test_set[ind_columns].values if ind_columns else test_set,
                                  test_set[dep_column].values, metric)
    metrics_sampled = np.sort(np.array([scorer_func(model, df_sample[ind_columns].values if ind_columns else df_sample,
                                                    df_sample[dep_column].values, metric) for df_sample in
                                        boot_samples]))

    #  par_res = Parallel(n_jobs=n_jobs)(delayed(scorer_func)(
    #         model, df_sample[ind_columns].values if ind_columns else df_sample, df_sample[dep_column].values, metric)
    #                                       for df_sample in boot_samples)
    return metric_test_set, \
           (2 * metric_test_set - np.percentile(metrics_sampled, (1 - alpha / 2) * 100),
            2 * metric_test_set - np.percentile(metrics_sampled, (alpha / 2) * 100))


def split_train_test(df, date, train_months, pred_months):
    split_date = jump_months_timedelta(date, train_months)
    to_date = jump_months_timedelta(split_date, pred_months)
    # print(date, split_date, to_date)
    df_train, df_test = split_set(df, date, split_date, to_date)
    return df_train, df_test


def split_set(df, from_date, split_date, to_date):
    mask_split = (df.event_date < pd.to_datetime(split_date))
    mask_from = (df.event_date >= pd.to_datetime(from_date))
    mask_to = (df.event_date < pd.to_datetime(to_date))
    df_train = df[mask_from & mask_split]
    df_test = df[~mask_split & mask_to]
    return df_train, df_test


def encode_train_test(df_train, df_test, factor_cols):
    df_test, fact_cols_all, encoder = encode_cols(df_test, factor_cols)
    df_train, fact_cols_all, encoder = encode_cols(df_train, factor_cols)
    return df_train, df_test, fact_cols_all, encoder


def jump_months_timedelta(curr_date, months):
    # watch out if more than 15 month jump
    return (curr_date + pd.Timedelta(32 * months, unit='d')).replace(day=1)


def encode_cols(df, factor_cols):
    df_code = df.copy()
    encoder = None
    if factor_cols and len(factor_cols) > 0:
        encoder = OneHotEncoder(cols=factor_cols, use_cat_names=True).fit(df_code)
        df_code = encoder.transform(df_code)

    return df_code, [feature for feature in encoder.get_feature_names() if
                     feature not in df.columns] if encoder else [], encoder


def load_rf_and_xgb_models(df_class, df_viewreg, df_editreg, prefix='noreg_', factor_cols=['code', 'cat'],
                           months_train=35, months_val=12, months_full_train=46, months_test=12,
                           model_dir='crossval_results'):
    models = {
        f'{prefix}edited_xgb': (XGBClassifier(n_jobs=16), df_class),
        f'{prefix}edits_xgb': (XGBRegressor(n_jobs=16), df_editreg),
        f'{prefix}viewed_xgb': (XGBClassifier(n_jobs=16), df_class),
        f'{prefix}views_xgb': (XGBRegressor(n_jobs=16), df_viewreg),
        f'{prefix}edited_rf': (RandomForestClassifier(n_jobs=16), df_class),
        f'{prefix}edits_rf': (RandomForestRegressor(n_jobs=16), df_editreg),
        f'{prefix}viewed_rf': (RandomForestClassifier(n_jobs=16), df_class),
        f'{prefix}views_rf': (RandomForestRegressor(n_jobs=16), df_viewreg),
    }
    model_eval = {}
    for path, model_info in models.items():
        if model_info:
            print(f'{model_dir}/{path}')
            model_eval[path] = ModelEvaluator(model_info[0], model_info[1],
                                              'noticed' if 'viewed' in path else 'edited' if 'edited' in path else
                                              'edits_7_sum_log' if 'edits' in path else 'views_7_sum_log',
                                              'f1_micro' if ('viewed' in path) or (
                                                      'edited' in path) else 'neg_mean_squared_error',
                                              factor_cols=factor_cols,
                                              params=None, months_train=months_train, months_val=months_val,
                                              months_full_train=months_full_train, months_test=months_test)
            model_eval[path].load_grid_search_result(path, path=model_dir)
    return models, model_eval
