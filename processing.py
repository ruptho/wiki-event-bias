import numpy as np
import pandas as pd


def build_regression_dataframe(df_crawled):
    d = {'pagetitle': df_crawled['pagetitle'],
         'views_before_sum': np.log1p(df_crawled['views_before_sum']),
         'views_7_sum': np.log1p(df_crawled['views_7_sum']),
         'diff_days': df_crawled['diff_days'].apply(lambda x: np.log(x + abs(min(df_crawled.diff_days)) + 1)),
         'economic_region': df_crawled['economic_region'],
         'cat': df_crawled['cat'],
         'planed': df_crawled['planed'],
         'surprising': df_crawled['surprising'],
         'factor': df_crawled['factor'],  # surprising or planned?
         'year': df_crawled['year'].astype(int),
         'code': df_crawled['code'],
         'bing_news': np.log1p(df_crawled['bing_hits']),
         'bing_attention': df_crawled.bing_hits.apply(lambda views: 'low' if views < 6 else 'high'),
         # this is probably not a good idea here: maybe get quantiles?
         'gdp_pc': np.log1p(df_crawled['GDP_pc']),
         'gdp_pc_raw': df_crawled['GDP_pc'],
         'gdp': np.log1p(df_crawled['GDP']),
         'gdp_raw': np.log1p(df_crawled['GDP']),
         'oecd': df_crawled['oecd'],
         'in_code_region': df_crawled['in_code_region'],  # factor, would also work with wikistats
         'in_code_lang': df_crawled['in_code_lang'],
         'edits_7_sum': np.log1p(df_crawled['edits_7_sum']),
         'gni_class': df_crawled['gni_class'],
         'gni_region': df_crawled['gni_region'],
         'population': np.log1p(df_crawled['population']),
         'views_diff_sum': np.log1p(df_crawled.views_7_sum - df_crawled.views_before_sum),
         'edits_diff_sum': np.log1p(df_crawled.edits_7_sum - df_crawled.edits_before_sum)
         }

    data = pd.DataFrame(data=d)
    # add views in year before
    wiki_views = pd.read_csv('viewsglobal/views_all.csv')
    wiki_views['year_after'] = wiki_views.date.str[:4].astype(int) + 1
    wiki_views_mean = np.log1p(wiki_views.groupby(['code', 'year_after']).mean()).rename({'views': 'mean_views_wiki'},
                                                                                         axis=1)
    data = data.merge(wiki_views_mean, left_on=['code', 'year'], right_on=['code', 'year_after'], how='left')
    return data


def filter_by_region_cat_code(df, gni_region, cat, code, n_largest=None, sorted=False):
    filtered = df[(df.gni_region == gni_region) & (df.cat == cat) & (df.code == code)]

    if sorted is not None:
        filtered = filtered.sort_values('views_7_sum', ascending=sorted)
    if n_largest is not None:
        filtered = filtered.nlargest(n_largest, 'views_7_sum')
    return filtered


def load_preprocessed_events(path='events/new/processed_manually_with_wikiviews.csv'):
    return pd.read_csv(path)


def compute_mean_and_variance(df: pd.DataFrame, group_col: str, val_col=('views_7_sum',)):
    group_col = group_col[0] if len(group_col) == 1 else group_col
    return df.groupby(group_col)[val_col].agg(['mean', 'var'])  # ({'mean': np.mean, 'var': np.var})


def calculate_uniqueness(df: pd.DataFrame, n_days, column):
    df_col_grouped = df.groupby(['code', column, 'event_date'])['event_id'].count().rename('n_articles').reset_index()
    df_col_grouped['event_date'] = pd.to_datetime(df_col_grouped.event_date)
    df_col_grouped['window_unique'] = df_col_grouped.event_date - pd.to_timedelta(f'{n_days} days')
    df_col_grouped[f'{column}_articles'] = df_col_grouped.apply(
        lambda row: np.sum(df_col_grouped[(df_col_grouped.code == row.code) & (
                df_col_grouped[column] == row[column]) & (df_col_grouped.event_date >= row.window_unique) & (
                                                  df_col_grouped.event_date < row.event_date)].n_articles),
        axis=1)
    df_col_grouped[f'{column}_articles_log'] = np.log1p(df_col_grouped[f'{column}_articles'])
    return df.merge(
        df_col_grouped[['code', column, 'event_date', f'{column}_articles', f'{column}_articles_log']],
        on=['code', column, 'event_date'], how='left')


def calculate_uniqueness_two_columns(df: pd.DataFrame, n_days, column1, column2):
    df_col_grouped = df.groupby(['code', column1, column2, 'event_date'])['event_id'].count().rename(
        'n_articles').reset_index()
    df_col_grouped['event_date'] = pd.to_datetime(df_col_grouped.event_date)
    df_col_grouped['window_unique'] = df_col_grouped.event_date - pd.to_timedelta(f'{n_days} days')
    df_col_grouped[f'{column1}_{column2}_articles'] = df_col_grouped.apply(
        lambda row: np.sum(df_col_grouped[(df_col_grouped.code == row.code) & (
                df_col_grouped[column1] == row[column1]) & (df_col_grouped[column2] == row[column2]) & (
                                                  df_col_grouped.event_date >= row.window_unique) & (
                                                  df_col_grouped.event_date < row.event_date)].n_articles), axis=1)
    df_col_grouped[f'{column1}_{column2}_articles_log'] = np.log1p(df_col_grouped[f'{column1}_{column2}_articles'])
    return df.merge(
        df_col_grouped[['code', column1, column2, 'event_date', f'{column1}_{column2}_articles',
                       f'{column1}_{column2}_articles_log']], on=['code', column1, column2, 'event_date'], how='left')
