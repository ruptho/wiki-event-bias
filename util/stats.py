from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests


def make_contingency_table(df, col_rows, col_cols):
    return df.groupby([col_rows])[col_cols].value_counts().rename('value').sort_index().reset_index().pivot(
        index=col_rows, columns=col_cols, values='value')


def calc_and_print_chi2(df_ct, text='Chi2'):
    chi2, p_value, dof, expected = chi2_contingency(df_ct)
    print(f'{text} {p_value:.5f} (val={chi2:.3f}, dof={dof})')
    return p_value, expected, df_ct


def chi2_pairwise(df_ct, col_x, pairwise=True):
    # Perform pairwise chi-squared tests
    p_values, expected, cont_tables = {}, {}, {}
    p, exp, _ = calc_and_print_chi2(df_ct, text='Chi2 Overall')
    p_values[(col_x, 'all')] = p
    expected[(col_x, 'all')] = exp
    cont_tables[(col_x, 'all')] = df_ct
    if len(df_ct) > 2 and pairwise:
        for pair1, pair2 in list(combinations(df_ct.index, 2)):
            print(f'{pair1} vs. {pair2}')
            p, exp, _ = calc_and_print_chi2(df_ct.loc[[pair1, pair2]])
            p_values[(col_x, pair1, pair2)] = p
            expected[(col_x, pair1, pair2)] = exp
            cont_tables[(col_x, pair1, pair2)] = df_ct.loc[[pair1, pair2]]

    return p_values, expected, cont_tables


def perform_chi2_2d(df_crawled, col_rows, col_cols, pairwise=True):
    df_ct = make_contingency_table(df_crawled, col_rows, col_cols)
    p_values, exp, ct = chi2_pairwise(df_ct, col_rows, pairwise)
    return p_values, exp, ct


def perform_chi2_3d(df_crawled, col_group, col_x, col_y, pairwise=True):
    p_values, expected, cont_tables = {}, {}, {}
    for g, df_g in df_crawled.groupby(col_group):
        print(f'Test separation by {g}')
        df_ct = make_contingency_table(df_g, col_x, col_y)
        ps, exts, ct = chi2_pairwise(df_ct, g, pairwise=pairwise)
        p_values.update(ps)
        expected.update(exts)
        cont_tables.update(ct)
    return p_values, expected, cont_tables


# statistical significance difference across regions?
def calc_catval_array(df_data, cat_col='gni_region', val_col='GDP_pc_log_SHAP'):
    return [df[val_col].values for cat, df in df_data.groupby(cat_col)]


def pairwise_mwu_test(df, categories, cat_column, val_column):
    df_test = pd.DataFrame(index=categories, columns=categories)
    for comb1, comb2 in combinations(categories, 2):
        df_test[comb1][comb2] = mannwhitneyu(df[df[cat_column] == comb1][val_column],
                                             df[df[cat_column] == comb2][val_column]).pvalue
        df_test[comb2][comb1] = np.nan
        df_test[comb2][comb2] = np.nan
        df_test[comb1][comb1] = np.nan
    return df_test


def pairwise_mwu_shap(dict_df, val_col, group_col, sep_col, do_mwu=False):
    for model, df_model in dict_df.items():
        print('============', model)
        print(f'Kruskal Wallis for diff in {val_col} in {group_col} across {sep_col} ')
        catvals = calc_catval_array(df_model, cat_col=sep_col, val_col=val_col)
        print(kruskal(*catvals))

        if do_mwu:
            print(pairwise_mwu_test(df_model, df_model[sep_col].unique(), sep_col, val_col + '_SHAP'))
        for group, df in df_model.groupby(group_col):
            calcres = calc_catval_array(df, cat_col=sep_col, val_col=val_col + '_SHAP')
            print('     ', group, kruskal(*calcres))
            if do_mwu:
                print(pairwise_mwu_test(df_model, df_model[sep_col].unique(), sep_col, val_col + '_SHAP'))


def three_way_mwu(df, y_col, group_cols=['code', 'cat'], comp_col='gni_region', adjust='fdr_bh'):
    # Grouping the DataFrame by specified group columns
    grouped = df.groupby(group_cols)

    # Creating an empty dictionary to store the matrices
    result_matrices = {}
    all_regions = list(df[comp_col].unique())

    # Performing Mann-Whitney U test for each group
    for group_keys, group in grouped:
        num_regions = len(all_regions)

        # Creating an empty matrix to store the p-values
        p_matrix = np.empty((num_regions, num_regions))

        # Performing Mann-Whitney U test for each pair of regions
        for i in range(num_regions - 1):
            for j in range(i + 1, num_regions):
                region1 = all_regions[i]
                region2 = all_regions[j]
                data1 = group.loc[group[comp_col] == region1, y_col]
                data2 = group.loc[group[comp_col] == region2, y_col]

                # Performing the Mann-Whitney U test if both groups have non-missing data
                if len(data1) > 0 and len(data2) > 0:
                    _, p_value = mannwhitneyu(data1, data2)
                    # p_value = np.round(p_value, 3)
                    # if region1 == region2:
                    #    print(region1, region2, p_value)
                else:
                    p_value = np.NaN
                p_matrix[j, i] = p_value
                p_matrix[i, j] = np.NaN
            p_matrix[i, i] = np.NaN

        # Storing the matrix in the dictionary
        result_matrices[group_keys] = p_matrix

    # Creating a DataFrame for each matrix and storing them in a list
    output_dfs = []
    for key, matrix in result_matrices.items():
        group_values = list(key) if not isinstance(key, str) else [key]
        matrix_df = pd.DataFrame(matrix, index=all_regions, columns=all_regions)
        matrix_df.index.name = comp_col
        matrix_df.columns.name = comp_col
        matrix_df[group_cols] = group_values
        output_dfs.append(matrix_df.reset_index())

    # Concatenating the DataFrames in the list into a single DataFrame
    results_df = pd.concat(output_dfs, ignore_index=True)

    # Reordering the columns
    results_df = results_df[group_cols + [comp_col] + all_regions]

    if adjust:
        # Adjusting p-values using Benjamini-Hochberg procedure
        p_values = results_df[all_regions].values
        p_values_no_nan = p_values[~np.isnan(p_values)]
        p_adjusted = np.round(multipletests(p_values_no_nan, method=adjust)[1], 3)

        # Putting the adjusted p-values back in the matrix
        p_adjusted_with_nan = np.empty(p_values.shape)
        p_adjusted_with_nan.fill(np.NaN)
        p_adjusted_with_nan[~np.isnan(p_values)] = p_adjusted
        results_df[all_regions] = p_adjusted_with_nan

    return results_df
