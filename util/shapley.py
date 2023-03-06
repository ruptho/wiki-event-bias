import numpy as np
import pandas as pd
import pickle


def get_indices(all_cols,
                baselines={'code_en': 4, 'cat_disaster': 4, 'gni_region_South Asia': 7}):
    cat_indices = []
    for baseline, cols_n in baselines.items():
        index_col = all_cols.index(baseline)
        cat_indices.append(list(range(index_col, index_col + cols_n)))
    return cat_indices


def compute_shap_values(predictor, df_train, df_vis, cols, cat_col_indices=[[]], is_tree=True):
    from acv_explainers import ACVTree
    acvtree = ACVTree(predictor, df_train[cols].values, C=cat_col_indices)
    forest_sv = acvtree.shap_values(df_vis[cols].values, C=cat_col_indices)
    df_acv_sv = pd.DataFrame(np.squeeze(forest_sv, axis=2), columns=cols)
    return df_acv_sv


def combine_categories_SHAP(shap_vals, col_baselines={'code': ['en', 'de', 'es', 'it'],
                                                      'cat': ['disaster', 'sports', 'politics', 'culture']},
                            mean=False):
    cat_indices = get_indices(shap_vals.feature_names,
                              {f'{key}_{vals[0]}': len(vals) for key, vals in col_baselines.items()})
    new_vals = []
    for i in range(len(shap_vals)):
        values, shapval = shap_vals.values[i], []
        for col_i, feature in enumerate(shap_vals.feature_names):
            val = values[col_i]
            for cat_index_list in cat_indices:
                if col_i in cat_index_list:
                    # print(feature, cat_index_list, values[cat_index_list], np.sum(values[cat_index_list]))
                    val = np.sum(values[cat_index_list]) if not mean else np.mean(values[cat_index_list])
            shapval.append(val)
        new_vals.append(shapval)
    return np.array(new_vals)


def load_shapval_results(shap_result_path):
    with open(shap_result_path, 'rb') as opf:
        shap_vals_acv, shap_vals, shap_vals_int = pickle.load(opf)
    return shap_vals, shap_vals_int, shap_vals_acv


def combine_categories_SHAP_all(shap_val_dict, col_baselines={'code': ['en', 'de', 'es', 'it'],
                                                              'cat': ['disaster', 'sports', 'politics',
                                                                      'culture']}):
    for key, shapvals in shap_val_dict.items():
        shap_val_dict[key].values = combine_categories_SHAP(shapvals, col_baselines)
    return shap_val_dict
