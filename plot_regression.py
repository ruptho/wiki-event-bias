import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from steinkasserer.regression import get_vals_for_coefficients, extract_coefficient_values_and_stderr, \
    is_param_categorical, extract_coefficient_values_and_stderr_single_code

colorblind_tol = ['#117733', '#88CCEE', '#E69F00', '#882255']
helper_langs = {"de": "German", "fr": "French", "it": "Italian", "en": "English"}
default_label_dict = {"code": "Language", "de": "German", "fr": "French", "it": "Italian", "en": "English",
                      "es": "Spanish", "gni_class": "Income", "gni_region": "Region", "H": "High", "L": "Low",
                      "LM": "Lower mid", "UM": "Upper mid", "cat": "Art.Category"}


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_label_if_in_dict(label_val, label_rename_dict):
    return label_val if (label_rename_dict is None) or (label_val not in label_rename_dict) else label_rename_dict[
        label_val]


def plot_regression_results_interactions(df_reg, reg_results, coefficients, coef_baselines, label_sort=None,
                                         cat_dict=None, cat_in_coeff='code', title='', figsize=(8, 8),
                                         x_limits=(-5.5, 5.5), label_rename_dict=None,
                                         include_counts=False) -> plt.Figure:
    all_coeffs = flatten([[coef] if (':' not in coef) else coef.split(':') for coef in coefficients])
    normal_coeffs = [coef for coef in coefficients if ':' not in coef]
    interaction_coeffs = [coef.split(':') for coef in coefficients if ':' in coef]
    vals_coefficients = get_vals_for_coefficients(df_reg, all_coeffs)
    vals_cats = df_reg[cat_in_coeff].unique()
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    rows_grid = sum([(len(vals_coefficients[coef]) - 1) if len(normal_coeffs) > 0 else 0 for coef in normal_coeffs]) + \
                sum([len(vals_coefficients[base_coef]) * len(vals_coefficients[int_coef]) \
                         if len(interaction_coeffs) > 0 else 0 for base_coef, int_coef in interaction_coeffs])
    single_box_ratio = 1 / rows_grid
    height_ratios = [single_box_ratio * (len(vals_coefficients[coef]) - 1) for coef in normal_coeffs] + [
        single_box_ratio * len(vals_coefficients[base_coef]) * len(vals_coefficients[int_coef])
        for base_coef, int_coef in interaction_coeffs]
    outer_grid = fig.add_gridspec(len(coefficients), 1, wspace=0.0, hspace=0.0, height_ratios=height_ratios)
    ylim_min, ylim_max, xlim_min, xlim_max = -1, 4, x_limits[0], x_limits[1]
    outer_grid_pos = outer_grid.get_grid_positions(fig)

    for i_coef, coef in enumerate(normal_coeffs + interaction_coeffs):
        is_int_coef = isinstance(coef, list) or isinstance(coef, tuple)
        bottom, top, left, right = outer_grid_pos[0][i_coef], outer_grid_pos[1][i_coef], outer_grid_pos[2][0], \
                                   outer_grid_pos[3][0]
        if is_int_coef:
            base_coef, int_coef = coef[0], coef[1]
            n_base_coeffs, n_int_coeffs = len(vals_coefficients[base_coef]), len(vals_coefficients[int_coef])
            coef_i_grid = outer_grid[i_coef].subgridspec(n_base_coeffs * n_int_coeffs, 1, wspace=0.0, hspace=0.0)
            inner_grid_pos = coef_i_grid.get_grid_positions(fig)
            base_coefficients_list = label_sort[base_coef] if label_sort is not None and base_coef in label_sort else \
                vals_coefficients[base_coef]
            int_coefficients_list = label_sort[int_coef] if label_sort is not None and int_coef in label_sort else \
                vals_coefficients[int_coef]

            for i_base_coef_val, base_coef_val in enumerate(base_coefficients_list):
                ax = None
                for i_int_coef_val, int_coef_val in enumerate(int_coefficients_list):
                    ax = setup_axis(fig, coef_i_grid, i_base_coef_val * n_int_coeffs + i_int_coef_val, xlim_min,
                                    xlim_max, ylim_min, ylim_max, get_label_if_in_dict(int_coef_val, label_rename_dict),
                                    (i_base_coef_val < n_base_coeffs - 1) or (i_int_coef_val < n_int_coeffs - 1))
                    # plot codes
                    coef_combo_counts = {
                        cat: len(df_reg[(df_reg[base_coef] == base_coef_val) & (df_reg[int_coef] == int_coef_val)
                                        & (df_reg[cat_in_coeff] == cat)]) for
                        cat in cat_dict} if include_counts else None
                    plot_separate_cats(vals_cats, reg_results, base_coef, base_coef_val, cat_in_coeff, coef_baselines,
                                       i_coef, i_base_coef_val, ax, cat_dict, title, int_coef, int_coef_val,
                                       i_int_coef_val, coef_combo_counts)

                    if i_coef == 0 and i_int_coef_val == 0 and i_base_coef_val == 0:
                        legend_elements = \
                            [Line2D([0], [1], color='black', lw=1, label='Significant (95% CI)', marker='.'),
                             Line2D([0], [1], color='black', lw=1, label='Not Significant (95% CI)', marker='x')]
                        ax.legend(handles=legend_elements, loc='center right')
                        ax.set_title(title, fontsize='x-large', pad=10)
                        # plot val and Cis in plot
                if i_base_coef_val < n_base_coeffs - 1:
                    ax.axhline(ylim_min, linewidth=3, color="k")

                bottom_i, top_i, left_i, right_i = inner_grid_pos[0][i_base_coef_val], \
                                                   inner_grid_pos[1][i_base_coef_val], \
                                                   inner_grid_pos[2][0], inner_grid_pos[3][0]
                # print(bottom_i, top_i, left_i, right_i, (n_int_coeffs / 2) * i_base_coef_val)
                y, x = top - ((top_i - bottom_i) * (n_int_coeffs / 2) * (i_base_coef_val + (i_base_coef_val + 1))), 0.07
                fig.text(x, y, get_label_if_in_dict(base_coef_val, label_rename_dict),
                         {'ha': 'center', 'va': 'center'}, rotation=90, fontsize='x-large')

            fig.text(right + 0.05, (bottom + top) / 2,
                     f'Baseline:\n{coef_baselines[base_coef]}:{coef_baselines[int_coef]}:{coef_baselines[cat_in_coeff]}',
                     {'ha': 'center', 'va': 'center'}, rotation=90,
                     fontsize='x-large')
            ax.set_xlabel('Coefficient Value ("Difference to Baseline")', fontsize='x-large')

    return fig


def plot_regression_results_interactions_from_dict(df_reg, dict_reg_results, coefficients, coef_baselines,
                                                   label_sort=None, cat_dict=None, cat_in_coeff='code', title='',
                                                   figsize=(8, 8),
                                                   x_limits=(-5.5, 5.5), label_rename_dict=None,
                                                   include_counts=False) -> plt.Figure:
    all_coeffs = flatten([[coef] if (':' not in coef) else coef.split(':') for coef in coefficients])
    normal_coeffs = [coef for coef in coefficients if ':' not in coef]
    interaction_coeffs = [coef.split(':') for coef in coefficients if ':' in coef]
    vals_coefficients = get_vals_for_coefficients(df_reg, all_coeffs)
    vals_cats = df_reg[cat_in_coeff].unique()
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    rows_grid = sum([(len(vals_coefficients[coef]) - 1) if len(normal_coeffs) > 0 else 0 for coef in normal_coeffs]) + \
                sum([len(vals_coefficients[base_coef]) * len(vals_coefficients[int_coef]) \
                         if len(interaction_coeffs) > 0 else 0 for base_coef, int_coef in interaction_coeffs])
    single_box_ratio = 1 / rows_grid
    height_ratios = [single_box_ratio * (len(vals_coefficients[coef]) - 1) for coef in normal_coeffs] + [
        single_box_ratio * len(vals_coefficients[base_coef]) * len(vals_coefficients[int_coef])
        for base_coef, int_coef in interaction_coeffs]
    outer_grid = fig.add_gridspec(len(coefficients), 1, wspace=0.0, hspace=0.0, height_ratios=height_ratios)
    ylim_min, ylim_max, xlim_min, xlim_max = -1, 4, x_limits[0], x_limits[1]
    outer_grid_pos = outer_grid.get_grid_positions(fig)

    for i_coef, coef in enumerate(normal_coeffs + interaction_coeffs):
        is_int_coef = isinstance(coef, list) or isinstance(coef, tuple)
        bottom, top, left, right = outer_grid_pos[0][i_coef], outer_grid_pos[1][i_coef], outer_grid_pos[2][0], \
                                   outer_grid_pos[3][0]
        if is_int_coef:
            base_coef, int_coef = coef[0], coef[1]
            n_base_coeffs, n_int_coeffs = len(vals_coefficients[base_coef]), len(vals_coefficients[int_coef])
            coef_i_grid = outer_grid[i_coef].subgridspec(n_base_coeffs * n_int_coeffs, 1, wspace=0.0, hspace=0.0)
            inner_grid_pos = coef_i_grid.get_grid_positions(fig)
            base_coefficients_list = label_sort[base_coef] if label_sort is not None and base_coef in label_sort else \
                vals_coefficients[base_coef]
            int_coefficients_list = label_sort[int_coef] if label_sort is not None and int_coef in label_sort else \
                vals_coefficients[int_coef]

            for i_base_coef_val, base_coef_val in enumerate(base_coefficients_list):
                ax = None
                for i_int_coef_val, int_coef_val in enumerate(int_coefficients_list):
                    ax = setup_axis(fig, coef_i_grid, i_base_coef_val * n_int_coeffs + i_int_coef_val, xlim_min,
                                    xlim_max, ylim_min, ylim_max, get_label_if_in_dict(int_coef_val, label_rename_dict),
                                    (i_base_coef_val < n_base_coeffs - 1) or (i_int_coef_val < n_int_coeffs - 1),
                                    i_int_coef_val == 0, ((i_base_coef_val + 1) % 2) == 0)
                    # plot codes
                    coef_combo_counts = {
                        cat: len(df_reg[(df_reg[base_coef] == base_coef_val) & (df_reg[int_coef] == int_coef_val)
                                        & (df_reg[cat_in_coeff] == cat)]) for
                        cat in cat_dict} if include_counts else None
                    plot_separate_cats_from_dict(dict_reg_results, base_coef, base_coef_val, cat_in_coeff,
                                                 coef_baselines, i_coef, i_base_coef_val, ax, cat_dict, title, int_coef,
                                                 int_coef_val, i_int_coef_val, coef_combo_counts)

                    if i_coef == 0 and i_int_coef_val == 0 and i_base_coef_val == 0:
                        legend_elements = \
                            [Line2D([0], [1], color='black', lw=1, label='Significant (95% CI)', marker='.'),
                             Line2D([0], [1], color='black', lw=1, label='Not Significant (95% CI)', marker='x')]
                        ax.legend(handles=legend_elements, loc='center right')
                        ax.set_title(title, fontsize='x-large', pad=10)
                        # plot val and Cis in plot

                if i_base_coef_val < n_base_coeffs - 1:
                    ax.axhline(ylim_min, linewidth=3, color="k")

                bottom_i, top_i, left_i, right_i = inner_grid_pos[0][i_base_coef_val], \
                                                   inner_grid_pos[1][i_base_coef_val], \
                                                   inner_grid_pos[2][0], inner_grid_pos[3][0]
                # print(bottom_i, top_i, left_i, right_i, (n_int_coeffs / 2) * i_base_coef_val)
                y, x = top - ((top_i - bottom_i) * (n_int_coeffs / 2) * (i_base_coef_val + (i_base_coef_val + 1))), 0.07
                fig.text(x, y, get_label_if_in_dict(base_coef_val, label_rename_dict),
                         {'ha': 'center', 'va': 'center'}, rotation=90, fontsize='x-large')

            ax.set_xlabel('Coefficient Value ("Difference to Baseline")', fontsize='x-large')

    return fig


def plot_cat(reg_results, i_cat, cat, base_coef, base_coef_val, cat_in_coeff, coef_baselines, i_coef, i_val,
             ax, cat_dict, title, int_coef, int_coef_val, i_int_coef_val, counts_dict,
             extract_coeff_func=extract_coefficient_values_and_stderr):
    x_coef_for_cat, stderr_coef_for_cat = extract_coeff_func(
        reg_results, base_coef, base_coef_val,
        base_coef_val == coef_baselines[base_coef] if is_param_categorical(base_coef, reg_results) else False,
        int_coef, int_coef_val, False if int_coef is None else int_coef_val == coef_baselines[int_coef],
        cat_in_coeff, cat, cat_is_baseline=cat == coef_baselines[cat_in_coeff])

    if counts_dict is not None and counts_dict[cat] < 1:
        x_coef_for_cat = 0
        stderr_coef_for_cat = 0.01

    ci_lower, ci_upper = x_coef_for_cat - stderr_coef_for_cat, x_coef_for_cat + stderr_coef_for_cat
    significant = not ((ci_lower < 0) and (ci_upper > 0))
    ax.plot((ci_lower, ci_upper), (i_cat, i_cat), color=colorblind_tol[i_cat])
    ax.plot(x_coef_for_cat, i_cat, '.' if significant else 'x', color=colorblind_tol[i_cat],
            markersize=10 if significant else 8)

    if counts_dict is not None:
        # below_0 = x_coef_for_cat < 0
        ax.text(ci_lower - 0.05, i_cat,
                f'{counts_dict[cat]}{" Articles " if i_coef == 0 and i_val == 0 and i_int_coef_val == 0 else ""}',
                va='center', ha='right', color=colorblind_tol[i_cat])

    if i_coef == 0 and i_val == 0 and i_int_coef_val == 0:
        ax.text(ci_upper + 0.05, i_cat, f' {cat_dict[cat]}', va='center', color=colorblind_tol[i_cat])
        legend_elements = [Line2D([0], [1], color='black', lw=1, label='Significant (95% CI)', marker='.'),
                           Line2D([0], [1], color='black', lw=1, label='Not Significant (95% CI)', marker='x')]
        ax.legend(handles=legend_elements, loc='center right')
        ax.set_title(title, fontsize='x-large', pad=10)


def plot_separate_cats_from_dict(dict_reg_results, base_coef, base_coef_val, cat_in_coeff, coef_baselines, i_coef,
                                 i_val, ax, cat_dict, title, int_coef=None, int_coef_val=None, i_int_coef_val=0,
                                 counts_dict=None):
    # plot
    for i_cat, cat in enumerate(dict_reg_results.keys()):
        reg_results = dict_reg_results[cat]
        plot_cat(reg_results, i_cat, cat, base_coef, base_coef_val, cat_in_coeff, coef_baselines, i_coef, i_val,
                 ax, cat_dict, title, int_coef, int_coef_val, i_int_coef_val, counts_dict,
                 extract_coefficient_values_and_stderr_single_code)


def plot_separate_cats(vals_cats, reg_results, base_coef, base_coef_val, cat_in_coeff, coef_baselines, i_coef, i_val,
                       ax, cat_dict, title, int_coef=None, int_coef_val=None, i_int_coef_val=0, counts_dict=None):
    # plot
    for i_cat, cat in enumerate(vals_cats):
        plot_cat(reg_results, i_cat, cat, base_coef, base_coef_val, cat_in_coeff, coef_baselines, i_coef, i_val,
                 ax, cat_dict, title, int_coef, int_coef_val, i_int_coef_val, counts_dict)


def setup_axis(fig, grid, i_grid, xlim_min, xlim_max, ylim_min, ylim_max, label, disable_xticks, write_baseline=False,
               do_color=False):
    ax = fig.add_subplot(grid[i_grid])
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)
    # ax.set_yticks([0, 1, 2, 3], vals_cats)
    ax.set_ylabel(label, rotation=90)
    ax.axvline(x=0, color='grey', linestyle=':')
    # ax.set_title(coef, loc='left')

    if write_baseline:
        ax.text(xlim_min + 0.05, (ylim_min + ylim_max / 2), f'Baseline',
                {'ha': 'left', 'va': 'center'}, fontsize='large', fontweight='bold')

    if do_color:
        ax.set_facecolor('whitesmoke')

    if disable_xticks:
        ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    return ax


# this works nicely so far with categorical values :-)
def plot_regression_results(df_reg, reg_results, coefficients, coef_baselines, label_sort=None, cat_dict=None,
                            cat_in_coeff='code', title='', figsize=(8, 8), x_limits=(-2, 2),
                            label_rename_dict=None) -> plt.Figure:
    n_coefficients = len(coefficients)
    vals_coefficients = {coef: df_reg[coef].unique() for coef in coefficients}
    vals_cats = df_reg[cat_in_coeff].unique()
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    rows_grid = sum(
        [(len(vals_coefficients[coef]) - 1 if is_param_categorical(coef, reg_results) else 1) for coef in coefficients])
    single_box_ratio = 1 / rows_grid
    height_ratios = [
        single_box_ratio * (len(vals_coefficients[coef]) - 1 if is_param_categorical(coef, reg_results) else 1) for coef
        in coefficients]
    outer_grid = fig.add_gridspec(n_coefficients, 1, wspace=0.0, hspace=0.0,
                                  height_ratios=height_ratios)
    # #valsincoefficients
    grid_pos = outer_grid.get_grid_positions(fig)
    ylim_min, ylim_max, xlim_min, xlim_max = -1, 4, x_limits[0], x_limits[1]

    for i_coef, coef in enumerate(coefficients):
        n_grids = len(vals_coefficients[coef]) - 1 if is_param_categorical(coef, reg_results) else 1
        coef_i_grid = outer_grid[i_coef].subgridspec(n_grids, 1, wspace=0.0, hspace=0.0)
        ax = None
        if is_param_categorical(coef, reg_results):
            coefficients_list = label_sort[coef] if label_sort is not None and coef in label_sort else \
                vals_coefficients[coef]
            i_val = 0
            for val_coef in coefficients_list:
                if val_coef == coef_baselines[coef]:
                    continue
                ax = setup_axis(fig, coef_i_grid, i_val, xlim_min, xlim_max, ylim_min, ylim_max,
                                get_label_if_in_dict(val_coef, label_rename_dict),
                                (i_coef < len(coefficients) - 1) or (i_val < len(coefficients_list) - 2))
                # plot cats for codes
                plot_separate_cats(vals_cats, reg_results, coef, val_coef, cat_in_coeff, coef_baselines, i_coef, i_val,
                                   ax, cat_dict, title)
                # plot val and Cis in plot
                i_val += 1
        else:
            ax = setup_axis(fig, coef_i_grid, 0, xlim_min, xlim_max, ylim_min, ylim_max, coef,
                            (i_coef < len(coefficients) - 1))
            # plot cats for codes
            plot_separate_cats(vals_cats, reg_results, coef, '', cat_in_coeff, coef_baselines, i_coef, 0,
                               ax, cat_dict, title)

        # if ((i_coef+1) % 2) == 0:
        #    ax.patch.set_facecolor('gray')
        #    ax.patch.set_alpha(0.25)

        if i_coef < len(coefficients) - 1:
            ax.axhline(ylim_min, linewidth=3, color="k")

        bottom, top, left, right = grid_pos[0][i_coef], grid_pos[1][i_coef], grid_pos[2][0], grid_pos[3][0]
        y, x = ((bottom + top) / 2), 0.05
        fig.text(x, y, coef, {'ha': 'center', 'va': 'center'}, rotation=90, fontsize='large')
        fig.text(right + 0.05, y,
                 f'Baseline:\n{coef_baselines[coef]}' if is_param_categorical(coef, reg_results) else '',
                 {'ha': 'center', 'va': 'center'}, rotation=90, fontsize='large')
        ax.set_xlabel('Coefficient Value: Difference to Baseline', fontsize='large')

    return fig


def plot_country_counts(df_crawled, n_countries=10, figsize=(22, 10)):
    val_country_counts = df_crawled[['code', 'country']].value_counts()
    fig, ax = plt.subplots(2, 2, figsize=figsize)
    axs = list(ax.flat)
    for i, code in enumerate(df_crawled.code.unique()):
        val_country_counts.loc[code, :].nlargest(n_countries).sort_values(ascending=False).plot.barh(
            title=f'Most popular country (by #articles) for {code}', ax=axs[i])
    return fig


def plot_cat_by_cat(df_inv, col_vis, col_plot, lang=('de', 'en', 'it', 'es'), figsize=(20, 4)):
    for lang in lang:
        values_col_plot = df_inv[col_plot].unique()
        fig, ax = plt.subplots(nrows=1, ncols=len(values_col_plot), figsize=figsize)
        axs = ax.flatten()
        overall_sum = 0
        for i, val in enumerate(values_col_plot):
            df_plt = df_inv[(df_inv.code == lang) & (df_inv[col_plot] == val)]

            if len(df_plt) == 0:
                continue

            col_vis_count = df_plt.groupby(col_vis).count()['views_7_sum']
            col_vis_count.plot.bar(ax=axs[i], sharex=True, sharey=False)
            overall_sum += col_vis_count.sum()
            axs[i].set_title(f'{val} ({col_vis_count.sum()} articles)')
        fig.suptitle(f'{lang}: {col_vis} by {col_plot} ({overall_sum} articles)')


def label_from_label_dict(label, label_dict=None):
    label_dict = default_label_dict if label_dict is None else None
    return label_dict[label] if label in label_dict else label


def plot_cat_by_cat_variable(df_inv, col_plot, col_x, col_bar, stacked=False, figsize=(15, 4), sharey=True,
                             label_dict=None):
    df_plot = df_inv.copy()
    label_dict = default_label_dict if label_dict is None else label_dict
    bar_plot_vals = df_inv[col_plot].unique()
    fig, ax = plt.subplots(nrows=1, ncols=len(bar_plot_vals), figsize=figsize, sharey=sharey)

    overall_sum = 0
    for i, c_plot in enumerate(bar_plot_vals):
        axs = ax.flatten()
        df_bar = df_plot[df_plot[col_plot] == c_plot].copy()
        df_bar.replace(label_dict, inplace=True)
        col_vis_count = df_bar.groupby([col_x, col_bar])['views_7_sum'].count().unstack().fillna(0)

        # col_vis_count.index.rename(label_dict, inplace=True)
        col_vis_count.plot.bar(ax=axs[i], stacked=stacked)
        curr_count = int(col_vis_count.sum().sum())
        overall_sum += curr_count
        axs[i].set_title(f'{label_from_label_dict(c_plot)}\n({curr_count} articles)')
        axs[i].set_xlabel(label_from_label_dict(col_x))
        if i == len(bar_plot_vals) - 1:
            axs[i].legend(title=label_from_label_dict(col_bar), bbox_to_anchor=(1, 0.5), loc='center left')
        else:
            axs[i].legend().set_visible(False)
    fig.suptitle(f'Number of articles for "{label_from_label_dict(col_bar)}" by "{label_from_label_dict(col_x)}" and '
                 f'"{label_from_label_dict(col_plot)}" ({overall_sum} overall articles)')
    plt.tight_layout()


def plot_pearson_residuals(df, model, exclude_n_outliers=0, col=None, ax=None, title='', log_scale=False):
    if exclude_n_outliers > 0:
        resid_outliers = model.resid_pearson.nlargest(exclude_n_outliers)
        outlier_filter = ~model.resid_pearson.isin(resid_outliers)
        x = df.loc[outlier_filter, col] if col is not None else model.fittedvalues[outlier_filter]
        y = model.resid_pearson[outlier_filter]
    else:
        x = df.loc[:, col] if col is not None else model.fittedvalues
        y = model.resid_pearson

    if ax is None:
        fig, ax = plt.subplots()

    x = np.log(x) if log_scale else x
    ax.scatter(x, y)
    ax.set_title(title)
    ax.set_xlabel('fitted values' + (': log(y)' if log_scale else ''))
    ax.set_ylabel('pearson residuals')


def plot_real_vs_fitted(df, model, outlier_filter=None, ax=None, title='', log_scale=False, exp_scale=False):
    x = model.fittedvalues if outlier_filter is None else model.fittedvalues.loc[outlier_filter]
    y = df.views_7_sum if outlier_filter is None else df.loc[outlier_filter, 'views_7_sum']

    if log_scale:
        x, y = np.log1p(x), np.log1p(y)
    elif exp_scale:
        x, y = np.expm1(x), np.expm1(y)
    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(x, y)
    ax.set_title(title)
    ax.set_xlabel('fitted values' + (': log(y)' if log_scale else ''))
    ax.set_ylabel('y (real values)')
    ax.ticklabel_format(useOffset=False)
    ax.ticklabel_format(style='plain')


def compute_regression_outliers_from_residual(model, exclude_n_outliers=10):
    resid_outliers = model.resid_pearson.nlargest(exclude_n_outliers)
    outlier_filter = ~model.resid_pearson.isin(resid_outliers)
    return resid_outliers, outlier_filter
