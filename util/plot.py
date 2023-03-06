import geojson
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib as mpl
import plotly.express as px

from util.preprocessing import replace_country_names

colorblind_tol4 = ['#117733', '#88CCEE', '#E69F00', '#882255']
colorblind_tol2 = ['#88CCEE', '#117733']
colorblind_tol2b = ['#E69F00', '#882255']
colorblind_wong7 = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
colorblind_tol8 = ['#882255', '#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499']
colorblind_tol7 = ['#882255', '#332288', '#117733', '#44AA99', '#88CCEE', '#AA4499', '#DDCC77']
colorblind_tol = colorblind_tol4

helper_langs = {"de": "German", "fr": "French", "it": "Italian", "en": "English"}
default_label_dict = {"code": "Language", "de": "German", "fr": "French", "it": "Italian", "en": "English",
                      "es": "Spanish", "gni_class": "Income Class", "gni_region": "Geographic Region", "H": "High",
                      "L": "Low", 'cat_sports': 'Category',
                      "LM": "Lower mid", "UM": "Upper mid", "cat": "Category", 'noticed': '>10 views',
                      'views_7_sum': 'Views Within 7 Days After an Event',
                      'views_7_sum_log': '7-Day-Views\n(log)', 'disaster': 'Disaster', 'sports': 'Sports',
                      'culture': 'Culture', 'politics': 'Politics', 'economic_region': 'Economic Region',
                      'edits_7_sum': 'Edits Within 7 Days After an Event',
                      'noted': '>10 Views', 'edited': '>0 Edits', 'GDP_pc_log': 'GDP per capita (log)'}

BASELINE_DICT = {'gni_class': 'H', 'planned': False, 'breaking': False,
                 'gni_region': 'North America', 'cat': 'sports', 'code': 'en', 'income_level': 'higher',
                 'continent': 'North America', 'economic_region': 'Global North', 'oecd': False,
                 'comb_region': 'North America', 'in_code_lang': False, 'cluster': '-1'}
CAT_DICT = {'en': 'English', 'it': 'Italian', 'es': 'Spanish', 'de': 'German'}
LABEL_RENAME_DICT = {'Middle East & North Africa': 'MENAf', 'Latin America & Caribbean': 'LatAmC',
                     "South Asia, East Asia, Pacific": 'SA,EA,P', "Africa & Middle East": "A&ME",
                     'Europe & Central Asia': 'EuCAs', 'East Asia & Pacific': 'EAsP', 'North America': 'NAm',
                     'Sub-Saharan Africa': 'SSAf', 'South Asia': 'SAs', 'GDP_pc_z': 'GDP pc (z.)',
                     'GDP_pc_log': 'GDP per capita (log)', 'gni_class': 'Income', 'gni_region': 'Region',
                     'bing_hits_log': 'Bing',
                     'view_country_article_log': 'Views to\nCountry\nArticle',
                     'views_baseline_log': 'Views\nfrom\nCountry\n(log)', 'bing_hits_z': 'Bing (z)',
                     'view_country_article_z': 'Views to\nCountry\nArticle (z)', 'cat': 'Article Category',
                     'views_baseline_z': 'Views\nfrom\nCountry (z)', 'Global North': 'North', 'Global South': 'South',
                     'country_articles_z': 'Past\nCountry\nArticles (z)',
                     'country_articles_log': '#Country\nArticles\nin Last\n30 Days\n(log)',
                     'cat_articles_log': '#Category\nArticles\nin Last\n30 Days\n(log)',
                     'cat_articles_z': 'Past\nCat\nArticles (z)',
                     'country_cat_articles_log': 'Past\nArticles\nCat+Cntry', 'population_log': 'Pop. (log)',
                     'population_z': 'Pop. (z)', 'country_cat_articles_z': 'Past\nArticles\nCat+Cntry (z)',
                     'economic_region': 'Economic\nRegion', 'planned': 'Article created\nbefore event',
                     'breaking': 'Article created\nwithin 1 day'}


def plot_shap_lineplot_for_model(shap_vals_alt_dict, model_dict, model_str, x_col='GDP_pc_log', x_cont=None,
                                 format_col='gni_region', sep_col='gni_region',
                                 path='figures/line/', regtype='poly', reg_ci=95, func_order=5, figsize=(15, 3),
                                 alpha=0.5, model_prefix='noreg_', model_postfix='_xgb'):
    full_model_str = f'{model_prefix}{model_str}{model_postfix}'
    model, shapvals = model_dict[full_model_str], shap_vals_alt_dict[full_model_str]
    df_full = model.get_full_dataset(decoded=False, all_cols=True)
    sep_cols = ['code_en', 'code_es', 'code_de', 'code_it'] if sep_col == 'gni_region' else \
        ['cat_sports', 'cat_disaster', 'cat_culture', 'cat_politics']
    shap_vals_plot = pd.Series(shapvals[:, x_cont if x_cont else x_col].values,
                               name=x_cont if x_cont else x_col).to_frame()

    fig1, axs = plot_contributions_sep(df_full, shap_vals_plot, x_col, sep_cols, x_cont=x_cont,
                                       format_col=format_col, hue_cols=None, encoder=model.encoder,
                                       regtype=regtype, reg_ci=reg_ci, func_order=func_order, figsize=figsize,
                                       alpha=alpha)
    if path:
        fig1.savefig(f'{path}SHAP_{model_str}_{format_col}.pdf', bbox_inches='tight')
    return fig1


def plot_contributions_sep(df_data, df_contribution, x, sep_cols=['code_en', 'code_es', 'code_de', 'code_it'],
                           sep_labels=['English', 'Spanish', 'German', 'Italian'],
                           x_cont=None, func_order=6, figsize=(10, 5), alpha=0.8, hue_cols=None,
                           y_label=None, regtype='poly', reg_ci=None,
                           format_col=None, marker_styles=['o', 'X', 's', '^', 'P', "d", "p"], encoder=None,
                           ylims=None, s=50):
    import seaborn as sns
    from sklearn.metrics import r2_score

    fig, axs = plt.subplots(ncols=len(sep_cols), figsize=figsize, sharey=True)
    format_values = ['East Asia & Pacific', 'Europe & Central Asia', 'Latin America & Caribbean',
                     'Middle East & North Africa', 'North America', 'South Asia',
                     'Sub-Saharan Africa'] if format_col == 'gni_region' else \
        ['Sports', 'Disaster', 'Politics', 'Culture']
    hue_cols = hue_cols if hue_cols else colorblind_tol4 if len(format_values) < 5 else colorblind_tol7 if len(
        format_values) < 8 else colorblind_tol8
    df_inv = encoder.inverse_transform(df_data) if encoder else df_data
    palette_dict = {region: hue_cols[i] for i, region in enumerate(format_values)}
    marker_dict = {region: marker_styles[i] for i, region in enumerate(format_values)}

    y_lim_min, y_lim_max = 9999, -9999
    for i, code in enumerate(sep_cols):
        ax = axs[i]
        mask = (df_data[code] == 1).values
        df_plot = df_data[mask].copy()
        df_contribution_sep = df_contribution[mask].copy()
        df_inv_trans = df_inv[mask].copy()
        if not x_cont:
            x_cont = x

        if format_col:
            df_inv_trans[format_col] = df_inv_trans[format_col].apply(lambda c: label_from_label_dict(c))

        ax.axhline(0, linestyle=':', color='grey', zorder=0)

        scatter_data = pd.concat([df_plot[x].reset_index(drop=True),
                                  df_contribution_sep[x_cont].rename('contributions').reset_index(drop=True)], axis=1)

        line_kws_reg = {'color': 'black', 'linewidth': 1, 'alpha': 0.8}

        g = sns.scatterplot(data=scatter_data, x=x, y='contributions', s=s, alpha=alpha,
                            palette=palette_dict if format_col != None else None, legend='auto' if i == 0 else None,
                            hue=df_inv_trans[format_col].values if format_col else None,
                            markers=marker_dict,
                            style=df_inv_trans[format_col].values if format_col else None, ax=ax)

        y_min, y_max = float(ax.get_ylim()[0]), float(ax.get_ylim()[1])
        y_lim_min, y_lim_max = y_min if y_min < y_lim_min else y_lim_min, y_max if y_max > y_lim_max else y_lim_max

        if regtype == 'poly':
            cv_line = grid_search_regline(scatter_data[x].values.reshape(-1, 1), scatter_data['contributions'].values,
                                          func_order)
            best_degrees = cv_line.best_params_["polynomialfeatures__degree"]
            fit_best = cv_line.best_estimator_.predict(scatter_data[x].values.reshape(-1, 1))

            sns.regplot(data=scatter_data, x=x, y='contributions', x_bins=1000, order=best_degrees, ax=ax,
                        line_kws=line_kws_reg,
                        scatter=False, ci=reg_ci)
            r2 = r2_score(df_contribution_sep[x_cont], fit_best)

            text = f'Reg. Line Degree={best_degrees}, r²={r2:.2f}'
            ax.text(0.95, 0.05, text, ha='right', va='center', transform=ax.transAxes)
        elif regtype == 'poly_force':
            cv_line = polynomial_regression(degree=func_order).fit(scatter_data[x].values.reshape(-1, 1),
                                                                   scatter_data['contributions'].values)
            fit_best = cv_line.predict(scatter_data[x].values.reshape(-1, 1))

            rg = sns.regplot(data=scatter_data, x=x, y='contributions', x_bins=1000, order=func_order, ax=ax,
                             line_kws=line_kws_reg, scatter=False, ci=reg_ci)
            r2 = r2_score(df_contribution_sep[x_cont], fit_best)

            text = f'Reg. Line Degree={func_order}, r²={r2:.2f}'
            # ax.text(0.95, 0.05, text, ha='right', va='center', transform=ax.transAxes)
        elif regtype == 'lowess':
            sns.regplot(data=scatter_data, x=x, y='contributions', ax=ax, line_kws=line_kws_reg,
                        lowess=True, scatter=False, ci=reg_ci)
        else:
            pass

        if i == 0:
            fig.text(0.45, 0, label_from_label_dict(x), va='top', fontsize='large')
            ax.set_xlabel('')
            ax.set_ylabel(y_label if y_label else f'SHAP Value for\n{label_from_label_dict(x)}', fontsize='large')
            # sns.move_legend(g, "lower left",  bbox_to_anchor=(0, 1), title=None, frameon=True,
            # ncol=len(df_inv_trans[format_col].unique()), columnspacing=0.2, handletextpad=0.05)
            h, l = ax.get_legend_handles_labels()
            l, h = zip(*sorted(zip(l, h), key=lambda t: t[0]))  # sort both labels and handles by labels
            n_hue = len(df_inv_trans[format_col].unique())
            ax.legend().set_visible(False)
            fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=n_hue, columnspacing=0.2,
                       handletextpad=0.05)
        else:
            ax.set_ylabel('')
            ax.set_xlabel('')

        ax.set_title(f'{sep_labels[i]} ({len(df_contribution_sep)} articles)', fontsize='large')

        # this is necessary!!!!
        if regtype:
            if not ylims:
                ax.set(ylim=(y_lim_min, y_lim_max))
            else:
                ax.set(ylim=ylims)
    # this is also necessary!!!!
    for ax in axs:
        if regtype:
            if not ylims:
                ax.set(ylim=(y_lim_min, y_lim_max + 0.075))
            else:
                ax.set(ylim=ylims)

    fig.subplots_adjust(wspace=0.05)
    return ax.get_figure(), ax


def label_from_label_dict(label, label_dict=None, log=False):
    label_dict = default_label_dict if label_dict is None else label_dict
    log_label = ' (log)'
    return (label_dict[label] if label in label_dict else label) + (log_label if log else '')


def polynomial_regression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(fit_intercept=False, **kwargs))


def grid_search_regline(X, y, func_order):
    reg_cv = GridSearchCV(polynomial_regression(),
                          param_grid={'polynomialfeatures__degree': list(range(1, func_order + 1))},
                          scoring='r2', cv=KFold(n_splits=100, shuffle=True,
                                                 random_state=20220101))  # just fix random date for now
    return reg_cv.fit(X, y)


def build_chloropleths(shap_dict, model_dict, model_type, column, group_col, color_data, color_map,
                       save_path='figures/maps', save_path_postfix='', model_prefix='noreg_', model_postfix='_xgb',
                       save_dims=(1600, 750), cm_chloropleth=mpl.cm.coolwarm_r, cb_horizontal=False, cb_label=None,
                       filter_mask=None):
    full_model_name = f'{model_prefix}{model_type}{model_postfix}'
    model = model_dict[full_model_name]
    df_full = pd.concat([model.df_train_full, model.df_test])
    df_full_enc = model.encoder.inverse_transform(df_full)
    shapvals = shap_dict[full_model_name]

    df_full_enc = (
        df_full_enc.iloc[filter_mask] if 'NoneType' not in str(type(filter_mask)) else df_full_enc).reset_index(
        drop=True)
    if 'DataFrame' not in str(type(shapvals)):
        shapvals = pd.Series(
            shapvals[filter_mask, column].values if 'NoneType' not in str(type(filter_mask)) else shapvals[:,
                                                                                                  column].values,
            name=f'{column}_shap').reset_index(
            drop=True)
    else:
        shapvals = (shapvals.loc[filter_mask, column] if 'NoneType' not in str(type(filter_mask))
                    else shapvals.loc[:, column]).rename(f'{column}_shap').reset_index(drop=True)

    df_full_shap = pd.concat([df_full_enc, shapvals], axis=1)
    max_abs_limit = np.abs(
        df_full_shap[[f'{column}_shap', group_col, color_data]].groupby([group_col, color_data]).mean()).max().values[0]
    prefix_files, figs_geo = f'{full_model_name}_{color_data}_{group_col}', []
    for i, (group, df) in enumerate(df_full_shap.groupby(group_col)):
        # print(group)
        df_chloro, fig_geo = plot_chloropleth(df, val_col=f'{column}_shap', color_data=color_data, color_map=color_map,
                                              range_color=[-max_abs_limit, max_abs_limit], show_legend=False,
                                              cmap=cm_chloropleth)
        if save_path:
            fig_geo.write_image(f'{save_path}/{prefix_files}_{group}{save_path_postfix}.pdf', width=save_dims[0],
                                height=save_dims[1],
                                engine='kaleido')
        figs_geo.append(fig_geo)

    fig_cmap = build_colormap(
        f'SHAP Value\nfor {label_from_label_dict(column) if column != "GDP_pc_log" else "GDP pc."}' if not cb_label
        else cb_label, max_limits=(-max_abs_limit, max_abs_limit), horizontal=cb_horizontal,
        save_path=f'{save_path}/{prefix_files}{save_path_postfix}_cbar.pdf' if save_path else None)
    return figs_geo, fig_cmap


def build_chloropleths_df(df, column, group_col, color_data, color_map,
                          save_path='figures/maps', save_dims=(1600, 750),
                          cm_chloropleth=mpl.cm.viridis, filter_mask=None, encoder=None,
                          relative_scale=False, metric=np.mean, cb_horizontal=False, cb_label=None):
    df_full = df.copy()
    if encoder:
        df_full = encoder.inverse_transform(df_full)

    df_full = (df_full.iloc[filter_mask] if 'NoneType' not in str(type(filter_mask)) else df_full).reset_index(
        drop=True)

    # compute global minimum for matching scales across plots
    df_minmax = df_full[[f'{column}', group_col, color_data]].groupby([group_col, color_data]).agg(metric)
    min_limit, max_limit = df_minmax.min().values[0], df_minmax.max().values[0]

    prefix_files = \
        f'{column}_{color_data}_{group_col}{"" if (metric == np.mean) or (metric == "mean") else f"_{metric}"}'
    figs_geo = []
    for i, (group, df) in enumerate(df_full.groupby(group_col)):
        print(f'{prefix_files}_{group}')
        df_chloro, fig_geo = plot_chloropleth(df, val_col=f'{column}', color_data=color_data, color_map=color_map,
                                              range_color=[min_limit, max_limit] if not relative_scale else [0, 1],
                                              show_legend=False, cmap=cm_chloropleth, relative_scale=relative_scale,
                                              metric=metric)
        if save_path:
            fig_geo.write_image(f'{save_path}/{prefix_files}_{group}.pdf', width=save_dims[0],
                                height=save_dims[1], engine='kaleido')
        figs_geo.append(fig_geo)

    cb_label = cb_label if cb_label else f'{label_from_label_dict(column) if column != "GDP_pc_log" else "GDP pc."}'
    fig_cmap = build_colormap(
        cb_label, max_limits=(min_limit, max_limit) if not relative_scale else (0, 1),
        save_path=f'{save_path}/{prefix_files}_cbar.pdf' if save_path else None, cmap=cm_chloropleth,
        horizontal=cb_horizontal)
    return figs_geo, fig_cmap


def plot_chloropleth(df_data, val_col='views_7_sum_log', color_data='gni_region', color_map='region_wb',
                     geojson_path='data/maps/custom.geo.json', metric='mean', range_color=(-0.4, 0.4),
                     cmap=mpl.cm.coolwarm_r, show_legend=True, relative_scale=False):
    with open(geojson_path) as f:
        gj = geojson.load(f)

    list_pd = []
    for f in gj["features"]:
        list_pd.append([f['properties']['name_en'], f['properties'][color_map]])
    df_countries = pd.DataFrame(list_pd, columns=['country_gs', color_data])

    if color_map == 'name_en':
        replace_country_names(df_countries[color_data], inplace=True)

    df_chloro = df_data.groupby([color_data])[val_col].agg(metric).reset_index()
    if relative_scale:
        df_chloro[val_col] = (df_chloro[val_col] - df_chloro[val_col].min()) / (
                df_chloro[val_col].max() - df_chloro[val_col].min())
    df_countries = df_countries.merge(df_chloro, on=color_data, how='left')

    # cNorm  = cmap.Normalize(vmin=range_color[0], vmax=range_color[1])
    fig_geo = px.choropleth(df_countries, geojson=gj, locations='country_gs', color=val_col, basemap_visible=True,
                            color_continuous_scale=get_continuous_scale(cmap),
                            featureidkey="properties.name_en",
                            range_color=range_color if not relative_scale else [0, 1])
    fig_geo.update_geos(lataxis_range=[-55, 82], projection_type="natural earth", showsubunits=False, subunitwidth=0,
                        showrivers=False, showlakes=False, showcountries=False, showland=False)
    fig_geo.update_traces(marker_line_width=0.0)
    fig_geo.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, coloraxis_showscale=show_legend)
    return df_countries, fig_geo


def build_colormap(label, max_limits, save_path=None, figsize=(0.1, 1), x=-0.4, y=1.1, labelpad=None,
                   cmap=mpl.cm.coolwarm_r, horizontal=False, adjust_limits=True, ticks=None):
    fig, ax = plt.subplots(figsize=figsize if not horizontal else (figsize[1] * 2, figsize[0]), dpi=100)
    norm = mpl.colors.Normalize(vmin=max_limits[0], vmax=max_limits[1])

    decs = 2 if horizontal else 1

    def round_format(x, pos):
        if abs(x) > 0:
            round_x = int(x * 10 ** decs) / 10 ** decs
            if round_x == 0:
                round_x = 1 / 10 ** decs
            return f'{round_x}'
        return '0'

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal' if horizontal else 'vertical',
                                    ticks=[max_limits[0] * 0.85, 0, max_limits[1] * 0.85] if
                                    adjust_limits and max_limits[0] != 0 else ticks if ticks else None,
                                    format=FuncFormatter(round_format))
    if not horizontal:
        plt.setp(cb1.ax.get_yticklabels(), ha="right")
        cb1.ax.tick_params(pad=22.5)
        cb1.set_label(label, fontsize='medium', labelpad=-18 if not labelpad else labelpad, y=y, ha='center',
                      va='bottom', rotation=0)
    else:
        # print('what') #  y=-0.5, x=-1, ha='center', va='top',
        cb1.ax.xaxis.set_label_position('top')
        cb1.set_label(label, fontsize='medium', labelpad=-19 if not labelpad else labelpad, x=x, rotation=0)
    if save_path:
        fig.savefig(f'{save_path}', bbox_inches='tight', dpi=100)
    return fig


def get_continuous_scale(cmap):
    # see https://www.kennethmoreland.com/color-advice/
    if cmap == mpl.cm.coolwarm_r:
        return [[0, 'rgb(180,4,38)'], [0.5, 'rgb(221, 221, 221)'],
                [1.0, 'rgb(59, 76, 192)']]
    elif cmap == mpl.cm.viridis:
        return [[0, 'rgb(68,1,84)'], [0.14, 'rgb(70, 50, 127)'], [0.29, 'rgb(54, 92, 141)'],
                [0.43, 'rgb(39, 127, 142)'], [0.57, 'rgb(31, 161, 135)'], [0.71, 'rgb(74, 194, 109)'],
                [0.86, 'rgb(159, 218, 58)'], [1, 'rgb(253, 231, 37)']]
    else:
        return None


def plot_disaster_results(df_shap, shap_values, shap_interaction, save_path='figures/shap/deaths_singlecol.pdf',
                          align_y=True, mark_indices=None):
    import seaborn as sns
    fig, axs = plt.subplots(ncols=2, figsize=(8, 2.75))  # figsize=(8, 2.75))
    colors = df_shap.GDP_pc_log.values
    marker_size = 60

    scatter_death = sns.scatterplot(x=df_shap['deaths_log'], y=shap_values[:, 'deaths_log'].values, c=colors,
                                    # style=df_shap.code_en if 'code_en' in df_shap.columns else None,
                                    alpha=0.33, ax=axs[0], cmap='viridis', legend=None, s=marker_size)
    scatter_int = sns.scatterplot(x=df_shap['deaths_log'], y=shap_interaction[:, 1, 0] * 2, c=colors, alpha=0.33,
                                  #  style=df_shap.code_en if 'code_en' in df_shap.columns else None,
                                  ax=axs[1], legend='brief', cmap='viridis', s=marker_size)
    scatter_death.set_ylabel('SHAP Value for Deaths (log1p)')
    scatter_int.set_ylabel('SHAP Interaction Value for\nGDP pc (log) and Deaths (log1p)')
    scatter_death.set_xlabel('Deaths (log1p)')
    scatter_int.set_xlabel('Deaths (log1p)')

    min_y, max_y = 1000, -1000
    if align_y:
        for ax in axs:
            lim_y_min, lim_y_max = ax.get_ylim()
            max_y = lim_y_max if lim_y_max > max_y else max_y
            min_y = lim_y_min if lim_y_min < min_y else min_y
            ax.set_ylabel(ax.get_ylabel(), fontsize='large')
            ax.set_xlabel(ax.get_xlabel(), fontsize='large')

    for ax in axs:
        if align_y:
            ax.set_ylim((min_y - 0.1, max_y))
        ax.axhline(0, color='gray', linestyle=':', zorder=-1)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)  # fig.subplots_adjust(wspace=0.42)
    if save_path:
        fig.savefig('figures/shap/deaths_singlecol.pdf', bbox_inches='tight')
    return fig
