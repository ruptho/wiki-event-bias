import re

import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


def fit_regression_and_rename_coeffs(df_reg, formula, robust=False):
    model = smf.ols(formula=formula, data=df_reg)
    model.data.xnames = [re.sub(r", Treatment\(reference='?[a-zA-Z& ]+'?\)\)", '', name.replace('C(', '')) for name in
                         model.data.xnames]
    return model.fit(cov_type='HC3') if robust else model.fit()


def fit_poisson_regression_and_rename_coeffs(df_reg, formula):
    model = sm.GLM.from_formula(formula=formula, family=sm.families.Poisson(), data=df_reg)
    model.data.xnames = [re.sub(r", Treatment\(reference='?[a-zA-Z& ]+'?\)\)", '', name.replace('C(', '')) for name in
                         model.data.xnames]
    return model.fit()


def fit_negative_binomial_regression_and_rename_coeffs(df_reg, formula):
    model = sm.GLM.from_formula(formula=formula, family=sm.families.NegativeBinomial(), data=df_reg)
    model.data.xnames = [re.sub(r", Treatment\(reference='?[a-zA-Z& ]+'?\)\)", '', name.replace('C(', '')) for name in
                         model.data.xnames]
    return model.fit()


def fit_regression_and_rename_coeffs_by_cat(df_reg, formula, cat_col='code', type='linear'):
    if type == 'linear':
        return {cat: fit_regression_and_rename_coeffs(df_reg[df_reg[cat_col] == cat], formula) for cat in
                df_reg[cat_col].unique()}
    elif type == 'poisson':
        return {cat: fit_poisson_regression_and_rename_coeffs(df_reg[df_reg[cat_col] == cat], formula) for cat in
                df_reg[cat_col].unique()}
    elif type == 'nb':
        return {cat: fit_negative_binomial_regression_and_rename_coeffs(df_reg[df_reg[cat_col] == cat], formula) for cat
                in df_reg[cat_col].unique()}


def write_reg_results(reg_results, filename, folder='.', method='csv'):
    file_ending = 'txt'
    results = 'Invalid method passed to write_reg_results.'
    if method == 'csv':
        file_ending = 'csv'
        results = reg_results.summary().as_csv()
    elif method == 'html':
        file_ending = 'html'
        results = reg_results.summary2().as_html()
    elif method == 'latex':
        results = reg_results.summary2().as_latex()
    elif method == 'text':
        results = reg_results.summary2().as_text()

    with open(f'{folder}/{filename}.{file_ending}', 'w') as f:
        f.write(results)


def get_vals_for_coefficient(df_reg, coefficient):
    return df_reg[coefficient].unique()


def get_vals_for_coefficients(df_reg, coefficients):
    return {coef: get_vals_for_coefficient(df_reg, coef) for coef in coefficients}


def get_standard_error_sum(results, covariates):
    # 95CI is approximated with +- 2 sum_variance_standard_erro
    # get the variance covariance matrix
    # print(covariates)
    # see, for example: https://stats.stackexchange.com/a/3657
    # this is what this does!
    # Note: diagonal of cov = var, se² = var.
    vcov = results.cov_params().loc[covariates, covariates].values

    # calculate the sum of all pair wise covariances by summing up off-diagonal entries
    off_dia_sum = np.sum(vcov)
    # variance of a sum of variables is the square root
    return np.sqrt(off_dia_sum)


def extract_interaction_coefficients_covidpaper(res, coefficient, lang_col, lang, is_baseline):
    # baseline => this is basically how it is realized here in the code. Since the overall result for the language
    # ... has to depend on "something" -> (increase from 0). This baseline is the first language in the codes
    # therefore, we have to grab the "pos_col":"treated_col", which basically equals the coefficient of
    # '{treated_col}:{pos_col}:{baseline_lang}". This is basically how it is technically implemented within
    # statsmodels
    # - beta_0 (intercept): danish wiki (language=0), year 2018/2019 (=0), pre-changepoint period (=0)
    # - beta_1 (language) - level change that given language  (categorical) introduces in comparison to the baseline
    #   language in 2018/2019 (=0, "basically intercept")
    # - beta_2 (year) - overall level change from 2018/2019 (=0) in comparison to 2020 (=1) for the baseline wiki
    # - beta_3 (period): seasonal effect for baseline language (pre-to-post over both years)
    # - beta_4 (year:language): How did overall level change from 2018/2019 to 2020 for the given language in comparison
    #   to the baseline wiki (interaction). "What were the language-specific effects for the year change"
    # - beta_5 (period:language): Seasonal effect for language in comparison to baseline language (pre-to-post)
    # - beta_6 (year:period): The change in V from pre- to post-changepoint date over all years for the baseline wiki,
    #   after change in year (year_dummy) or period alone (which is captured by period_dummy).
    #    + What is the additional effect in 2020 for this period of time, that was not pre-existing in 2019?
    # - beta_7 (year:period:language): the change in V after the changepoint in 2020 in comparison to the baseline wiki,
    #   after change in year (year_dummy) or period alone (which is captured by period_dummy)

    # relative starting point for language = intercept + language = beta_0 + beta_1
    #   (what is the intercept for the given language model here, basically)
    # relative seasonal effect for non-baseline language (over both years) = period + period:language = beta_3 + beta_5
    # relative yearly effect for non-baseline language = year + year_language = beta_2 + beta_4
    # relative change from pre- to- post-changepoint levels in 2020 for non-baseline language =
    #   year:period + year:period:language = beta_6 + beta_7

    # The categorical encoding of using n-1 (for n categories) predictors is necessary, because otherwise we would
    # introduce multicollinearity ("the dummy trap" -
    # e.g., https://en.wikipedia.org/wiki/Dummy_variable_(statistics)#Incorporating_a_dummy_independent,
    # http://facweb.cs.depaul.edu/sjost/csc423/documents/dummy-variable-trap.htm)
    if is_baseline:
        if coefficient == lang_col:
            coefficient = 'Intercept'  # see notes above
        val = res.params[coefficient]
        std = get_standard_error_sum(res, [coefficient])
    else:
        if coefficient == lang_col:
            # note that the lang parameter has to be interpreted slightly differently
            val = res.params['Intercept'] + res.params[f'{lang_col}[T.{lang}]']
            std = get_standard_error_sum(res, ['Intercept', f'{lang_col}[T.{lang}]'])
        else:
            val = res.params[coefficient] + res.params[f'{coefficient}:{lang_col}[T.{lang}]']
            std = get_standard_error_sum(res, [coefficient, f'{coefficient}:{lang_col}[T.{lang}]'])
    return val, std


def extract_coefficient_values_and_stderr(res, coeff_col, coeff_val, coeff_is_baseline=False,
                                          coeff_int_col=None, coeff_int_val=None,
                                          coeff_int_is_baseline=False, cat_col='code', cat_val='en',
                                          cat_is_baseline=False, add_cat_coeff=True):
    # coeff_col = gni_class, coef_val = 'L', coef_is_baseline=False
    # we want the change o
    # Intercept = english wikipedia, countries that speak english, and are rich (=U)
    # Examples:
    # gni_class[T.L] => change for the english wikipedia and sports when the region of the event is .U
    # code[T.de] => difference between english and german
    # gni_class[T.L]:code[T.de] =>
    # this function does not allow to grab the code parameters directly
    baseline_coefficient = (f'{coeff_col}' if not coeff_is_baseline else '') + (
        f'[T.{coeff_val}]' if is_param_categorical(coeff_col, res) else '')
    interaction_coefficient = (f'{coeff_int_col}' if coeff_int_col is not None else '') + (
        f'[T.{coeff_int_val}]' if is_param_categorical(coeff_int_col, res) else '')

    coefficient_string = ''
    if not coeff_is_baseline:
        coefficient_string += baseline_coefficient
        if not coeff_int_is_baseline and coeff_int_col is not None:
            coefficient_string += f':{interaction_coefficient}'
    else:
        if not coeff_int_is_baseline and coeff_int_col is not None:
            coefficient_string += interaction_coefficient

    # print('Hi!')
    if cat_is_baseline:
        if coefficient_string != '':
            val, std = res.params[coefficient_string], get_standard_error_sum(res, [coefficient_string])
            print(coefficient_string)
        else:
            # val, std = res.params['Intercept'], get_standard_error_sum(res, ['Intercept'])
            val, std = 0, get_standard_error_sum(res, ['Intercept'])  # - res.params['Intercept']
    else:
        cat_interaction = f'{cat_col}[T.{cat_val}]'
        coefficient_cat_string = f'{coefficient_string}{":" if len(coefficient_string) > 0 else ""}{cat_interaction}'
        if not add_cat_coeff:
            print(coefficient_cat_string)
            val, std = res.params[coefficient_cat_string], get_standard_error_sum(res, [coefficient_cat_string])
        else:
            # print(coefficient_string, is_param_categorical(coeff_col, res), coeff_val, coeff_int_val,
            #      coeff_is_baseline, coeff_int_is_baseline, coefficient_cat_string)
            # to get the absolute value of the interacted coefficient, we have to add the baseline to it for vals,
            # and must account for the covariates in the standard errors as of: https://stats.stackexchange.com/a/3657
            # print(cat_interaction, coefficient_cat_string)
            print(coefficient_string, coefficient_cat_string)
            if coefficient_string != '':
                # print(f'{coefficient_string} + {coefficient_cat_string}')
                val = res.params[coefficient_string] + res.params[coefficient_cat_string]
                std = get_standard_error_sum(res, [coefficient_string, coefficient_cat_string])
            else:
                val, std = res.params[coefficient_cat_string], get_standard_error_sum(res, [coefficient_cat_string])

    return val, std


def extract_coefficient_values_and_stderr_single_code(res, coeff_col, coeff_val, coeff_is_baseline=False,
                                                      coeff_int_col=None, coeff_int_val=None,
                                                      coeff_int_is_baseline=False, cat_col='code', cat_val='en',
                                                      cat_is_baseline=False, add_cat_coeff=True):
    # difference
    # print(coeff_val, coeff_int_val, coeff_is_baseline, coeff_int_is_baseline, cat_val)
    if coeff_is_baseline and coeff_int_is_baseline:
        return 0, 0  # => reference for this "box"
    else:
        baseline_coefficient = (f'{coeff_col}' if not coeff_is_baseline else '') + (
            f'[T.{coeff_val}]' if is_param_categorical(coeff_col, res) else '')
        interaction_coefficient = (f'{coeff_int_col}' if coeff_int_col is not None else '') + (
            f'[T.{coeff_int_val}]' if is_param_categorical(coeff_int_col, res) else '')

        if coeff_is_baseline:
            val, std = res.params[interaction_coefficient], get_standard_error_sum(res, [interaction_coefficient])
            # print(f'{cat_val}: {interaction_coefficient} = {val}')
        elif coeff_int_is_baseline:  # coef_int_is_baseline
            val, std = 0, 0  # res.params[baseline_coefficient], get_standard_error_sum(res, [baseline_coefficient])
        else:
            coeff_interaction_string = f'{baseline_coefficient}:{interaction_coefficient}'
            val = res.params[interaction_coefficient] + res.params[coeff_interaction_string]
            std = get_standard_error_sum(res, [interaction_coefficient, coeff_interaction_string])
            # print(f'{cat_val}: {interaction_coefficient} + {coeff_interaction_string} = {val}')

    return val, std


def is_param_categorical(coef_name, reg_results):
    return np.any([True if f'{coef_name}[T.' in param else False for param in reg_results.params.index])
