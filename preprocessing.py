import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from bing_helper import load_lang_news_hits

countries_version = {'en': ['United States'], 'it': ['Italy'], 'de': ['Germany'], 'es': ['Spain']}
version_language = {'en': 'English', 'it': 'Italian', 'de': 'German', 'es': 'Spanish'}

country_replace_dict = {
    'Bahamas, The': 'Bahamas',
    'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
    'Congo, the Democratic Republic of the': 'Democratic Republic of the Congo',
    'Egypt, Arab Rep.': 'Egypt',
    'Faroe Islands': 'Faeroe Islands',
    'Gambia, The': 'Gambia',
    'Hong Kong SAR, China': 'Hong Kong',
    'Iran, Islamic Rep.': 'Iran',
    "Korea, Dem. People's Rep.": 'North Korea',
    "Korea, Democratic People's Republic of": 'North Korea',
    'Korea, Rep.': 'South Korea',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Lao PDR': 'Laos',
    'Micronesia, Fed. Sts.': 'Federated States of Micronesia',
    'Micronesia, Federated States of': 'Federated States of Micronesia',
    'North Macedonia': 'Macedonia',
    'Russian Federation': 'Russia',
    'S?o Tomé and Principe': 'Sao Tome and Principe',
    'São Tomé and Príncipe': 'Sao Tome and Principe',
    'Slovak Republic': 'Slovakia',
    'St. Kitts and Nevis': 'Saint Kitts and Nevis',
    'St. Lucia': 'Saint Lucia',
    'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'Syrian Arab Republic': 'Syria',
    'Venezuela, RB': 'Venezuela',
    'Venezuela, Bolivarian Republic of': 'Venezuela',

    'Yemen, Rep.': 'Yemen',
    'Taiwan, China': 'Taiwan',
    "Taiwan, Province of China": "Taiwan",
    'Faeroe Islands': 'Faroe Islands',
    'Congo, Rep.': 'Republic of the Congo',
    'Guernsey': 'United Kingdom',  # this is kind of a reach
    'Cabo Verde': 'Cape Verde',
    'Sint Maarten (Dutch part)': 'Sint Maarten',
    'Vatican City': 'Italy',  # this is kind of a reach
    'Curacao': 'Curaçao',
    "Cote d'Ivoire": "Côte d'Ivoire",
    "China (mainland)": "China",
    "Viet Nam": "Vietnam",
    "Iran, Islamic Republic of": "Iran",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Czechia": "Czech Republic",
    "Republic of Korea": "South Korea",
    "Korea, Republic of": "South Korea",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "United Republic of Tanzania: Mainland": "Tanzania",
    "Tanzania, United Republic of": "Tanzania",
    "Congo": "Republic of the Congo",
    "Republic of Moldova": "Moldova",
    "Moldova, Republic of": "Moldova",
    "Republic of North Macedonia": "Macedonia",
    "China, Hong Kong SAR": "Hong Kong",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Bolivia, Plurinational State of": "Bolivia",
    "Macedonia, the Former Yugoslav Republic of": "Macedonia",
    "Lao People's Democratic Republic": "Laos",
    "Democratic People's Republic of Korea": "North Korea",
    "Micronesia (Federated States of)": "Federated States of Micronesia",
    "Niue": "Tonga",
    "CÃ´te d'Ivoire": "Côte d'Ivoire",
    "CuraĂ§ao": "Curaçao",
    "CuraÃ§ao": "Curaçao",
    'Holy See (Vatican City State)': 'Italy',
    "Virgin Islands, British": "British Virgin Islands"
}


def load_events(path='events/new'):
    df_de = pd.read_csv(f"{path}/events_dataframe_de.csv",
                        converters={'redirects': pd.eval, 'list_views_7_days': pd.eval})
    df_de['code'] = 'de'
    df_en = pd.read_csv(f"{path}/events_dataframe_en.csv",
                        converters={'redirects': pd.eval, 'list_views_7_days': pd.eval})
    df_en['code'] = 'en'
    df_it = pd.read_csv(f"{path}/events_dataframe_it.csv",
                        converters={'redirects': pd.eval, 'list_views_7_days': pd.eval})
    df_it['code'] = 'it'
    df_es = pd.read_csv(f"{path}/events_dataframe_es.csv",
                        converters={'redirects': pd.eval, 'list_views_7_days': pd.eval})
    df_es['code'] = 'es'
    df = pd.concat([df_en, df_de, df_it, df_es])
    df.cat.replace({np.nan: 'undefined', '': 'undefined'}, inplace=True)  # replace nan events
    return df[df.views_7_sum > 0]


def load_bing_results(path=f'scraping/results'):
    files_en = [f'{path}/en_bingnews.json', f'{path}/en_bingnews_rest.json', f'{path}/en_bingnews_missing.json']
    files_de = [f'{path}/de_bingnews.json', f'{path}//de_bingnews_missing.json']
    files_it = [f'{path}/it_bingnews.json', f'{path}//it_bingnews_missing.json']
    files_es = [f'{path}//es_bingnews_es-ES.json', f'{path}//es_bingnews_es-ES_missing.json']

    df_news = load_lang_news_hits({'de': files_de, 'en': files_en, 'it': files_it, 'es': files_es})
    df_es_corrected = load_lang_news_hits({'es': [f'{path}/es_bingnews_es-ES_corrected.json']})
    df_es_corrected['pagetitle_original'] = df_es_corrected.pagetitle.apply(lambda s: 'Anexo:' + str(s))
    # this could be nicer, but writing this code was easier.
    df_news.loc[:, 'bing_hits'] = df_news.apply(
        lambda row: df_es_corrected[df_es_corrected.pagetitle_original == row.pagetitle].bing_hits.values[
            0] if row.pagetitle.startswith('Anexo:') else row.bing_hits, axis=1)
    return df_news


def check_counts_and_merge(df, df_news, dropna=True):
    print('de', len(df_news[df_news.code == 'de']), len(df[df.code == 'de']))
    print('it', len(df_news[df_news.code == 'it']), len(df[df.code == 'it']))
    print('es', len(df_news[df_news.code == 'es']), len(df[df.code == 'es']))
    print('en', len(df_news[df_news.code == 'en']), len(df[df.code == 'en']))
    print('total', len(df), len(df_news))

    # check duplicates
    df_crawled = df.merge(df_news, on=['pagetitle', 'code'], how='outer')
    df_crawled = df_crawled.loc[~pd.isna(df_crawled.bing_hits)] if dropna else df_crawled
    print(f'With duplicates, but dropped na from bing {len(df_crawled)}')
    df_crawled = df_crawled.loc[~df_crawled.pagetitle.duplicated(keep=False), :]
    print(f'Without duplicates {len(df_crawled)}')
    print('de', len(df_crawled[df_crawled.code == 'de']))
    print('it', len(df_crawled[df_crawled.code == 'it']))
    print('es', len(df_crawled[df_crawled.code == 'es']))
    print('en', len(df_crawled[df_crawled.code == 'en']))
    df_crawled.replace(country_replace_dict, inplace=True)

    return df_crawled


def load_gni_class(df_crawled, gni_path='worldbank/gni_export.csv'):
    gni = pd.read_csv(gni_path, sep=',', encoding='latin1')
    gni = gni.drop(['code'], axis=1).set_index('name')
    gni = gni.unstack(fill_value='..').reset_index().rename(
        {'level_0': 'year', 'name': 'country', 0: 'gni_class'}, axis=1)
    gni.year = gni.year.astype(int)
    gni.replace(country_replace_dict, inplace=True)
    df_crawled = df_crawled.merge(gni, on=['country', 'year'], how='left')
    df_missing = df_crawled[pd.isna(df_crawled.gni_class)]
    print('Lost Events for class:', len(df_missing))
    return df_crawled[~pd.isna(df_crawled.gni_class)], df_missing


def load_gni_region(df_crawled, gni_path='worldbank/economies_class.csv'):
    regional_class = pd.read_csv(gni_path, sep=';', encoding='latin1')
    # regional_groups.rename({'GroupName': 'gni_region', 'CountryName': 'country'}, axis=1, inplace=True)
    regional_class.replace(country_replace_dict, inplace=True)
    regional_class = regional_class[~pd.isna(regional_class.Region)].rename(
        {'Economy': 'country', 'Region': 'gni_region', 'Income group': 'gni_group'}, axis=1)
    df_crawled = df_crawled.merge(regional_class[['country', 'gni_region']], on='country', how='left')
    df_missing = df_crawled[pd.isna(df_crawled.gni_region)]
    print('Lost Events for region: ', len(df_missing))
    return df_crawled[~pd.isna(df_crawled.gni_region)], df_missing


def load_population(df_crawled, pop_path='worldbank/population/population.csv'):
    df_population = pd.read_csv(pop_path, sep=';')
    df_population = df_population.set_index('Country Name')
    df_population = df_population[['2014', '2015', '2016', '2017', '2018', '2019', '2020']]
    df_population = df_population.unstack().reset_index().rename(
        {'Country Name': 'country', 'level_0': 'year', 0: 'population'}, axis=1)
    df_population.replace(country_replace_dict, inplace=True)
    df_population = df_population.dropna()
    df_population.year = df_population.year.astype(int)
    df_population.population = df_population.population.astype(int)

    df_crawled = df_crawled.merge(df_population, on=['country', 'year'], how='left')
    df_missing = df_crawled[df_crawled.population < 1]
    print('Lost Events for population: ', len(df_missing))
    return df_crawled[df_crawled.population > 0].copy(), df_missing


def load_gdp(df_crawled, path='gdp/gdp_all.csv'):
    gdp_all = pd.read_csv(path)
    gdp_all.replace(country_replace_dict, inplace=True)
    df_crawled = df_crawled.merge(
        gdp_all[['Country/Area', 'Year', 'GDP_pc', 'GDP', 'oecd']], left_on=['country', 'year'],
        right_on=['Country/Area', 'Year'], how='left', copy=True)
    df_missing = df_crawled[df_crawled.country.isin(df_crawled[pd.isna(df_crawled.Year)].country.unique())]
    df_crawled = df_crawled[~df_crawled.country.isin(df_crawled[pd.isna(df_crawled.Year)].country.unique())]

    print('Lost Events for gdp: ', len(df_missing))
    return df_crawled, df_missing


def load_if_in_country_or_lang(df_crawled, path_langs='languages/langs.csv'):
    lang_map = pd.read_csv(path_langs)
    # whether it is an official language in the target country
    df_crawled['in_code_lang'] = df_crawled.apply(lambda row: version_language[row.code] in (
        lang_map[lang_map.Country == row.country]['Official language'].values if len(
            lang_map[lang_map.Country == row.country]) > 0 else ['0'])[0], axis=1)
    # whether it matches the prespecified region - not good feature
    df_crawled['in_code_region'] = df_crawled.apply(lambda row: row.country in countries_version[row.code], axis=1)
    return df_crawled


def load_views(path_views='viewsglobal/all_views_by_country.csv'):
    df_views = pd.read_csv(path_views)
    df_views['month_year'] = df_views.apply(lambda row: f'{row.year}-{row.month:02d}', axis=1)
    df_views['date_month'] = df_views.apply(lambda row: pd.to_datetime(f'{row.month_year}-01'), axis=1)
    df_views.replace(country_replace_dict, inplace=True)
    return df_views


def categorize_views_for_df(df, views_col='views_baseline'):
    '''
    Original Wikipedia classification # our classification
    1000000000-9999999999 #  >10^9
    100000000-999999999 #  >10^8
    10000000-99999999 # >10^7
    1000000-9999999 # >10^6
    100000-999999 # >10^5
    10000-99999 # >10^4
    1000-9999 # >10^3
    100-999 # >10^2
    0 # >=0
    '''

    def categorize_views(views):
        if views > 10 ** 9:
            return '>10^9'
        if views > 10 ** 8:
            return '>10^8'
        if views > 10 ** 7:
            return '>10^7'
        if views > 10 ** 6:
            return '>10^6'
        if views > 10 ** 5:
            return '>10^5'
        if views > 10 ** 4:
            return '>10^4'
        if views > 10 ** 3:
            return '>10^3'
        if views > 10 ** 2:
            return '>10^2'
        return '>=0'

    df['views_baseline_cat'] = df[views_col].apply(lambda t: categorize_views(t))
    return df


def compute_view_baseline(df_views, start_date_views='2015-05-01', months_before=5, func_agg=np.median):
    df_views.dropna(inplace=True)
    min_date = pd.to_datetime(start_date_views) + relativedelta(months=months_before)
    df_agg_views = df_views[['code', 'date_month', 'month_year', 'country', 'views_ceil']].copy()
    df_agg_views['phase_before'] = pd.to_datetime(df_agg_views.date_month.dt.date - relativedelta(months=months_before))
    df_views_rel = df_agg_views[df_agg_views.date_month >= min_date].copy()
    views_baseline = df_views_rel.apply(
        lambda row: func_agg(df_agg_views[(df_agg_views.code == row.code) & (df_agg_views.country == row.country) &
                                          (df_agg_views.date_month >= row.phase_before) &
                                          (df_agg_views.date_month < row.date_month)].views_ceil), axis=1)
    df_views_rel['views_baseline'] = views_baseline.fillna(0)
    df_views_rel.replace(country_replace_dict, inplace=True)
    return df_views_rel, min_date


def replace_country_names(df, inplace=False):
    if inplace:
        df.replace(country_replace_dict, inplace=True)
    else:
        return df.replace(country_replace_dict)
