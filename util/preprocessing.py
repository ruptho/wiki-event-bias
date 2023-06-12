import numpy as np
import pandas as pd

countries_version = {'en': ['United States'], 'it': ['Italy'], 'de': ['Germany'], 'es': ['Spain']}
version_language = {'en': 'English', 'it': 'Italian', 'de': 'German', 'es': 'Spanish'}

country_replace_dict = {
    'Bahamas, The': 'Bahamas',
    'The Bahamas': 'Bahamas',
    'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
    'Congo, the Democratic Republic of the': 'Democratic Republic of the Congo',
    'Egypt, Arab Rep.': 'Egypt',
    'Faroe Islands': 'Faeroe Islands',
    'Gambia, The': 'Gambia',
    'The Gambia': 'Gambia',
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
    "People's Republic of China": "China",
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
    "Virgin Islands, British": "British Virgin Islands",
    "United States of America": "United States"
}


def replace_country_names(df, inplace=False):
    if inplace:
        df.replace(country_replace_dict, inplace=True)
    else:
        return df.replace(country_replace_dict)


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


def label_language(df, path_langs='data/supp/langs.csv'):
    lang_map = pd.read_csv(path_langs)

    def has_lang(row):
        row_country = lang_map[lang_map.Country == row.country]
        if len(row_country) > 0:
            languages = list(
                set(row_country['Official language'].values))  # | set(row_country['National language'].values)
            return np.any([version_language[row.code] in str(l) for l in languages])
        return False

    df['in_code_lang'] = df.apply(has_lang, axis=1)
    return df
