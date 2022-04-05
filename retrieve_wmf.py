import pandas as pd
import requests as rq
from mw.lib import title

WM_API = 'https://wikimedia.org/api/rest_v1'
MW_API = f"https://WIKI.wikipedia.org/w/api.php"

headers = {"User-Agent": "th.ruprechter@gmail.com"}

EDITOR_ACT_ALL = 'all-activity-levels'
EDITOR_ACT_1_4 = '1..4-edits'
EDITOR_ACT_5_24 = '5..24-edits'
EDITOR_ACT_25_99 = '25..99-edits'
EDITOR_ACT_100 = '100..-edits'
EDITOR_ALL_ACTIVITY_LEVELS = [EDITOR_ACT_ALL, EDITOR_ACT_1_4, EDITOR_ACT_5_24, EDITOR_ACT_25_99, EDITOR_ACT_100]

EDITOR_TYPE_ANON = 'anonymous'
EDITOR_TYPE_USER = 'user'
EDITOR_TYPE_GBOT = 'group-bot'
EDITOR_TYPE_NBOT = 'name-bot'
EDITOR_ALL_TYPES = [EDITOR_TYPE_ANON, EDITOR_TYPE_USER, EDITOR_TYPE_GBOT, EDITOR_TYPE_NBOT]

PV_ACCESS_ALL = 'all-access'
PV_ACCESS_DESKTOP = 'desktop'
PV_ACCESS_MOBAPP = 'mobile-app'
PV_ACCESS_MOBWEB = 'mobile-web'

PV_AGENT_ALL = 'all-agents'
PV_AGENT_USER = 'user'
PV_AGENT_SPIDER = 'spider'
PV_AGENT_AUTOMATED = 'automated'

PV_GRANULARITY_HOUR = 'hourly'
PV_GRANULARITY_DAY = 'daily'
PV_GRANULARITY_MONTH = 'monthly'


def retrieve_pageviews_aggregate(lang, start=20140101, end=20220101, access=PV_ACCESS_ALL,
                                 granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL, legacy=False):
    if not legacy:
        url = f'{WM_API}/metrics/pageviews/aggregate/{lang}.wikipedia.org/{access}/{agent}/{granularity}/{start}/{end}'
    else:
        url = f'{WM_API}/metrics/legacy/pagecounts/aggregate/{lang}.wikipedia.org/all-sites/{granularity}/{start}00/{end}00'

    response = rq.get(url, headers=headers)
    lang_result = {'date': [], 'views': []}
    # print(url, response.text)
    string = response.json()

    for res in string['items']:
        lang_result['date'].append(pd.to_datetime(res['timestamp'][:-2], format='%Y%m%d'))
        lang_result['views'].append(res['views'] if not legacy else res['count'])
    return pd.DataFrame(lang_result)


def retrieve_pageviews_aggregate_all_langs(codes, start=20140101, end=20220101, access=PV_ACCESS_ALL,
                                           granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL, legacy=False):
    df_lang_list = []
    for code in codes:
        df_lang = retrieve_pageviews_aggregate(code, start, end, access, granularity, agent, legacy)
        df_lang['code'] = code
        df_lang_list.append(df_lang)
    return pd.concat(df_lang_list)


def retrieve_pageviews_by_country_for_project(lang, year: int, month: int, access=PV_ACCESS_ALL):
    # Lists the pageviews to this project, split by country of origin for a given month.
    # Because of privacy reasons, pageviews are given in a bucketed format,
    # and countries with less than 100 views do not get reported. Stability: experimental
    # https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews#Pageviews_split_by_country

    # https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews/Pageviews_per_project
    #  With the buckets we were unable to answer basic questions as "how many pageviews there were from the U.S. in
    #  English Wikipedia"? So we had to come up with a solution that prevented PII attacks on small wikis/countries
    #  but also didn't harm the value of the metric in bigger wikis where there is no such threat.
    #  We decided to round all pageview values to the nearest "thousand ceiling" using the following expression in SQL:
    url = f'{WM_API}/metrics/pageviews/top-by-country/{lang}.wikipedia.org/{access}/{year}/{month:02}'
    response_string = rq.get(url, headers=headers).json()
    results = response_string['items'][0]
    base_res = {'code': lang, 'year': results['year'], 'month': results['month'], 'access': results['access']}
    df_result = pd.DataFrame([(base_res | country_result) for country_result in results['countries']])
    return df_result.rename({'country': 'country_code'}, axis=1)


def retrieve_pageviews_by_country_for_projects(langs, year: int, month: int, access=PV_ACCESS_ALL):
    return pd.concat(
        [retrieve_pageviews_by_country_for_project(lang, year, month, access=access) for lang in langs])


def retrieve_pageviews_by_country_for_projects_and_months(langs, start_year: int, start_month: int, end_year=2020,
                                                          end_month=12, access=PV_ACCESS_ALL):
    results_month = []
    for year in range(start_year, end_year + 1):
        for month in range(start_month if year == start_year else 1, end_month + 1 if year == end_year else 13):
            print(f'Retrieving {year}-{month} for {langs}')
            results_month.append(
                pd.concat([retrieve_pageviews_by_country_for_project(lang, year, month, access) for lang in langs]))
    return pd.concat(results_month).sort_values(['code', 'year', 'month', 'rank'])


def retrieve_pageviews_per_article(article, lang, start=20140101, end=20220101, access=PV_ACCESS_ALL,
                                   granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL):
    url = f'{WM_API}/metrics/pageviews/per-article/{lang}.wikipedia.org/{access}/{agent}/{title.normalize(article)}/' \
          f'{granularity}/{start}/{end}'

    response = rq.get(url, headers=headers)
    lang_result = {'date': [], 'views': []}
    # print(url, response.text)
    string = response.json()

    if 'items' in string:
        for res in string['items']:
            lang_result['date'].append(pd.to_datetime(res['timestamp'][:-2], format='%Y%m%d'))
            lang_result['views'].append(res['views'])

    return pd.DataFrame(lang_result)


def retrieve_pageviews_for_articles(articles, lang, start=20140101, end=20220101, access=PV_ACCESS_ALL,
                                    granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL):
    df_retrieved = []
    for article in articles:
        df_art = retrieve_pageviews_per_article(article, lang, start, end, access, granularity, agent)
        df_art['article'] = article
        df_art['norm_article'] = title.normalize(article)
        df_retrieved.append(df_art)
    return pd.concat(df_retrieved)


def get_langlinks_for_country_articles_and_langs(country_articles, langs, name_synonyms={}, session=None):
    if session is None:
        session = rq.Session()
    retrieved_data = []
    retrieved_dfs = []
    for country_article in country_articles:
        api_path = MW_API.replace('WIKI', 'en')
        params = {
            "action": "query",
            "titles": title.normalize(
                country_article if country_article not in name_synonyms else name_synonyms[country_article]),
            "prop": "langlinks",
            "format": "json",
            "lllimit": 500
        }
        response = session.get(url=api_path, params=params)
        data = response.json()
        retrieved_data.append(data)

        for lang in langs:
            result_dict = list(data['query']['pages'].values())[0]
            if 'langlinks' in result_dict:
                found = False
                for langlinks in result_dict['langlinks']:
                    if langlinks['lang'] == lang:
                        retrieved_dfs.append([country_article, lang, langlinks['*']])
                        found = True
                        break
                if not found:
                    retrieved_dfs.append([country_article, lang, None])
            else:
                retrieved_dfs.append([country_article, lang, None])

    return retrieved_data, pd.DataFrame(retrieved_dfs, columns=['en_article', 'lang', 'langlink'])


def retrieve_pageviews_for_articles_across_langs(df_langlinks, start=20140101, end=20220101, combine_rows=True,
                                                 access=PV_ACCESS_ALL, granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL):
    # use english as baseline, then store row-wise
    df_retrieved = []

    for en_article, val_ll in df_langlinks.groupby('en_article'):
        df_art = retrieve_pageviews_per_article(en_article, 'en', start, end, access, granularity, agent)
        if combine_rows:
            df_art.insert(0, 'en_article', en_article)
            df_art.rename({'views': 'views_en'}, axis=1, inplace=True)
        else:
            df_art['code'] = 'en'
            df_retrieved.append(df_art)

        for lang, langlink in zip(val_ll['lang'], val_ll['langlink']):
            df_lang = retrieve_pageviews_per_article(langlink, lang, start, end, access, granularity, agent)
            print(lang, langlink)
            if combine_rows:
                df_lang.rename({'views': f'views_{lang}'}, axis=1, inplace=True)
                df_art = df_art.merge(df_lang, on='date', how='left').fillna(0)
            else:
                df_lang['code'] = lang
                df_retrieved.append(df_lang)
        if combine_rows:
            df_retrieved.append(df_art)
    return pd.concat(df_retrieved)


def retrieve_pageviews_for_articles_and_langs(articles, langs, start=20140101, end=20220101, access=PV_ACCESS_ALL,
                                              granularity=PV_GRANULARITY_DAY, agent=PV_AGENT_ALL):
    df_retrieved = []
    for lang in langs:
        df_lang = retrieve_pageviews_for_articles(articles, lang, start, end, access, granularity, agent)
        df_lang['code'] = lang
        df_retrieved.append(df_lang)
    return pd.concat(df_retrieved)
