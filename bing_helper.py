import json
import re
import pandas as pd


def load_json_files(files):
    data = {}
    for file in files:
        with open(f'{file}') as json_data:
            data = {**data, **json.load(json_data)}
            print(f'LOAD FROM {file}, Articles now: {len(data)}')
    return data


def get_hits_from_json(json_data, serp='bing'):
    bing_news_hits = []
    i = 0
    for title, crawl_res in json_data.items():
        # get result string and clean
        if '1' not in crawl_res:
            print(f'Error in results for {title}: |{crawl_res}|')
            continue

        res_string = crawl_res['1']['num_results']
        num_string = re.sub('[^0-9]', '', res_string)

        if len(num_string) == 0:
            print(f'Error when parsing hits for {title}: |{res_string}|')
            continue

        bing_news_hits.append([title, int(num_string)])

        # debug output
        # if i%500 == 0:
        #    print(title, res_string, num_string)
        # i+=1
    return pd.DataFrame(bing_news_hits, columns=['pagetitle', f'{serp}_hits'])


def load_lang_news_hits(lang_files: dict, serp='bing'):
    df_langs = []
    for lang, files in lang_files.items():
        print(f'Loading {lang}')
        json_data = load_json_files(files)
        df_hits = get_hits_from_json(json_data, serp)
        df_hits['code'] = lang
        df_langs.append(df_hits)

    return pd.concat(df_langs)
