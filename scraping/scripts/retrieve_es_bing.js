const se_scraper = require('se-scraper');
// https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/market-codes
(async () => {
    let scrape_job = {
        search_engine: 'bing_news',
        output_file: '../results/es_bingnews_es-ES_corrected.json',
        num_pages: 1,
        keyword_file: 'es_articles_corrected.txt', 
        // keywords: ['Mar del Plata'],
        screen_output: false,
        block_assets: true,
        bing_settings: {
            'setlang': 'es',
            'setMkt': 'es-ES'
        }
    };

    var results = await se_scraper.scrape({}, scrape_job);

    console.dir(results, {depth: null, colors: true});
})();