const se_scraper = require('se-scraper');

(async () => {
    let scrape_job = {
        search_engine: 'bing_news',
        output_file: '../results/de_bingnews_missing.json',
        num_pages: 1,
        keyword_file: 'de_articles_missing.txt', 
        screen_output: false,
        block_assets: true,
        bing_settings: {
            'setlang': 'de',
            'setMkt': 'de-DE'
        }
    };

    var results = await se_scraper.scrape({}, scrape_job);

    console.dir(results, {depth: null, colors: true});
})();