const se_scraper = require('se-scraper');

(async () => {
    let scrape_job = {
        search_engine: 'bing_news',
        output_file: '../results/en_bingnews_missing.json',
        num_pages: 1,
        keyword_file: 'en_articles_missing.txt',
        //keywords: ['2017 Preakness Stakes'],
        screen_output: false,
        block_assets: true,
        bing_settings: {
            setlang: 'en',
            setMkt: 'en-US'
        },
    };

    var results = await se_scraper.scrape({}, scrape_job);

    console.dir(results, {depth: null, colors: true});
})();