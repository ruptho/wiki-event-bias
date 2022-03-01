const se_scraper = require('se-scraper');

(async () => {
    let scrape_job = {
        search_engine: 'bing_news',
        output_file: '../results/it_bingnews_missing.json',
        num_pages: 1,
        keyword_file: 'it_articles_missing.txt', 
        //keywords: ['Test'],
        screen_output: false,
        block_assets: true,
        bing_settings: {
            'setlang': 'it',
            'setMkt': 'it-IT'
        }
    };

    var results = await se_scraper.scrape({}, scrape_job);

    console.dir(results, {depth: null, colors: true});
})();