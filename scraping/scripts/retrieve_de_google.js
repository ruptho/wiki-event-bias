const se_scraper = require('se-scraper');

(async () => {
    let scrape_job = {
        search_engine: 'google_news_old',
        output_file: '../results/de_google.json',
        num_pages: 1,
        keyword_file: 'de_articles.txt', 
        screen_output: false,
        block_assets: true,
        sleep_range: [1, 5], 
        google_settings: {
            gl: 'de', // The gl parameter determines the Google country to use for the query.
            hl: 'de', // The hl parameter determines the Google UI language to return results.
        }
    };
com
    var results = await se_scraper.scrape({}, scrape_job);

    console.dir(results, {depth: null, colors: true});
})();