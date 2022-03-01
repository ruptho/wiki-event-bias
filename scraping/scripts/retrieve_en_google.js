const se_scraper = require('se-scraper');

(async () => {
    let scrape_job = {
        search_engine: 'google_news_old',
        output_file: '../results/en_google_3.json',
        num_pages: 1,
        keyword_file: 'en_articles_3.txt', 
        screen_output: false,
        block_assets: true,
        google_settings: {
            gl: 'us', // The gl parameter determines the Google country to use for the query.
            hl: 'en', // The hl parameter determines the Google UI language to return results.
        }
    };

    var results = await se_scraper.scrape({}, scrape_job);

    console.dir(results, {depth: null, colors: true});
})();