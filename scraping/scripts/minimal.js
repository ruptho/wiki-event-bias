const se_scraper = require('se-scraper');

(async () => {
    let scrape_job = {
        search_engine: 'google_news_old',
        output_file: '../results/google_news_test.json',
        keywords: ['Trigana-Air-Service-Flug 267', 'Test'],
        num_pages: 1,
        screen_output: false,
        sleep_range: [1, 3], 
        block_assets: true,
        google_settings: {
            gl: 'de', // The gl parameter determines the Google country to use for the query.
            hl: 'en', // The hl parameter determines the Google UI language to return results.
        }
    };

    var results = await se_scraper.scrape({}, scrape_job);

    console.dir(results, {depth: null, colors: true});
})();
