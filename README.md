# Poor attention: The wealth and regional gaps in event attention and coverage on Wikipedia

Repository for "Poor attention: The wealth and regional gaps in event attention and coverage on Wikipedia" as currently under review.

This repository contains the preprocessed data files as well as code for the main experiments conducted during the creation of the paper.

## Dataset and intermediate results
- Event data: Relevant event articles used for computation, as crawled from Wikipedia through Wikidata identifiers
  - `data/events/all_events.csv.gz`: All Wikipedia event articles collected for this study
  - `data/events/disasters_with_deaths.csv.gz`: Articles about disasters with additional metadata (e.g., deaths)
- `crossval_results`: Cross validaiton results and final models for SVC, SVM, Random Forests, and XGBoosted Trees.
- `shap`: SHAP values as computed from the best model configurations as by the best-performing model identified through cross-validation
- `figures`: All figures displayed in the paper, along with supplementary results

As `crossval_results` and `shap` folders are too large for GitHub, we provide them on Zenodo [1] alongside the other content of this GitHub repository.

## Notebooks and Scripts
- `util`-Directory: Contains helper and util files used by the notebooks in the main directory
- Experiments and Visualizations: 
  - `model_cross_validation.ipynb`: Model cross validation (grid search) to find best configurations for XGBClassifier/XGBRegressors, RandomForestClassifiers/RandomForestRegressors, and SVC/SVR for predicting views and edits. Results of this step are available under the corresponding zenodo link and should be extracted into the `crossval_results/`-Folder.
  - `generate_SHAP.ipynb`: SHAP value generation for the best-performing XGBoostClassifiers and XGBRegressors. We compute both SHAP [2] as well as ACV Shapley values [3]. Results of this step are available under the corresponding zenodo link and should be extracted into the `shap/`-Folder for this repository.
  - `plot_metrics.ipynb`: Chloropleth (map) plots for article metrics such as article count or median views/edits to articles per region and country
  - `plot_shapval.ipynb`:
    - Line plots for GDP per capita SHAP values across language editions, with regions as markers
    - Chloropleth (map) plots for the effect of regions across article categories
  - `disaster_analysis.ipynb`: Experiment in regards to category-specific features, in this case the number of deaths during a disaster (terror attack, earthquake, etc.)
  
[1] https://doi.org/10.5281/zenodo.7701969<br />
[2] https://github.com/slundberg/shap<br/>
[3] https://github.com/salimamoukou/acv00
