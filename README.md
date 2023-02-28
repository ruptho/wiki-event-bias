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

As `crossval_results` and `shap` folders are too large for GitHub, we provide them here: <TODO>  zenodo links

## Notebooks and Scripts
- `scripts`-Directory: Contains helper and util files called by the notebooks in the main directory
- `notebook1.ipynb` TODO describe notebooks