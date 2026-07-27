# Technical notes

## Result records

`results/reference/published_metrics.json` contains the classical-model results reported in the article. Analysis runs write their metrics to `results/generated/`.

Every run records the configuration, software versions, dataset checksum, shapes, class counts, selected variables, and randomness settings in `run_manifest.json`.

## Analysis notebook

`notebooks/paper/published_experiment.ipynb` contains the end-to-end analysis notebook. Its checksum and workflow contract are verified by the repository test suite. The package implementation in `src/life_satisfaction/` provides the maintained command-line workflow.
