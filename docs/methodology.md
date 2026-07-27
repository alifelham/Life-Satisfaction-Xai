# Methodology

The analysis follows these stages:

1. Load the SHILD 2012 Stata dataset.
2. Map `Content` and `Very content` to class 0 and `Discontent` and `Very discontent` to class 1; remove other target responses.
3. Remove the target, identifiers, survey weights, response method, and study-excluded variables.
4. Drop features with at least 18% missing values and remove duplicate rows.
5. Apply the categorical encodings defined in `src/life_satisfaction/mappings.py`.
6. Split the data 80/20 with `random_state=20`, without stratification.
7. Remove zero-variance variables and variables above the 0.8 correlation threshold.
8. Replace training values beyond mean ±2 standard deviations with the training median.
9. Fit `IterativeImputer(max_iter=10, random_state=21)` on the training partition and transform both partitions.
10. Apply `SMOTETomek(sampling_strategy=0.4, random_state=21)` followed by `RandomUnderSampler(sampling_strategy=1)`.
11. Run five-fold RFECV with a random forest and ROC-AUC scoring.
12. Run the 0.0094 random-forest importance analysis and use the study's 27 selected variables for model analysis.
13. Train the model benchmark and soft-voting ensemble.
14. Generate classification metrics, statistical summaries, explainability outputs, and diagnostic figures.

All analysis parameters are centralized in [`../configs/paper.yaml`](../configs/paper.yaml).
