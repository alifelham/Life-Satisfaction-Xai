# Model card

## Purpose

The models estimate binary life-satisfaction classes from SHILD survey variables for research analysis. They are not diagnostic or decision-support systems.

## Primary study model

The soft-voting ensemble contains:

- a random forest with `class_weight={0: 5, 1: 0.1}`, 400 trees, and `random_state=21`;
- a default `GradientBoostingClassifier`;
- the recorded LightGBM configuration.

## Inputs

The analysis uses 27 survey-derived variables following preprocessing, dual resampling, RFECV support, and feature-importance filtering. Their machine-readable names are defined in `src/life_satisfaction/constants.py`.

## Performance records

The article reports 93.6% accuracy and 73.0% macro-F1 for the ensemble. The full model table is available in `docs/results.md`.

## Constraints

- The data are imbalanced and geographically and temporally specific.
- LIME examples are local model explanations, not causal explanations.
