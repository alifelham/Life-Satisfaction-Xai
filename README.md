# Predicting Life Satisfaction Using Machine Learning and Explainable AI

[![Paper](https://img.shields.io/badge/Heliyon-10.1016%2Fj.heliyon.2024.e31158-blue)](https://doi.org/10.1016/j.heliyon.2024.e31158)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of:

> Alif Elham Khan, Mohammad Junayed Hasan, Humayra Anjum, Nabeel Mohammed, and Sifat Momen. “Predicting life satisfaction using machine learning and explainable AI.” *Heliyon* 10 (2024), e31158.

The project provides the data-processing pipeline, feature-selection workflow,
classical machine-learning models, soft-voting ensemble, evaluation utilities,
explainability analyses, figures, and interactive demonstration used in the study.

## Motivation

Life satisfaction is an important component of subjective well-being, with relevance
to mental health, productivity, social research, and public policy. Population-scale
assessment, however, often depends on lengthy questionnaires and resource-intensive
data collection, making it difficult to determine which factors are most informative.

This study investigates three related questions:

1. Which health, psychological, social, and economic factors are most informative
   for understanding life satisfaction?
2. How effectively can machine-learning models distinguish between content and
   discontent respondents?
3. Can explainable AI make those predictions more understandable to researchers
   and practitioners?

Using responses from approximately 19,000 participants in the Danish SHILD survey,
the study analyzes 243 survey variables and identifies a focused set of 27 questions.
It then evaluates multiple machine-learning models, constructs a soft-voting
ensemble, and applies statistical and explainability techniques to examine the
factors influencing individual predictions.

## Method overview

```text
SHILD 2012 survey
  → binary life-satisfaction target
  → study exclusions and missing-data filtering
  → categorical encoding
  → 80/20 train–test split
  → variance, correlation, and outlier processing
  → iterative imputation
  → SMOTE–Tomek and random undersampling
  → RFECV and feature-importance filtering
  → 27 selected survey variables
  → model benchmark and soft-voting ensemble
  → evaluation, statistical analysis, LIME, and diagnostic figures
```

The full methodology and configuration are documented in
[`docs/methodology.md`](docs/methodology.md) and
[`configs/paper.yaml`](configs/paper.yaml).

## Published results

| Model | Accuracy | Macro F1 | Macro precision | Macro recall |
|---|---:|---:|---:|---:|
| Random forest | 93.8% | 70.6% | 72.0% | 69.3% |
| Gradient boosting | 92.2% | 70.3% | 67.9% | 73.7% |
| XGBoost | 93.0% | 68.5% | 68.7% | 68.2% |
| **Soft-voting ensemble** | **93.6%** | **73.0%** | **71.9%** | **74.3%** |

See [`docs/results.md`](docs/results.md) for the complete model table.

## Installation

Python 3.10 is recommended.

```bash
git clone https://github.com/alifelham/life-
[![CI](https://github.com/alifelham/Life-Satisfaction-Xai/actions/workflows/ci.yml/badge.svg)](https://github.com/alifelham/Life-Satisfaction-Xai/actions/workflows/ci.yml)satisfaction-xai.git
cd life-satisfaction-xai
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-paper.txt
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Docker and Conda setup options are also provided in
[`docs/usage.md`](docs/usage.md).

## Dataset

The study uses the Survey of Health, Impairment and Living Conditions in Denmark
(SHILD):

1. Open the [Dryad dataset record](https://datadryad.org/dataset/doi:10.5061/dryad.qd2nj).
2. Select **Download dataset**.
3. Locate `SHILD2012_cens_eng_16-64.dta`.
4. Place the file at `data/raw/SHILD2012_cens_eng_16-64.dta`.

The dataset is governed by the terms of its Dryad record and is not distributed in
this repository. See [`data/README.md`](data/README.md) and
[`DATA_CARD.md`](DATA_CARD.md) for schema and handling details.

## Run the analysis

```bash
python -m life_satisfaction.pipeline \
  --data data/raw/SHILD2012_cens_eng_16-64.dta \
  --config configs/paper.yaml \
  --output results/generated
```

Run the complete analysis suite, including supplementary models, cross-validation,
parameter search, LIME explanations, learning curves, and calibration diagrams:

```bash
python -m life_satisfaction.pipeline \
  --data data/raw/SHILD2012_cens_eng_16-64.dta \
  --config configs/paper.yaml \
  --output results/generated \
  --all-analyses
```

The pipeline writes model metrics, predictions, selected features, cross-validation
summaries, statistical analyses, figures, serialized estimators, and a run manifest
to `results/generated/`.

Study figures and diagnostic outputs are catalogued in the
[`figure catalog`](results/reference/figures/README.md).

## Interactive demonstration

After running the analysis:

```bash
python -m pip install -e ".[app]"
life-satisfaction-demo --model results/generated/best_model.pkl
```

The interface accepts encoded survey-variable values. It is intended for research
demonstration and is not a clinical, psychological, or individual decision-support
tool.

## Quality checks

```bash
python scripts/verify_repository.py
python scripts/verify_notebook_contract.py
pytest -q
ruff check .
```

Continuous integration runs the repository checks, linting, and test suite on every
push and pull request.

## Repository structure

```text
configs/                 analysis configuration
data/                    dataset acquisition and placement instructions
docs/                    methodology, usage, results, and technical notes
metadata/                machine-readable paper and workflow metadata
notebooks/paper/         paper analysis notebook
results/reference/       published metrics and paper figures
scripts/                 command-line utilities
src/life_satisfaction/   analysis package
tests/                   automated software tests
```

## Citation

```bibtex
@article{khan2024predicting,
  title   = {Predicting life satisfaction using machine learning and explainable AI},
  author  = {Khan, Alif Elham and Hasan, Mohammad Junayed and Anjum, Humayra and Mohammed, Nabeel and Momen, Sifat},
  journal = {Heliyon},
  volume  = {10},
  year    = {2024},
  pages   = {e31158},
  doi     = {10.1016/j.heliyon.2024.e31158}
}
```

## License and responsible use

Code is released under the [MIT License](LICENSE). Before using the software or
model outputs, review [`MODEL_CARD.md`](MODEL_CARD.md) and
[`DATA_CARD.md`](DATA_CARD.md).
