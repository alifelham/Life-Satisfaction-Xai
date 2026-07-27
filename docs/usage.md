# Usage guide

## Local environment

Python 3.10 is recommended. The dependency files provide pinned versions for the analysis environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-paper.txt
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

## Dataset

Open the [Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.qd2nj), select **Download dataset**, and place `SHILD2012_cens_eng_16-64.dta` in `data/raw/`.

## Analysis pipeline

```bash
python -m life_satisfaction.pipeline \
  --data data/raw/SHILD2012_cens_eng_16-64.dta \
  --config configs/paper.yaml \
  --output results/generated
```

The command generates metrics, selected-feature metadata, test predictions, odds
ratios, Wilcoxon signed-rank statistics, trained model files, core figures, and a
run manifest.

To run every analysis used in the study, including supplementary models,
20-fold validation, the XGBoost parameter search, local LIME explanations,
learning curves, and calibration diagrams:

```bash
python -m life_satisfaction.pipeline \
  --data data/raw/SHILD2012_cens_eng_16-64.dta \
  --config configs/paper.yaml \
  --output results/generated \
  --all-analyses
```

## Docker

```bash
docker build -t life-satisfaction-xai:1.2.0 .
docker run --rm \
  -v "$PWD/data/raw:/workspace/data/raw:ro" \
  -v "$PWD/results/generated:/workspace/results/generated" \
  life-satisfaction-xai:1.2.0 \
  --data data/raw/SHILD2012_cens_eng_16-64.dta \
  --config configs/paper.yaml \
  --output results/generated
```

The dataset remains outside the image and is mounted read-only.

## Conda

```bash
conda env create -f environment.yml
conda activate life-satisfaction-xai
python -m pip install -e .
```

## Verification

```bash
python scripts/verify_repository.py
python scripts/verify_notebook_contract.py
pytest -q
ruff check .
```
