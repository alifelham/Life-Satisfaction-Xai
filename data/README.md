# Dataset

The study uses the **Survey of Health, Impairment and Living Conditions in Denmark (SHILD)** dataset.

- Dataset DOI: https://doi.org/10.5061/dryad.qd2nj
- License reported by the original repository: CC0 1.0
- Expected paper filename: `SHILD2012_cens_eng_16-64.dta`
- paper notebook input shape: 18,957 rows × 243 columns

Download it manually from the authoritative record:

1. Visit https://datadryad.org/dataset/doi:10.5061/dryad.qd2nj.
2. Select **Download dataset**.
3. Extract the download when necessary.
4. Copy `SHILD2012_cens_eng_16-64.dta` to:

```text
data/raw/SHILD2012_cens_eng_16-64.dta
```

Do not rename a different file to match this path. The dataset is intentionally not committed. A full pipeline execution run records the selected file's SHA-256 checksum and observed dimensions in `run_manifest.json`.
