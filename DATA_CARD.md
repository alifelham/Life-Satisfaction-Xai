# Dataset card: SHILD 2012

## Source

The publication pipeline uses `SHILD2012_cens_eng_16-64.dta` from the Survey of Health Impairment and Living Conditions in Denmark. The authoritative dataset record is Dryad DOI `10.5061/dryad.qd2nj`.

## Population and paper shape

The paper notebook reads 18,957 rows and 243 columns before target filtering. The survey covers people aged 16–64 in Denmark.

## Acquisition and repository policy

Open the [Dryad dataset page](https://datadryad.org/dataset/doi:10.5061/dryad.qd2nj), select **Download dataset**, and place the downloaded `SHILD2012_cens_eng_16-64.dta` file at `data/raw/SHILD2012_cens_eng_16-64.dta`. The repository does not use a private mirror or depend on a programmatic Dryad API.

The dataset is not committed. Every full pipeline execution run records the local file's SHA-256 checksum and original shape in `run_manifest.json`.

## Target construction

The notebook maps `A1` as follows:

- `Content` and `Very content` → class 0
- `Discontent` and `Very discontent` → class 1
- all other responses → excluded

