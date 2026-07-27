from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PaperConfig:
    """Configuration values for the paper analysis workflow."""

    random_seed_split: int
    random_seed_preprocessing: int
    random_seed_undersampling: int | None
    test_size: float
    missing_column_threshold_percent: int
    correlation_threshold: float
    outlier_standard_deviations: float
    smote_tomek_sampling_strategy: float
    undersampling_strategy: float
    rfecv_folds: int
    rfecv_scoring: str
    feature_importance_threshold: float
    emulate_stata_csv_roundtrip: bool


def load_config(path: str | Path) -> tuple[PaperConfig, dict[str, Any]]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ValueError(f"Invalid configuration file: {source}")
    return PaperConfig(**payload["pipeline"]), payload
