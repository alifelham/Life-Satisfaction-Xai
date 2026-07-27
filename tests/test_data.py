from pathlib import Path

import pytest

from life_satisfaction.data import load_raw_data, sha256


def test_load_csv_and_checksum(tmp_path: Path):
    path = tmp_path / "sample.csv"
    path.write_text("A1,age\nContent,40\n", encoding="utf-8")
    frame = load_raw_data(path)
    assert frame.to_dict("records") == [{"A1": "Content", "age": 40}]
    assert len(sha256(path)) == 64


def test_missing_dataset_has_actionable_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_raw_data(tmp_path / "missing.dta")


def test_unsupported_dataset_format(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("not a dataset", encoding="utf-8")
    with pytest.raises(ValueError, match="Stata"):
        load_raw_data(path)
