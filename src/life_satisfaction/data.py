from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_data(
    path: str | Path,
    *,
    emulate_notebook_csv_roundtrip: bool = True,
) -> pd.DataFrame:
    """Load SHILD using the same Stata→CSV path used by the notebook.

    The publication notebook first loaded the Stata file, wrote an unindexed CSV,
    then read that CSV back into pandas. The in-memory round trip below preserves
    that conversion while avoiding an undocumented intermediate file.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found: {source}. See data/README.md for acquisition instructions."
        )

    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix != ".dta":
        raise ValueError("Expected a Stata .dta file or the notebook's CSV export.")

    frame = pd.read_stata(source)
    if not emulate_notebook_csv_roundtrip:
        return frame

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return pd.read_csv(buffer)
