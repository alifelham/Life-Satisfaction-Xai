#!/usr/bin/env python
"""Verify repository structure, notebook integrity, and study figures."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "CITATION.cff",
    "DATA_CARD.md",
    "MODEL_CARD.md",
    "pyproject.toml",
    "configs/paper.yaml",
    "src/life_satisfaction/pipeline.py",
    "notebooks/paper/published_experiment.ipynb",
    "results/reference/README.md",
    "results/reference/published_metrics.json",
    "results/reference/figure_manifest.json",
    "docs/methodology.md",
    "docs/results.md",
    "docs/technical_notes.md",
    "metadata/paper_artifacts.json",
    "metadata/notebook_contract.json",
    "scripts/verify_notebook_contract.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing required repository files: " + ", ".join(missing))

    published = json.loads((ROOT / "results/reference/published_metrics.json").read_text())
    if len(published) != 10:
        raise SystemExit("Published metrics must contain all ten benchmark entries.")

    metadata = json.loads((ROOT / "metadata/paper_artifacts.json").read_text())
    notebook = ROOT / metadata["paper_notebook"]["path"]
    actual_hash = sha256(notebook)
    expected_hash = metadata["paper_notebook"]["sha256"]
    if actual_hash != expected_hash:
        raise SystemExit(
            f"paper notebook checksum mismatch: {actual_hash} != {expected_hash}"
        )

    figure_manifest = json.loads(
        (ROOT / "results/reference/figure_manifest.json").read_text()
    )
    if len(figure_manifest) != 22:
        raise SystemExit("paper figure manifest must contain 22 notebook figures.")
    for filename, record in figure_manifest.items():
        figure = ROOT / "results/reference/figures" / filename
        if not figure.exists():
            raise SystemExit(f"paper figure is missing: {filename}")
        if figure.stat().st_size != record["bytes"]:
            raise SystemExit(f"paper figure size mismatch: {filename}")
        if sha256(figure) != record["sha256"]:
            raise SystemExit(f"paper figure checksum mismatch: {filename}")
        if not isinstance(record.get("source_cell_index"), int):
            raise SystemExit(f"paper figure lacks its analysis-cell identifier: {filename}")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_notebook_contract.py")],
        cwd=ROOT,
        check=True,
    )
    print(
        "Repository structure, published results, notebook contract, and "
        "study-figure checksums are valid."
    )


if __name__ == "__main__":
    main()
