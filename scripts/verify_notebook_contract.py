"""Verify the package against the paper notebook contract."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from life_satisfaction.constants import EXCLUDED_COLUMNS, SELECTED_FEATURES, TARGET_MAPPING
from life_satisfaction.mappings import CATEGORY_MAPPINGS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_mappings(source: str) -> dict[str, dict[object, object]]:
    tree = ast.parse(source)
    mappings: dict[str, dict[object, object]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "replace":
            continue
        subject = call.func.value
        if not isinstance(subject, ast.Subscript):
            continue
        if not isinstance(subject.value, ast.Name) or subject.value.id != "df":
            continue
        if not isinstance(subject.slice, ast.Constant) or not isinstance(subject.slice.value, str):
            continue
        if call.args:
            mappings[subject.slice.value] = ast.literal_eval(call.args[0])
    return mappings


def verify(root: Path = ROOT) -> None:
    contract = json.loads((root / "metadata/notebook_contract.json").read_text(encoding="utf-8"))
    notebook_path = root / contract["source"]["path"]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert _sha256(notebook_path) == contract["source"]["sha256"]
    assert len(notebook["cells"]) == contract["source"]["cells"]
    assert sum(c["cell_type"] == "code" for c in notebook["cells"]) == contract["source"]["code_cells"]
    markdown_cells = sum(c["cell_type"] == "markdown" for c in notebook["cells"])
    assert markdown_cells == contract["source"]["markdown_cells"]
    assert notebook["metadata"]["language_info"]["version"] == contract["source"]["python_version_metadata"]

    mapping_source = "".join(notebook["cells"][31]["source"])
    notebook_mappings = _extract_mappings(mapping_source)
    assert len(notebook_mappings) == contract["pipeline"]["categorical_mapping_columns"]
    assert notebook_mappings == CATEGORY_MAPPINGS

    assert TARGET_MAPPING == contract["pipeline"]["target_mapping"]
    assert EXCLUDED_COLUMNS == contract["pipeline"]["excluded_columns"]
    assert SELECTED_FEATURES == contract["selected_features"]

    split_source = "".join(notebook["cells"][35]["source"])
    assert "test_size = 0.2" in split_source and "random_state = 20" in split_source
    balance_source = "".join(notebook["cells"][54]["source"])
    assert "sampling_strategy=.4" in balance_source and "random_state=21" in balance_source
    undersample_source = "".join(notebook["cells"][56]["source"])
    assert "sampling_strategy=1" in undersample_source and "random_state" not in undersample_source
    reassignment_source = "".join(notebook["cells"][58]["source"])
    assert "X_train = Xb" in reassignment_source and "y_train = y_b" in reassignment_source
    rfecv_source = "".join(notebook["cells"][60]["source"])
    assert "StratifiedKFold(5)" in rfecv_source and "scoring='roc_auc'" in rfecv_source
    assert "rfecv.fit(X_train, y_train)" in rfecv_source

    primary_model_sources = {
        88: "SVC(class_weight={0: 1, 1: 1.4})",
        91: "LGBMClassifier(**params)",
        94: "GaussianNB()",
        97: "DecisionTreeClassifier(random_state=21)",
        100: "RandomForestClassifier(oob_score=True)",
        108: "GradientBoostingClassifier(n_estimators=500, learning_rate=1,",
        111: "DecisionTreeClassifier(max_depth=1)",
        114: "LogisticRegression(solver = 'liblinear', penalty = 'l2')",
        124: "n_estimators=600",
    }
    for cell_index, expected in primary_model_sources.items():
        assert expected in "".join(notebook["cells"][cell_index]["source"])

    search_source = "".join(notebook["cells"][121]["source"])
    assert "n_iter=5" in search_source
    assert "scoring='roc_auc'" in search_source
    assert "cv=5" in search_source

    supplementary_sources = {
        104: "RandomForestClassifier(class_weight={0: 10, 1: .1})",
        128: "make_pipeline(StandardScaler(), SVC(gamma='auto'))",
        131: "KNeighborsClassifier(n_neighbors = 5)",
        134: "n_estimators=10, random_state=21, n_jobs = -1",
    }
    for cell_index, expected in supplementary_sources.items():
        assert expected in "".join(notebook["cells"][cell_index]["source"])

    for cell_index in (102, 106):
        validation_source = "".join(notebook["cells"][cell_index]["source"])
        assert "StratifiedKFold(n_splits=20)" in validation_source
        assert "DecisionTreeClassifier()" in validation_source
        assert "scoring = 'f1_macro'" in validation_source
        assert "scoring = 'accuracy'" in validation_source

    ensemble_source = "".join(notebook["cells"][141]["source"])
    assert "class_weight={0: 5, 1: .1}" in ensemble_source
    assert "n_estimators = 400" in ensemble_source
    assert "random_state=21" in ensemble_source
    assert "voting ='soft'" in ensemble_source

    model_save_source = "".join(notebook["cells"][169]["source"])
    assert "joblib.dump(lgb, 'best_model.pkl')" in model_save_source

    print("Notebook contract verified.")


if __name__ == "__main__":
    verify()
