from __future__ import annotations

import argparse
import logging
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .analysis import decision_tree_cross_validation, run_xgboost_search
from .config import load_config
from .constants import SELECTED_FEATURES
from .data import load_raw_data, sha256
from .diagnostics import save_calibration_curves, save_error_analysis, save_learning_curve
from .evaluation import evaluate_predictions, save_json
from .explainability import generate_lime_cases
from .feature_selection import balance_training_data, select_features
from .figures import (
    save_confusion_matrix,
    save_feature_importance,
    save_rfecv_curve,
    save_target_distribution,
)
from .models import (
    calibration_models,
    learning_curve_models,
    publication_ensemble,
    publication_models,
    supplementary_models,
)
from .preprocessing import prepare_data
from .statistics import fit_logit_odds_ratios, notebook_wilcoxon_tests, save_forest_plot

LOGGER = logging.getLogger("life_satisfaction")


def _package_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "imbalanced-learn",
        "lightgbm",
        "xgboost",
        "statsmodels",
        "lime",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def run(
    data_path: Path,
    config_path: Path,
    output_dir: Path,
    *,
    with_lime: bool = False,
    with_diagnostics: bool = False,
    with_supplementary: bool = False,
    with_search: bool = False,
) -> dict[str, Any]:
    """Run the paper's classical machine-learning analysis workflow."""

    config, raw_config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"

    LOGGER.info("Loading %s", data_path)
    raw = load_raw_data(
        data_path,
        emulate_notebook_csv_roundtrip=config.emulate_stata_csv_roundtrip,
    )
    prepared = prepare_data(
        raw,
        test_size=config.test_size,
        split_seed=config.random_seed_split,
        imputer_seed=config.random_seed_preprocessing,
        missing_threshold_percent=config.missing_column_threshold_percent,
        correlation_threshold=config.correlation_threshold,
        outlier_standard_deviations=config.outlier_standard_deviations,
        figure_dir=figure_dir,
    )
    save_target_distribution(prepared.y_full, figure_dir / "target_distribution.png")

    X_balanced, y_balanced = balance_training_data(
        prepared.X_train,
        prepared.y_train,
        smote_tomek_strategy=config.smote_tomek_sampling_strategy,
        undersampling_strategy=config.undersampling_strategy,
        smote_random_state=config.random_seed_preprocessing,
        undersampling_random_state=config.random_seed_undersampling,
        figure_dir=figure_dir,
    )
    selected = select_features(
        X_balanced,
        prepared.X_test,
        y_balanced,
        folds=config.rfecv_folds,
        scoring=config.rfecv_scoring,
        random_state=config.random_seed_preprocessing,
        importance_threshold=config.feature_importance_threshold,
        final_features=SELECTED_FEATURES,
    )
    selection_record = {
        "rfecv_feature_count": int(selected.rfecv.n_features_),
        "stage_one_features": selected.stage_one_features,
        "stage_one_importances": dict(
            zip(selected.stage_one_features, selected.stage_one_importances)
        ),
        "selected_features": selected.selected_features,
    }
    save_json(selection_record, output_dir / "feature_selection.json")
    save_rfecv_curve(selected.rfecv, figure_dir / "rfecv_curve.png")
    save_feature_importance(
        selected.stage_one_features,
        selected.stage_one_importances,
        figure_dir / "rfecv_feature_importance.png",
    )

    results: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    fitted_models: dict[str, Any] = {}
    for name, model in publication_models().items():
        LOGGER.info("Training %s", name)
        model.fit(selected.X_train, selected.y_train)
        model_predictions = model.predict(selected.X_test)
        predictions[name] = model_predictions
        results[name] = evaluate_predictions(prepared.y_test, model_predictions)
        fitted_models[name] = model
        save_confusion_matrix(
            prepared.y_test,
            model_predictions,
            figure_dir / f"confusion_{name}.png",
            name,
        )

    supplementary_results: dict[str, Any] = {}
    fitted_supplementary: dict[str, Any] = {}
    if with_supplementary:
        for name, model in supplementary_models().items():
            LOGGER.info("Training supplementary model %s", name)
            model.fit(selected.X_train, selected.y_train)
            model_predictions = model.predict(selected.X_test)
            predictions[name] = model_predictions
            supplementary_results[name] = evaluate_predictions(
                prepared.y_test,
                model_predictions,
            )
            fitted_supplementary[name] = model
            save_confusion_matrix(
                prepared.y_test,
                model_predictions,
                figure_dir / f"confusion_{name}.png",
                name,
            )
        save_json(supplementary_results, output_dir / "supplementary_metrics.json")
        save_json(
            decision_tree_cross_validation(
                selected.X_train,
                selected.y_train,
            ),
            output_dir / "decision_tree_cross_validation.json",
        )

    ensemble = publication_ensemble()
    LOGGER.info("Training soft-voting ensemble")
    ensemble.fit(selected.X_train, selected.y_train)
    ensemble_predictions = ensemble.predict(selected.X_test)
    predictions["ensemble"] = ensemble_predictions
    results["ensemble"] = evaluate_predictions(prepared.y_test, ensemble_predictions)
    save_confusion_matrix(
        prepared.y_test,
        ensemble_predictions,
        figure_dir / "confusion_ensemble.png",
        "Soft-voting ensemble",
    )

    save_error_analysis(
        {name: metrics["confusion_matrix"] for name, metrics in results.items()},
        figure_dir / "error_analysis.png",
    )

    training_results = {
        "ensemble": evaluate_predictions(
            selected.y_train,
            ensemble.predict(selected.X_train),
        )
    }
    if "bagging_svc" in fitted_supplementary:
        training_results["bagging_svc"] = evaluate_predictions(
            selected.y_train,
            fitted_supplementary["bagging_svc"].predict(selected.X_train),
        )
    save_json(training_results, output_dir / "training_metrics.json")

    _, odds = fit_logit_odds_ratios(selected.X_train, selected.y_train)
    odds.to_csv(output_dir / "odds_ratios.csv", index_label="feature")
    save_forest_plot(odds, figure_dir / "forest_plot.png")

    wilcoxon_results = notebook_wilcoxon_tests(prepared.y_test, predictions)
    save_json(wilcoxon_results, output_dir / "wilcoxon_tests.json")

    prediction_table = pd.DataFrame({"true_class": prepared.y_test})
    for name, values in predictions.items():
        prediction_table[name] = values
    prediction_table.to_csv(output_dir / "test_predictions.csv", index=False)

    # Save the LightGBM deployment model and the ensemble analysis model.
    joblib.dump(fitted_models["lightgbm"], output_dir / "best_model.pkl")
    joblib.dump(ensemble, output_dir / "publication_ensemble.joblib")
    joblib.dump(prepared.imputer, output_dir / "iterative_imputer.joblib")
    save_json(results, output_dir / "metrics.json")

    if with_lime:
        generate_lime_cases(
            ensemble,
            selected.X_train,
            selected.X_test,
            prepared.y_test,
            output_dir / "lime",
        )

    if with_diagnostics:
        LOGGER.info("Generating paper learning curves and calibration diagrams")
        for name, model in learning_curve_models().items():
            save_learning_curve(
                model,
                selected.X_train,
                selected.y_train,
                figure_dir / f"learning_curve_{name}.png",
            )
        save_calibration_curves(
            calibration_models(include_ensemble=False),
            selected.X_train,
            selected.y_train,
            selected.X_test,
            prepared.y_test,
            figure_dir / "calibration_models.png",
        )
        save_calibration_curves(
            calibration_models(include_ensemble=True),
            selected.X_train,
            selected.y_train,
            selected.X_test,
            prepared.y_test,
            figure_dir / "calibration_ensemble.png",
        )

    if with_search:
        LOGGER.info("Running five-candidate randomized XGBoost search")
        save_json(
            run_xgboost_search(selected.X_train, selected.y_train),
            output_dir / "xgboost_parameter_search.json",
        )

    manifest = {
        "analysis": "life-satisfaction classical machine-learning workflow",
        "dataset": {
            "path": str(data_path),
            "sha256": sha256(data_path),
            "raw_shape": list(raw.shape),
            "target_filtered_rows": prepared.target_filtered_rows,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "config": raw_config,
        "requested_optional_artifacts": {
            "lime": with_lime,
            "diagnostics": with_diagnostics,
            "supplementary_models_and_cross_validation": with_supplementary,
            "xgboost_parameter_search": with_search,
        },
        "data_shapes": {
            "post_missing_feature_count": prepared.post_missing_feature_count,
            "train_before_balancing": list(prepared.X_train.shape),
            "test": list(prepared.X_test.shape),
            "train_after_balancing_and_selection": list(selected.X_train.shape),
        },
        "class_counts": {
            "target_full": prepared.y_full.value_counts().sort_index().to_dict(),
            "train_before_balancing": prepared.y_train.value_counts().sort_index().to_dict(),
            "train_after_balancing": selected.y_train.value_counts().sort_index().to_dict(),
            "test": prepared.y_test.value_counts().sort_index().to_dict(),
        },
        "dropped_columns": {
            "missing": prepared.dropped_missing_columns,
            "constant": prepared.dropped_constant_columns,
            "correlated": prepared.dropped_correlated_columns,
        },
        "feature_selection": selection_record,
    }
    save_json(manifest, output_dir / "run_manifest.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the life-satisfaction analysis pipeline."
    )
    parser.add_argument("--data", type=Path, required=True, help="SHILD .dta or CSV path")
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/generated"))
    parser.add_argument(
        "--with-lime",
        action="store_true",
        help="Generate the two LIME case analyses.",
    )
    parser.add_argument(
        "--with-diagnostics",
        action="store_true",
        help="Generate learning curves and calibration diagrams.",
    )
    parser.add_argument(
        "--with-supplementary",
        action="store_true",
        help="Run supplementary models and the two 20-fold validation analyses.",
    )
    parser.add_argument(
        "--with-search",
        action="store_true",
        help="Run the five-candidate randomized XGBoost parameter search.",
    )
    parser.add_argument(
        "--all-analyses",
        action="store_true",
        help="Generate every model, statistical analysis, explanation, and diagnostic.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    if args.all_analyses:
        args.with_lime = True
        args.with_diagnostics = True
        args.with_supplementary = True
        args.with_search = True
    run(
        args.data,
        args.config,
        args.output,
        with_lime=args.with_lime,
        with_diagnostics=args.with_diagnostics,
        with_supplementary=args.with_supplementary,
        with_search=args.with_search,
    )


if __name__ == "__main__":
    main()
