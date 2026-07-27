from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon


def fit_logit_odds_ratios(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Any, pd.DataFrame]:
    """Fit the notebook's no-intercept logistic model and return odds ratios."""
    model = sm.Logit(y_train, X_train).fit(disp=False)
    conf = model.conf_int()
    conf["Odds Ratio"] = model.params
    conf.columns = ["2.5%", "97.5%", "Odds Ratio"]
    odds = np.exp(conf)
    odds["p_value"] = model.pvalues
    odds["significant_at_0.05"] = model.pvalues <= 0.05
    return model, odds


def save_forest_plot(odds: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = odds.iloc[::-1]
    errors = [
        ordered["Odds Ratio"] - ordered["2.5%"],
        ordered["97.5%"] - ordered["Odds Ratio"],
    ]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.errorbar(
        x=ordered["Odds Ratio"],
        y=ordered.index,
        xerr=errors,
        color="black",
        capsize=3,
        linestyle="none",
        linewidth=1,
        marker="o",
        markersize=5,
        markerfacecolor="black",
        markeredgecolor="black",
    )
    ax.axvline(x=1, linewidth=0.8, linestyle="--", color="black")
    ax.set_xlabel("Odds Ratio and 95% Confidence Interval")
    fig.tight_layout()
    fig.savefig(target, dpi=300)
    plt.close(fig)


def notebook_wilcoxon_tests(
    y_true: pd.Series,
    predictions: dict[str, np.ndarray],
    *,
    alpha: float = 5e-8,
) -> dict[str, Any]:
    """Run the two signed-rank comparisons defined in the analysis workflow."""
    required = {"svc", "random_forest_oob", "ensemble"}
    missing = required - predictions.keys()
    if missing:
        raise KeyError(f"Missing predictions for Wilcoxon comparisons: {sorted(missing)}")

    y = np.asarray(y_true)
    errors = {name: np.asarray(predictions[name]) - y for name in required}
    pairs = [
        ("svc_vs_random_forest", "svc", "random_forest_oob"),
        ("random_forest_vs_ensemble", "random_forest_oob", "ensemble"),
    ]
    results: dict[str, Any] = {"alpha": alpha, "comparisons": {}}
    for label, left, right in pairs:
        statistic, p_value = wilcoxon(errors[left], errors[right])
        results["comparisons"][label] = {
            "left": left,
            "right": right,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": bool(p_value < alpha),
        }
    return results
