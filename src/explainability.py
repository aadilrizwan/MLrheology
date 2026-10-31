"""
SHAP-based model explainability module for the Rheology-ML project.

Generates:
• Global feature importance
• SHAP summary (beeswarm) plots
• SHAP dependence plots
• Feature importance bar charts

IMPORTANT: SHAP describes model behaviour — how the model uses features
to make predictions.  It does NOT independently establish causal
relationships or validate physical correctness.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, get_logger

logger = get_logger(__name__)

try:
    import shap
except ImportError:
    shap = None
    logger.warning("SHAP not installed. Explainability features unavailable.")


def _feature_names_from_pipeline(pipeline, X: pd.DataFrame) -> List[str]:
    """Extract feature names from a sklearn Pipeline with ColumnTransformer."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        names = preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        return list(X.columns)


def compute_shap_values(
    pipeline,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    model_name: str = "Model",
    max_samples: int = 200,
) -> Optional[shap.Explanation]:
    """
    Compute SHAP values for a tree-based pipeline.

    Uses TreeExplainer for Random Forest and XGBoost.

    Parameters
    ----------
    pipeline : sklearn Pipeline with preprocessor + tree regressor
    X_train : training data (for background / reference)
    X_explain : data to explain
    model_name : label
    max_samples : subsample X_explain if larger

    Returns
    -------
    shap.Explanation or None if SHAP unavailable
    """
    if shap is None:
        logger.warning("SHAP not available. Skipping explainability.")
        return None

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        regressor = pipeline.named_steps["regressor"]

        X_train_transformed = preprocessor.transform(X_train)
        X_explain_transformed = preprocessor.transform(X_explain)

        if len(X_explain_transformed) > max_samples:
            idx = np.random.default_rng(42).choice(
                len(X_explain_transformed), max_samples, replace=False
            )
            X_explain_transformed = X_explain_transformed[idx]

        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_explain_transformed)

        feature_names = _feature_names_from_pipeline(pipeline, X_train)
        explanation = shap.Explanation(
            values=shap_values,
            data=X_explain_transformed,
            feature_names=feature_names,
        )

        logger.info("SHAP values computed for %s (%d samples).",
                     model_name, len(X_explain_transformed))
        return explanation

    except Exception as e:
        logger.error("SHAP computation failed for %s: %s", model_name, e)
        return None


def plot_shap_summary(
    explanation: shap.Explanation,
    model_name: str = "Model",
    save_name: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Generate SHAP beeswarm summary plot."""
    if explanation is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        explanation.values, explanation.data,
        feature_names=explanation.feature_names,
        show=False, plot_type="dot",
    )
    plt.title(f"SHAP Summary — {model_name}", fontsize=14)
    plt.tight_layout()

    if save_name:
        path = FIGURES_DIR / f"{save_name}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved SHAP summary → %s", path)

    plt.close()
    return fig


def plot_shap_bar(
    explanation: shap.Explanation,
    model_name: str = "Model",
    save_name: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Generate SHAP mean absolute value bar chart."""
    if explanation is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        explanation.values, explanation.data,
        feature_names=explanation.feature_names,
        show=False, plot_type="bar",
    )
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=14)
    plt.tight_layout()

    if save_name:
        path = FIGURES_DIR / f"{save_name}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved SHAP bar plot → %s", path)

    plt.close()
    return fig


def plot_shap_dependence(
    explanation: shap.Explanation,
    feature_name: str,
    model_name: str = "Model",
    save_name: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Generate SHAP dependence plot for a specific feature."""
    if explanation is None:
        return None

    if feature_name not in explanation.feature_names:
        logger.warning("Feature '%s' not found in SHAP explanation.", feature_name)
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    idx = explanation.feature_names.index(feature_name)
    shap.dependence_plot(
        idx, explanation.values, explanation.data,
        feature_names=explanation.feature_names,
        show=False, ax=ax,
    )
    ax.set_title(f"SHAP Dependence: {feature_name} — {model_name}", fontsize=14)
    fig.tight_layout()

    if save_name:
        path = FIGURES_DIR / f"{save_name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved SHAP dependence plot → %s", path)

    plt.close()
    return fig


def full_shap_analysis(
    pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str = "Model",
    save_prefix: Optional[str] = None,
) -> Optional[shap.Explanation]:
    """
    Run complete SHAP analysis: compute values + generate all plots.

    Returns the SHAP Explanation for further use.
    """
    explanation = compute_shap_values(pipeline, X_train, X_test, model_name)
    if explanation is None:
        return None

    prefix = save_prefix or model_name.lower().replace(" ", "_")

    plot_shap_summary(explanation, model_name, f"{prefix}_shap_summary")
    plot_shap_bar(explanation, model_name, f"{prefix}_shap_bar")

    # Dependence plots for key features
    for feat in explanation.feature_names:
        clean_feat = feat.replace("num__", "").replace("cat__", "")
        plot_shap_dependence(
            explanation, feat, model_name,
            f"{prefix}_shap_dep_{clean_feat}",
        )

    return explanation
