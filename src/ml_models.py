"""
Conventional Machine Learning module for the Rheology-ML project.

Implements:
• Random Forest Regressor
• XGBoost Regressor

Predicts apparent viscosity from available input features.
Handles categorical encoding, feature scaling, grouped splitting,
and generalization experiments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DEFAULT_ML_CONFIG, MLConfig, RANDOM_SEED, get_logger
from src.evaluation import compute_metrics

logger = get_logger(__name__)

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logger.warning("XGBoost not installed. XGBoost models will be unavailable.")


# Feature engineering

def prepare_features(
    df: pd.DataFrame,
    target_col: str = "apparent_viscosity",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepare feature matrix X and target vector y.

    Parameters
    ----------
    df : DataFrame with canonical column names.
    target_col : target column name.
    feature_cols : explicit list of feature columns.
        If None, auto-detect from available columns.

    Returns
    -------
    X, y, numeric_features, categorical_features
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Auto-detect features
    if feature_cols is None:
        potential = ["shear_rate", "concentration", "temperature", "fluid_type"]
        feature_cols = [c for c in potential if c in df.columns]

    if not feature_cols:
        raise ValueError("No input features available.")

    # Separate numeric and categorical
    numeric_features = []
    categorical_features = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    # Drop rows with NaN in features or target
    cols_needed = feature_cols + [target_col]
    df_clean = df[cols_needed].dropna()
    if len(df_clean) < len(df):
        logger.warning(
            "Dropped %d rows with missing values in features or target.",
            len(df) - len(df_clean),
        )

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    return X, y, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for mixed-type features."""
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_features)
        )
    return ColumnTransformer(transformers, remainder="drop")


# Train/test splitting

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    group_col: Optional[str] = None,
    groups: Optional[pd.Series] = None,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test.

    If group_col or groups is provided, uses GroupShuffleSplit
    to prevent data leakage from same experimental condition.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if groups is not None or (group_col is not None and group_col in X.columns):
        if groups is None:
            groups = X[group_col]
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return (
            X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
        )
    else:
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state)


def create_group_labels(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Create group labels for grouped splitting.
    Groups = (fluid_type, concentration, temperature) — i.e. one flow curve.
    """
    group_cols = [c for c in ["fluid_type", "concentration", "temperature"]
                  if c in df.columns]
    if not group_cols:
        return None
    return df[group_cols].astype(str).agg("_".join, axis=1)


# Model training

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    config: Optional[MLConfig] = None,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a Random Forest regressor pipeline.

    Returns
    -------
    (pipeline, train_metrics)
    """
    config = config or DEFAULT_ML_CONFIG
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=config.n_estimators_rf,
            random_state=config.random_state,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    train_metrics = compute_metrics(y_train.values, y_pred_train, prefix="train_")

    logger.info("Random Forest trained. Train R² = %.4f", train_metrics["train_R2"])
    return pipeline, train_metrics


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    config: Optional[MLConfig] = None,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train an XGBoost regressor pipeline.

    Returns
    -------
    (pipeline, train_metrics)
    """
    if xgb is None:
        raise ImportError("XGBoost is not installed.")

    config = config or DEFAULT_ML_CONFIG
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", xgb.XGBRegressor(
            n_estimators=config.n_estimators_xgb,
            random_state=config.random_state,
            learning_rate=0.1,
            max_depth=6,
            verbosity=0,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred_train = pipeline.predict(X_train)
    train_metrics = compute_metrics(y_train.values, y_pred_train, prefix="train_")

    logger.info("XGBoost trained. Train R² = %.4f", train_metrics["train_R2"])
    return pipeline, train_metrics


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate a trained pipeline on test data.

    Returns
    -------
    (y_pred, test_metrics)
    """
    y_pred = pipeline.predict(X_test)

    # Physical plausibility check
    n_negative = (y_pred < 0).sum()
    if n_negative > 0:
        logger.warning(
            "%s: %d predictions are negative (physically implausible). "
            "Clipping to 1e-10.", model_name, n_negative
        )
        y_pred = np.clip(y_pred, 1e-10, None)

    metrics = compute_metrics(y_test.values, y_pred, prefix="test_")
    logger.info("%s test R² = %.4f, RMSE = %.6f",
                model_name, metrics["test_R2"], metrics["test_RMSE"])
    return y_pred, metrics


# Generalization experiment

def generalization_experiment(
    df: pd.DataFrame,
    holdout_col: str = "concentration",
    target_col: str = "apparent_viscosity",
    config: Optional[MLConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Hold out entire conditions (e.g. a concentration level) and evaluate.

    Returns
    -------
    List of dicts with holdout_value, model, metrics.
    """
    config = config or DEFAULT_ML_CONFIG

    if holdout_col not in df.columns:
        logger.warning("Holdout column '%s' not in data. Skipping.", holdout_col)
        return []

    unique_values = df[holdout_col].unique()
    if len(unique_values) < 3:
        logger.warning(
            "Only %d unique values in '%s'. Need ≥3 for generalization experiment.",
            len(unique_values), holdout_col,
        )
        return []

    results = []

    for holdout_val in unique_values:
        train_df = df[df[holdout_col] != holdout_val]
        test_df = df[df[holdout_col] == holdout_val]

        if len(train_df) < 10 or len(test_df) < 5:
            continue

        X_train, y_train, num_f, cat_f = prepare_features(train_df, target_col)
        X_test, y_test, _, _ = prepare_features(test_df, target_col)

        # Train RF
        rf_pipe, _ = train_random_forest(X_train, y_train, num_f, cat_f, config)
        y_pred_rf, rf_metrics = evaluate_model(rf_pipe, X_test, y_test, "RF")

        # Train XGB
        xgb_pipe, _ = train_xgboost(X_train, y_train, num_f, cat_f, config)
        y_pred_xgb, xgb_metrics = evaluate_model(xgb_pipe, X_test, y_test, "XGB")

        results.append({
            "holdout_col": holdout_col,
            "holdout_value": holdout_val,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "RF_metrics": rf_metrics,
            "XGB_metrics": xgb_metrics,
        })

    return results
