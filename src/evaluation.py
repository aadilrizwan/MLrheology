"""
Evaluation module for the Rheology-ML project.

Provides consistent metric calculations across all models.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute RMSE, MAE, R², and log-space RMSE.

    Parameters
    ----------
    y_true, y_pred : array-like
    prefix : str
        Optional prefix for metric keys (e.g. "PL_").

    Returns
    -------
    dict with keys like RMSE, MAE, R2, RMSE_log
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics: Dict[str, float] = {
        f"{prefix}RMSE": rmse,
        f"{prefix}MAE": mae,
        f"{prefix}R2": r2,
    }

    # Log-space RMSE (only where both are positive)
    mask = (y_true > 0) & (y_pred > 0)
    if mask.sum() > 0:
        log_rmse = float(np.sqrt(
            np.mean((np.log10(y_true[mask]) - np.log10(y_pred[mask])) ** 2)
        ))
        metrics[f"{prefix}RMSE_log10"] = log_rmse

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "") -> None:
    """Pretty-print a metrics dictionary."""
    if title:
        logger.info("── %s ──", title)
    for k, v in metrics.items():
        logger.info("  %-20s : %.6f", k, v)
