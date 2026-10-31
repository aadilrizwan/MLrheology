"""
Unit tests for the evaluation module (metrics computation).
"""

import numpy as np
import pytest

from src.evaluation import compute_metrics, print_metrics


class TestEvaluationMetrics:

    def test_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_metrics(y_true, y_pred)
        assert abs(metrics["RMSE"]) < 1e-10
        assert abs(metrics["MAE"]) < 1e-10
        assert abs(metrics["R2"] - 1.0) < 1e-10
        assert abs(metrics["RMSE_log10"]) < 1e-10

    def test_known_deviation(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 32.0])  # diffs: +2, -2, +2
        metrics = compute_metrics(y_true, y_pred)
        assert abs(metrics["MAE"] - 2.0) < 1e-10
        assert abs(metrics["RMSE"] - 2.0) < 1e-10
        assert metrics["R2"] > 0.9

    def test_prefix_support(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        metrics = compute_metrics(y_true, y_pred, prefix="test_")
        assert "test_RMSE" in metrics
        assert "test_MAE" in metrics
        assert "test_R2" in metrics
        assert "test_RMSE_log10" in metrics

    def test_non_positive_values_in_log(self):
        """Values <= 0 should be safely masked in log-space RMSE without crashing."""
        y_true = np.array([-1.0, 0.0, 10.0, 100.0])
        y_pred = np.array([1.0, 0.0, 10.0, 100.0])
        metrics = compute_metrics(y_true, y_pred)
        # Should still compute linear metrics
        assert np.isfinite(metrics["RMSE"])
        # Only [10, 100] are positive in both
        assert "RMSE_log10" in metrics
        assert abs(metrics["RMSE_log10"]) < 1e-10

    def test_multidimensional_arrays(self):
        """Should ravel 2D column vectors properly."""
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        metrics = compute_metrics(y_true, y_pred)
        assert abs(metrics["R2"] - 1.0) < 1e-10

