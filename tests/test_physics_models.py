"""
Tests for physics_models module.

Covers:
• Power Law and Herschel–Bulkley forward equations
• Curve fitting on known synthetic data
• Parameter recovery accuracy
• Edge cases: insufficient data, convergence
• Model selection logic
"""

import numpy as np
import pytest

from src.physics_models import (
    FitResult,
    fit_herschel_bulkley,
    fit_power_law,
    herschel_bulkley_stress,
    herschel_bulkley_viscosity,
    power_law_stress,
    power_law_viscosity,
    select_best_model,
)


# ── Forward equation tests ────────────────────────────────────────────

class TestForwardEquations:

    def test_power_law_stress_basic(self):
        sr = np.array([1.0, 10.0, 100.0])
        K, n = 2.0, 0.5
        result = power_law_stress(sr, K, n)
        expected = K * sr ** n
        np.testing.assert_allclose(result, expected)

    def test_power_law_newtonian(self):
        """n=1 → Newtonian fluid, τ = K·γ̇"""
        sr = np.array([1.0, 5.0, 10.0])
        result = power_law_stress(sr, K=0.001, n=1.0)
        np.testing.assert_allclose(result, 0.001 * sr)

    def test_power_law_viscosity_basic(self):
        sr = np.array([1.0, 10.0, 100.0])
        K, n = 2.0, 0.5
        result = power_law_viscosity(sr, K, n)
        expected = K * sr ** (n - 1)
        np.testing.assert_allclose(result, expected)

    def test_hb_stress_zero_yield(self):
        """τ_y = 0 → reduces to Power Law."""
        sr = np.array([1.0, 10.0, 100.0])
        K, n = 2.0, 0.5
        hb = herschel_bulkley_stress(sr, 0.0, K, n)
        pl = power_law_stress(sr, K, n)
        np.testing.assert_allclose(hb, pl)

    def test_hb_stress_with_yield(self):
        sr = np.array([1.0, 10.0])
        tau_y, K, n = 0.5, 1.0, 0.6
        result = herschel_bulkley_stress(sr, tau_y, K, n)
        expected = tau_y + K * sr ** n
        np.testing.assert_allclose(result, expected)

    def test_hb_viscosity(self):
        sr = np.array([1.0, 10.0])
        tau_y, K, n = 0.5, 1.0, 0.6
        eta = herschel_bulkley_viscosity(sr, tau_y, K, n)
        expected = (tau_y + K * sr ** n) / sr
        np.testing.assert_allclose(eta, expected)


# ── Fitting tests ─────────────────────────────────────────────────────

class TestFitting:

    def test_power_law_recovery(self):
        """Fit should recover known K and n from noiseless data."""
        sr = np.logspace(-1, 3, 50)
        K_true, n_true = 1.5, 0.6
        ss = power_law_stress(sr, K_true, n_true)
        result = fit_power_law(sr, ss, mode="stress")
        assert result.converged
        assert abs(result.parameters["K"] - K_true) < 0.01
        assert abs(result.parameters["n"] - n_true) < 0.01
        assert result.metrics["R2"] > 0.999

    def test_hb_recovery(self):
        """Fit should recover known τ_y, K, n from noiseless data."""
        sr = np.logspace(-1, 3, 50)
        tau_y_true, K_true, n_true = 0.5, 2.0, 0.4
        ss = herschel_bulkley_stress(sr, tau_y_true, K_true, n_true)
        result = fit_herschel_bulkley(sr, ss, mode="stress")
        assert result.converged
        assert abs(result.parameters["tau_y"] - tau_y_true) < 0.05
        assert abs(result.parameters["K"] - K_true) < 0.1
        assert abs(result.parameters["n"] - n_true) < 0.05
        assert result.metrics["R2"] > 0.999

    def test_power_law_noisy(self):
        """Fit should handle noisy data gracefully."""
        rng = np.random.default_rng(42)
        sr = np.logspace(-1, 3, 50)
        K_true, n_true = 1.0, 0.5
        ss = power_law_stress(sr, K_true, n_true) * rng.normal(1.0, 0.03, 50)
        result = fit_power_law(sr, ss, mode="stress")
        assert result.converged
        assert abs(result.parameters["K"] - K_true) < 0.2
        assert abs(result.parameters["n"] - n_true) < 0.1

    def test_insufficient_data(self):
        """Should gracefully handle < 2 data points."""
        result = fit_power_law(np.array([1.0]), np.array([1.0]))
        assert not result.converged
        assert "Insufficient" in result.notes

    def test_hb_insufficient_data(self):
        result = fit_herschel_bulkley(np.array([1.0, 2.0]), np.array([1.0, 1.5]))
        assert not result.converged

    def test_log_stress_mode(self):
        """Log-space fitting should also recover parameters."""
        sr = np.logspace(-1, 3, 50)
        K_true, n_true = 3.0, 0.4
        ss = power_law_stress(sr, K_true, n_true)
        result = fit_power_law(sr, ss, mode="log_stress")
        assert result.converged
        assert abs(result.parameters["K"] - K_true) < 0.1
        assert abs(result.parameters["n"] - n_true) < 0.05

    def test_parameter_errors_present(self):
        sr = np.logspace(-1, 3, 50)
        ss = power_law_stress(sr, 1.0, 0.5)
        result = fit_power_law(sr, ss)
        assert result.parameter_errors is not None
        assert "K" in result.parameter_errors


# ── Model selection tests ─────────────────────────────────────────────

class TestModelSelection:

    def test_prefer_simpler_when_similar(self):
        """Power Law preferred when HB yield stress ≈ 0."""
        pl = FitResult("Power Law", {"K": 1.0, "n": 0.5}, metrics={"R2": 0.99}, converged=True)
        hb = FitResult("HB", {"tau_y": 0.001, "K": 1.0, "n": 0.5}, metrics={"R2": 0.991}, converged=True)
        best, reason = select_best_model(pl, hb)
        assert best.model_name == "Power Law"

    def test_prefer_hb_when_yield_stress_significant(self):
        """HB preferred when yield stress is meaningful and fit is better."""
        pl = FitResult("Power Law", {"K": 1.0, "n": 0.5}, metrics={"R2": 0.90}, converged=True)
        hb = FitResult("HB", {"tau_y": 1.0, "K": 2.0, "n": 0.4}, metrics={"R2": 0.99}, converged=True)
        best, reason = select_best_model(pl, hb)
        assert best.model_name == "HB"

    def test_fallback_on_failure(self):
        pl = FitResult("Power Law", {"K": 1.0, "n": 0.5}, metrics={"R2": 0.95}, converged=True)
        hb = FitResult("HB", {"tau_y": np.nan, "K": np.nan, "n": np.nan}, converged=False)
        best, reason = select_best_model(pl, hb)
        assert best.model_name == "Power Law"
