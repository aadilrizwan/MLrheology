"""
Classical rheological models for the Rheology-ML project.

Implements:
• Power Law:          τ = K γ̇ⁿ
• Herschel–Bulkley:   τ = τᵧ + K γ̇ⁿ

Each model provides:
  - Forward evaluation
  - Nonlinear curve-fitting with bounded parameters
  - Metrics (RMSE, MAE, R²)
  - Uncertainty estimates where feasible (from Jacobian)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from src.config import DEFAULT_FITTING_CONFIG, FittingConfig, get_logger
from src.evaluation import compute_metrics

logger = get_logger(__name__)


# Model equations

def power_law_stress(shear_rate: np.ndarray, K: float, n: float) -> np.ndarray:
    """τ = K γ̇ⁿ"""
    return K * np.power(shear_rate, n)


def power_law_viscosity(shear_rate: np.ndarray, K: float, n: float) -> np.ndarray:
    """η_a = K γ̇⁽ⁿ⁻¹⁾"""
    return K * np.power(shear_rate, n - 1)


def herschel_bulkley_stress(
    shear_rate: np.ndarray, tau_y: float, K: float, n: float
) -> np.ndarray:
    """τ = τᵧ + K γ̇ⁿ"""
    return tau_y + K * np.power(shear_rate, n)


def herschel_bulkley_viscosity(
    shear_rate: np.ndarray, tau_y: float, K: float, n: float
) -> np.ndarray:
    """η_a = τ / γ̇ = (τᵧ + K γ̇ⁿ) / γ̇"""
    stress = herschel_bulkley_stress(shear_rate, tau_y, K, n)
    return stress / shear_rate


# Fitting result container

@dataclass
class FitResult:
    """Container for curve-fitting results."""
    model_name: str
    parameters: Dict[str, float]
    parameter_errors: Optional[Dict[str, float]] = None  # 1σ from Jacobian
    metrics: Dict[str, float] = field(default_factory=dict)
    converged: bool = True
    condition_label: str = ""
    fitting_mode: str = "stress"  # "stress" or "viscosity" or "log_stress"
    notes: str = ""

    def summary_str(self) -> str:
        lines = [f"Model: {self.model_name}  ({self.condition_label})"]
        lines.append(f"  Fitting mode: {self.fitting_mode}")
        lines.append(f"  Converged: {self.converged}")
        lines.append("  Parameters:")
        for k, v in self.parameters.items():
            err_str = ""
            if self.parameter_errors and k in self.parameter_errors:
                err_str = f" ± {self.parameter_errors[k]:.6g}"
            lines.append(f"    {k} = {v:.6g}{err_str}")
        lines.append("  Metrics:")
        for k, v in self.metrics.items():
            lines.append(f"    {k} = {v:.6f}")
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)


# Fitting functions

def fit_power_law(
    shear_rate: np.ndarray,
    shear_stress: np.ndarray,
    apparent_viscosity: Optional[np.ndarray] = None,
    mode: str = "stress",
    config: Optional[FittingConfig] = None,
    condition_label: str = "",
) -> FitResult:
    """
    Fit the Power Law model: τ = K γ̇ⁿ.

    Parameters
    ----------
    shear_rate : array, s⁻¹
    shear_stress : array, Pa
    apparent_viscosity : array, Pa·s (optional, needed for mode='viscosity')
    mode : "stress", "viscosity", or "log_stress"
    config : FittingConfig
    condition_label : descriptive string for this condition

    Returns
    -------
    FitResult
    """
    config = config or DEFAULT_FITTING_CONFIG
    sr = np.asarray(shear_rate, dtype=float)
    ss = np.asarray(shear_stress, dtype=float)

    # Filter positive values
    mask = sr > 0
    if mode in ("viscosity",) and apparent_viscosity is not None:
        av = np.asarray(apparent_viscosity, dtype=float)
        mask &= av > 0
    sr, ss = sr[mask], ss[mask]

    if len(sr) < 2:
        return FitResult(
            model_name="Power Law",
            parameters={"K": np.nan, "n": np.nan},
            converged=False,
            condition_label=condition_label,
            fitting_mode=mode,
            notes="Insufficient data points (< 2).",
        )

    try:
        if mode == "stress":
            popt, pcov = curve_fit(
                power_law_stress, sr, ss,
                p0=config.power_law_p0,
                bounds=config.power_law_bounds,
                maxfev=config.max_iterations,
            )
            y_obs, y_pred = ss, power_law_stress(sr, *popt)

        elif mode == "log_stress":
            # Fit in log-space for better low-stress accuracy
            log_ss = np.log10(ss)

            def log_pl(sr_val, K, n):
                return np.log10(power_law_stress(sr_val, K, n))

            popt, pcov = curve_fit(
                log_pl, sr, log_ss,
                p0=config.power_law_p0,
                bounds=config.power_law_bounds,
                maxfev=config.max_iterations,
            )
            y_obs, y_pred = ss, power_law_stress(sr, *popt)

        elif mode == "viscosity":
            av = np.asarray(apparent_viscosity, dtype=float)[mask[: len(apparent_viscosity)]] \
                if apparent_viscosity is not None else ss / sr
            popt, pcov = curve_fit(
                power_law_viscosity, sr, av,
                p0=config.power_law_p0,
                bounds=config.power_law_bounds,
                maxfev=config.max_iterations,
            )
            y_obs, y_pred = av, power_law_viscosity(sr, *popt)
        else:
            raise ValueError(f"Unknown fitting mode: {mode}")

        # Standard errors from covariance
        perr = np.sqrt(np.diag(pcov))
        param_errors = {"K": float(perr[0]), "n": float(perr[1])}

        # Always compute stress-based metrics for fair comparison
        stress_pred = power_law_stress(sr, *popt)
        metrics = compute_metrics(ss, stress_pred, prefix="")

        return FitResult(
            model_name="Power Law",
            parameters={"K": float(popt[0]), "n": float(popt[1])},
            parameter_errors=param_errors,
            metrics=metrics,
            converged=True,
            condition_label=condition_label,
            fitting_mode=mode,
        )

    except (RuntimeError, ValueError) as e:
        logger.warning("Power Law fit failed for %s: %s", condition_label, e)
        return FitResult(
            model_name="Power Law",
            parameters={"K": np.nan, "n": np.nan},
            converged=False,
            condition_label=condition_label,
            fitting_mode=mode,
            notes=f"Convergence failure: {e}",
        )


def fit_herschel_bulkley(
    shear_rate: np.ndarray,
    shear_stress: np.ndarray,
    apparent_viscosity: Optional[np.ndarray] = None,
    mode: str = "stress",
    config: Optional[FittingConfig] = None,
    condition_label: str = "",
) -> FitResult:
    """
    Fit the Herschel–Bulkley model: τ = τᵧ + K γ̇ⁿ.

    Parameters
    ----------
    shear_rate : array, s⁻¹
    shear_stress : array, Pa
    apparent_viscosity : array, Pa·s (optional)
    mode : "stress" or "log_stress"
    config : FittingConfig
    condition_label : descriptive string

    Returns
    -------
    FitResult
    """
    config = config or DEFAULT_FITTING_CONFIG
    sr = np.asarray(shear_rate, dtype=float)
    ss = np.asarray(shear_stress, dtype=float)

    mask = sr > 0
    sr, ss = sr[mask], ss[mask]

    if len(sr) < 3:
        return FitResult(
            model_name="Herschel-Bulkley",
            parameters={"tau_y": np.nan, "K": np.nan, "n": np.nan},
            converged=False,
            condition_label=condition_label,
            fitting_mode=mode,
            notes="Insufficient data points (< 3).",
        )

    try:
        if mode == "stress":
            popt, pcov = curve_fit(
                herschel_bulkley_stress, sr, ss,
                p0=config.hb_p0,
                bounds=config.hb_bounds,
                maxfev=config.max_iterations,
            )
            y_pred = herschel_bulkley_stress(sr, *popt)

        elif mode == "log_stress":
            log_ss = np.log10(ss)

            def log_hb(sr_val, tau_y, K, n):
                return np.log10(herschel_bulkley_stress(sr_val, tau_y, K, n))

            popt, pcov = curve_fit(
                log_hb, sr, log_ss,
                p0=config.hb_p0,
                bounds=config.hb_bounds,
                maxfev=config.max_iterations,
            )
            y_pred = herschel_bulkley_stress(sr, *popt)
        else:
            raise ValueError(f"Unknown fitting mode: {mode}")

        perr = np.sqrt(np.diag(pcov))
        param_errors = {
            "tau_y": float(perr[0]),
            "K": float(perr[1]),
            "n": float(perr[2]),
        }

        metrics = compute_metrics(ss, y_pred, prefix="")

        tau_y_val = float(popt[0])
        notes = ""
        if tau_y_val < 1e-6:
            notes = (
                "Fitted yield stress ≈ 0; this fluid may not exhibit yield stress. "
                "Power Law may be more appropriate."
            )

        return FitResult(
            model_name="Herschel-Bulkley",
            parameters={
                "tau_y": tau_y_val,
                "K": float(popt[1]),
                "n": float(popt[2]),
            },
            parameter_errors=param_errors,
            metrics=metrics,
            converged=True,
            condition_label=condition_label,
            fitting_mode=mode,
            notes=notes,
        )

    except (RuntimeError, ValueError) as e:
        logger.warning("Herschel-Bulkley fit failed for %s: %s", condition_label, e)
        return FitResult(
            model_name="Herschel-Bulkley",
            parameters={"tau_y": np.nan, "K": np.nan, "n": np.nan},
            converged=False,
            condition_label=condition_label,
            fitting_mode=mode,
            notes=f"Convergence failure: {e}",
        )


def select_best_model(
    pl_result: FitResult,
    hb_result: FitResult,
    criterion: str = "R2",
) -> Tuple[FitResult, str]:
    """
    Select the best model for a given condition based on a metric.

    Uses a parsimony heuristic: prefer Power Law unless HB is
    substantially better (R² improvement > 0.01) and yield stress
    is meaningfully non-zero.

    Returns
    -------
    (best_result, reason_string)
    """
    if not pl_result.converged and not hb_result.converged:
        return pl_result, "Both models failed to converge."
    if not pl_result.converged:
        return hb_result, "Power Law failed to converge."
    if not hb_result.converged:
        return pl_result, "Herschel-Bulkley failed to converge."

    pl_score = pl_result.metrics.get(criterion, -np.inf)
    hb_score = hb_result.metrics.get(criterion, -np.inf)

    # Check if HB yield stress is meaningful
    tau_y = hb_result.parameters.get("tau_y", 0.0)
    tau_y_significant = tau_y > 0.01  # threshold for meaningful yield stress

    improvement = hb_score - pl_score

    if tau_y_significant and improvement > 0.01:
        reason = (
            f"HB selected: R² improvement = {improvement:.4f}, "
            f"τᵧ = {tau_y:.4g} Pa (meaningful yield stress)."
        )
        return hb_result, reason
    else:
        reason = (
            f"Power Law selected: HB R² improvement = {improvement:.4f}, "
            f"τᵧ = {tau_y:.4g} Pa (insufficient evidence for yield stress)."
        )
        return pl_result, reason


def fit_all_conditions(
    df,
    mode: str = "stress",
    config: Optional[FittingConfig] = None,
) -> List[Dict]:
    """
    Fit Power Law and Herschel–Bulkley to every (fluid, conc, temp) condition.

    Returns
    -------
    List of dicts with keys: condition, pl_result, hb_result, best, reason
    """
    from src.data_processing import get_condition_groups

    config = config or DEFAULT_FITTING_CONFIG
    group_cols = [c for c in ["fluid_type", "concentration", "temperature"]
                  if c in df.columns]

    results = []

    if group_cols:
        for name, grp in df.groupby(group_cols):
            label = " / ".join(str(p) for p in (name if isinstance(name, tuple) else (name,)))
            sr = grp["shear_rate"].values
            ss = grp["shear_stress"].values
            av = grp["apparent_viscosity"].values if "apparent_viscosity" in grp.columns else None

            pl = fit_power_law(sr, ss, av, mode=mode, config=config, condition_label=label)
            hb = fit_herschel_bulkley(sr, ss, av, mode=mode, config=config, condition_label=label)
            best, reason = select_best_model(pl, hb)

            results.append({
                "condition": name,
                "condition_label": label,
                "pl_result": pl,
                "hb_result": hb,
                "best_result": best,
                "selection_reason": reason,
            })
    else:
        sr = df["shear_rate"].values
        ss = df["shear_stress"].values
        av = df["apparent_viscosity"].values if "apparent_viscosity" in df.columns else None
        pl = fit_power_law(sr, ss, av, mode=mode, config=config, condition_label="All data")
        hb = fit_herschel_bulkley(sr, ss, av, mode=mode, config=config, condition_label="All data")
        best, reason = select_best_model(pl, hb)
        results.append({
            "condition": (),
            "condition_label": "All data",
            "pl_result": pl,
            "hb_result": hb,
            "best_result": best,
            "selection_reason": reason,
        })

    return results
