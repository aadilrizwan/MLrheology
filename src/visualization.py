"""
Visualization module for the Rheology-ML project.

Publication-quality plotting utilities for:
• Rheological flow curves (τ vs γ̇, η vs γ̇)
• Model fits
• ML prediction plots
• Residuals
• Training curves
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, get_logger

logger = get_logger(__name__)

# Global style
STYLE_DEFAULTS = {
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "font.family": "serif",
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
}

FLUID_COLORS = {"CMC": "#1f77b4", "XG": "#d62728"}
FLUID_MARKERS = {"CMC": "o", "XG": "s"}


def apply_style() -> None:
    """Apply project-wide matplotlib style."""
    plt.rcParams.update(STYLE_DEFAULTS)
    sns.set_palette("deep")


apply_style()


def _save(fig: plt.Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Save figure as PNG and return path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure → %s", path)
    return path


# Exploratory flow-curve plots

def plot_flow_curves(
    df: pd.DataFrame,
    title: str = "Rheological Flow Curves",
    save_name: Optional[str] = "flow_curves",
) -> plt.Figure:
    """
    Plot shear stress vs shear rate (log-log) grouped by fluid, conc, temp.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- τ vs γ̇ ---
    ax = axes[0]
    _plot_grouped_loglog(df, "shear_rate", "shear_stress", ax)
    ax.set_xlabel("Shear Rate (s⁻¹)")
    ax.set_ylabel("Shear Stress (Pa)")
    ax.set_title("Shear Stress vs Shear Rate")
    ax.legend(fontsize=8, loc="upper left")

    # --- η vs γ̇ ---
    ax = axes[1]
    if "apparent_viscosity" in df.columns:
        _plot_grouped_loglog(df, "shear_rate", "apparent_viscosity", ax)
    ax.set_xlabel("Shear Rate (s⁻¹)")
    ax.set_ylabel("Apparent Viscosity (Pa·s)")
    ax.set_title("Apparent Viscosity vs Shear Rate")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_name:
        _save(fig, save_name)
    return fig


def _plot_grouped_loglog(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: plt.Axes,
) -> None:
    """Plot log-log grouped by available fluid/conc/temp columns."""
    group_cols = [c for c in ["fluid_type", "concentration", "temperature"]
                  if c in df.columns]
    if not group_cols:
        ax.loglog(df[x_col], df[y_col], "o", ms=4, alpha=0.7)
        return

    for name, grp in df.groupby(group_cols):
        label_parts = name if isinstance(name, tuple) else (name,)
        label = " / ".join(str(p) for p in label_parts)
        fluid = label_parts[0] if group_cols[0] == "fluid_type" else None
        color = FLUID_COLORS.get(fluid, None)
        marker = FLUID_MARKERS.get(fluid, "o")
        ax.loglog(
            grp[x_col], grp[y_col],
            marker=marker, linestyle="none", ms=4, alpha=0.7,
            color=color, label=label,
        )


def plot_flow_curves_by_fluid(
    df: pd.DataFrame,
    save_name: Optional[str] = "flow_curves_by_fluid",
) -> plt.Figure:
    """One subplot per fluid type showing conc/temp variation."""
    fluids = df["fluid_type"].unique() if "fluid_type" in df.columns else ["All"]
    n_fluids = len(fluids)
    fig, axes = plt.subplots(1, n_fluids, figsize=(7 * n_fluids, 6), squeeze=False)

    for i, fluid in enumerate(fluids):
        ax = axes[0, i]
        sub = df[df["fluid_type"] == fluid] if "fluid_type" in df.columns else df
        group_cols = [c for c in ["concentration", "temperature"] if c in sub.columns]
        if group_cols:
            for name, grp in sub.groupby(group_cols):
                label = " / ".join(f"{v}" for v in (name if isinstance(name, tuple) else (name,)))
                ax.loglog(grp["shear_rate"], grp["shear_stress"], "o-", ms=4, label=label)
        else:
            ax.loglog(sub["shear_rate"], sub["shear_stress"], "o-", ms=4)
        ax.set_xlabel("Shear Rate (s⁻¹)")
        ax.set_ylabel("Shear Stress (Pa)")
        ax.set_title(f"{fluid} — Flow Curves")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.3)

    fig.tight_layout()
    if save_name:
        _save(fig, save_name)
    return fig


# Model-fit plots

def plot_model_fit(
    shear_rate: np.ndarray,
    stress_obs: np.ndarray,
    stress_pred: np.ndarray,
    model_name: str,
    condition_label: str = "",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Plot observed vs fitted stress on log-log axes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(shear_rate, stress_obs, "ko", ms=5, alpha=0.7, label="Observed")
    sr_smooth = np.logspace(np.log10(shear_rate.min()), np.log10(shear_rate.max()), 200)
    # If stress_pred has same length as shear_rate, interpolate
    if len(stress_pred) == len(shear_rate):
        ax.loglog(shear_rate, stress_pred, "r-", lw=2, label=f"{model_name} fit")
    else:
        ax.loglog(sr_smooth, stress_pred, "r-", lw=2, label=f"{model_name} fit")
    ax.set_xlabel("Shear Rate (s⁻¹)")
    ax.set_ylabel("Shear Stress (Pa)")
    ax.set_title(f"{model_name} Fit — {condition_label}")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    if save_name:
        _save(fig, save_name)
    return fig


def plot_actual_vs_predicted(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_label: str = "Apparent Viscosity (Pa·s)",
    log_scale: bool = True,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Parity (actual vs predicted) plot."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_actual, y_pred, alpha=0.5, s=20, edgecolors="k", linewidths=0.3)

    all_vals = np.concatenate([y_actual, y_pred])
    lo, hi = all_vals.min() * 0.8, all_vals.max() * 1.2
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(f"Actual {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")
    ax.set_title(f"{model_name} — Actual vs Predicted")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    if save_name:
        _save(fig, save_name)
    return fig


def plot_residuals(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Residual distribution and residual vs predicted."""
    residuals = y_actual - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.5, s=15)
    ax.axhline(0, color="r", ls="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(f"{model_name} — Residuals vs Predicted")
    ax.grid(True, ls="--", alpha=0.3)

    ax = axes[1]
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{model_name} — Residual Distribution")
    ax.grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    if save_name:
        _save(fig, save_name)
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Curves",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Plot training (and optional validation) loss vs epoch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", lw=2)
    if val_losses is not None:
        ax.plot(epochs, val_losses, label="Val Loss", lw=2, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    if save_name:
        _save(fig, save_name)
    return fig


def plot_comparison_table(
    results: Dict[str, Dict[str, float]],
    save_name: Optional[str] = "model_comparison",
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Create a model comparison bar chart and return the table."""
    df = pd.DataFrame(results).T
    df.index.name = "Model"

    fig, axes = plt.subplots(1, len(df.columns), figsize=(5 * len(df.columns), 5))
    if len(df.columns) == 1:
        axes = [axes]

    for ax, metric in zip(axes, df.columns):
        bars = ax.bar(df.index, df[metric], color=sns.color_palette("deep", len(df)))
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_name:
        _save(fig, save_name)
    return fig, df
