"""
Data processing module for the Rheology-ML project.

Responsibilities
────────────────
• Load CSV / Excel datasets with configurable column mapping.
• Validate columns, dtypes, physical bounds, and consistency.
• Generate an explicitly labelled synthetic demonstration dataset.
• Produce data summaries and exploratory statistics.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_COLUMN_MAPPING,
    DEFAULT_PHYSICAL_BOUNDS,
    VISCOSITY_CONSISTENCY_RTOL,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    ColumnMapping,
    PhysicalBounds,
    get_logger,
)

logger = get_logger(__name__)

# Loading

def load_dataset(
    filepath: Union[str, Path],
    column_mapping: Optional[ColumnMapping] = None,
    sheet_name: Union[str, int] = 0,
) -> pd.DataFrame:
    """
    Load a CSV or Excel file and rename columns to internal canonical names.

    Parameters
    ----------
    filepath : path-like
        Path to .csv or .xlsx / .xls file.
    column_mapping : ColumnMapping, optional
        Custom mapping.  Falls back to ``DEFAULT_COLUMN_MAPPING``.
    sheet_name : str or int
        Sheet to read when loading Excel files.

    Returns
    -------
    pd.DataFrame with canonical column names where mapping was found.

    Raises
    ------
    FileNotFoundError, ValueError
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    mapping = column_mapping or DEFAULT_COLUMN_MAPPING
    rename_map = mapping.raw_to_internal()

    ext = filepath.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .xlsx")

    # Rename columns that exist in the file
    found_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=found_renames)

    logger.info(
        "Loaded %d rows × %d columns from %s", len(df), len(df.columns), filepath.name
    )
    return df


# Validation

class ValidationReport:
    """Collects validation warnings and errors."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"  ✗ ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  ⚠ WARNING: {w}")
        for i in self.info:
            lines.append(f"  ✓ INFO: {i}")
        return "\n".join(lines) if lines else "  ✓ All checks passed."


def validate_dataset(
    df: pd.DataFrame,
    mapping: Optional[ColumnMapping] = None,
    bounds: Optional[PhysicalBounds] = None,
    rtol: float = VISCOSITY_CONSISTENCY_RTOL,
) -> ValidationReport:
    """
    Run all validation checks on a rheology dataframe.

    Checks
    ------
    1. Required columns present.
    2. Numeric dtype for measurement columns.
    3. No null / NaN in required columns.
    4. Duplicate row detection.
    5. Physical range for each measurement column.
    6. Consistency: η_a ≈ τ / γ̇  (where all three are present).

    Returns
    -------
    ValidationReport
    """
    mapping = mapping or DEFAULT_COLUMN_MAPPING
    bounds = bounds or DEFAULT_PHYSICAL_BOUNDS
    report = ValidationReport()

    # --- 1. Column presence ---
    internal_names = mapping.raw_to_internal()
    for req in mapping.REQUIRED:
        if req not in df.columns:
            report.errors.append(
                f"Required column '{req}' not found. "
                f"Available: {list(df.columns)}"
            )

    for opt in mapping.OPTIONAL:
        if opt not in df.columns:
            report.info.append(f"Optional column '{opt}' not present.")

    if not report.is_valid:
        return report  # can't continue without required cols

    # --- 2. Numeric dtypes ---
    numeric_cols = ["shear_rate", "shear_stress", "apparent_viscosity"]
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                report.errors.append(f"Column '{col}' is not numeric (dtype={df[col].dtype}).")

    # --- 3. Missing values ---
    for req in mapping.REQUIRED:
        if req in df.columns:
            n_missing = df[req].isna().sum()
            if n_missing > 0:
                report.errors.append(
                    f"Column '{req}' has {n_missing} missing values."
                )

    for opt in ["apparent_viscosity", "concentration", "temperature"]:
        if opt in df.columns:
            n_missing = df[opt].isna().sum()
            if n_missing > 0:
                report.warnings.append(
                    f"Optional column '{opt}' has {n_missing} missing values."
                )

    # --- 4. Duplicates ---
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        report.warnings.append(f"{n_dup} duplicate rows detected.")

    # --- 5. Physical bounds ---
    _check_bounds(df, "shear_rate", bounds.shear_rate_min, bounds.shear_rate_max, report)
    _check_bounds(df, "shear_stress", bounds.shear_stress_min, bounds.shear_stress_max, report)
    _check_bounds(df, "apparent_viscosity", bounds.viscosity_min, bounds.viscosity_max, report)
    _check_bounds(df, "temperature", bounds.temperature_min, bounds.temperature_max, report)
    _check_bounds(df, "concentration", bounds.concentration_min, bounds.concentration_max, report)

    # --- 6. Consistency η_a ≈ τ / γ̇  ---
    if all(c in df.columns for c in ["shear_rate", "shear_stress", "apparent_viscosity"]):
        mask = df["shear_rate"] > 0
        if mask.any():
            computed = df.loc[mask, "shear_stress"] / df.loc[mask, "shear_rate"]
            reported = df.loc[mask, "apparent_viscosity"]
            rel_err = np.abs(computed - reported) / (reported + 1e-30)
            n_inconsistent = (rel_err > rtol).sum()
            if n_inconsistent > 0:
                report.warnings.append(
                    f"{n_inconsistent} rows where |η_a − τ/γ̇|/η_a > {rtol:.0%} "
                    f"(max relative error = {rel_err.max():.4f})."
                )
            else:
                report.info.append("Viscosity consistency check passed.")

    n_zero_sr = (df["shear_rate"] == 0).sum() if "shear_rate" in df.columns else 0
    if n_zero_sr > 0:
        report.warnings.append(
            f"{n_zero_sr} rows with shear_rate = 0 (will be excluded from fitting)."
        )

    return report


def _check_bounds(
    df: pd.DataFrame,
    col: str,
    lo: float,
    hi: float,
    report: ValidationReport,
) -> None:
    if col not in df.columns:
        return
    series = df[col].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return
    n_below = (series < lo).sum()
    n_above = (series > hi).sum()
    if n_below > 0:
        report.warnings.append(
            f"Column '{col}': {n_below} values below physical minimum {lo}."
        )
    if n_above > 0:
        report.warnings.append(
            f"Column '{col}': {n_above} values above physical maximum {hi}."
        )


# Data summary

def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    summary = num_df.describe().T
    summary["missing"] = df.isnull().sum()
    summary["dtype"] = df.dtypes
    return summary


def group_summary(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Summarize data grouped by fluid_type, concentration, temperature
    (whichever columns are present).
    """
    group_cols = [c for c in ["fluid_type", "concentration", "temperature"] if c in df.columns]
    if not group_cols:
        return None
    return (
        df.groupby(group_cols)
        .agg(
            n_points=("shear_rate", "size"),
            shear_rate_min=("shear_rate", "min"),
            shear_rate_max=("shear_rate", "max"),
            stress_min=("shear_stress", "min"),
            stress_max=("shear_stress", "max"),
        )
        .reset_index()
    )


# Synthetic demonstration dataset

def generate_synthetic_dataset(
    save_path: Optional[Union[str, Path]] = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate an **explicitly labelled synthetic** demonstration dataset
    for Xanthan Gum (XG) and Carboxymethyl Cellulose (CMC).

    ⚠  WARNING
    ──────────
    This dataset is **synthetic** — generated from idealized Power-Law and
    Herschel–Bulkley models with Gaussian noise.  It is provided solely
    to exercise and validate the software pipeline.

    **Scientific conclusions require real or traceable published data.**

    Rheological parameters are loosely inspired by published ranges but
    are NOT measured values.

    Model basis
    ───────────
    • CMC solutions → Power Law  (τ = K γ̇ⁿ)
    • XG solutions  → Herschel–Bulkley  (τ = τ_y + K γ̇ⁿ)

    Returns
    -------
    pd.DataFrame  with provenance column ``data_source`` = "synthetic".
    """
    rng = np.random.default_rng(seed)

    # Shear rates: log-spaced to mimic rheometer protocol (0.1 → 1000 s⁻¹)
    shear_rates = np.logspace(-1, 3, 30)

    records: List[Dict] = []

    # ---- CMC (Power Law) ----
    # Parameters loosely inspired by literature for CMC solutions
    cmc_params = [
        # (concentration_wt%, temperature_°C, K, n)
        (0.5, 25, 0.15, 0.75),
        (0.5, 40, 0.10, 0.78),
        (1.0, 25, 0.85, 0.60),
        (1.0, 40, 0.55, 0.64),
        (1.5, 25, 3.20, 0.48),
        (1.5, 40, 2.10, 0.52),
    ]
    for conc, temp, K, n in cmc_params:
        for sr in shear_rates:
            tau = K * sr ** n
            # Add ≈ 2 % multiplicative noise
            noise = rng.normal(1.0, 0.02)
            tau_noisy = max(tau * noise, 1e-6)
            eta = tau_noisy / sr
            records.append(
                {
                    "fluid_type": "CMC",
                    "concentration": conc,
                    "temperature": temp,
                    "shear_rate": round(sr, 6),
                    "shear_stress": round(tau_noisy, 6),
                    "apparent_viscosity": round(eta, 6),
                    "data_source": "synthetic",
                    "model_basis": "Power Law",
                    "true_K": K,
                    "true_n": n,
                    "true_tau_y": 0.0,
                }
            )

    # ---- Xanthan Gum (Herschel–Bulkley) ----
    xg_params = [
        # (concentration_wt%, temperature_°C, τ_y, K, n)
        (0.25, 25, 0.05, 0.50, 0.42),
        (0.25, 40, 0.03, 0.35, 0.45),
        (0.50, 25, 0.30, 2.50, 0.35),
        (0.50, 40, 0.18, 1.80, 0.38),
        (1.00, 25, 1.20, 8.00, 0.28),
        (1.00, 40, 0.80, 5.50, 0.32),
    ]
    for conc, temp, tau_y, K, n in xg_params:
        for sr in shear_rates:
            tau = tau_y + K * sr ** n
            noise = rng.normal(1.0, 0.02)
            tau_noisy = max(tau * noise, 1e-6)
            eta = tau_noisy / sr
            records.append(
                {
                    "fluid_type": "XG",
                    "concentration": conc,
                    "temperature": temp,
                    "shear_rate": round(sr, 6),
                    "shear_stress": round(tau_noisy, 6),
                    "apparent_viscosity": round(eta, 6),
                    "data_source": "synthetic",
                    "model_basis": "Herschel-Bulkley",
                    "true_K": K,
                    "true_n": n,
                    "true_tau_y": tau_y,
                }
            )

    df = pd.DataFrame(records)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Synthetic dataset saved to %s (%d rows).", save_path, len(df))

    logger.info(
        "Generated synthetic dataset: %d rows, %d conditions.",
        len(df),
        len(cmc_params) + len(xg_params),
    )
    return df


# Pre-processing helpers

def filter_positive_shear_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with shear_rate <= 0 (required for log-space fitting)."""
    mask = df["shear_rate"] > 0
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        logger.warning("Dropped %d rows with shear_rate ≤ 0.", n_dropped)
    return df.loc[mask].copy()


def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log₁₀ columns for shear rate, stress, and viscosity."""
    df = df.copy()
    for col, log_col in [
        ("shear_rate", "log_shear_rate"),
        ("shear_stress", "log_shear_stress"),
        ("apparent_viscosity", "log_apparent_viscosity"),
    ]:
        if col in df.columns:
            pos = df[col] > 0
            df.loc[pos, log_col] = np.log10(df.loc[pos, col])
    return df


def get_condition_groups(df: pd.DataFrame) -> List[Tuple]:
    """
    Return unique (fluid_type, concentration, temperature) tuples present.
    Handles missing grouping columns gracefully.
    """
    group_cols = [c for c in ["fluid_type", "concentration", "temperature"]
                  if c in df.columns]
    if not group_cols:
        return [()]
    return list(df.groupby(group_cols).groups.keys())
