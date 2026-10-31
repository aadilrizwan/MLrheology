"""
Tests for data_processing module.

Covers:
• Synthetic dataset generation
• Column validation
• Numeric / dtype validation
• Physical bounds checking
• Viscosity consistency checking
• Missing value and duplicate detection
• Data summary utilities
• Loading from CSV
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    ValidationReport,
    add_log_columns,
    data_summary,
    filter_positive_shear_rate,
    generate_synthetic_dataset,
    get_condition_groups,
    group_summary,
    load_dataset,
    validate_dataset,
)
from src.config import ColumnMapping, DEFAULT_COLUMN_MAPPING


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Generate a fresh synthetic dataset."""
    return generate_synthetic_dataset(seed=42)


@pytest.fixture
def minimal_valid_df() -> pd.DataFrame:
    """Minimal valid dataframe with only required columns."""
    return pd.DataFrame({
        "shear_rate": [0.1, 1.0, 10.0, 100.0],
        "shear_stress": [0.05, 0.3, 1.5, 8.0],
    })


# ── Synthetic dataset tests ──────────────────────────────────────────

class TestSyntheticDataset:

    def test_shape_and_columns(self, synthetic_df):
        assert len(synthetic_df) > 0
        required = {"shear_rate", "shear_stress", "apparent_viscosity",
                     "fluid_type", "concentration", "temperature", "data_source"}
        assert required.issubset(set(synthetic_df.columns))

    def test_all_values_positive(self, synthetic_df):
        for col in ["shear_rate", "shear_stress", "apparent_viscosity"]:
            assert (synthetic_df[col] > 0).all(), f"{col} has non-positive values"

    def test_data_source_label(self, synthetic_df):
        assert (synthetic_df["data_source"] == "synthetic").all()

    def test_both_fluids_present(self, synthetic_df):
        fluids = set(synthetic_df["fluid_type"].unique())
        assert fluids == {"CMC", "XG"}

    def test_reproducibility(self):
        df1 = generate_synthetic_dataset(seed=42)
        df2 = generate_synthetic_dataset(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_synthetic_dataset(seed=42)
        df2 = generate_synthetic_dataset(seed=99)
        assert not df1["shear_stress"].equals(df2["shear_stress"])

    def test_viscosity_consistency(self, synthetic_df):
        """η_a should equal τ / γ̇ within rounding."""
        computed = synthetic_df["shear_stress"] / synthetic_df["shear_rate"]
        np.testing.assert_allclose(
            synthetic_df["apparent_viscosity"], computed, rtol=1e-4
        )

    def test_save_to_csv(self, tmp_path):
        path = tmp_path / "test_synth.csv"
        df = generate_synthetic_dataset(save_path=path, seed=42)
        assert path.exists()
        df_loaded = pd.read_csv(path)
        assert len(df_loaded) == len(df)


# ── Validation tests ─────────────────────────────────────────────────

class TestValidation:

    def test_valid_synthetic_passes(self, synthetic_df):
        report = validate_dataset(synthetic_df)
        assert report.is_valid, report.summary()

    def test_missing_required_column(self):
        df = pd.DataFrame({"shear_rate": [1.0], "other": [2.0]})
        report = validate_dataset(df)
        assert not report.is_valid
        assert any("shear_stress" in e for e in report.errors)

    def test_non_numeric_column(self):
        df = pd.DataFrame({
            "shear_rate": ["a", "b"],
            "shear_stress": [1.0, 2.0],
        })
        report = validate_dataset(df)
        assert not report.is_valid

    def test_missing_values_in_required(self):
        df = pd.DataFrame({
            "shear_rate": [1.0, np.nan, 10.0],
            "shear_stress": [0.5, 1.0, 5.0],
        })
        report = validate_dataset(df)
        assert not report.is_valid

    def test_duplicate_warning(self):
        df = pd.DataFrame({
            "shear_rate": [1.0, 1.0],
            "shear_stress": [0.5, 0.5],
        })
        report = validate_dataset(df)
        assert any("duplicate" in w.lower() for w in report.warnings)

    def test_out_of_bounds_warning(self):
        df = pd.DataFrame({
            "shear_rate": [1.0, 10.0],
            "shear_stress": [-5.0, 10.0],  # negative stress
        })
        report = validate_dataset(df)
        assert any("below" in w.lower() for w in report.warnings)

    def test_inconsistent_viscosity_warning(self):
        df = pd.DataFrame({
            "shear_rate": [10.0],
            "shear_stress": [5.0],
            "apparent_viscosity": [1.0],  # should be 0.5
        })
        report = validate_dataset(df)
        assert any("η_a" in w or "inconsist" in w.lower() for w in report.warnings)

    def test_minimal_valid(self, minimal_valid_df):
        report = validate_dataset(minimal_valid_df)
        assert report.is_valid


# ── Loading tests ─────────────────────────────────────────────────────

class TestLoading:

    def test_load_csv(self, tmp_path):
        df = pd.DataFrame({
            "shear_rate_1_per_s": [1.0, 10.0],
            "shear_stress_Pa": [0.5, 3.0],
        })
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        loaded = load_dataset(path)
        assert "shear_rate" in loaded.columns
        assert "shear_stress" in loaded.columns

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_file.csv")

    def test_load_unsupported_format(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset(path)


# ── Utility tests ─────────────────────────────────────────────────────

class TestUtilities:

    def test_filter_positive_shear_rate(self):
        df = pd.DataFrame({
            "shear_rate": [0.0, 1.0, -1.0, 10.0],
            "shear_stress": [0.0, 0.5, 0.1, 5.0],
        })
        filtered = filter_positive_shear_rate(df)
        assert (filtered["shear_rate"] > 0).all()
        assert len(filtered) == 2

    def test_add_log_columns(self, synthetic_df):
        df_log = add_log_columns(synthetic_df)
        assert "log_shear_rate" in df_log.columns
        assert "log_shear_stress" in df_log.columns

    def test_data_summary(self, synthetic_df):
        s = data_summary(synthetic_df)
        assert isinstance(s, pd.DataFrame)
        assert "shear_rate" in s.index

    def test_group_summary(self, synthetic_df):
        gs = group_summary(synthetic_df)
        assert gs is not None
        assert "n_points" in gs.columns

    def test_get_condition_groups(self, synthetic_df):
        groups = get_condition_groups(synthetic_df)
        assert len(groups) > 0

    def test_get_condition_groups_no_grouping(self):
        df = pd.DataFrame({"shear_rate": [1.0], "shear_stress": [0.5]})
        groups = get_condition_groups(df)
        assert groups == [()]
