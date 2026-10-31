"""
Configuration module for the Rheology-ML project.

Centralizes paths, constants, column mappings, default hyperparameters,
and physical units so that every other module stays DRY.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories on import
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random seeds for reproducibility
RANDOM_SEED = 42

# Column mapping — maps canonical internal names to user-facing names

@dataclass
class ColumnMapping:
    """
    Configurable mapping from raw dataset column names to internal names.
    
    Internal canonical names:
        fluid_type, concentration, temperature,
        shear_rate, shear_stress, apparent_viscosity
    """
    fluid_type: str = "fluid_type"
    concentration: str = "concentration_wt_pct"
    temperature: str = "temperature_C"
    shear_rate: str = "shear_rate_1_per_s"
    shear_stress: str = "shear_stress_Pa"
    apparent_viscosity: str = "apparent_viscosity_Pa_s"

    # Required columns for minimal operation
    REQUIRED: List[str] = field(
        default_factory=lambda: ["shear_rate", "shear_stress"]
    )
    # Optional columns
    OPTIONAL: List[str] = field(
        default_factory=lambda: [
            "fluid_type", "concentration", "temperature", "apparent_viscosity"
        ]
    )

    def raw_to_internal(self) -> Dict[str, str]:
        """Return {raw_column_name: internal_canonical_name}."""
        return {
            getattr(self, attr): attr
            for attr in [
                "fluid_type", "concentration", "temperature",
                "shear_rate", "shear_stress", "apparent_viscosity",
            ]
        }

    def internal_to_raw(self) -> Dict[str, str]:
        """Return {internal_canonical_name: raw_column_name}."""
        return {v: k for k, v in self.raw_to_internal().items()}


DEFAULT_COLUMN_MAPPING = ColumnMapping()

# Physical constants and valid ranges

@dataclass
class PhysicalBounds:
    """Physically plausible ranges for rheological measurements (SI)."""
    shear_rate_min: float = 0.0        # s⁻¹  (can be zero in principle)
    shear_rate_max: float = 1e6        # s⁻¹
    shear_stress_min: float = 0.0      # Pa
    shear_stress_max: float = 1e6      # Pa
    viscosity_min: float = 1e-6        # Pa·s  (lower than water)
    viscosity_max: float = 1e8         # Pa·s
    temperature_min: float = -50.0     # °C
    temperature_max: float = 300.0     # °C
    concentration_min: float = 0.0     # wt%
    concentration_max: float = 100.0   # wt%


DEFAULT_PHYSICAL_BOUNDS = PhysicalBounds()

# Consistency tolerance
VISCOSITY_CONSISTENCY_RTOL = 0.05   # 5 % relative tolerance

# Classical model fitting defaults

@dataclass
class FittingConfig:
    """Defaults for nonlinear curve-fitting."""
    max_iterations: int = 10_000
    # Power Law initial guesses: K, n
    power_law_p0: List[float] = field(default_factory=lambda: [1.0, 0.5])
    # Herschel–Bulkley initial guesses: tau_y, K, n
    hb_p0: List[float] = field(default_factory=lambda: [0.1, 1.0, 0.5])
    # Bounds: (lower, upper) per parameter
    power_law_bounds: tuple = ((1e-6, 1e-3), (1e4, 2.0))
    hb_bounds: tuple = ((0.0, 1e-6, 1e-3), (1e4, 1e4, 2.0))


DEFAULT_FITTING_CONFIG = FittingConfig()

# ML training defaults

@dataclass
class MLConfig:
    """Defaults for tree-based ML models."""
    test_size: float = 0.2
    n_estimators_rf: int = 200
    n_estimators_xgb: int = 200
    random_state: int = RANDOM_SEED


DEFAULT_ML_CONFIG = MLConfig()

# Physics-guided NN defaults

@dataclass
class PGNNConfig:
    """Defaults for the physics-guided neural network."""
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64, 32])
    learning_rate: float = 1e-3
    epochs: int = 500
    batch_size: int = 32
    physics_lambda: float = 0.1
    patience: int = 30      # early stopping patience
    random_state: int = RANDOM_SEED


DEFAULT_PGNN_CONFIG = PGNNConfig()

# Pipe-flow engineering defaults

@dataclass
class PipeFlowDefaults:
    """Default pipe-flow case study parameters (SI units)."""
    diameter_m: float = 0.05        # 50 mm pipe
    length_m: float = 10.0          # 10 m pipe
    flow_rate_m3_per_s: float = 1e-4  # 0.1 L/s
    density_kg_per_m3: float = 1000.0


DEFAULT_PIPE_FLOW = PipeFlowDefaults()

# Logging
import logging

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
