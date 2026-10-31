"""
Non-Newtonian pipe-flow engineering module.

Implements pressure-drop calculations for fully developed laminar flow
of Power Law and Herschel–Bulkley fluids in a circular pipe.

Key equations (Power Law, laminar):
────────────────────────────────
  V       = 4Q / (π D²)
  γ̇_w     = [(3n+1)/(4n)] × (8V/D)
  τ_w     = K γ̇_wⁿ
  ΔP      = 4 L τ_w / D

Generalized Reynolds number (Metzner–Reed):
  Re_MR = ρ V² / τ_w    (alternative form)
  Re_MR = (ρ V^(2−n) Dⁿ) / (K × 8^(n−1) × ((3n+1)/(4n))ⁿ)

Fanning friction factor (laminar):
  f_F = 16 / Re_MR

Darcy friction factor:
  f_D = 4 f_F = 64 / Re_MR

⚠  All calculations assume:
  1. Fully developed, steady-state, laminar flow.
  2. Incompressible fluid.
  3. Smooth, straight, horizontal pipe.
  4. No entrance/exit effects.
  5. Time-independent rheology.

Herschel–Bulkley pipe flow is NOT implemented via the simple Power Law
equations.  A separate (more complex) analytical/numerical approach would
be required for rigorous HB pipe flow.  This module flags HB as not
supported and raises an error rather than silently mis-applying PL equations.

Units: SI throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import DEFAULT_PIPE_FLOW, PipeFlowDefaults, get_logger

logger = get_logger(__name__)


# Data containers

@dataclass
class PipeFlowInput:
    """Input parameters for a pipe-flow calculation."""
    diameter_m: float
    length_m: float
    flow_rate_m3_per_s: float
    K: float           # consistency index (Pa·sⁿ)
    n: float           # flow behavior index
    density_kg_per_m3: float = 1000.0
    model: str = "power_law"   # "power_law" only (see module docstring)
    tau_y: float = 0.0         # yield stress — only for validation / warning

    def validate(self) -> None:
        """Raise ValueError for physically invalid inputs."""
        if self.diameter_m <= 0:
            raise ValueError(f"Pipe diameter must be > 0, got {self.diameter_m}")
        if self.length_m <= 0:
            raise ValueError(f"Pipe length must be > 0, got {self.length_m}")
        if self.flow_rate_m3_per_s <= 0:
            raise ValueError(f"Flow rate must be > 0, got {self.flow_rate_m3_per_s}")
        if self.K <= 0:
            raise ValueError(f"Consistency index K must be > 0, got {self.K}")
        if self.n <= 0:
            raise ValueError(f"Flow behavior index n must be > 0, got {self.n}")
        if self.density_kg_per_m3 <= 0:
            raise ValueError(f"Density must be > 0, got {self.density_kg_per_m3}")
        if self.model != "power_law":
            raise ValueError(
                f"Only 'power_law' is supported for pipe flow. Got '{self.model}'. "
                "Herschel–Bulkley pipe flow requires different equations "
                "and is not implemented in this module."
            )
        if self.tau_y > 0:
            logger.warning(
                "Yield stress τ_y = %.4g Pa is non-zero, but Power Law pipe-flow "
                "equations are being used. Results may be inaccurate for HB fluids.",
                self.tau_y,
            )


@dataclass
class PipeFlowResult:
    """Results of a pipe-flow calculation."""
    # Inputs echoed back
    diameter_m: float
    length_m: float
    flow_rate_m3_per_s: float
    K: float
    n: float
    density_kg_per_m3: float

    # Computed
    velocity_m_per_s: float
    wall_shear_rate_1_per_s: float
    wall_shear_stress_Pa: float
    pressure_drop_Pa: float
    reynolds_number_MR: float
    fanning_friction_factor: float
    darcy_friction_factor: float
    is_laminar: bool

    def summary_str(self) -> str:
        lines = [
            "═══ Power Law Pipe Flow Results ═══",
            f"  Pipe:   D = {self.diameter_m*1000:.1f} mm,  L = {self.length_m:.2f} m",
            f"  Flow:   Q = {self.flow_rate_m3_per_s:.6g} m³/s",
            f"  Fluid:  K = {self.K:.4g} Pa·sⁿ,  n = {self.n:.4g}",
            f"  ρ = {self.density_kg_per_m3:.1f} kg/m³",
            "",
            f"  Mean velocity        V   = {self.velocity_m_per_s:.6g} m/s",
            f"  Wall shear rate      γ̇_w = {self.wall_shear_rate_1_per_s:.6g} s⁻¹",
            f"  Wall shear stress    τ_w = {self.wall_shear_stress_Pa:.6g} Pa",
            f"  Pressure drop        ΔP  = {self.pressure_drop_Pa:.6g} Pa "
            f"({self.pressure_drop_Pa/1000:.4g} kPa)",
            f"  Reynolds number (MR) Re  = {self.reynolds_number_MR:.4g}",
            f"  Fanning f_F              = {self.fanning_friction_factor:.6g}",
            f"  Darcy   f_D              = {self.darcy_friction_factor:.6g}",
            f"  Flow regime: {'LAMINAR' if self.is_laminar else '⚠ TURBULENT (results invalid)'}",
        ]
        return "\n".join(lines)


# Core calculations

def mean_velocity(Q: float, D: float) -> float:
    """V = 4Q / (π D²)"""
    return 4.0 * Q / (math.pi * D ** 2)


def wall_shear_rate_power_law(V: float, D: float, n: float) -> float:
    """
    Rabinowitsch-corrected wall shear rate for Power Law fluid:
      γ̇_w = [(3n + 1) / (4n)] × (8V / D)
    """
    return ((3.0 * n + 1.0) / (4.0 * n)) * (8.0 * V / D)


def wall_shear_stress_power_law(gamma_dot_w: float, K: float, n: float) -> float:
    """τ_w = K γ̇_wⁿ"""
    return K * gamma_dot_w ** n


def pressure_drop(tau_w: float, L: float, D: float) -> float:
    """ΔP = 4 L τ_w / D"""
    return 4.0 * L * tau_w / D


def reynolds_number_metzner_reed(
    rho: float, V: float, D: float, K: float, n: float
) -> float:
    """
    Metzner–Reed generalised Reynolds number:
      Re_MR = (ρ V^(2−n) Dⁿ) / (K × 8^(n−1) × ((3n+1)/(4n))ⁿ)
    """
    factor = ((3.0 * n + 1.0) / (4.0 * n)) ** n
    return (rho * V ** (2.0 - n) * D ** n) / (K * 8.0 ** (n - 1.0) * factor)


def fanning_friction(Re_MR: float) -> float:
    """f_F = 16 / Re_MR  (laminar only)"""
    if Re_MR <= 0:
        raise ValueError("Reynolds number must be > 0.")
    return 16.0 / Re_MR


def darcy_friction(Re_MR: float) -> float:
    """f_D = 64 / Re_MR = 4 × f_F  (laminar only)"""
    return 4.0 * fanning_friction(Re_MR)


# Main calculator

# Critical Re for laminar-turbulent transition (Metzner–Reed convention)
LAMINAR_RE_LIMIT = 2100.0


def calculate_pipe_flow(inp: PipeFlowInput) -> PipeFlowResult:
    """
    Compute pressure drop for fully developed laminar Power Law pipe flow.

    Parameters
    ----------
    inp : PipeFlowInput

    Returns
    -------
    PipeFlowResult

    Raises
    ------
    ValueError  for invalid inputs or unsupported model.
    """
    inp.validate()

    V = mean_velocity(inp.flow_rate_m3_per_s, inp.diameter_m)
    gamma_w = wall_shear_rate_power_law(V, inp.diameter_m, inp.n)
    tau_w = wall_shear_stress_power_law(gamma_w, inp.K, inp.n)
    dP = pressure_drop(tau_w, inp.length_m, inp.diameter_m)
    Re = reynolds_number_metzner_reed(
        inp.density_kg_per_m3, V, inp.diameter_m, inp.K, inp.n
    )
    f_F = fanning_friction(Re)
    f_D = darcy_friction(Re)
    is_laminar = Re < LAMINAR_RE_LIMIT

    if not is_laminar:
        logger.warning(
            "Re_MR = %.1f > %d — flow is likely turbulent. "
            "Laminar correlations are NOT valid. Results are unreliable.",
            Re, LAMINAR_RE_LIMIT,
        )

    result = PipeFlowResult(
        diameter_m=inp.diameter_m,
        length_m=inp.length_m,
        flow_rate_m3_per_s=inp.flow_rate_m3_per_s,
        K=inp.K,
        n=inp.n,
        density_kg_per_m3=inp.density_kg_per_m3,
        velocity_m_per_s=V,
        wall_shear_rate_1_per_s=gamma_w,
        wall_shear_stress_Pa=tau_w,
        pressure_drop_Pa=dP,
        reynolds_number_MR=Re,
        fanning_friction_factor=f_F,
        darcy_friction_factor=f_D,
        is_laminar=is_laminar,
    )

    logger.info("Pipe flow: ΔP = %.4g Pa, Re = %.4g, %s",
                dP, Re, "laminar" if is_laminar else "TURBULENT")
    return result


# Sensitivity analysis

def sensitivity_analysis(
    base_input: PipeFlowInput,
    param_name: str,
    values: np.ndarray,
) -> np.ndarray:
    """
    Compute pressure drop over a range of values for one parameter.

    Parameters
    ----------
    base_input : PipeFlowInput (used as template)
    param_name : one of "K", "n", "flow_rate_m3_per_s", "diameter_m"
    values : array of parameter values

    Returns
    -------
    pressure_drops : array
    """
    import copy
    pressure_drops = np.zeros(len(values))
    for i, val in enumerate(values):
        inp = copy.copy(base_input)
        setattr(inp, param_name, float(val))
        try:
            inp.validate()
            result = calculate_pipe_flow(inp)
            pressure_drops[i] = result.pressure_drop_Pa
        except ValueError:
            pressure_drops[i] = np.nan
    return pressure_drops


def flow_rate_vs_pressure_drop(
    base_input: PipeFlowInput,
    flow_rates: np.ndarray,
) -> np.ndarray:
    """Compute ΔP over a range of flow rates."""
    return sensitivity_analysis(base_input, "flow_rate_m3_per_s", flow_rates)
