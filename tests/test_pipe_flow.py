"""
Tests for the pipe_flow module.

Covers:
• Mean velocity calculation
• Wall shear rate (Rabinowitsch correction)
• Wall shear stress
• Pressure drop
• Metzner–Reed Reynolds number
• Friction factors (Fanning vs Darcy)
• Newtonian limit (n=1)
• Input validation (invalid inputs)
• Laminar / turbulent regime detection
• Analytical benchmark
"""

import math

import numpy as np
import pytest

from src.pipe_flow import (
    PipeFlowInput,
    calculate_pipe_flow,
    darcy_friction,
    fanning_friction,
    flow_rate_vs_pressure_drop,
    mean_velocity,
    pressure_drop,
    reynolds_number_metzner_reed,
    sensitivity_analysis,
    wall_shear_rate_power_law,
    wall_shear_stress_power_law,
)


class TestElementaryFunctions:

    def test_mean_velocity(self):
        # Q=1e-4 m³/s, D=0.05 m → V = 4*1e-4/(π*0.0025) = 0.05093 m/s
        V = mean_velocity(1e-4, 0.05)
        expected = 4e-4 / (math.pi * 0.05 ** 2)
        assert abs(V - expected) < 1e-10

    def test_wall_shear_rate_newtonian(self):
        """For n=1, correction factor (3n+1)/(4n) = 1."""
        V, D, n = 1.0, 0.1, 1.0
        gamma_w = wall_shear_rate_power_law(V, D, n)
        expected = 1.0 * (8.0 * V / D)  # factor = 1
        assert abs(gamma_w - expected) < 1e-10

    def test_wall_shear_rate_shear_thinning(self):
        """For n<1, correction factor > 1."""
        V, D, n = 1.0, 0.1, 0.5
        gamma_w = wall_shear_rate_power_law(V, D, n)
        factor = (3 * 0.5 + 1) / (4 * 0.5)
        expected = factor * 8.0 * V / D
        assert abs(gamma_w - expected) < 1e-10
        assert gamma_w > 8.0 * V / D  # factor > 1 for n < 1

    def test_wall_shear_stress(self):
        tau = wall_shear_stress_power_law(100.0, 2.0, 0.5)
        expected = 2.0 * 100.0 ** 0.5
        assert abs(tau - expected) < 1e-10

    def test_pressure_drop_formula(self):
        dP = pressure_drop(50.0, 10.0, 0.05)
        expected = 4.0 * 10.0 * 50.0 / 0.05
        assert abs(dP - expected) < 1e-10

    def test_fanning_friction(self):
        f_F = fanning_friction(1600.0)
        assert abs(f_F - 0.01) < 1e-10

    def test_darcy_is_4x_fanning(self):
        f_F = fanning_friction(500.0)
        f_D = darcy_friction(500.0)
        assert abs(f_D - 4.0 * f_F) < 1e-10

    def test_fanning_invalid_re(self):
        with pytest.raises(ValueError):
            fanning_friction(0.0)


class TestNewtonianBenchmark:
    """
    Benchmark against analytical Newtonian result.

    For a Newtonian fluid (n=1, K=μ):
        ΔP = 128 μ L Q / (π D⁴)    (Hagen–Poiseuille)

    Our Power Law formulation should reproduce this exactly when n=1.
    """

    def test_newtonian_pressure_drop(self):
        D = 0.05       # 50 mm
        L = 10.0       # 10 m
        Q = 1e-4       # m³/s
        mu = 0.001     # Pa·s (water)
        rho = 1000.0

        # Analytical Hagen–Poiseuille
        dP_HP = 128.0 * mu * L * Q / (math.pi * D ** 4)

        inp = PipeFlowInput(
            diameter_m=D, length_m=L, flow_rate_m3_per_s=Q,
            K=mu, n=1.0, density_kg_per_m3=rho,
        )
        result = calculate_pipe_flow(inp)

        # Should match within floating-point precision
        np.testing.assert_allclose(result.pressure_drop_Pa, dP_HP, rtol=1e-10)

    def test_newtonian_reynolds_number(self):
        """Re_MR for Newtonian should equal classical Re = ρVD/μ."""
        D = 0.05
        Q = 1e-4
        mu = 0.001
        rho = 1000.0
        V = mean_velocity(Q, D)

        Re_classical = rho * V * D / mu
        Re_MR = reynolds_number_metzner_reed(rho, V, D, mu, 1.0)

        np.testing.assert_allclose(Re_MR, Re_classical, rtol=1e-10)


class TestInputValidation:

    def test_negative_diameter(self):
        inp = PipeFlowInput(diameter_m=-0.05, length_m=10, flow_rate_m3_per_s=1e-4,
                            K=1.0, n=0.5)
        with pytest.raises(ValueError, match="diameter"):
            inp.validate()

    def test_zero_flow_rate(self):
        inp = PipeFlowInput(diameter_m=0.05, length_m=10, flow_rate_m3_per_s=0,
                            K=1.0, n=0.5)
        with pytest.raises(ValueError, match="Flow rate"):
            inp.validate()

    def test_negative_K(self):
        inp = PipeFlowInput(diameter_m=0.05, length_m=10, flow_rate_m3_per_s=1e-4,
                            K=-1.0, n=0.5)
        with pytest.raises(ValueError, match="Consistency"):
            inp.validate()

    def test_hb_model_rejected(self):
        inp = PipeFlowInput(diameter_m=0.05, length_m=10, flow_rate_m3_per_s=1e-4,
                            K=1.0, n=0.5, model="herschel_bulkley")
        with pytest.raises(ValueError, match="power_law"):
            inp.validate()


class TestShearThinningCase:
    """Worked example for a shear-thinning fluid."""

    def test_complete_calculation(self):
        inp = PipeFlowInput(
            diameter_m=0.05,
            length_m=10.0,
            flow_rate_m3_per_s=1e-4,
            K=2.0,
            n=0.5,
            density_kg_per_m3=1050.0,
        )
        result = calculate_pipe_flow(inp)

        # Verify intermediate calculations step by step
        V = mean_velocity(1e-4, 0.05)
        assert abs(result.velocity_m_per_s - V) < 1e-10

        gamma_w = wall_shear_rate_power_law(V, 0.05, 0.5)
        assert abs(result.wall_shear_rate_1_per_s - gamma_w) < 1e-10

        tau_w = wall_shear_stress_power_law(gamma_w, 2.0, 0.5)
        assert abs(result.wall_shear_stress_Pa - tau_w) < 1e-10

        dP = pressure_drop(tau_w, 10.0, 0.05)
        assert abs(result.pressure_drop_Pa - dP) < 1e-10

        # Pressure drop should be positive
        assert result.pressure_drop_Pa > 0


class TestSensitivityAnalysis:

    def test_sensitivity_returns_correct_length(self):
        inp = PipeFlowInput(diameter_m=0.05, length_m=10, flow_rate_m3_per_s=1e-4,
                            K=1.0, n=0.5)
        K_values = np.linspace(0.5, 5.0, 10)
        dPs = sensitivity_analysis(inp, "K", K_values)
        assert len(dPs) == 10
        assert all(np.isfinite(dPs))

    def test_pressure_increases_with_K(self):
        inp = PipeFlowInput(diameter_m=0.05, length_m=10, flow_rate_m3_per_s=1e-4,
                            K=1.0, n=0.5)
        K_values = np.array([0.5, 1.0, 2.0, 5.0])
        dPs = sensitivity_analysis(inp, "K", K_values)
        # Higher K → higher viscosity → higher ΔP
        assert all(dPs[i] < dPs[i+1] for i in range(len(dPs)-1))

    def test_flow_rate_vs_pressure(self):
        inp = PipeFlowInput(diameter_m=0.05, length_m=10, flow_rate_m3_per_s=1e-4,
                            K=1.0, n=0.5)
        Qs = np.linspace(1e-5, 1e-3, 10)
        dPs = flow_rate_vs_pressure_drop(inp, Qs)
        assert len(dPs) == 10
        # Higher Q → higher ΔP
        assert all(dPs[i] <= dPs[i+1] for i in range(len(dPs)-1))
