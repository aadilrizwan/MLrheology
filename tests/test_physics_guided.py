"""
Unit tests for the Physics-Guided Neural Network module.
"""

import numpy as np
import pytest
import torch

from src.config import PGNNConfig
from src.physics_guided import (
    ParameterPredictionNet,
    ResidualPenaltyNet,
    evaluate_pgnn,
    predict_parameter_model,
    predict_residual_model,
    train_parameter_prediction_model,
    train_residual_penalty_model,
)


@pytest.fixture
def dummy_pgnn_data():
    """Generate small synthetic tensors for fast network testing."""
    rng = np.random.default_rng(42)
    n = 60
    # 2 condition features: concentration (0.5 to 1.5), temperature (25 to 40)
    X = rng.uniform(0.5, 2.0, size=(n, 2))
    # shear rate: 0.1 to 100 s^-1
    sr = 10 ** rng.uniform(-1, 2, size=n)
    # Power Law stress: tau = 2.0 * sr^0.5
    stress = 2.0 * np.power(sr, 0.5) * rng.normal(1.0, 0.05, size=n)
    return X, sr, stress


class TestNetworkArchitectures:

    def test_parameter_prediction_net_constraints(self):
        """Softplus must ensure K > 0 and n > 0."""
        net = ParameterPredictionNet(n_input_features=3, hidden_sizes=[16, 16], include_yield_stress=False)
        x_cond = torch.randn(10, 3)
        sr = torch.rand(10, 1) * 100.0 + 0.1

        stress, params = net(x_cond, sr)
        assert stress.shape == (10, 1)
        assert (params["K"] > 0).all()
        assert (params["n"] > 0).all()
        assert (stress > 0).all()

    def test_parameter_prediction_with_yield_stress(self):
        """tau_y, K, and n must all be strictly positive."""
        net = ParameterPredictionNet(n_input_features=2, hidden_sizes=[16, 16], include_yield_stress=True)
        x_cond = torch.randn(5, 2)
        sr = torch.rand(5, 1) * 50.0 + 0.1

        stress, params = net(x_cond, sr)
        assert "tau_y" in params
        assert (params["tau_y"] >= 0).all()
        assert (params["K"] > 0).all()
        assert (params["n"] > 0).all()
        assert (stress >= params["tau_y"]).all()

    def test_residual_penalty_net_forward(self):
        net = ResidualPenaltyNet(n_input_features=2, hidden_sizes=[16, 16])
        x_cond = torch.randn(8, 2)
        sr = torch.rand(8, 1) * 20.0 + 0.1

        stress, residual = net(x_cond, sr)
        assert stress.shape == (8, 1)
        assert residual.shape == (8, 1)


class TestTrainingAndInference:

    def test_train_parameter_model_short(self, dummy_pgnn_data):
        X, sr, stress = dummy_pgnn_data
        config = PGNNConfig(hidden_sizes=[16, 16], epochs=10, batch_size=16)

        model, history = train_parameter_prediction_model(
            X, sr, stress, X_val=X[:20], sr_val=sr[:20], stress_val=stress[:20],
            include_yield_stress=False, config=config
        )
        assert len(history["train_loss"]) == 10
        assert len(history["val_loss"]) == 10

        pred_stress, params = predict_parameter_model(model, X[:10], sr[:10])
        assert len(pred_stress) == 10
        assert (pred_stress > 0).all()

    def test_train_residual_model_short(self, dummy_pgnn_data):
        X, sr, stress = dummy_pgnn_data
        config = PGNNConfig(hidden_sizes=[16, 16], epochs=10, batch_size=16)

        model, history = train_residual_penalty_model(
            X, sr, stress, X_val=X[:20], sr_val=sr[:20], stress_val=stress[:20],
            physics_lambda=0.05, config=config
        )
        assert len(history["train_loss"]) == 10
        assert "data_loss" in history
        assert "physics_loss" in history

        pred_stress = predict_residual_model(model, X[:10], sr[:10])
        assert len(pred_stress) == 10

    def test_evaluate_pgnn(self):
        sr = np.array([1.0, 10.0, 100.0])
        stress_true = np.array([2.0, 6.32, 20.0])
        stress_pred = np.array([2.1, 6.20, 20.2])

        metrics = evaluate_pgnn(stress_true, stress_pred, sr, model_name="TestPGNN")
        assert "TestPGNN_stress_R2" in metrics
        assert "TestPGNN_visc_R2" in metrics
        assert metrics["TestPGNN_stress_R2"] > 0.95
