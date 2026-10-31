"""
Physics-Guided Neural Network for Rheological Modelling.

Implements a neural network in PyTorch that predicts rheological parameters
(K, n, and optionally τᵧ) from fluid conditions, then computes stress and
viscosity through the constitutive equation.

Two approaches implemented:
1. Parameter-prediction network (architecture enforces physics)
2. Residual-penalty network (adds physics loss term)

Design notes
────────────
• The parameter-prediction approach predicts K, n (and τᵧ for HB) from
  input features.  Stress is computed as τ = K γ̇ⁿ (or τ = τᵧ + K γ̇ⁿ).
  Because the constitutive equation IS the forward pass, the physics is
  enforced by construction — a separate physics loss would be redundant.

• The residual-penalty approach predicts stress directly and adds a penalty
  for deviations from the constitutive equation.  This is mathematically
  distinct and useful when the constitutive equation is approximate.

Important limitations
─────────────────────
• This is a physics-guided CONSTITUTIVE model, NOT a PDE-based PINN.
• A physics penalty does NOT guarantee physical correctness.
• With small datasets, classical models may outperform neural networks.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DEFAULT_PGNN_CONFIG, PGNNConfig, get_logger
from src.evaluation import compute_metrics

logger = get_logger(__name__)


# Network architectures

class ParameterPredictionNet(nn.Module):
    """
    Predicts rheological parameters (K, n) from fluid conditions.
    Stress = K γ̇ⁿ is computed in the forward pass.

    Physical constraints:
    • K > 0 : enforced via softplus
    • n > 0 : enforced via softplus (allows n > 1 for shear thickening)
    """

    def __init__(
        self,
        n_input_features: int,
        hidden_sizes: List[int],
        include_yield_stress: bool = False,
    ):
        super().__init__()
        self.include_yield_stress = include_yield_stress
        n_output = 3 if include_yield_stress else 2  # (K, n) or (τᵧ, K, n)

        layers = []
        in_size = n_input_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            in_size = h
        layers.append(nn.Linear(in_size, n_output))

        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus(beta=5)

    def forward(
        self,
        x_conditions: torch.Tensor,
        shear_rate: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        x_conditions : (batch, n_features) — fluid type, conc, temp encoded
        shear_rate : (batch, 1) — shear rate values

        Returns
        -------
        stress_pred : (batch, 1) — predicted shear stress
        params : dict with K, n, and optionally tau_y tensors
        """
        raw = self.net(x_conditions)

        if self.include_yield_stress:
            tau_y = self.softplus(raw[:, 0:1])
            K = self.softplus(raw[:, 1:2])
            n = self.softplus(raw[:, 2:3])
            stress = tau_y + K * torch.pow(shear_rate + 1e-8, n)
            return stress, {"tau_y": tau_y, "K": K, "n": n}
        else:
            K = self.softplus(raw[:, 0:1])
            n = self.softplus(raw[:, 1:2])
            stress = K * torch.pow(shear_rate + 1e-8, n)
            return stress, {"K": K, "n": n}


class ResidualPenaltyNet(nn.Module):
    """
    Directly predicts stress, with an optional physics residual penalty.

    The physics residual is:  r = τ_pred − (K γ̇ⁿ)
    where K, n are also predicted by the network.

    This is useful when the constitutive equation is only approximate.
    """

    def __init__(
        self,
        n_input_features: int,
        hidden_sizes: List[int],
    ):
        super().__init__()

        # Main stress prediction branch
        layers = []
        in_size = n_input_features + 1  # conditions + shear_rate
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.stress_net = nn.Sequential(*layers)

        # Parameter branch (for physics residual)
        param_layers = []
        in_size = n_input_features
        for h in hidden_sizes[:2]:  # smaller
            param_layers.append(nn.Linear(in_size, h))
            param_layers.append(nn.ReLU())
            in_size = h
        param_layers.append(nn.Linear(in_size, 2))  # K, n
        self.param_net = nn.Sequential(*param_layers)

        self.softplus = nn.Softplus(beta=5)

    def forward(
        self,
        x_conditions: torch.Tensor,
        shear_rate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        stress_pred : (batch, 1)
        physics_residual : (batch, 1)
        """
        x_full = torch.cat([x_conditions, shear_rate], dim=1)
        stress_pred = self.softplus(self.stress_net(x_full))

        raw_params = self.param_net(x_conditions)
        K = self.softplus(raw_params[:, 0:1])
        n = self.softplus(raw_params[:, 1:2])

        stress_physics = K * torch.pow(shear_rate + 1e-8, n)
        residual = stress_pred - stress_physics

        return stress_pred, residual


# Training utilities

def prepare_torch_data(
    X: np.ndarray,
    shear_rate: np.ndarray,
    y_stress: np.ndarray,
) -> TensorDataset:
    """Convert numpy arrays to a TensorDataset."""
    X_t = torch.FloatTensor(X)
    sr_t = torch.FloatTensor(shear_rate.reshape(-1, 1))
    y_t = torch.FloatTensor(y_stress.reshape(-1, 1))
    return TensorDataset(X_t, sr_t, y_t)


def train_parameter_prediction_model(
    X_train: np.ndarray,
    sr_train: np.ndarray,
    stress_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    sr_val: Optional[np.ndarray] = None,
    stress_val: Optional[np.ndarray] = None,
    include_yield_stress: bool = False,
    config: Optional[PGNNConfig] = None,
) -> Tuple[ParameterPredictionNet, Dict[str, List[float]]]:
    """
    Train the parameter-prediction physics-guided model.

    Since the constitutive equation is embedded in the forward pass,
    no separate physics loss is needed (it would be redundant).

    Returns
    -------
    model, history (dict with train_loss, val_loss lists)
    """
    config = config or DEFAULT_PGNN_CONFIG
    torch.manual_seed(config.random_state)

    n_features = X_train.shape[1]
    model = ParameterPredictionNet(n_features, config.hidden_sizes, include_yield_stress)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    train_ds = prepare_torch_data(X_train, sr_train, stress_train)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    has_val = X_val is not None and sr_val is not None and stress_val is not None
    if has_val:
        X_val_t = torch.FloatTensor(X_val)
        sr_val_t = torch.FloatTensor(sr_val.reshape(-1, 1))
        stress_val_t = torch.FloatTensor(stress_val.reshape(-1, 1))

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, sr_batch, y_batch in train_loader:
            optimizer.zero_grad()
            stress_pred, params = model(X_batch, sr_batch)

            # Data loss: MSE on log-scale for better behavior across magnitudes
            loss = torch.mean(
                (torch.log(stress_pred + 1e-8) - torch.log(y_batch + 1e-8)) ** 2
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        avg_train_loss = epoch_loss / len(train_ds)
        history["train_loss"].append(avg_train_loss)

        # Validation
        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred, _ = model(X_val_t, sr_val_t)
                val_loss = torch.mean(
                    (torch.log(val_pred + 1e-8) - torch.log(stress_val_t + 1e-8)) ** 2
                ).item()
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

            if (epoch + 1) % 50 == 0:
                logger.info(
                    "Epoch %d: train=%.6f, val=%.6f",
                    epoch + 1, avg_train_loss, val_loss,
                )
        else:
            if (epoch + 1) % 50 == 0:
                logger.info("Epoch %d: train=%.6f", epoch + 1, avg_train_loss)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def train_residual_penalty_model(
    X_train: np.ndarray,
    sr_train: np.ndarray,
    stress_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    sr_val: Optional[np.ndarray] = None,
    stress_val: Optional[np.ndarray] = None,
    physics_lambda: float = 0.1,
    config: Optional[PGNNConfig] = None,
) -> Tuple[ResidualPenaltyNet, Dict[str, List[float]]]:
    """
    Train the residual-penalty physics-guided model.

    L_total = L_data + λ L_physics
    L_data = MSE(log(τ_pred), log(τ_obs))
    L_physics = MSE(residual)  where residual = τ_pred − K γ̇ⁿ

    The physics loss penalises deviations from the Power Law form.

    λ choice
    ────────
    λ = 0.1 is a reasonable starting point.  Higher λ constrains the
    model closer to Power Law behaviour.  λ sensitivity analysis is
    recommended.

    Returns
    -------
    model, history (with train_loss, val_loss, data_loss, physics_loss)
    """
    config = config or DEFAULT_PGNN_CONFIG
    torch.manual_seed(config.random_state)

    n_features = X_train.shape[1]
    model = ResidualPenaltyNet(n_features, config.hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_ds = prepare_torch_data(X_train, sr_train, stress_train)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    has_val = X_val is not None and sr_val is not None and stress_val is not None
    if has_val:
        X_val_t = torch.FloatTensor(X_val)
        sr_val_t = torch.FloatTensor(sr_val.reshape(-1, 1))
        stress_val_t = torch.FloatTensor(stress_val.reshape(-1, 1))

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "data_loss": [], "physics_loss": [],
    }
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        n_samples = 0

        for X_batch, sr_batch, y_batch in train_loader:
            optimizer.zero_grad()
            stress_pred, residual = model(X_batch, sr_batch)

            data_loss = torch.mean(
                (torch.log(stress_pred + 1e-8) - torch.log(y_batch + 1e-8)) ** 2
            )
            physics_loss = torch.mean(residual ** 2)

            total_loss = data_loss + physics_lambda * physics_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = len(X_batch)
            epoch_data_loss += data_loss.item() * batch_size
            epoch_phys_loss += physics_loss.item() * batch_size
            n_samples += batch_size

        avg_data_loss = epoch_data_loss / n_samples
        avg_phys_loss = epoch_phys_loss / n_samples
        avg_train_loss = avg_data_loss + physics_lambda * avg_phys_loss
        history["train_loss"].append(avg_train_loss)
        history["data_loss"].append(avg_data_loss)
        history["physics_loss"].append(avg_phys_loss)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred, val_residual = model(X_val_t, sr_val_t)
                val_data = torch.mean(
                    (torch.log(val_pred + 1e-8) - torch.log(stress_val_t + 1e-8)) ** 2
                ).item()
                val_phys = torch.mean(val_residual ** 2).item()
                val_loss = val_data + physics_lambda * val_phys
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

            if (epoch + 1) % 50 == 0:
                logger.info(
                    "Epoch %d: total=%.6f, data=%.6f, physics=%.6f, val=%.6f",
                    epoch + 1, avg_train_loss, avg_data_loss, avg_phys_loss, val_loss,
                )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


# Prediction and evaluation

def predict_parameter_model(
    model: ParameterPredictionNet,
    X: np.ndarray,
    shear_rate: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Predict stress and extract parameters."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        sr_t = torch.FloatTensor(shear_rate.reshape(-1, 1))
        stress, params = model(X_t, sr_t)
    stress_np = stress.numpy().ravel()
    params_np = {k: v.numpy().ravel() for k, v in params.items()}
    return stress_np, params_np


def predict_residual_model(
    model: ResidualPenaltyNet,
    X: np.ndarray,
    shear_rate: np.ndarray,
) -> np.ndarray:
    """Predict stress from residual-penalty model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        sr_t = torch.FloatTensor(shear_rate.reshape(-1, 1))
        stress, _ = model(X_t, sr_t)
    return stress.numpy().ravel()


def evaluate_pgnn(
    stress_true: np.ndarray,
    stress_pred: np.ndarray,
    shear_rate: np.ndarray,
    model_name: str = "PGNN",
) -> Dict[str, float]:
    """Evaluate PGNN predictions with standard metrics + viscosity metrics."""
    metrics = compute_metrics(stress_true, stress_pred, prefix=f"{model_name}_stress_")

    # Also compute viscosity metrics
    mask = shear_rate > 0
    if mask.sum() > 0:
        visc_true = stress_true[mask] / shear_rate[mask]
        visc_pred = stress_pred[mask] / shear_rate[mask]
        visc_metrics = compute_metrics(visc_true, visc_pred, prefix=f"{model_name}_visc_")
        metrics.update(visc_metrics)

    return metrics


# Lambda sensitivity analysis

def lambda_sensitivity_analysis(
    X_train: np.ndarray,
    sr_train: np.ndarray,
    stress_train: np.ndarray,
    X_val: np.ndarray,
    sr_val: np.ndarray,
    stress_val: np.ndarray,
    lambdas: List[float] = None,
    config: Optional[PGNNConfig] = None,
) -> Dict[float, Dict[str, float]]:
    """
    Train residual-penalty models for multiple λ values and compare.

    Returns
    -------
    {lambda_value: {metric_name: value}}
    """
    if lambdas is None:
        lambdas = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]

    results = {}
    for lam in lambdas:
        logger.info("Training residual-penalty model with λ = %.4f", lam)
        model, history = train_residual_penalty_model(
            X_train, sr_train, stress_train,
            X_val, sr_val, stress_val,
            physics_lambda=lam,
            config=config,
        )
        stress_pred = predict_residual_model(model, X_val, sr_val)
        metrics = compute_metrics(stress_val, stress_pred, prefix="")
        results[lam] = metrics
        logger.info("  λ=%.4f: R²=%.4f, RMSE=%.6f", lam, metrics["R2"], metrics["RMSE"])

    return results
