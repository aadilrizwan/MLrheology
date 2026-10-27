# Physics-Guided Machine Learning for Rheological Modelling

> **Non-Newtonian Fluid Flow Prediction using Classical, Data-Driven, and Physics-Guided Approaches**

A complete, modular, reproducible Python research project for Chemical Engineering that combines classical rheological models, conventional machine learning, and physics-guided neural networks for predicting non-Newtonian fluid behaviour.

---

## 🎯 Research Objectives

1. **Analyse** rheological flow data for Xanthan Gum (XG) and Carboxymethyl Cellulose (CMC)
2. **Fit** classical Power Law and Herschel–Bulkley models
3. **Train** Random Forest and XGBoost regression models
4. **Develop** a physics-guided neural network incorporating rheological equations
5. **Compare** physical, data-driven, and physics-guided approaches
6. **Interpret** ML predictions using SHAP
7. **Predict** pressure drop in non-Newtonian pipe flow
8. **Generate** publication-quality plots and reproducible results

---

## 📁 Project Structure

```
rheology-ml/
├── data/
│   ├── raw/                          # Raw input datasets
│   └── processed/                    # Validated & preprocessed data
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Data loading, validation, EDA
│   ├── 02_classical_models.ipynb     # Power Law & Herschel–Bulkley fitting
│   ├── 03_machine_learning.ipynb     # RF & XGBoost training
│   ├── 04_physics_guided_learning.ipynb  # PGNN training
│   └── 05_pipe_flow.ipynb           # Engineering case study
├── src/
│   ├── config.py                     # Paths, constants, defaults
│   ├── data_processing.py            # Loading, validation, synthetic data
│   ├── visualization.py              # Publication-quality plotting
│   ├── physics_models.py             # Power Law & Herschel–Bulkley
│   ├── ml_models.py                  # RF & XGBoost pipelines
│   ├── physics_guided.py             # PyTorch PGNN models
│   ├── evaluation.py                 # Metrics (RMSE, MAE, R², log-RMSE)
│   ├── explainability.py             # SHAP integration
│   ├── pipe_flow.py                  # Pipe-flow pressure drop calculator
│   └── main.py                       # Full pipeline orchestrator
├── tests/
│   ├── test_data_processing.py       # Data validation tests
│   ├── test_physics_models.py        # Rheological model tests
│   └── test_pipe_flow.py             # Pipe-flow benchmark tests
├── results/
│   ├── figures/                      # Generated plots
│   └── tables/                       # CSV result tables
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
cd MLrheology
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Tests

```bash
python -m pytest tests/ -v --tb=short
```

### 3. Run Full Pipeline (Synthetic Data)

```bash
python -m src.main
```

### 4. Run with Your Own Data

```bash
python -m src.main --data data/raw/your_data.csv
```

### 5. Options

```bash
python -m src.main --help
python -m src.main --skip-pgnn     # Skip neural network
python -m src.main --skip-shap     # Skip SHAP analysis
```

---

## 📊 Dataset Requirements

Your CSV/Excel dataset should contain these columns (names are configurable):

| Column | Required | Description | Unit |
|--------|----------|-------------|------|
| `shear_rate_1_per_s` | ✅ | Shear rate | s⁻¹ |
| `shear_stress_Pa` | ✅ | Shear stress | Pa |
| `apparent_viscosity_Pa_s` | ❌ | Apparent viscosity | Pa·s |
| `fluid_type` | ❌ | Fluid name (e.g. "XG", "CMC") | — |
| `concentration_wt_pct` | ❌ | Polymer concentration | wt% |
| `temperature_C` | ❌ | Temperature | °C |

⚠ **If no dataset is provided**, a synthetic demonstration dataset is generated. This dataset is explicitly labelled as synthetic and should NOT be used for scientific conclusions.

---

## 🔬 Models Implemented

### Classical Rheological Models

| Model | Equation | Parameters |
|-------|----------|------------|
| **Power Law** | τ = K γ̇ⁿ | K (consistency index), n (flow behaviour index) |
| **Herschel–Bulkley** | τ = τᵧ + K γ̇ⁿ | τᵧ (yield stress), K, n |

### Machine Learning Models

- **Random Forest** — ensemble of decision trees
- **XGBoost** — gradient-boosted trees

### Physics-Guided Neural Network

- **Parameter-Prediction** — network predicts K, n (and τᵧ), stress via constitutive equation
- **Residual-Penalty** — network predicts stress directly with physics regularisation

---

## 🔧 Engineering Module

Fully developed laminar pipe flow for Power Law fluids:

```
V       = 4Q / (πD²)
γ̇_w     = [(3n+1)/(4n)] × (8V/D)
τ_w     = K γ̇_wⁿ
ΔP      = 4Lτ_w/D
Re_MR   = ρV^(2−n)Dⁿ / [K × 8^(n−1) × ((3n+1)/(4n))ⁿ]
```

**Assumptions**: Steady-state, laminar, incompressible, smooth horizontal pipe, time-independent rheology.

---

## ⚠ Limitations

1. **Synthetic data**: Default dataset is synthetic. Scientific conclusions require real data.
2. **Small datasets**: Neural networks may underperform with < 100 data points.
3. **Laminar only**: Pipe-flow module applies to laminar regime only (Re_MR < 2100).
4. **Time-independent**: Models assume time-independent rheology (no thixotropy).
5. **Physics penalty ≠ correctness**: A physics loss term does not guarantee physical validity.
6. **SHAP ≠ causality**: SHAP explains model behaviour, not causal mechanisms.

---

## 📝 License

This project is for academic and research purposes.
