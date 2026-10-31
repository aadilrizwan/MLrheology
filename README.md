# Physics-Guided Machine Learning for Rheological Modelling

> **Non-Newtonian Fluid Flow Prediction using Classical, Data-Driven, and Physics-Guided Approaches**

A modular, reproducible, and scientifically validated Python research framework designed for Chemical Engineering. The system investigates the non-Newtonian flow behavior of **Xanthan Gum (XG)** and **Carboxymethyl Cellulose (CMC)** across varying concentrations and temperatures, contrasting classical rheometry, conventional machine learning (Random Forest, XGBoost), and physics-guided neural networks (PGNN), coupled to an engineering pipeline hydraulics simulator.

---

## 🎯 Research Objectives

1. **Analyze** non-Newtonian flow data for Xanthan Gum (pseudoplastic with yield stress) and Carboxymethyl Cellulose (pure pseudoplastic).
2. **Fit** classical Ostwald–de Waele (Power Law) and Herschel–Bulkley models with bounded nonlinear least-squares and automated model selection.
3. **Train** Random Forest and XGBoost regression pipelines using grouped validation (`GroupShuffleSplit`) over whole flow curves to prevent data leakage.
4. **Develop** Physics-Guided Neural Networks (PGNN) enforcing physical constraints via:
   - **Parameter-Prediction PGNN**: $\text{Softplus}$ parameterized $(K, n, \tau_y)$ evaluated through the constitutive law in the forward pass.
   - **Residual-Penalty PGNN**: Physics loss regularization $\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}$.
5. **Interpret** model decisions using SHAP (SHapley Additive exPlanations) for global feature importance and interaction dynamics.
6. **Simulate** non-Newtonian laminar pipe flow using the Rabinowitsch–Mooney wall shear rate correction and Metzner–Reed generalized Reynolds number ($Re_{MR}$).
7. **Benchmark** cross-paradigm predictive fidelity and quantify sensitivity to rheological parameter uncertainty.

---

## 📁 Project Structure

```
MLrheology/
├── data/
│   ├── raw/                              # User experimental CSV/Excel datasets
│   │   └── synthetic_rheology_data.csv   # Explicitly labeled demonstration dataset (360 rows)
│   └── processed/
│       └── processed_data.csv            # Validated data with logarithmic transformations
├── notebooks/
│   ├── 01_data_exploration.ipynb         # Data ingestion, unit checks, EDA, flow curves
│   ├── 02_classical_models.ipynb         # Power Law & Herschel–Bulkley fitting & model selection
│   ├── 03_machine_learning.ipynb         # RF & XGBoost training, grouped CV, SHAP explainability
│   ├── 04_physics_guided_learning.ipynb  # PGNN architectures, loss dynamics, lambda sensitivity
│   └── 05_pipe_flow.ipynb               # Hydraulic pipeline simulation & sensitivity analysis
├── src/
│   ├── config.py                         # Hyperparameters, column mapping, physical bounds
│   ├── data_processing.py                # Dataset loading, validation (bounds + consistency), synthesis
│   ├── visualization.py                  # Publication-grade plotting routines (flow curves, fits, residuals)
│   ├── physics_models.py                 # Power Law & Herschel–Bulkley fitting with SciPy
│   ├── ml_models.py                      # Scikit-learn & XGBoost pipelines, grouped splitting
│   ├── physics_guided.py                 # PyTorch PGNN (Parameter-Prediction & Residual-Penalty)
│   ├── evaluation.py                     # Metric calculation (RMSE, MAE, R², log-RMSE)
│   ├── explainability.py                 # SHAP TreeExplainer integration & summary plots
│   ├── pipe_flow.py                      # Metzner–Reed pipe flow calculator & sensitivity analysis
│   └── main.py                           # Full orchestrator executing Phases 1 through 8
├── tests/
│   ├── test_data_processing.py           # Data ingestion, bounds & consistency check tests
│   ├── test_physics_models.py            # Forward constitutive laws & bounded fitting tests
│   ├── test_ml_models.py                 # Pipeline feature encoding, grouped split & clipping tests
│   ├── test_physics_guided.py            # PyTorch network architecture, loss & training tests
│   ├── test_pipe_flow.py                 # Hydraulic formulas, Metzner–Reed Re & analytical benchmark tests
│   └── test_evaluation.py                # Metric computation (linear and log-space) tests
├── results/
│   ├── figures/                          # 51 publication-quality plots (PNG, 300 DPI)
│   ├── tables/                           # Benchmark CSV tables
│   │   ├── classical_model_fits.csv      # Parameter estimates and model selection per condition
│   │   ├── generalization_results.csv    # Leave-one-concentration-out holdout metrics
│   │   ├── lambda_sensitivity.csv        # Physics loss weight (lambda) impact on PGNN validation
│   │   └── model_comparison.csv          # Cross-paradigm benchmark table
│   └── RESEARCH_REPORT.md                # Comprehensive chemical engineering research report
├── website/                              # Next.js interactive conference presentation web app
│   ├── src/components/                   # Presentation mode, simulators, benchmarks, SHAP, gallery
│   ├── src/app/                          # Layout, styling, and main presentation page
│   └── public/figures/                   # 51 publication-grade figure assets
├── create_notebooks.py                   # Script to regenerate interactive Jupyter notebooks
├── requirements.txt                      # Complete pinned dependencies
├── README.md                             # Project documentation
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
cd MLrheology

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (78 Tests)

```bash
python -m pytest tests/ -v
```

### 3. Run Full Research Pipeline

```bash
# Execute end-to-end pipeline with synthetic demonstration data
python -m src.main
```

### 4. Run with Custom Laboratory Viscometry Data

```bash
python -m src.main --data data/raw/your_viscometry_data.csv
```

Pipeline execution options:
```bash
python -m src.main --skip-pgnn     # Skip PyTorch neural network training
python -m src.main --skip-shap     # Skip SHAP explainability analysis
```

### 5. Launch Interactive Jupyter Notebooks

```bash
jupyter lab
# Or:
jupyter notebook
```
Navigate to `notebooks/` and run `01_data_exploration.ipynb` through `05_pipe_flow.ipynb`.

### 6. Launch Interactive Conference Website (Next.js)

```bash
cd website
pnpm dev
# Open in browser: http://localhost:3000
```
*Features interactive rheology curve simulator, live Metzner–Reed pipeline pressure drop calculator, model benchmark explorer, SHAP feature attribution viewer, 51-figure publication gallery, and a **fullscreen Presentation / Slide Deck mode** with presentation timer and keyboard arrow navigation for conferences.*

---

## 📊 Dataset Schema & Validation

Input datasets (`.csv` or `.xlsx`) accept the following columns (configurable via `ColumnMapping` in `src/config.py`):

| Canonical Column | Required | Description | Physical Bounds | Units |
|---|---|---|---|---|
| `shear_rate_1_per_s` | ✅ Yes | Shear rate ($\dot{\gamma}$) | $0.0 < \dot{\gamma} \le 10^6$ | $\text{s}^{-1}$ |
| `shear_stress_Pa` | ✅ Yes | Shear stress ($\tau$) | $0.0 < \tau \le 10^6$ | $\text{Pa}$ |
| `apparent_viscosity_Pa_s` | ❌ Optional | Apparent viscosity ($\eta_a$) | $10^{-6} \le \eta_a \le 10^8$ | $\text{Pa}\cdot\text{s}$ |
| `fluid_type` | ❌ Optional | Polymer identifier (`XG` / `CMC`) | Categorical | — |
| `concentration_wt_pct` | ❌ Optional | Concentration ($c$) | $0.0 < c \le 100.0$ | $\text{wt}\%$ |
| `temperature_C` | ❌ Optional | Temperature ($T$) | $-50 \le T \le 300$ | ${}^\circ\text{C}$ |

### Validation Safeguards (`validate_dataset`):
1. **Physical Validity**: Flags values outside physically realistic intervals.
2. **Internal Consistency**: Verifies that reported viscosity satisfies $\left|\eta_a - \frac{\tau}{\dot{\gamma}}\right| / \eta_a \le 5\%$.
3. **Data Provenance**: If no dataset is supplied, a demonstration dataset is generated with an explicit `data_source = 'synthetic'` column to ensure transparent scientific reporting.

---

## 🔬 Mathematical Models & Cross-Paradigm Benchmark

### Governing Rheological Equations

- **Ostwald–de Waele Power Law**:
  $$\tau = K \dot{\gamma}^n, \quad \eta_a = K \dot{\gamma}^{n-1}$$
- **Herschel–Bulkley**:
  $$\tau = \tau_y + K \dot{\gamma}^n, \quad \eta_a = \frac{\tau_y}{\dot{\gamma}} + K \dot{\gamma}^{n-1}$$

### Cross-Paradigm Benchmark Results

Evaluated on held-out test data across 12 experimental conditions:

| Model Architecture | Modeling Paradigm | Test $R^2$ | RMSE ($\text{Pa}$) | MAE ($\text{Pa}$) | Physical Plausibility |
|---|---|---|---|---|---|
| **Classical Power Law / HB** | Analytical Regression | **0.9995** | **0.2140** | **0.1280** | Guaranteed by analytical equation |
| **PGNN (Parameter-Prediction)** | Physics-Guided Neural Network | **0.9771** | **1.5570** | **0.8464** | Guaranteed via Softplus forward pass |
| **XGBoost Regressor** | Gradient-Boosted Trees | **0.8442** | **3.4454** | **1.4715** | Empirical (requires boundary clipping) |
| **Random Forest Regressor** | Bagged Decision Trees | **0.7506** | **4.3601** | **1.6608** | Empirical (piecewise-constant steps) |
| **PGNN (Residual-Penalty, $\lambda=0.1$)** | Physics-Regularized NN | **0.3698** | **8.1711** | **3.9708** | Penalized (optimization trade-off) |

### Key Scientific Findings

1. **Parameter-Prediction PGNN ($R^2 = 0.977$)**: Incorporating the constitutive equation directly into the PyTorch network graph forces positive parameters ($K > 0, n > 0, \tau_y \ge 0$) via $\text{Softplus}$, outperforming unconstrained ML architectures.
2. **Residual Penalty Limitations**: Adding a soft penalty $\lambda \mathcal{L}_{physics}$ does **not** guarantee physical validity; excessive penalty weights ($\lambda \ge 1.0$) introduce severe gradient conflicts with data loss.
3. **Model Selection**: CMC is strictly pseudoplastic (Power Law, $\tau_y \approx 0$), whereas concentrated XG solutions ($c \ge 0.5\text{ wt}\%$) develop measurable yield stress ($\tau_y = 0.18\text{--}1.20\text{ Pa}$), requiring Herschel–Bulkley fitting.

---

## 🔧 Engineering Pipeline Hydraulics Module

Calculates laminar frictional pressure drop for Power Law fluids in horizontal circular pipes:

```
V       = 4Q / (πD²)                                    # Mean velocity
γ̇_w     = [(3n + 1) / (4n)] × (8V / D)                  # Rabinowitsch–Mooney wall shear rate
τ_w     = K (γ̇_w)ⁿ                                      # Wall shear stress
ΔP      = 4Lτ_w / D                                     # Pressure drop
Re_MR   = ρ V^(2−n) Dⁿ / [K × 8^(n−1) × ((3n+1)/(4n))ⁿ] # Generalized Metzner–Reed Reynolds number
```

### Flow Regime & Friction Factor Conventions
- **Laminar Regime**: Strictly valid for $Re_{MR} < 2100$.
- **Fanning Friction Factor**: $f = 16 / Re_{MR}$
- **Darcy Friction Factor**: $f_D = 4f = 64 / Re_{MR}$
- **Newtonian Benchmark**: At $n = 1.0, K = \mu$, the formulation exactly reduces to the classical Hagen–Poiseuille equation:
  $$\Delta P = \frac{32 \mu L V}{D^2} = \frac{128 \mu L Q}{\pi D^4}$$
  *(Verified analytically to machine precision: discrepancy $< 10^{-14}\text{ Pa}$).*

### Hydraulic Pumping Case Study ($D = 50\text{ mm}$, $L = 50\text{ m}$, $Q = 0.5\text{ L/s}$)
- **CMC 0.5% (25°C)** ($K=0.10, n=0.78$): $\Delta P = 0.62\text{ kPa}$, $Re_{MR} = 345.2$ (Laminar)
- **CMC 1.5% (25°C)** ($K=3.50, n=0.61$): $\Delta P = 14.81\text{ kPa}$, $Re_{MR} = 18.2$ (Laminar)
- **XG 1.0% (25°C)** ($K=8.00, n=0.28$): $\Delta P = 10.12\text{ kPa}$, $Re_{MR} = 15.6$ (Laminar)
- **Sensitivity Insight**: Pressure drop scales linearly with consistency index ($\Delta P \propto K$), but non-linearly with shear-thinning exponent ($\Delta P \propto V^n$). Underestimating $n$ leads to severe undersizing of pipeline pump motors.

---

## 🔍 Model Explainability (SHAP)

SHAP TreeExplainer decomposes model behavior into additive feature attributions:
- **Shear Rate ($\dot{\gamma}$)**: Exhibits the highest dynamic importance; high shear rate induces large negative SHAP values, capturing shear-thinning viscosity reduction.
- **Polymer Concentration ($c$)**: Strongest positive driver of consistency; increases viscosity nonlinearly due to intermolecular chain entanglement.
- **Temperature ($T$)**: Negative attribution; thermal agitation increases free volume, lowering apparent viscosity.
- **Methodological Distinction**: SHAP identifies feature reliance within the tree partition graph; it explains model behavior rather than establishing physical causality.

---

## ⚠ Limitations & Engineering Guidance

1. **Synthetic vs. Real Data**: Default demonstration data is parametrically generated. Defensible engineering designs must be calibrated against experimental viscometer measurements.
2. **Small-Dataset Regimes**: Neural networks require sufficient condition combinations to generalize reliably. For sparse laboratory designs ($< 50$ measurements), classical nonlinear regression remains superior.
3. **Laminar Flow Assumption**: The pipe flow module is valid strictly for $Re_{MR} < 2100$. Turbulent flows require empirical correlations (e.g. Dodge–Metzner).
4. **Time-Independence**: The models do not account for thixotropy (time-dependent breakdown) or viscoelastic normal stress differences ($N_1, N_2$).

---

## 🌐 Conference Presentation Web Platform (Next.js)

An interactive, editorial-grade web application is built into the repository under `website/` for conference presentations, oral defenses, and research demonstrations.

### Features
- **Keynote Slide Deck (`P`)**: Fullscreen presentation mode with an active stopwatch, slide index drawer, and keyboard navigation (`←` / `→` / `Space`).
- **Interactive Rheometer Lab**: Live constitutive flow curve bench with real-time model evaluation ($\tau = K \dot{\gamma}^n$) across XG and CMC biopolymer concentrations.
- **Pipe Flow Hydraulics Workstation**: Dynamic velocity profile $u(r)$ plotting pseudoplastic blunt profiles vs. Newtonian parabolas, with Metzner–Reed $Re_{MR}$ regime verification.
- **Model Benchmark Matrix**: Quantitative comparison across all 5 modeling paradigms, with interactive analysis of the ML extrapolation collapse on held-out conditions ($c = 0.25\text{ wt}\%$).
- **Mechanistic SHAP Interpretability**: Game-theoretic feature attribution grounded in molecular chain entanglement and Arrhenius thinning.
- **300-DPI Publication Figure Gallery**: Filterable repository of all 51 generated publication plots with high-resolution lightbox inspection.

### Launching the Website
```bash
cd website
pnpm install
pnpm dev
# Open http://localhost:3000 in your browser
```

---

## 📝 License & Citation

This project is prepared for the Chemical Engineering Minor Project curriculum. Free for academic, educational, and research use.
