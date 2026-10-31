"""
Main orchestrator for the Rheology-ML project.

Runs the complete pipeline:
  1. Data loading / generation + validation
  2. Exploratory visualization
  3. Classical model fitting
  4. ML training (RF + XGBoost)
  5. SHAP analysis
  6. Physics-guided neural network training
  7. Pipe-flow engineering case study
  8. Model comparison and report generation

Usage
─────
  python -m src.main                  # run full pipeline with synthetic data
  python -m src.main --data path.csv  # use a real dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_FITTING_CONFIG,
    DEFAULT_ML_CONFIG,
    DEFAULT_PGNN_CONFIG,
    DEFAULT_PIPE_FLOW,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RANDOM_SEED,
    TABLES_DIR,
    get_logger,
)
from src.data_processing import (
    add_log_columns,
    data_summary,
    filter_positive_shear_rate,
    generate_synthetic_dataset,
    group_summary,
    load_dataset,
    validate_dataset,
)
from src.evaluation import compute_metrics, print_metrics
from src.visualization import (
    plot_actual_vs_predicted,
    plot_comparison_table,
    plot_flow_curves,
    plot_flow_curves_by_fluid,
    plot_model_fit,
    plot_residuals,
    plot_training_curves,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Rheology-ML Pipeline")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to input CSV/Excel dataset.")
    parser.add_argument("--skip-pgnn", action="store_true",
                        help="Skip physics-guided neural network training.")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP analysis.")
    return parser.parse_args()


def run_pipeline(data_path: str = None, skip_pgnn: bool = False, skip_shap: bool = False):
    """Run the complete Rheology-ML pipeline."""

    all_metrics = {}  # {model_name: {metric: value}}

    # PHASE 1: Data Loading + Validation
    logger.info("═" * 60)
    logger.info("PHASE 1: Data Loading & Validation")
    logger.info("═" * 60)

    if data_path is not None:
        logger.info("Loading user dataset: %s", data_path)
        df = load_dataset(data_path)
    else:
        logger.info("No dataset provided — generating synthetic demonstration data.")
        logger.info("⚠  Scientific conclusions require real or published data.")
        save_path = RAW_DATA_DIR / "synthetic_rheology_data.csv"
        df = generate_synthetic_dataset(save_path=save_path)

    # Validate
    report = validate_dataset(df)
    logger.info("Validation report:\n%s", report.summary())
    if not report.is_valid:
        logger.error("Dataset validation FAILED. Aborting.")
        sys.exit(1)

    # Summary
    logger.info("\n%s", data_summary(df).to_string())
    gs = group_summary(df)
    if gs is not None:
        logger.info("\nGroup summary:\n%s", gs.to_string(index=False))

    # Pre-process
    df = filter_positive_shear_rate(df)
    df = add_log_columns(df)
    df.to_csv(PROCESSED_DATA_DIR / "processed_data.csv", index=False)
    logger.info("Processed data saved to %s", PROCESSED_DATA_DIR)

    # PHASE 2: Exploratory Visualization + Classical Models
    logger.info("\n" + "═" * 60)
    logger.info("PHASE 2: Exploratory Plots & Classical Model Fitting")
    logger.info("═" * 60)

    plot_flow_curves(df, save_name="flow_curves")
    plot_flow_curves_by_fluid(df, save_name="flow_curves_by_fluid")

    from src.physics_models import (
        fit_all_conditions,
        power_law_stress,
        herschel_bulkley_stress,
    )

    fit_results = fit_all_conditions(df, mode="stress")

    # Save fitting results table
    fit_rows = []
    for res in fit_results:
        pl = res["pl_result"]
        hb = res["hb_result"]
        best = res["best_result"]

        row = {"condition": res["condition_label"], "best_model": best.model_name}
        row.update({f"PL_{k}": v for k, v in pl.parameters.items()})
        row.update({f"PL_{k}": v for k, v in pl.metrics.items()})
        row.update({f"HB_{k}": v for k, v in hb.parameters.items()})
        row.update({f"HB_{k}": v for k, v in hb.metrics.items()})
        row["selection_reason"] = res["selection_reason"]
        fit_rows.append(row)

        logger.info("\n%s", pl.summary_str())
        logger.info("\n%s", hb.summary_str())
        logger.info("  → %s", res["selection_reason"])

    fit_table = pd.DataFrame(fit_rows)
    fit_table.to_csv(TABLES_DIR / "classical_model_fits.csv", index=False)
    logger.info("Classical fit table saved.")

    # Plot fits for each condition
    group_cols = [c for c in ["fluid_type", "concentration", "temperature"]
                  if c in df.columns]
    if group_cols:
        for res in fit_results:
            label = res["condition_label"]
            safe_label = label.replace(" ", "_").replace("/", "_")
            cond = res["condition"]
            if not isinstance(cond, tuple):
                cond = (cond,)

            mask = pd.Series([True] * len(df), index=df.index)
            for col, val in zip(group_cols, cond):
                mask &= (df[col] == val)
            sub = df[mask]

            sr = sub["shear_rate"].values
            ss = sub["shear_stress"].values

            pl = res["pl_result"]
            if pl.converged:
                pl_pred = power_law_stress(sr, pl.parameters["K"], pl.parameters["n"])
                plot_model_fit(sr, ss, pl_pred, "Power Law", label,
                             save_name=f"fit_PL_{safe_label}")

            hb = res["hb_result"]
            if hb.converged:
                hb_pred = herschel_bulkley_stress(
                    sr, hb.parameters["tau_y"], hb.parameters["K"], hb.parameters["n"]
                )
                plot_model_fit(sr, ss, hb_pred, "Herschel-Bulkley", label,
                             save_name=f"fit_HB_{safe_label}")

    # PHASE 3: ML Training (RF + XGBoost)
    logger.info("\n" + "═" * 60)
    logger.info("PHASE 3: ML Model Training")
    logger.info("═" * 60)

    from src.ml_models import (
        prepare_features,
        split_data,
        create_group_labels,
        train_random_forest,
        train_xgboost,
        evaluate_model,
        generalization_experiment,
    )

    X, y, num_features, cat_features = prepare_features(df)
    groups = create_group_labels(df)

    X_train, X_test, y_train, y_test = split_data(
        X, y, groups=groups, test_size=0.2
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # Random Forest
    rf_pipe, rf_train_met = train_random_forest(X_train, y_train, num_features, cat_features)
    y_pred_rf, rf_test_met = evaluate_model(rf_pipe, X_test, y_test, "Random Forest")
    print_metrics(rf_test_met, "Random Forest — Test")
    all_metrics["Random Forest"] = rf_test_met
    plot_actual_vs_predicted(y_test.values, y_pred_rf, "Random Forest",
                            save_name="rf_actual_vs_pred")
    plot_residuals(y_test.values, y_pred_rf, "Random Forest",
                   save_name="rf_residuals")

    # XGBoost
    xgb_pipe, xgb_train_met = train_xgboost(X_train, y_train, num_features, cat_features)
    y_pred_xgb, xgb_test_met = evaluate_model(xgb_pipe, X_test, y_test, "XGBoost")
    print_metrics(xgb_test_met, "XGBoost — Test")
    all_metrics["XGBoost"] = xgb_test_met
    plot_actual_vs_predicted(y_test.values, y_pred_xgb, "XGBoost",
                            save_name="xgb_actual_vs_pred")
    plot_residuals(y_test.values, y_pred_xgb, "XGBoost",
                   save_name="xgb_residuals")

    # Generalization experiment
    gen_results = generalization_experiment(df)
    if gen_results:
        gen_rows = []
        for gr in gen_results:
            gen_rows.append({
                "holdout": f"{gr['holdout_col']}={gr['holdout_value']}",
                "n_train": gr["n_train"],
                "n_test": gr["n_test"],
                **{f"RF_{k}": v for k, v in gr["RF_metrics"].items()},
                **{f"XGB_{k}": v for k, v in gr["XGB_metrics"].items()},
            })
        gen_table = pd.DataFrame(gen_rows)
        gen_table.to_csv(TABLES_DIR / "generalization_results.csv", index=False)
        logger.info("Generalization results saved.")

    # PHASE 4: SHAP Analysis
    if not skip_shap:
        logger.info("\n" + "═" * 60)
        logger.info("PHASE 4: SHAP Explainability")
        logger.info("═" * 60)

        from src.explainability import full_shap_analysis

        full_shap_analysis(rf_pipe, X_train, X_test, "Random Forest", "rf")
        full_shap_analysis(xgb_pipe, X_train, X_test, "XGBoost", "xgb")
    else:
        logger.info("Skipping SHAP analysis (--skip-shap).")

    # PHASE 5: Physics-Guided Neural Network
    if not skip_pgnn:
        logger.info("\n" + "═" * 60)
        logger.info("PHASE 5: Physics-Guided Neural Network")
        logger.info("═" * 60)

        from src.physics_guided import (
            train_parameter_prediction_model,
            train_residual_penalty_model,
            predict_parameter_model,
            predict_residual_model,
            evaluate_pgnn,
            lambda_sensitivity_analysis,
        )
        from sklearn.preprocessing import StandardScaler, OneHotEncoder

        # Prepare features for PGNN (need raw numpy)
        feature_cols = [c for c in ["concentration", "temperature"] if c in df.columns]
        cat_cols = [c for c in ["fluid_type"] if c in df.columns]

        pgnn_df = df.dropna(subset=["shear_rate", "shear_stress"] + feature_cols)

        # Encode categorical
        X_num = pgnn_df[feature_cols].values.astype(float) if feature_cols else np.empty((len(pgnn_df), 0))
        if cat_cols:
            enc = OneHotEncoder(sparse_output=False)
            X_cat = enc.fit_transform(pgnn_df[cat_cols])
            X_all = np.hstack([X_num, X_cat])
        else:
            X_all = X_num

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        sr_all = pgnn_df["shear_rate"].values
        stress_all = pgnn_df["shear_stress"].values

        # Split
        n_total = len(X_scaled)
        idx = np.random.default_rng(RANDOM_SEED).permutation(n_total)
        n_train = int(0.8 * n_total)
        train_idx, val_idx = idx[:n_train], idx[n_train:]

        X_tr, X_va = X_scaled[train_idx], X_scaled[val_idx]
        sr_tr, sr_va = sr_all[train_idx], sr_all[val_idx]
        st_tr, st_va = stress_all[train_idx], stress_all[val_idx]

        dataset_size_warning = ""
        if n_total < 100:
            dataset_size_warning = (
                "⚠  Dataset has only %d samples. Neural network results may be "
                "unreliable. Classical and tree-based models are more appropriate "
                "for this dataset size." % n_total
            )
            logger.warning(dataset_size_warning)

        # Train parameter-prediction model
        logger.info("Training Parameter-Prediction PGNN...")
        pp_model, pp_history = train_parameter_prediction_model(
            X_tr, sr_tr, st_tr,
            X_va, sr_va, st_va,
            include_yield_stress=True,
            config=DEFAULT_PGNN_CONFIG,
        )
        pp_pred, pp_params = predict_parameter_model(pp_model, X_va, sr_va)
        pp_metrics = evaluate_pgnn(st_va, pp_pred, sr_va, "ParamPred")
        print_metrics(pp_metrics, "Parameter-Prediction PGNN — Validation")
        all_metrics["PGNN (Param-Pred)"] = {
            k.replace("ParamPred_stress_", "test_"): v for k, v in pp_metrics.items()
            if "stress" in k
        }

        plot_training_curves(
            pp_history["train_loss"], pp_history.get("val_loss"),
            "Parameter-Prediction PGNN Training",
            save_name="pgnn_param_pred_training",
        )
        plot_actual_vs_predicted(
            st_va, pp_pred, "PGNN (Param-Pred)",
            target_label="Shear Stress (Pa)",
            save_name="pgnn_param_pred_actual_vs_pred",
        )

        # Train residual-penalty model
        logger.info("Training Residual-Penalty PGNN...")
        rp_model, rp_history = train_residual_penalty_model(
            X_tr, sr_tr, st_tr,
            X_va, sr_va, st_va,
            physics_lambda=0.1,
            config=DEFAULT_PGNN_CONFIG,
        )
        rp_pred = predict_residual_model(rp_model, X_va, sr_va)
        rp_metrics = evaluate_pgnn(st_va, rp_pred, sr_va, "ResPenalty")
        print_metrics(rp_metrics, "Residual-Penalty PGNN — Validation")
        all_metrics["PGNN (Res-Penalty)"] = {
            k.replace("ResPenalty_stress_", "test_"): v for k, v in rp_metrics.items()
            if "stress" in k
        }

        plot_training_curves(
            rp_history["train_loss"], rp_history.get("val_loss"),
            "Residual-Penalty PGNN Training",
            save_name="pgnn_res_penalty_training",
        )

        # Lambda sensitivity
        logger.info("Running λ sensitivity analysis...")
        lambda_results = lambda_sensitivity_analysis(
            X_tr, sr_tr, st_tr,
            X_va, sr_va, st_va,
            lambdas=[0.0, 0.01, 0.1, 0.5, 1.0, 5.0],
            config=DEFAULT_PGNN_CONFIG,
        )
        lambda_df = pd.DataFrame(lambda_results).T
        lambda_df.index.name = "lambda"
        lambda_df.to_csv(TABLES_DIR / "lambda_sensitivity.csv")
        logger.info("Lambda sensitivity saved.")

    else:
        logger.info("Skipping PGNN (--skip-pgnn).")

    # PHASE 6: Pipe-Flow Engineering
    logger.info("\n" + "═" * 60)
    logger.info("PHASE 6: Pipe-Flow Engineering Case Study")
    logger.info("═" * 60)

    from src.pipe_flow import (
        PipeFlowInput,
        calculate_pipe_flow,
        sensitivity_analysis,
        flow_rate_vs_pressure_drop,
    )

    # Use fitted parameters from the best model for each condition
    for res in fit_results:
        best = res["best_result"]
        if not best.converged:
            continue

        K = best.parameters["K"]
        n = best.parameters["n"]
        label = res["condition_label"]

        inp = PipeFlowInput(
            diameter_m=0.05,
            length_m=10.0,
            flow_rate_m3_per_s=1e-4,
            K=K,
            n=n,
            density_kg_per_m3=1000.0,
        )
        result = calculate_pipe_flow(inp)
        logger.info("\n%s:\n%s", label, result.summary_str())

    # Pressure drop vs flow rate plot
    example_res = [r for r in fit_results if r["best_result"].converged]
    if example_res:
        best_ex = example_res[0]["best_result"]
        base_inp = PipeFlowInput(
            diameter_m=0.05, length_m=10.0, flow_rate_m3_per_s=1e-4,
            K=best_ex.parameters["K"], n=best_ex.parameters["n"],
        )
        flow_rates = np.logspace(-5, -2, 50)
        dPs = flow_rate_vs_pressure_drop(base_inp, flow_rates)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(flow_rates * 1e6, dPs / 1000, "b-", lw=2)
        ax.set_xlabel("Flow Rate (mL/s)")
        ax.set_ylabel("Pressure Drop (kPa)")
        ax.set_title(f"ΔP vs Q — {example_res[0]['condition_label']}")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "pressure_drop_vs_flow_rate.png", dpi=300)
        plt.close(fig)

        # Sensitivity to K
        K_values = np.linspace(best_ex.parameters["K"] * 0.5,
                               best_ex.parameters["K"] * 2.0, 20)
        dP_K = sensitivity_analysis(base_inp, "K", K_values)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(K_values, dP_K / 1000, "r-o", ms=4, lw=2)
        ax.set_xlabel("Consistency Index K (Pa·sⁿ)")
        ax.set_ylabel("Pressure Drop (kPa)")
        ax.set_title("Sensitivity of ΔP to K")
        ax.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "sensitivity_K.png", dpi=300)
        plt.close(fig)

        # Sensitivity to n
        n_values = np.linspace(0.2, 1.0, 20)
        dP_n = sensitivity_analysis(base_inp, "n", n_values)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(n_values, dP_n / 1000, "g-s", ms=4, lw=2)
        ax.set_xlabel("Flow Behavior Index n")
        ax.set_ylabel("Pressure Drop (kPa)")
        ax.set_title("Sensitivity of ΔP to n")
        ax.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "sensitivity_n.png", dpi=300)
        plt.close(fig)

    # PHASE 7: Model Comparison
    logger.info("\n" + "═" * 60)
    logger.info("PHASE 7: Model Comparison")
    logger.info("═" * 60)

    if all_metrics:
        # Normalize metric keys
        comparison = {}
        for model_name, mets in all_metrics.items():
            comparison[model_name] = {
                "R²": mets.get("test_R2", np.nan),
                "RMSE": mets.get("test_RMSE", np.nan),
                "MAE": mets.get("test_MAE", np.nan),
            }
        comp_fig, comp_table = plot_comparison_table(comparison, "model_comparison")
        comp_table.to_csv(TABLES_DIR / "model_comparison.csv")
        logger.info("\nModel comparison:\n%s", comp_table.to_string())

    logger.info("\n" + "═" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("═" * 60)
    logger.info("Figures: %s", FIGURES_DIR)
    logger.info("Tables:  %s", TABLES_DIR)


def main():
    args = parse_args()
    run_pipeline(
        data_path=args.data,
        skip_pgnn=args.skip_pgnn,
        skip_shap=args.skip_shap,
    )


if __name__ == "__main__":
    main()
