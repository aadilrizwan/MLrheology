"use client";

import React, { useState } from "react";
import { Download, Maximize2, X } from "lucide-react";

interface FigureItem {
  id: string;
  name: string;
  category: "Flow Curves" | "Model Fits" | "Residuals" | "SHAP" | "Hydraulics";
  filename: string;
  description: string;
}

const GALLERY_FIGURES: FigureItem[] = [
  { id: "1", name: "Flow Curves (All Conditions)", category: "Flow Curves", filename: "flow_curves.png", description: "Log-log shear stress vs shear rate across all concentrations and temperatures." },
  { id: "2", name: "Flow Curves Grouped by Fluid", category: "Flow Curves", filename: "flow_curves_by_fluid.png", description: "Side-by-side comparison of pseudoplastic behavior between CMC and XG." },
  { id: "3", name: "Cross-Paradigm Model Comparison", category: "Model Fits", filename: "model_comparison.png", description: "Comprehensive bar chart comparing R², RMSE, and MAE across all 5 paradigms." },
  { id: "4", name: "PGNN (Param-Pred) Actual vs Predicted", category: "Model Fits", filename: "pgnn_param_pred_actual_vs_pred.png", description: "Validation scatter plot showing R² = 0.977 on shear stress." },
  { id: "5", name: "PGNN Param-Pred Loss History", category: "Model Fits", filename: "pgnn_param_pred_training.png", description: "Training and validation loss progression with early stopping." },
  { id: "6", name: "PGNN Res-Penalty Loss History", category: "Model Fits", filename: "pgnn_res_penalty_training.png", description: "Evolution of total, data, and physics regularization loss terms." },
  { id: "7", name: "Random Forest Actual vs Predicted", category: "Residuals", filename: "rf_actual_vs_pred.png", description: "Viscosity test predictions from 200 bagged decision trees." },
  { id: "8", name: "Random Forest Residual Distribution", category: "Residuals", filename: "rf_residuals.png", description: "Residual error histogram and scatter for Random Forest." },
  { id: "9", name: "XGBoost Actual vs Predicted", category: "Residuals", filename: "xgb_actual_vs_pred.png", description: "Viscosity test predictions from gradient boosted trees (R² = 0.844)." },
  { id: "10", name: "XGBoost Residual Distribution", category: "Residuals", filename: "xgb_residuals.png", description: "Residual distribution for XGBoost regression pipeline." },
  { id: "11", name: "SHAP Feature Importance (RF)", category: "SHAP", filename: "rf_shap_bar.png", description: "Mean absolute SHAP value ranking for Random Forest." },
  { id: "12", name: "SHAP Beeswarm Summary (RF)", category: "SHAP", filename: "rf_shap_summary.png", description: "Beeswarm plot showing feature impact directions for Random Forest." },
  { id: "13", name: "SHAP Feature Importance (XGB)", category: "SHAP", filename: "xgb_shap_bar.png", description: "Global feature importance ranking for XGBoost." },
  { id: "14", name: "SHAP Beeswarm Summary (XGB)", category: "SHAP", filename: "xgb_shap_summary.png", description: "Beeswarm plot highlighting shear rate dominance in XGBoost." },
  { id: "15", name: "SHAP Dependence: Shear Rate", category: "SHAP", filename: "xgb_shap_dep_shear_rate.png", description: "Monotonic negative attribution curve showing shear-thinning behavior." },
  { id: "16", name: "SHAP Dependence: Concentration", category: "SHAP", filename: "xgb_shap_dep_concentration.png", description: "Positive nonlinear attribution scaling with polymer entanglement." },
  { id: "17", name: "SHAP Dependence: Temperature", category: "SHAP", filename: "xgb_shap_dep_temperature.png", description: "Negative thermal attribution reflecting Arrhenius thinning." },
  { id: "18", name: "Pressure Drop vs Flow Rate", category: "Hydraulics", filename: "pressure_drop_vs_flow_rate.png", description: "Log-log pipeline pressure drop curve for non-Newtonian pipe flow." },
  { id: "19", name: "Hydraulic Sensitivity to K", category: "Hydraulics", filename: "sensitivity_K.png", description: "Linear scaling of pressure drop with consistency index K." },
  { id: "20", name: "Hydraulic Sensitivity to n", category: "Hydraulics", filename: "sensitivity_n.png", description: "Nonlinear exponential sensitivity of pressure drop to flow index n." },
  { id: "21", name: "Fit: HB XG 1.0wt% 25°C", category: "Model Fits", filename: "fit_HB_XG___1.0___25.png", description: "Herschel-Bulkley non-linear regression fit for concentrated Xanthan Gum." },
  { id: "22", name: "Fit: PL CMC 1.0wt% 25°C", category: "Model Fits", filename: "fit_PL_CMC___1.0___25.png", description: "Power Law non-linear regression fit for Carboxymethyl Cellulose." },
  { id: "23", name: "Fit: HB XG 0.5wt% 25°C", category: "Model Fits", filename: "fit_HB_XG___0.5___25.png", description: "Herschel-Bulkley fit for semi-dilute Xanthan Gum exhibiting yield stress." },
  { id: "24", name: "Fit: PL CMC 0.5wt% 25°C", category: "Model Fits", filename: "fit_PL_CMC___0.5___25.png", description: "Power Law fit for dilute CMC." },
];

export const FigureGallery: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [activeModalFigure, setActiveModalFigure] = useState<FigureItem | null>(null);

  const categories = ["All", "Flow Curves", "Model Fits", "Residuals", "SHAP", "Hydraulics"];

  const filteredFigures =
    selectedCategory === "All"
      ? GALLERY_FIGURES
      : GALLERY_FIGURES.filter((f) => f.category === selectedCategory);

  return (
    <div className="surface-card rounded-2xl p-4 sm:p-6 md:p-8 relative overflow-hidden">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between pb-6 border-b border-white/[0.06] gap-4">
        <div>
          <div className="flex items-center space-x-2 text-[11px] font-mono tracking-widest uppercase text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span>High-Resolution Graphics</span>
          </div>
          <h2 className="text-xl md:text-2xl font-semibold tracking-tight text-white mt-1">
            Publication Figure Gallery
          </h2>
          <p className="text-xs text-zinc-400 mt-1 max-w-xl font-light">
            All 51 generated plots in 300 DPI vector-ready raster formats across curve fitting,
            machine learning residuals, SHAP attributions, and hydraulics.
          </p>
        </div>

        {/* Categories: Horizontal scrollable on mobile, wrapping on sm+ */}
        <div className="w-full sm:w-auto flex overflow-x-auto p-1 rounded-lg bg-zinc-900 border border-white/[0.06] gap-1">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1.5 text-xs font-mono rounded-md whitespace-nowrap transition-all ${
                selectedCategory === cat
                  ? "bg-zinc-800 text-white font-medium shadow-sm"
                  : "text-zinc-400 hover:text-white"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-6">
        {filteredFigures.map((fig) => (
          <div
            key={fig.id}
            onClick={() => setActiveModalFigure(fig)}
            className="group cursor-pointer rounded-xl bg-zinc-900/40 border border-white/[0.06] hover:border-white/[0.18] p-3 transition-all flex flex-col justify-between"
          >
            <div className="relative rounded-lg overflow-hidden bg-zinc-950 aspect-[4/3] flex items-center justify-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`/figures/${fig.filename}`}
                alt={fig.name}
                className="w-full h-full object-contain p-2 group-hover:scale-105 transition-transform duration-300"
                loading="lazy"
              />
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <Maximize2 className="w-4 h-4 text-white" />
              </div>
            </div>

            <div className="mt-2.5 space-y-1">
              <div className="flex justify-between items-center text-[10px] font-mono text-zinc-500">
                <span className="text-zinc-400">{fig.category}</span>
                <span>300 DPI</span>
              </div>
              <h4 className="font-sans font-medium text-xs text-zinc-100 truncate">{fig.name}</h4>
              <p className="text-[10px] text-zinc-400 font-light line-clamp-2 leading-tight">
                {fig.description}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox Modal */}
      {activeModalFigure && (
        <div
          className="fixed inset-0 z-50 bg-black/85 backdrop-blur-md flex items-center justify-center p-3 sm:p-4"
          onClick={() => setActiveModalFigure(null)}
        >
          <div
            className="max-w-4xl w-full max-h-[92vh] overflow-y-auto bg-zinc-900 border border-white/[0.1] rounded-2xl p-4 sm:p-6 shadow-2xl flex flex-col space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <div>
                <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">
                  {activeModalFigure.category}
                </span>
                <h3 className="text-sm sm:text-base font-semibold text-white mt-0.5">{activeModalFigure.name}</h3>
              </div>
              <button
                onClick={() => setActiveModalFigure(null)}
                className="p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-white/[0.04] transition-colors"
                aria-label="Close image modal"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="max-h-[50vh] sm:max-h-[60vh] rounded-xl bg-zinc-950 flex items-center justify-center p-2 sm:p-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`/figures/${activeModalFigure.filename}`}
                alt={activeModalFigure.name}
                className="max-h-[45vh] sm:max-h-[55vh] w-auto object-contain"
              />
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs text-zinc-400 font-light pt-1 gap-3">
              <p className="leading-relaxed">{activeModalFigure.description}</p>
              <a
                href={`/figures/${activeModalFigure.filename}`}
                download={activeModalFigure.filename}
                className="inline-flex items-center justify-center space-x-1.5 px-3.5 py-2 rounded-lg bg-zinc-100 text-zinc-950 hover:bg-white text-xs font-semibold shrink-0 transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download Figure</span>
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
