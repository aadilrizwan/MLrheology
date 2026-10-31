"use client";

import React, { useState } from "react";
import {
  MODEL_BENCHMARKS,
  LAMBDA_SENSITIVITY,
  GENERALIZATION_EXPERIMENT,
} from "@/data/rheologyData";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { AlertCircle, CheckCircle2, ShieldCheck, HelpCircle } from "lucide-react";

export const BenchmarkDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"models" | "generalization" | "lambda">("models");

  return (
    <div className="surface-card rounded-2xl p-4 sm:p-6 md:p-8 relative overflow-hidden">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between pb-6 border-b border-white/[0.06] gap-4">
        <div>
          <div className="flex items-center space-x-2 text-[11px] font-mono tracking-widest uppercase text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span>Cross-Paradigm Scientific Benchmarks</span>
          </div>
          <h2 className="text-xl md:text-2xl font-semibold tracking-tight text-white mt-1">
            Empirical vs. Data-Driven vs. Physics-Guided Architecture
          </h2>
          <p className="text-xs text-zinc-400 mt-1 max-w-xl font-light">
            Comprehensive statistical evaluation comparing nonlinear parameter optimization,
            tree ensembles, and deep neural networks on shear stress and apparent viscosity.
          </p>
        </div>

        {/* Minimal Responsive Segmented Tabs */}
        <div className="w-full sm:w-auto flex overflow-x-auto p-1 rounded-lg bg-zinc-900 border border-white/[0.06] gap-1">
          <button
            onClick={() => setActiveTab("models")}
            className={`px-3 py-1.5 text-xs font-mono rounded-md whitespace-nowrap transition-all ${
              activeTab === "models" ? "bg-zinc-800 text-white shadow-sm font-medium" : "text-zinc-400 hover:text-white"
            }`}
          >
            Model Matrix
          </button>
          <button
            onClick={() => setActiveTab("generalization")}
            className={`px-3 py-1.5 text-xs font-mono rounded-md whitespace-nowrap transition-all ${
              activeTab === "generalization" ? "bg-zinc-800 text-white shadow-sm font-medium" : "text-zinc-400 hover:text-white"
            }`}
          >
            Extrapolation Holdouts
          </button>
          <button
            onClick={() => setActiveTab("lambda")}
            className={`px-3 py-1.5 text-xs font-mono rounded-md whitespace-nowrap transition-all ${
              activeTab === "lambda" ? "bg-zinc-800 text-white shadow-sm font-medium" : "text-zinc-400 hover:text-white"
            }`}
          >
            λ-Loss Sensitivity
          </button>
        </div>
      </div>

      {/* Tab 1: Master Model Benchmark Matrix */}
      {activeTab === "models" && (
        <div className="pt-6 space-y-6">
          <div className="text-[10px] font-mono text-zinc-500 sm:hidden">
            ← Scroll horizontally to view all metrics →
          </div>
          <div className="w-full max-w-full overflow-x-auto pb-2">
            <table className="w-full min-w-[580px] text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-white/[0.08] text-zinc-400 font-mono text-[11px]">
                  <th className="pb-3 font-medium">Model Architecture</th>
                  <th className="pb-3 font-medium">Paradigm</th>
                  <th className="pb-3 font-medium text-right">R² Score</th>
                  <th className="pb-3 font-medium text-right">RMSE (Pa)</th>
                  <th className="pb-3 font-medium text-right">MAE (Pa)</th>
                  <th className="pb-3 font-medium">Physical Guarantee</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04] font-mono">
                {MODEL_BENCHMARKS.map((m, idx) => (
                  <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                    <td className="py-3.5 font-semibold text-white flex items-center space-x-2">
                      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: m.color }} />
                      <span className="font-sans font-medium text-sm">{m.name}</span>
                    </td>
                    <td className="py-3.5 text-zinc-400 text-xs font-sans">{m.paradigm}</td>
                    <td className="py-3.5 text-right font-semibold text-emerald-400">
                      {(m.r2 * 100).toFixed(2)}%
                    </td>
                    <td className="py-3.5 text-right text-zinc-200">{m.rmse.toFixed(3)}</td>
                    <td className="py-3.5 text-right text-zinc-200">{m.mae.toFixed(3)}</td>
                    <td className="py-3.5">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-zinc-900 border border-white/[0.08] text-zinc-300">
                        {m.physicsGuarantee}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="p-4 rounded-xl bg-zinc-900/40 border border-white/[0.06] text-xs text-zinc-300 flex items-start space-x-3">
            <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-semibold text-white">Why Parameter-Prediction PGNN is the Winner:</span>
              <p className="text-zinc-400 font-light leading-relaxed">
                By enforcing non-negative parameters ($K &gt; 0, n &gt; 0, \tau_y \ge 0$) via Softplus
                and evaluating the physical constitutive law within the forward computation graph,
                the network achieves $R^2 = 0.9771$ while eliminating unphysical negative viscosities.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Generalization Holdouts */}
      {activeTab === "generalization" && (
        <div className="pt-6 space-y-6">
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-200 flex items-start space-x-3">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-semibold text-rose-300">The Machine Learning Extrapolation Crisis:</span>
              <p className="text-zinc-300 font-light leading-relaxed">
                Decision trees partition space into orthogonal constant leaves. When testing against unseen
                dilute concentrations (c = 0.25 wt%), both Random Forest and XGBoost exhibit severe
                collapse (R² = -38.39 and -30.74). This empirically confirms that purely data-driven models
                cannot extrapolate non-Newtonian flow without physics guidance.
              </p>
            </div>
          </div>

          <div className="text-[10px] font-mono text-zinc-500 sm:hidden">
            ← Scroll horizontally to view holdout results →
          </div>
          <div className="w-full max-w-full overflow-x-auto pb-2">
            <table className="w-full min-w-[620px] text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="border-b border-white/[0.08] text-zinc-400 text-[11px]">
                  <th className="pb-3 font-medium">Holdout Condition</th>
                  <th className="pb-3 font-medium">Domain</th>
                  <th className="pb-3 font-medium text-right">RF R²</th>
                  <th className="pb-3 font-medium text-right">XGB R²</th>
                  <th className="pb-3 font-medium text-right">RF RMSE</th>
                  <th className="pb-3 font-medium text-right">XGB RMSE</th>
                  <th className="pb-3 font-medium pl-4">Physical Diagnosis</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {GENERALIZATION_EXPERIMENT.map((g, idx) => (
                  <tr key={idx} className="hover:bg-white/[0.02]">
                    <td className="py-3 font-semibold text-white">{g.holdout}</td>
                    <td className="py-3">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                        g.type === "Interpolation" ? "text-teal-400 bg-teal-500/10" : "text-rose-400 bg-rose-500/10"
                      }`}>
                        {g.type}
                      </span>
                    </td>
                    <td className={`py-3 text-right font-bold ${g.rfR2 < 0 ? "text-rose-400" : "text-emerald-400"}`}>
                      {g.rfR2.toFixed(2)}
                    </td>
                    <td className={`py-3 text-right font-bold ${g.xgbR2 < 0 ? "text-rose-400" : "text-emerald-400"}`}>
                      {g.xgbR2.toFixed(2)}
                    </td>
                    <td className="py-3 text-right text-zinc-300">{g.rfRmse.toFixed(2)}</td>
                    <td className="py-3 text-right text-zinc-300">{g.xgbRmse.toFixed(2)}</td>
                    <td className="py-3 pl-4 font-sans text-zinc-400 font-light text-xs">{g.insight}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 3: Lambda Loss Sensitivity */}
      {activeTab === "lambda" && (
        <div className="pt-6 space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-center">
            <div className="lg:col-span-7 min-w-0 w-full bg-zinc-950 p-3 sm:p-4 rounded-xl border border-white/[0.06] overflow-hidden">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs font-mono text-zinc-400 mb-3 gap-1">
                <span>Validation R² vs. Physics Penalty Weight (λ)</span>
                <span className="text-zinc-500 text-[10px]">L_total = L_data + λ·L_physics</span>
              </div>
              <div className="h-48 sm:h-56 w-full min-w-0">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={LAMBDA_SENSITIVITY} margin={{ top: 10, right: 15, left: -20, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="2 4" stroke="#27272a" />
                    <XAxis dataKey="lambda" stroke="#52525b" fontSize={10} />
                    <YAxis stroke="#52525b" fontSize={10} domain={[-0.2, 0.5]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0d0d11",
                        border: "1px solid rgba(255,255,255,0.1)",
                        fontSize: "11px",
                        fontFamily: "monospace",
                      }}
                      formatter={(val: number) => [`${(val * 100).toFixed(2)}%`, "Validation R²"]}
                    />
                    <Line type="monotone" dataKey="r2" stroke="#10b981" strokeWidth={2} dot={{ r: 4, fill: "#34d399" }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="lg:col-span-5 min-w-0 w-full space-y-3 text-xs font-light text-zinc-300">
              <h4 className="font-sans font-semibold text-sm text-white">
                Gradient Interference in Soft-Penalty PINNs
              </h4>
              <p className="leading-relaxed">
                When $\lambda$ is increased above 0.1, the physics residual penalty aggressively competes with the empirical loss gradient.
                At $\lambda = 5.0$, the network degenerates to negative $R^2$ (-0.14).
              </p>
              <div className="p-3 rounded-lg bg-zinc-900 border border-white/[0.06] font-mono text-[11px] text-emerald-400">
                Key Defense Point: Inductive Parameter-Prediction enforces exact physics without optimization instability.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
