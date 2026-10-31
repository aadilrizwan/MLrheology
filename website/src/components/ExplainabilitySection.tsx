"use client";

import React, { useState } from "react";
import { ArrowUpRight, HelpCircle } from "lucide-react";

export const ExplainabilitySection: React.FC = () => {
  const [selectedFeature, setSelectedFeature] = useState<string>("shear_rate");

  const featureExplanations: Record<
    string,
    {
      title: string;
      rank: number;
      importance: string;
      direction: string;
      chemicalMechanism: string;
      figure: string;
    }
  > = {
    shear_rate: {
      title: "Shear Rate (γ̇, s⁻¹)",
      rank: 1,
      importance: "68% Global Impact",
      direction: "Monotonic Viscous Thinning",
      chemicalMechanism:
        "Increasing shear rates physically unravel coiled macromolecules and align them with flow streamlines. Intermolecular hydrodynamic drag decreases, yielding the characteristic shear-thinning slope n - 1 < 0.",
      figure: "/figures/xgb_shap_dep_shear_rate.png",
    },
    concentration: {
      title: "Polymer Concentration (c, wt%)",
      rank: 2,
      importance: "22% Global Impact",
      direction: "Consistency Shift (K ∝ c^α)",
      chemicalMechanism:
        "Elevating concentration from 0.25% to 1.5% increases intermolecular coil overlap and topological entanglement density. The consistency index K scales as c^alpha, shifting flow curves upward.",
      figure: "/figures/xgb_shap_dep_concentration.png",
    },
    temperature: {
      title: "Temperature (T, °C)",
      rank: 3,
      importance: "7% Global Impact",
      direction: "Arrhenius Thermal Thinning",
      chemicalMechanism:
        "Thermal agitation increases polymer chain kinetic mobility and free volume, weakening non-covalent hydrogen bonding and lowering viscosity according to the Arrhenius–Andrade equation.",
      figure: "/figures/xgb_shap_dep_temperature.png",
    },
    fluid_type: {
      title: "Polymer Chemistry (XG vs. CMC)",
      rank: 4,
      importance: "3% Global Impact",
      direction: "Helical Yield Stress vs. Linear Chain",
      chemicalMechanism:
        "Carboxymethyl Cellulose is a flexible linear polymer with pure pseudoplastic flow (tau_y ≈ 0). Xanthan Gum possesses a rigid 5-fold helical backbone forming weak 3D networks with distinct yield stresses (tau_y up to 0.58 Pa).",
      figure: "/figures/xgb_shap_dep_fluid_type_XG.png",
    },
  };

  const active = featureExplanations[selectedFeature];

  return (
    <div className="surface-card rounded-2xl p-4 sm:p-6 md:p-8 relative overflow-hidden">
      {/* Header */}
      <div className="pb-6 border-b border-white/[0.06]">
        <div className="flex items-center space-x-2 text-[11px] font-mono tracking-widest uppercase text-emerald-400">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          <span>Mechanistic Interpretability</span>
        </div>
        <h2 className="text-xl md:text-2xl font-semibold tracking-tight text-white mt-1">
          SHAP Game-Theoretic Feature Attribution
        </h2>
        <p className="text-xs text-zinc-400 mt-1 max-w-xl font-light">
          Quantifying decision boundary importance in tree models and grounding statistical contributions
          in chemical engineering transport phenomena.
        </p>
      </div>

      {/* Grid: Feature Selector (Responsive 2x2 on mobile, vertical stack on lg) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 pt-6 items-start">
        {/* Left Column: Feature List */}
        <div className="lg:col-span-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-2">
          {Object.entries(featureExplanations).map(([key, item]) => (
            <button
              key={key}
              onClick={() => setSelectedFeature(key)}
              className={`w-full p-3 sm:p-3.5 rounded-xl border text-left transition-all ${
                selectedFeature === key
                  ? "border-emerald-500/40 bg-emerald-950/20 text-white"
                  : "border-white/[0.06] bg-zinc-900/40 text-zinc-400 hover:border-white/[0.12]"
              }`}
            >
              <div className="flex items-center justify-between text-[10px] font-mono">
                <span className="text-emerald-400 font-semibold">RANK #{item.rank}</span>
                <span className="text-zinc-400">{item.importance}</span>
              </div>
              <div className="font-semibold text-xs text-white mt-1">{item.title}</div>
              <div className="text-[10px] text-zinc-500 font-mono mt-0.5 truncate">{item.direction}</div>
            </button>
          ))}
        </div>

        {/* Right Column: Deep Explanation & SHAP Curve */}
        <div className="lg:col-span-8 space-y-4">
          <div className="p-4 sm:p-5 rounded-xl bg-zinc-900/50 border border-white/[0.06] space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-white/[0.06] pb-3 gap-2">
              <div>
                <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">
                  Chemical Transport Mechanism
                </span>
                <h3 className="text-base font-semibold text-white mt-0.5">{active.title}</h3>
              </div>
              <span className="self-start sm:self-auto px-2.5 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                {active.importance}
              </span>
            </div>

            <p className="text-xs text-zinc-300 font-light leading-relaxed">
              {active.chemicalMechanism}
            </p>

            {/* Embedded SHAP Curve */}
            <div className="rounded-lg bg-zinc-950 border border-white/[0.06] p-2 sm:p-3 flex flex-col items-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={active.figure}
                alt={active.title}
                className="max-h-56 sm:max-h-64 w-full sm:w-auto object-contain rounded"
              />
              <span className="text-[10px] font-mono text-zinc-500 mt-2 text-center">
                SHAP Dependence Response Curve &bull; XGBoost Pipeline
              </span>
            </div>

            <div className="p-3 rounded-lg bg-zinc-950 border border-white/[0.06] text-[11px] text-zinc-400 font-light">
              <strong className="text-zinc-200 font-medium">Conference Defense Note:</strong> SHAP values
              describe how the ensemble partitions its tree splits. They identify feature reliance,
              complementing rather than replacing physical conservation laws.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
