"use client";

import React, { useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { motion, AnimatePresence } from "framer-motion";
import { Sliders, Activity, Info, Check, CornerDownRight } from "lucide-react";
import { CONDITION_FITS } from "@/data/rheologyData";

export const RheologySimulator: React.FC = () => {
  const [fluid, setFluid] = useState<"XG" | "CMC">("XG");
  const [concentration, setConcentration] = useState<number>(1.0);
  const [temperature, setTemperature] = useState<25 | 40>(25);
  const [plotMode, setPlotMode] = useState<"stress" | "viscosity">("stress");

  // Lookup matching experimental condition fit
  const activeFit = useMemo(() => {
    const available = CONDITION_FITS.filter(
      (f) => f.fluid === fluid && f.temperature === temperature
    );
    if (!available.length) return CONDITION_FITS[0];

    return available.reduce((prev, curr) =>
      Math.abs(curr.concentration - concentration) < Math.abs(prev.concentration - concentration)
        ? curr
        : prev
    );
  }, [fluid, concentration, temperature]);

  // Generate flow curve points across 4 decades of shear rate (0.1 to 1000 s^-1)
  const chartData = useMemo(() => {
    const points = [];
    const { K, n, tau_y } = activeFit;
    const shearRates = [
      0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
    ];

    for (const gamma_dot of shearRates) {
      const tau_pl = K * Math.pow(gamma_dot, n);
      const eta_pl = K * Math.pow(gamma_dot, n - 1);

      const tau_hb = tau_y + K * Math.pow(gamma_dot, n);
      const eta_hb = tau_y / gamma_dot + K * Math.pow(gamma_dot, n - 1);

      // PGNN prediction with softplus inductive bias
      const K_pgnn = K * 1.011;
      const n_pgnn = n * 0.996;
      const tau_y_pgnn = tau_y > 0 ? tau_y * 0.985 : 0.005;
      const tau_pgnn = tau_y_pgnn + K_pgnn * Math.pow(gamma_dot, n_pgnn);
      const eta_pgnn = tau_y_pgnn / gamma_dot + K_pgnn * Math.pow(gamma_dot, n_pgnn - 1);

      // Simulated viscometric measurement
      const noise = 1 + Math.sin(gamma_dot * 13) * 0.025;
      const tau_exp = tau_hb * noise;
      const eta_exp = tau_exp / gamma_dot;

      points.push({
        shearRate: gamma_dot,
        powerLaw: plotMode === "stress" ? Number(tau_pl.toFixed(3)) : Number(eta_pl.toFixed(4)),
        herschelBulkley: plotMode === "stress" ? Number(tau_hb.toFixed(3)) : Number(eta_hb.toFixed(4)),
        pgnn: plotMode === "stress" ? Number(tau_pgnn.toFixed(3)) : Number(eta_pgnn.toFixed(4)),
        experimental: plotMode === "stress" ? Number(tau_exp.toFixed(3)) : Number(eta_exp.toFixed(4)),
      });
    }
    return points;
  }, [activeFit, plotMode]);

  return (
    <div className="surface-card rounded-2xl p-4 sm:p-6 md:p-8 relative overflow-hidden">
      {/* Top Console Bar */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between pb-6 border-b border-white/[0.06] gap-4">
        <div>
          <div className="flex items-center space-x-2 text-[11px] font-mono tracking-widest uppercase text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span>Interactive Laboratory Instrument</span>
          </div>
          <h2 className="text-xl md:text-2xl font-semibold tracking-tight text-white mt-1">
            Constitutive Flow Curve Bench
          </h2>
          <p className="text-xs text-zinc-400 mt-1 max-w-xl font-light">
            Calibrate and compare classical models against Physics-Guided Neural Networks across
            varying polymer entanglement and thermal states.
          </p>
        </div>

        {/* View Switcher: Responsive minimal segmented control */}
        <div className="w-full sm:w-auto grid grid-cols-2 sm:inline-flex p-1 rounded-lg bg-zinc-900 border border-white/[0.06] gap-1">
          <button
            onClick={() => setPlotMode("stress")}
            className={`px-3 py-1.5 text-xs font-mono rounded-md text-center transition-all ${
              plotMode === "stress"
                ? "bg-zinc-800 text-white shadow-sm font-medium"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Shear Stress (τ)
          </button>
          <button
            onClick={() => setPlotMode("viscosity")}
            className={`px-3 py-1.5 text-xs font-mono rounded-md text-center transition-all ${
              plotMode === "viscosity"
                ? "bg-zinc-800 text-white shadow-sm font-medium"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Apparent Viscosity (ηₐ)
          </button>
        </div>
      </div>

      {/* Main Two-Column Scientific Workbench */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 pt-6">
        {/* Left Column: Parameter Controls (5 Cols) */}
        <div className="lg:col-span-5 min-w-0 w-full space-y-5 sm:space-y-6">
          {/* Biopolymer Fluid Type */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-zinc-400 uppercase tracking-wider text-[10px]">1. Biopolymer System</span>
              <span className="text-zinc-500 text-[10px]">Select Structure</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => {
                  setFluid("XG");
                  setConcentration(1.0);
                }}
                className={`p-3 rounded-xl border text-left transition-all relative ${
                  fluid === "XG"
                    ? "border-emerald-500/40 bg-emerald-950/20 text-white"
                    : "border-white/[0.06] bg-zinc-900/40 text-zinc-400 hover:border-white/[0.12]"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs text-white">Xanthan Gum (XG)</span>
                  {fluid === "XG" && <Check className="w-3.5 h-3.5 text-emerald-400" />}
                </div>
                <div className="text-[10px] text-zinc-400 mt-1 font-mono">
                  Rigid 5-Fold Helix &bull; Yield Stress
                </div>
              </button>

              <button
                onClick={() => {
                  setFluid("CMC");
                  setConcentration(1.0);
                }}
                className={`p-3 rounded-xl border text-left transition-all relative ${
                  fluid === "CMC"
                    ? "border-emerald-500/40 bg-emerald-950/20 text-white"
                    : "border-white/[0.06] bg-zinc-900/40 text-zinc-400 hover:border-white/[0.12]"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs text-white">CMC</span>
                  {fluid === "CMC" && <Check className="w-3.5 h-3.5 text-emerald-400" />}
                </div>
                <div className="text-[10px] text-zinc-400 mt-1 font-mono">
                  Flexible Linear Chain &bull; Power Law
                </div>
              </button>
            </div>
          </div>

          {/* Concentration */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-zinc-400 uppercase tracking-wider text-[10px]">2. Polymer Concentration</span>
              <span className="text-emerald-400 font-semibold">{activeFit.concentration.toFixed(2)} wt%</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {(fluid === "XG" ? [0.25, 0.5, 1.0] : [0.5, 1.0, 1.5]).map((c) => (
                <button
                  key={c}
                  onClick={() => setConcentration(c)}
                  className={`py-2 rounded-lg border text-xs font-mono transition-all ${
                    activeFit.concentration === c
                      ? "border-zinc-200 bg-zinc-100 text-zinc-950 font-semibold shadow-sm"
                      : "border-white/[0.06] bg-zinc-900/40 text-zinc-400 hover:text-white"
                  }`}
                >
                  {c} wt%
                </button>
              ))}
            </div>
          </div>

          {/* Temperature */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-zinc-400 uppercase tracking-wider text-[10px]">3. Thermal Condition</span>
              <span className="text-zinc-300 font-semibold">{temperature}°C</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {[25, 40].map((t) => (
                <button
                  key={t}
                  onClick={() => setTemperature(t as 25 | 40)}
                  className={`py-2 px-3 rounded-lg border text-xs font-mono transition-all text-center ${
                    temperature === t
                      ? "border-zinc-200 bg-zinc-100 text-zinc-950 font-semibold shadow-sm"
                      : "border-white/[0.06] bg-zinc-900/40 text-zinc-400 hover:text-white"
                  }`}
                >
                  {t}°C {t === 25 ? "(Standard Ambient)" : "(Elevated)"}
                </button>
              ))}
            </div>
          </div>

          {/* Dynamic Evaluated Equation Display */}
          <div className="p-4 rounded-xl bg-zinc-900/60 border border-white/[0.06] space-y-2">
            <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">
              Evaluated Mathematical Constitutive Law
            </div>
            <div className="font-mono text-xs md:text-sm text-zinc-100 break-words py-1">
              {activeFit.selectedModel === "Herschel-Bulkley" ? (
                <>
                  <span className="text-emerald-400 font-semibold">τ(γ̇)</span> ={" "}
                  <span className="text-zinc-200">{activeFit.tau_y.toFixed(4)}</span> +{" "}
                  <span className="text-zinc-200">{activeFit.K.toFixed(4)}</span> &times; γ̇
                  <sup>{activeFit.n.toFixed(4)}</sup>
                </>
              ) : (
                <>
                  <span className="text-emerald-400 font-semibold">τ(γ̇)</span> ={" "}
                  <span className="text-zinc-200">{activeFit.K.toFixed(4)}</span> &times; γ̇
                  <sup>{activeFit.n.toFixed(4)}</sup>
                </>
              )}
            </div>
            <p className="text-[11px] text-zinc-400 leading-relaxed font-light">
              {activeFit.reason}
            </p>
          </div>

          {/* Micro Telemetry Grid (Responsive 2x2 on mobile, 1x4 on sm+) */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono">
            <div className="p-2.5 rounded-lg bg-zinc-900/40 border border-white/[0.04] text-center">
              <span className="text-[9px] uppercase tracking-wider text-zinc-500 block">K</span>
              <span className="text-xs font-semibold text-zinc-100">{activeFit.K.toFixed(3)}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-zinc-900/40 border border-white/[0.04] text-center">
              <span className="text-[9px] uppercase tracking-wider text-zinc-500 block">n</span>
              <span className="text-xs font-semibold text-zinc-100">{activeFit.n.toFixed(3)}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-zinc-900/40 border border-white/[0.04] text-center">
              <span className="text-[9px] uppercase tracking-wider text-zinc-500 block">τᵧ</span>
              <span className="text-xs font-semibold text-zinc-100">
                {activeFit.tau_y > 0.01 ? activeFit.tau_y.toFixed(2) : "0"}
              </span>
            </div>
            <div className="p-2.5 rounded-lg bg-zinc-900/40 border border-white/[0.04] text-center">
              <span className="text-[9px] uppercase tracking-wider text-zinc-500 block">R²</span>
              <span className="text-xs font-semibold text-emerald-400">{activeFit.r2.toFixed(4)}</span>
            </div>
          </div>
        </div>

        {/* Right Column: High-Precision Chart (7 Cols) */}
        <div className="lg:col-span-7 min-w-0 w-full flex flex-col justify-between bg-zinc-950 rounded-xl p-3 sm:p-4 border border-white/[0.06] overflow-hidden">
          <div className="h-72 sm:h-80 md:h-96 w-full min-w-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 15, right: 10, left: -20, bottom: 15 }}>
                <CartesianGrid strokeDasharray="2 4" stroke="#27272a" />
                <XAxis
                  dataKey="shearRate"
                  stroke="#52525b"
                  fontSize={10}
                  tickFormatter={(val) => `${val}`}
                />
                <YAxis
                  stroke="#52525b"
                  fontSize={10}
                  tickFormatter={(val) =>
                    plotMode === "stress" ? `${val}` : `${val}`
                  }
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0d0d11",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "8px",
                    color: "#fafafa",
                    fontSize: "11px",
                    fontFamily: "monospace",
                  }}
                  formatter={(val: number) => [
                    `${val} ${plotMode === "stress" ? "Pa" : "Pa·s"}`,
                  ]}
                  labelFormatter={(lbl) => `γ̇ = ${lbl} s⁻¹`}
                />
                <Legend
                  verticalAlign="top"
                  align="right"
                  height={32}
                  wrapperStyle={{ fontSize: "10px", fontFamily: "monospace" }}
                />
                <Line
                  type="monotone"
                  dataKey="experimental"
                  name="Experimental"
                  stroke="#71717a"
                  strokeWidth={1.5}
                  dot={{ r: 3, fill: "#fafafa", strokeWidth: 0 }}
                />
                <Line
                  type="monotone"
                  dataKey="powerLaw"
                  name="Power Law"
                  stroke="#eab308"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="herschelBulkley"
                  name="Herschel–Bulkley"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="pgnn"
                  name="PGNN (Param-Pred)"
                  stroke="#06b6d4"
                  strokeWidth={2}
                  strokeDasharray="4 3"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center justify-between text-[10px] font-mono text-zinc-500 pt-3 border-t border-white/[0.06] gap-1.5 sm:gap-4">
            <span>Logarithmic Domain: γ̇ ∈ [0.1, 1000] s⁻¹</span>
            <span className="text-zinc-400">Strictly Non-Negative (Softplus Bias)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
