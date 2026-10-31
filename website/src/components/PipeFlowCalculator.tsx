"use client";

import React, { useState, useMemo } from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { ShieldCheck, AlertTriangle, ArrowRight, Gauge } from "lucide-react";

export const PipeFlowCalculator: React.FC = () => {
  const [diameterMm, setDiameterMm] = useState<number>(50);
  const [lengthM, setLengthM] = useState<number>(30);
  const [flowRateLps, setFlowRateLps] = useState<number>(0.5);
  const [K, setK] = useState<number>(2.48);
  const [n, setN] = useState<number>(0.35);
  const [density, setDensity] = useState<number>(1000);

  const hydraulics = useMemo(() => {
    const D = diameterMm / 1000.0;
    const L = lengthM;
    const Q = flowRateLps / 1000.0;

    const V = (4 * Q) / (Math.PI * Math.pow(D, 2));
    const rabinowitsch = (3 * n + 1) / (4 * n);
    const gamma_w = rabinowitsch * ((8 * V) / D);
    const tau_w = K * Math.pow(gamma_w, n);

    const deltaP_Pa = (4 * L * tau_w) / D;
    const deltaP_kPa = deltaP_Pa / 1000.0;
    const deltaP_bar = deltaP_Pa / 100000.0;

    const numerator = density * Math.pow(V, 2 - n) * Math.pow(D, n);
    const denominator = K * Math.pow(8, n - 1) * Math.pow(rabinowitsch, n);
    const Re_MR = numerator / denominator;

    const f_Fanning = 16.0 / Re_MR;
    const powerWatts = deltaP_Pa * Q;

    return {
      V,
      gamma_w,
      tau_w,
      deltaP_Pa,
      deltaP_kPa,
      deltaP_bar,
      Re_MR,
      isLaminar: Re_MR < 2100,
      f_Fanning,
      powerWatts,
      rabinowitsch,
    };
  }, [diameterMm, lengthM, flowRateLps, K, n, density]);

  // Profile data across normalized radius -1.0 to +1.0
  const profileData = useMemo(() => {
    const points = [];
    const R = diameterMm / 2000.0;
    const L = lengthM;
    const deltaP = hydraulics.deltaP_Pa;
    const factor = (n / (n + 1)) * Math.pow(deltaP / (2 * K * L), 1 / n);

    for (let r_norm = -1.0; r_norm <= 1.05; r_norm += 0.05) {
      const r = Math.abs(r_norm) * R;
      const u_powerlaw = Math.max(
        0,
        factor * (Math.pow(R, (n + 1) / n) - Math.pow(r, (n + 1) / n))
      );
      const u_newtonian = Math.max(0, 2 * hydraulics.V * (1 - Math.pow(r_norm, 2)));

      points.push({
        r_norm: Number(r_norm.toFixed(2)),
        u_pl: Number(u_powerlaw.toFixed(4)),
        u_newt: Number(u_newtonian.toFixed(4)),
      });
    }
    return points;
  }, [diameterMm, lengthM, K, n, hydraulics.deltaP_Pa, hydraulics.V]);

  return (
    <div className="surface-card rounded-2xl p-4 sm:p-6 md:p-8 relative overflow-hidden">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between pb-6 border-b border-white/[0.06] gap-4">
        <div>
          <div className="flex items-center space-x-2 text-[11px] font-mono tracking-widest uppercase text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span>Transport Phenomena &bull; Pipe Hydraulics</span>
          </div>
          <h2 className="text-xl md:text-2xl font-semibold tracking-tight text-white mt-1">
            Metzner–Reed Pipeline Pressure Drop Simulator
          </h2>
          <p className="text-xs text-zinc-400 mt-1 max-w-xl font-light">
            Exact laminar solutions for circular conduits using the Rabinowitsch–Mooney
            wall correction and generalized non-Newtonian Reynolds formulation.
          </p>
        </div>

        {/* Regime Indicator */}
        <div className="shrink-0">
          {hydraulics.isLaminar ? (
            <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-mono text-xs">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>Laminar Regime (Re_MR = {hydraulics.Re_MR.toFixed(1)} &lt; 2100)</span>
            </div>
          ) : (
            <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 font-mono text-xs">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Transition / Turbulent (Re_MR = {hydraulics.Re_MR.toFixed(1)})</span>
            </div>
          )}
        </div>
      </div>

      {/* Main Grid: Parameters on Left, Output Canvas on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 pt-6">
        {/* Controls Column (5 Cols) */}
        <div className="lg:col-span-5 min-w-0 w-full space-y-5">
          <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">
            Pipeline Geometry &amp; Flow Conditions
          </div>

          {/* Presets */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <button
              onClick={() => {
                setK(0.15);
                setN(0.75);
              }}
              className="py-1.5 px-2 rounded-lg border border-white/[0.06] bg-zinc-900/60 text-[10px] font-mono text-zinc-300 hover:text-white hover:border-white/[0.14] transition-all text-center"
            >
              CMC 0.5% (n=0.75)
            </button>
            <button
              onClick={() => {
                setK(2.48);
                setN(0.35);
              }}
              className="py-1.5 px-2 rounded-lg border border-emerald-500/30 bg-emerald-950/20 text-[10px] font-mono text-emerald-300 transition-all text-center"
            >
              XG 0.5% (n=0.35)
            </button>
            <button
              onClick={() => {
                setK(8.71);
                setN(0.27);
              }}
              className="py-1.5 px-2 rounded-lg border border-white/[0.06] bg-zinc-900/60 text-[10px] font-mono text-zinc-300 hover:text-white hover:border-white/[0.14] transition-all text-center"
            >
              XG 1.0% (n=0.27)
            </button>
          </div>

          {/* Pipe Diameter */}
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-zinc-400">Diameter (D):</span>
              <span className="text-zinc-100 font-semibold">{diameterMm} mm</span>
            </div>
            <input
              type="range"
              min={15}
              max={150}
              step={5}
              value={diameterMm}
              onChange={(e) => setDiameterMm(Number(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Pipe Length */}
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-zinc-400">Length (L):</span>
              <span className="text-zinc-100 font-semibold">{lengthM} m</span>
            </div>
            <input
              type="range"
              min={5}
              max={100}
              step={5}
              value={lengthM}
              onChange={(e) => setLengthM(Number(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Flow Rate */}
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-zinc-400">Flow Rate (Q):</span>
              <span className="text-emerald-400 font-semibold">{flowRateLps.toFixed(2)} L/s</span>
            </div>
            <input
              type="range"
              min={0.05}
              max={3.0}
              step={0.05}
              value={flowRateLps}
              onChange={(e) => setFlowRateLps(Number(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Consistency K */}
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-zinc-400">Consistency (K):</span>
              <span className="text-zinc-100 font-semibold">{K.toFixed(2)} Pa·sⁿ</span>
            </div>
            <input
              type="range"
              min={0.05}
              max={10.0}
              step={0.05}
              value={K}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Flow Index n */}
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-zinc-400">Flow Behavior Index (n):</span>
              <span className="text-zinc-100 font-semibold">{n.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.2}
              max={1.0}
              step={0.02}
              value={n}
              onChange={(e) => setN(Number(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        {/* Calculations & Velocity Profile (7 Cols) */}
        <div className="lg:col-span-7 min-w-0 w-full space-y-4">
          {/* Telemetry Matrix */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 sm:gap-2.5 font-mono">
            <div className="p-2.5 sm:p-3 rounded-xl bg-zinc-900/60 border border-white/[0.04] min-w-0">
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider block">Pressure Drop</span>
              <span className="text-base sm:text-lg font-semibold text-emerald-400 block truncate">
                {hydraulics.deltaP_kPa.toFixed(2)}{" "}
                <span className="text-xs text-zinc-400 font-normal">kPa</span>
              </span>
              <span className="text-[10px] text-zinc-500 block truncate">
                {(hydraulics.deltaP_bar).toFixed(4)} bar
              </span>
            </div>

            <div className="p-2.5 sm:p-3 rounded-xl bg-zinc-900/60 border border-white/[0.04] min-w-0">
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider block">Mean Velocity</span>
              <span className="text-base sm:text-lg font-semibold text-zinc-100 block truncate">
                {hydraulics.V.toFixed(3)}{" "}
                <span className="text-xs text-zinc-400 font-normal">m/s</span>
              </span>
              <span className="text-[10px] text-zinc-500 block truncate">V = 4Q/(πD²)</span>
            </div>

            <div className="p-2.5 sm:p-3 rounded-xl bg-zinc-900/60 border border-white/[0.04] min-w-0">
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider block">Wall Shear Rate</span>
              <span className="text-base sm:text-lg font-semibold text-zinc-100 block truncate">
                {hydraulics.gamma_w.toFixed(1)}{" "}
                <span className="text-xs text-zinc-400 font-normal">s⁻¹</span>
              </span>
              <span className="text-[10px] text-zinc-500 block truncate">
                {hydraulics.rabinowitsch.toFixed(2)}&times; Rabinowitsch
              </span>
            </div>

            <div className="p-2.5 sm:p-3 rounded-xl bg-zinc-900/60 border border-white/[0.04] min-w-0">
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider block">Wall Shear Stress</span>
              <span className="text-base sm:text-lg font-semibold text-zinc-100 block truncate">
                {hydraulics.tau_w.toFixed(2)}{" "}
                <span className="text-xs text-zinc-400 font-normal">Pa</span>
              </span>
              <span className="text-[10px] text-zinc-500 block truncate">τ_w = K·γ̇_wⁿ</span>
            </div>

            <div className="p-2.5 sm:p-3 rounded-xl bg-zinc-900/60 border border-white/[0.04] min-w-0">
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider block">Reynolds (Re_MR)</span>
              <span className="text-base sm:text-lg font-semibold text-zinc-100 block truncate">
                {hydraulics.Re_MR.toFixed(1)}
              </span>
              <span className="text-[10px] text-zinc-500 block truncate">
                f_Fanning = {hydraulics.f_Fanning.toFixed(4)}
              </span>
            </div>

            <div className="p-2.5 sm:p-3 rounded-xl bg-zinc-900/60 border border-white/[0.04] min-w-0">
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider block">Pumping Power</span>
              <span className="text-base sm:text-lg font-semibold text-zinc-100 block truncate">
                {hydraulics.powerWatts.toFixed(2)}{" "}
                <span className="text-xs text-zinc-400 font-normal">W</span>
              </span>
              <span className="text-[10px] text-zinc-500 block truncate">P = ΔP &times; Q</span>
            </div>
          </div>

          {/* Velocity Profile Graph */}
          <div className="p-3 sm:p-4 rounded-xl bg-zinc-950 border border-white/[0.06] space-y-2 overflow-hidden">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs font-mono gap-1">
              <span className="text-zinc-300">
                Cross-Sectional Velocity Streamlines u(r)
              </span>
              <span className="text-zinc-500 text-[10px]">r/R = 0 (Pipe Axis)</span>
            </div>
            <div className="h-44 sm:h-52 w-full min-w-0">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={profileData} margin={{ top: 5, right: 10, left: -25, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke="#27272a" />
                  <XAxis dataKey="r_norm" stroke="#52525b" fontSize={10} />
                  <YAxis stroke="#52525b" fontSize={10} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0d0d11",
                      border: "1px solid rgba(255,255,255,0.1)",
                      fontSize: "10px",
                      fontFamily: "monospace",
                    }}
                    formatter={(val: number) => [`${val} m/s`]}
                  />
                  <Area
                    type="monotone"
                    dataKey="u_pl"
                    name={`Pseudoplastic (n=${n})`}
                    stroke="#10b981"
                    fill="#10b981"
                    fillOpacity={0.15}
                    strokeWidth={1.5}
                  />
                  <Area
                    type="monotone"
                    dataKey="u_newt"
                    name="Newtonian (n=1)"
                    stroke="#71717a"
                    fill="none"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-zinc-400 font-light leading-relaxed">
              For pseudoplastic flow ($n &lt; 1$), velocity is blunted across the core with steep wall gradients.
              This reduces viscous shear dissipation along the pipe center, yielding significantly lower pumping energy.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
