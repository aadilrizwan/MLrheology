"use client";

import React, { useState, useEffect } from "react";
import { Header } from "@/components/Header";
import { PresentationMode } from "@/components/PresentationMode";
import { RheologySimulator } from "@/components/RheologySimulator";
import { PipeFlowCalculator } from "@/components/PipeFlowCalculator";
import { BenchmarkDashboard } from "@/components/BenchmarkDashboard";
import { ExplainabilitySection } from "@/components/ExplainabilitySection";
import { FigureGallery } from "@/components/FigureGallery";
import { motion, AnimatePresence } from "framer-motion";
import {
  Presentation,
  FlaskConical,
  Gauge,
  Activity,
  Terminal,
  FileText,
  CheckCircle2,
  Copy,
  Check,
  ChevronRight,
  ArrowRight,
  ExternalLink,
  ShieldCheck,
  Cpu,
  Layers,
  Sparkles,
} from "lucide-react";

export default function Home() {
  const [isPresentationOpen, setIsPresentationOpen] = useState(false);
  const [activeSection, setActiveSection] = useState("overview");
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const copyCommand = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-[#f4f4f5] flex flex-col font-sans relative selection:bg-zinc-200 selection:text-zinc-950">
      {/* Background Subtle Micro-Grid */}
      <div className="fixed inset-0 bg-grid-pattern pointer-events-none opacity-40 z-0" />

      {/* Top Header */}
      <Header
        onOpenPresentation={() => setIsPresentationOpen(true)}
        activeSection={activeSection}
      />

      {/* Hero Section */}
      <section
        id="overview"
        className="relative z-10 pt-16 pb-20 md:pt-24 md:pb-28 border-b border-white/[0.06] overflow-hidden"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="space-y-6"
          >
            {/* Academic Monograph Kicker */}
            <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono tracking-wider text-zinc-400">
              <span className="px-2.5 py-1 rounded bg-zinc-900 border border-white/[0.08] text-zinc-300 font-medium">
                MNNIT ALLAHABAD
              </span>
              <span className="text-zinc-600">//</span>
              <span className="text-zinc-400">
                DEPARTMENT OF CHEMICAL ENGINEERING
              </span>
              <span className="text-zinc-600">//</span>
              <div className="inline-flex items-center space-x-1.5 text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="font-medium">
                  DEFENSE READY &bull; 78 TESTS PASSING
                </span>
              </div>
            </div>

            {/* Editorial Title */}
            <h1 className="text-3xl sm:text-5xl md:text-6xl lg:text-7xl font-light tracking-[-0.035em] text-[#fafafa] leading-[1.1] max-w-5xl">
              Physics-Guided Machine Learning for{" "}
              <span className="font-medium text-white underline decoration-white/20 underline-offset-8">
                Non-Newtonian Fluid Flow
              </span>
            </h1>

            {/* Editorial Abstract */}
            <p className="text-sm sm:text-base md:text-lg text-zinc-400 font-light leading-relaxed max-w-3xl">
              A scientifically grounded framework bridging classical viscometry,
              tree-based machine learning, and physics-guided neural networks.
              Resolving out-of-domain extrapolation failure by embedding
              constitutive conservation laws for Xanthan Gum (XG) and
              Carboxymethyl Cellulose (CMC) biopolymers into pipeline hydraulic
              predictions.
            </p>

            {/* Call to Actions Bar */}
            <div className="flex flex-wrap items-center gap-2.5 sm:gap-3 pt-2">
              <button
                onClick={() => setIsPresentationOpen(true)}
                className="w-full sm:w-auto justify-center group inline-flex items-center space-x-2 px-5 py-2.5 rounded-lg text-xs font-semibold bg-zinc-100 hover:bg-white text-zinc-950 transition-all shadow-sm active:scale-[0.98] cursor-pointer"
              >
                <Presentation className="w-4 h-4 text-zinc-950" />
                <span>Launch Presentation Deck</span>
                <kbd className="hidden sm:inline-block ml-1 px-1.5 py-0.5 text-[10px] font-mono bg-zinc-200 text-zinc-800 rounded border border-zinc-300">
                  P
                </kbd>
              </button>

              <a
                href="#simulator"
                className="flex-1 sm:flex-initial justify-center inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg text-xs font-medium bg-zinc-900 hover:bg-zinc-800 border border-white/[0.08] text-zinc-300 transition-colors"
              >
                <FlaskConical className="w-4 h-4 text-emerald-400" />
                <span>Rheometer Lab</span>
                <ArrowRight className="w-3.5 h-3.5 text-zinc-500 ml-0.5" />
              </a>

              <a
                href="#hydraulics"
                className="flex-1 sm:flex-initial justify-center inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg text-xs font-medium bg-zinc-900 hover:bg-zinc-800 border border-white/[0.08] text-zinc-300 transition-colors"
              >
                <Gauge className="w-4 h-4 text-sky-400" />
                <span>Hydraulics</span>
              </a>

              <a
                href="#benchmarks"
                className="w-full sm:w-auto justify-center inline-flex items-center space-x-2 px-4 py-2.5 rounded-lg text-xs font-medium bg-zinc-900 hover:bg-zinc-800 border border-white/[0.08] text-zinc-300 transition-colors"
              >
                <Activity className="w-4 h-4 text-amber-400" />
                <span>Model Benchmarks</span>
              </a>
            </div>

            {/* Quantitative Telemetry Strip */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5 sm:gap-3 pt-6 sm:pt-8 border-t border-white/[0.06]">
              <div className="p-3 sm:p-4 rounded-xl bg-zinc-900/40 border border-white/[0.06] flex flex-col justify-between">
                <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
                  PGNN Accuracy
                </span>
                <div className="text-xl sm:text-2xl font-mono font-medium text-emerald-400 mt-1 sm:mt-2">
                  R² = 0.977
                </div>
                <span className="text-[10px] sm:text-[11px] text-zinc-400 mt-1 font-light">
                  Inductive Softplus constraint strictly enforces τ &gt; 0, K
                  &gt; 0
                </span>
              </div>

              <div className="p-3 sm:p-4 rounded-xl bg-zinc-900/40 border border-white/[0.06] flex flex-col justify-between">
                <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
                  Extrapolation Crisis
                </span>
                <div className="text-xl sm:text-2xl font-mono font-medium text-rose-400 mt-1 sm:mt-2">
                  R² = -38.39
                </div>
                <span className="text-[10px] sm:text-[11px] text-zinc-400 mt-1 font-light">
                  Tree ensembles collapse when holding out c = 0.25 wt%
                </span>
              </div>

              <div className="p-3 sm:p-4 rounded-xl bg-zinc-900/40 border border-white/[0.06] flex flex-col justify-between">
                <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
                  Experimental Span
                </span>
                <div className="text-xl sm:text-2xl font-mono font-medium text-zinc-100 mt-1 sm:mt-2">
                  4 Decades
                </div>
                <span className="text-[10px] sm:text-[11px] text-zinc-400 mt-1 font-light">
                  Continuous shear rate coverage from 0.1 to 1000 s⁻¹
                </span>
              </div>

              <div className="p-3 sm:p-4 rounded-xl bg-zinc-900/40 border border-white/[0.06] flex flex-col justify-between">
                <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-400">
                  Hydraulic Error
                </span>
                <div className="text-xl sm:text-2xl font-mono font-medium text-sky-400 mt-1 sm:mt-2">
                  &lt; 10⁻¹⁴ Pa
                </div>
                <span className="text-[10px] sm:text-[11px] text-zinc-400 mt-1 font-light">
                  Zero deviation against Hagen–Poiseuille analytical solution
                </span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Interactive 5-Stage Scientific Pipeline Workflow */}
      <section className="relative z-10 py-12 border-b border-white/[0.06] bg-[#0c0d10]/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center space-x-2 text-[10px] font-mono tracking-widest uppercase text-zinc-500 mb-4">
            <span>[ SYSTEM ARCHITECTURE // 5-STAGE PIPELINE ]</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {[
              {
                step: "01",
                title: "Data Filtering",
                desc: "Ingestion of XG & CMC viscometry. Consistency filter |η - τ/γ̇| / η ≤ 5% and bounds validation.",
                badge: "Data Integrity",
                link: "#simulator",
              },
              {
                step: "02",
                title: "Constitutive Fitting",
                desc: "Bounded nonlinear least squares for Power Law & Herschel–Bulkley. Occam's criterion selection.",
                badge: "Analytical",
                link: "#simulator",
              },
              {
                step: "03",
                title: "Tree Ensembles",
                desc: "Random Forest & XGBoost regressing log(η). GroupShuffleSplit prevents flow curve leakage.",
                badge: "Machine Learning",
                link: "#benchmarks",
              },
              {
                step: "04",
                title: "Physics-Guided NN",
                desc: "Softplus parameter prediction network & residual penalty loss L = L_data + λ L_physics.",
                badge: "PyTorch PGNN",
                link: "#benchmarks",
              },
              {
                step: "05",
                title: "Pipe Hydraulics",
                desc: "Metzner–Reed generalized Reynolds number & Rabinowitsch wall shear rate pressure drop.",
                badge: "Engineering",
                link: "#hydraulics",
              },
            ].map((s, i) => (
              <a
                key={i}
                href={s.link}
                className="group p-4 rounded-xl bg-zinc-900/30 border border-white/[0.06] hover:border-white/[0.16] hover:bg-zinc-900/60 transition-all flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-zinc-500 font-semibold">
                      {s.step}
                    </span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">
                      {s.badge}
                    </span>
                  </div>
                  <h4 className="text-sm font-medium text-zinc-200 group-hover:text-white mt-2">
                    {s.title}
                  </h4>
                  <p className="text-[11px] text-zinc-400 mt-1 font-light leading-relaxed">
                    {s.desc}
                  </p>
                </div>
                <div className="pt-3 flex items-center text-[10px] font-mono text-zinc-500 group-hover:text-emerald-400 transition-colors">
                  <span>Inspect module</span>
                  <ChevronRight className="w-3 h-3 ml-0.5" />
                </div>
              </a>
            ))}
          </div>
        </div>
      </section>

      {/* Main Interactive Stations */}
      <main className="relative z-10 flex-1 max-w-7xl w-full mx-auto px-3 sm:px-6 lg:px-8 py-10 sm:py-16 space-y-12 sm:space-y-20 min-w-0">
        {/* Station 01: Rheometer Lab */}
        <motion.section
          id="simulator"
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="space-y-3 w-full min-w-0 overflow-hidden"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase">
              01 // CONSTITUTIVE VISCOMETRY BENCH
            </span>
          </div>
          <RheologySimulator />
        </motion.section>

        {/* Station 02: Model Benchmarks & Extrapolation Crisis */}
        <motion.section
          id="benchmarks"
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="space-y-3 w-full min-w-0 overflow-hidden"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase">
              02 // CROSS-PARADIGM BENCHMARKS &amp; INDUCTIVE BIAS
            </span>
          </div>
          <BenchmarkDashboard />
        </motion.section>

        {/* Station 03: Engineering Pipe Hydraulics */}
        <motion.section
          id="hydraulics"
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="space-y-3 w-full min-w-0 overflow-hidden"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase">
              03 // TRANSPORT PHENOMENA &amp; PIPELINE HYDRAULICS
            </span>
          </div>
          <PipeFlowCalculator />
        </motion.section>

        {/* Station 04: Mechanistic Interpretability & SHAP */}
        <motion.section
          id="shap"
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="space-y-3 w-full min-w-0 overflow-hidden"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase">
              04 // GAME-THEORETIC ATTRIBUTION &amp; CHEMICAL MECHANISMS
            </span>
          </div>
          <ExplainabilitySection />
        </motion.section>

        {/* Station 05: Publication Figure Gallery */}
        <motion.section
          id="gallery"
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="space-y-3 w-full min-w-0 overflow-hidden"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono tracking-widest text-zinc-500 uppercase">
              05 // 300-DPI PUBLICATION FIGURE REPOSITORY
            </span>
          </div>
          <FigureGallery />
        </motion.section>

        {/* Station 06: Deterministic Execution & Notebooks */}
        <motion.section
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="surface-card rounded-2xl p-4 sm:p-6 md:p-8 relative overflow-hidden space-y-6 w-full min-w-0"
        >
          <div className="flex flex-col sm:flex-row sm:items-end justify-between pb-6 border-b border-white/[0.06] gap-4">
            <div>
              <div className="flex items-center space-x-2 text-[11px] font-mono tracking-widest uppercase text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span>Deterministic Research Pipeline</span>
              </div>
              <h2 className="text-xl md:text-2xl font-semibold tracking-tight text-white mt-1">
                CLI Reproduction &amp; Interactive Notebooks
              </h2>
              <p className="text-xs text-zinc-400 mt-1 max-w-xl font-light">
                Every calculation, model weight, and publication plot is
                reproducible with deterministic random seeds and explicit
                command line interfaces.
              </p>
            </div>
            <div className="text-left sm:text-right">
              <span className="text-[10px] font-mono text-zinc-500 block">
                Python Environment
              </span>
              <span className="text-xs font-mono font-medium text-zinc-200">
                Python 3.11 / 3.13 &bull; PyTorch 2.6
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-start">
            {/* Terminal Commands */}
            <div className="lg:col-span-6 min-w-0 w-full space-y-4">
              <span className="text-xs font-mono uppercase text-zinc-400 block tracking-wider">
                Reproduce Results from Terminal
              </span>

              <div className="space-y-3 font-mono text-xs">
                {[
                  {
                    label: "Run All 78 Unit Tests",
                    cmd: "python -m pytest tests/ -v",
                    desc: "Verifies bounds, consistency checks, model fits, and analytical hydraulic benchmarks.",
                  },
                  {
                    label: "Execute Complete Research Pipeline",
                    cmd: "python -m src.main",
                    desc: "Trains RF, XGBoost, and PGNN models; produces 51 figures in results/figures/.",
                  },
                  {
                    label: "Ingest Custom Laboratory Viscometry",
                    cmd: "python -m src.main --data data/raw/your_viscometry.csv",
                    desc: "Accepts CSV/Excel data with column mapping, consistency verification, and reports.",
                  },
                ].map((item, idx) => (
                  <div
                    key={idx}
                    className="p-3.5 rounded-xl bg-zinc-950 border border-white/[0.06] space-y-2 group min-w-0"
                  >
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-zinc-400 font-sans font-medium">
                        {item.label}
                      </span>
                      <button
                        onClick={() => copyCommand(item.cmd, idx)}
                        className="inline-flex items-center space-x-1 text-zinc-500 hover:text-zinc-200 transition-colors"
                        title="Copy command"
                      >
                        {copiedIndex === idx ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-emerald-400" />
                            <span className="text-[10px] text-emerald-400">
                              Copied
                            </span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            <span className="text-[10px]">Copy</span>
                          </>
                        )}
                      </button>
                    </div>
                    <div className="p-2 rounded bg-zinc-900/60 text-emerald-400 font-mono text-[11px] overflow-x-auto select-all border border-white/[0.03] whitespace-nowrap">
                      {item.cmd}
                    </div>
                    <p className="text-[11px] text-zinc-500 font-sans font-light leading-normal">
                      {item.desc}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Notebooks Directory */}
            <div className="lg:col-span-6 min-w-0 w-full space-y-4">
              <span className="text-xs font-mono uppercase text-zinc-400 block tracking-wider">
                Jupyter Research Notebooks (notebooks/)
              </span>

              <div className="space-y-2 font-mono text-xs">
                {[
                  {
                    file: "01_data_exploration.ipynb",
                    title: "Ingestion, Units & Consistency",
                    desc: "Physical validation |η - τ/γ̇| / η ≤ 5%, column mappings, flow curve plots.",
                  },
                  {
                    file: "02_classical_models.ipynb",
                    title: "Classical Constitutive Models",
                    desc: "Bounded nonlinear regression for Power Law and Herschel–Bulkley.",
                  },
                  {
                    file: "03_machine_learning.ipynb",
                    title: "Tree Ensembles & SHAP Explainability",
                    desc: "Random Forest & XGBoost pipelines, GroupShuffleSplit, TreeExplainer attributions.",
                  },
                  {
                    file: "04_physics_guided_learning.ipynb",
                    title: "Physics-Guided Neural Networks",
                    desc: "PyTorch PGNN parameter-prediction (Softplus) and residual penalty λ sensitivity.",
                  },
                  {
                    file: "05_pipe_flow.ipynb",
                    title: "Pipe Hydraulics & Uncertainty",
                    desc: "Rabinowitsch wall shear rate, Metzner–Reed Re_MR, and pressure drop sensitivity.",
                  },
                ].map((nb, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl bg-zinc-950 border border-white/[0.06] hover:border-white/[0.12] transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-200 font-medium font-mono text-[11px]">
                        {nb.file}
                      </span>
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-zinc-900 border border-white/[0.06] text-zinc-400">
                        Executable
                      </span>
                    </div>
                    <div className="text-xs text-zinc-300 font-sans font-medium mt-1">
                      {nb.title}
                    </div>
                    <div className="text-[11px] text-zinc-500 font-sans font-light mt-0.5">
                      {nb.desc}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.section>
      </main>

      {/* Editorial Footer */}
      <footer className="border-t border-white/[0.06] bg-[#09090b] py-12 text-xs text-zinc-500 relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="flex items-center space-x-2 text-zinc-300 font-mono text-xs">
                <span>RHEO.ML</span>
                <span className="text-zinc-600">&bull;</span>
                <span>
                  MOTILAL NEHRU NATIONAL INSTITUTE OF TECHNOLOGY ALLAHABAD
                </span>
              </div>
              <p className="text-[11px] text-zinc-500 font-light">
                Department of Chemical Engineering &bull; Minor Research Project
                &bull; B.Tech 7th Semester
              </p>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={() => setIsPresentationOpen(true)}
                className="px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-white/[0.08] text-zinc-300 text-xs font-mono transition-colors"
              >
                Keynote Presentation (P)
              </button>
              <a
                href="#overview"
                className="px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-white/[0.08] text-zinc-300 text-xs font-mono transition-colors"
              >
                Back to Top ↑
              </a>
            </div>
          </div>

          <div className="pt-4 border-t border-white/[0.04] flex flex-col sm:flex-row sm:items-center justify-between text-[10px] font-mono text-zinc-600 gap-2">
            <span>
              Authored for Academic Defense, Technical Conferences &amp; Open
              Science
            </span>
            <span>
              PyTorch 2.6 &bull; XGBoost 2.1 &bull; Scikit-Learn 1.6 &bull;
              Next.js 14 &bull; MARQ
            </span>
          </div>
        </div>
      </footer>

      {/* Fullscreen Keynote Presentation Modal */}
      <PresentationMode
        isOpen={isPresentationOpen}
        onClose={() => setIsPresentationOpen(false)}
      />
    </div>
  );
}
