"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  Clock,
  RotateCcw,
  Layers,
  ArrowRight,
} from "lucide-react";
import { PRESENTATION_SLIDES, PresentationSlide } from "@/data/rheologyData";

interface PresentationModeProps {
  isOpen: boolean;
  onClose: () => void;
}

export const PresentationMode: React.FC<PresentationModeProps> = ({ isOpen, onClose }) => {
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [isTimerRunning, setIsTimerRunning] = useState(true);
  const [showSlideList, setShowSlideList] = useState(false);

  const totalSlides = PRESENTATION_SLIDES.length;
  const currentSlide: PresentationSlide = PRESENTATION_SLIDES[currentSlideIndex];

  // Timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isOpen && isTimerRunning) {
      interval = setInterval(() => {
        setSeconds((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isOpen, isTimerRunning]);

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown") {
        e.preventDefault();
        setCurrentSlideIndex((prev) => (prev < totalSlides - 1 ? prev + 1 : prev));
      } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
        e.preventDefault();
        setCurrentSlideIndex((prev) => (prev > 0 ? prev - 1 : prev));
      } else if (e.key === "Escape") {
        onClose();
      }
    },
    [isOpen, totalSlides, onClose]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-[#09090b] flex flex-col text-zinc-100 select-none overflow-hidden">
      {/* Top Presentation Bar */}
      <div className="h-14 px-3 sm:px-6 border-b border-white/[0.06] flex items-center justify-between bg-zinc-950/80 backdrop-blur-md">
        <div className="flex items-center space-x-2 sm:space-x-4">
          <span className="font-mono text-xs font-semibold text-zinc-400">
            {String(currentSlide.id).padStart(2, "0")}{" "}
            <span className="text-zinc-600">/</span> {String(totalSlides).padStart(2, "0")}
          </span>
          <span className="text-[10px] font-mono uppercase tracking-widest text-emerald-400 truncate max-w-[120px] sm:max-w-none">
            {currentSlide.category}
          </span>
        </div>

        {/* Stopwatch & Action Bar */}
        <div className="flex items-center space-x-1.5 sm:space-x-3">
          <div className="flex items-center space-x-1.5 bg-zinc-900 px-2 sm:px-3 py-1 rounded-md border border-white/[0.06] font-mono text-xs text-zinc-300">
            <Clock className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
            <span>{formatTime(seconds)}</span>
            <button
              onClick={() => setSeconds(0)}
              title="Reset Timer"
              className="text-zinc-500 hover:text-zinc-300 ml-0.5 sm:ml-1"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          </div>

          <button
            onClick={() => setShowSlideList(!showSlideList)}
            className="p-1.5 rounded-md text-zinc-400 hover:text-white hover:bg-white/[0.04] transition-colors"
            title="Slide Index"
          >
            <Layers className="w-4 h-4" />
          </button>

          <button
            onClick={toggleFullscreen}
            className="hidden sm:flex p-1.5 rounded-md text-zinc-400 hover:text-white hover:bg-white/[0.04] transition-colors"
            title="Toggle Fullscreen"
          >
            <Maximize2 className="w-4 h-4" />
          </button>

          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-zinc-400 hover:text-white hover:bg-white/[0.04] transition-colors"
            title="Exit Presentation (Esc)"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Slide Stage with Smooth AnimatePresence */}
      <div className="flex-1 p-4 sm:p-8 md:p-14 max-w-6xl w-full mx-auto flex flex-col justify-start md:justify-center overflow-y-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentSlide.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            className="space-y-4 sm:space-y-6 py-2"
          >
            {/* Header Titles */}
            <div className="space-y-1.5">
              <h1 className="text-2xl sm:text-3xl md:text-5xl font-light tracking-tight text-white leading-tight">
                {currentSlide.title}
              </h1>
              <p className="text-sm sm:text-base md:text-lg text-zinc-400 font-light">
                {currentSlide.subtitle}
              </p>
            </div>

            {/* Content Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-center pt-2">
              <div className={`space-y-4 ${currentSlide.figurePath ? "lg:col-span-7" : "lg:col-span-12"}`}>
                <ul className="space-y-2.5 sm:space-y-3.5 text-xs sm:text-sm md:text-base text-zinc-300 font-light leading-relaxed">
                  {currentSlide.bulletPoints.map((point, idx) => (
                    <li key={idx} className="flex items-start space-x-2.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-2 shrink-0" />
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>

                {currentSlide.keyMetric && (
                  <div className="mt-4 sm:mt-6 inline-flex flex-wrap items-center gap-2 p-2.5 sm:p-3 rounded-xl bg-zinc-900 border border-white/[0.08]">
                    <span className="text-[10px] sm:text-[11px] font-mono uppercase text-zinc-400">
                      {currentSlide.keyMetric.label}:
                    </span>
                    <span className="font-mono text-sm sm:text-base font-bold text-emerald-400">
                      {currentSlide.keyMetric.value}
                    </span>
                    <span className="px-2 py-0.5 rounded text-[9px] sm:text-[10px] font-mono bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                      {currentSlide.keyMetric.badge}
                    </span>
                  </div>
                )}
              </div>

              {/* Embedded Figure */}
              {currentSlide.figurePath && (
                <div className="lg:col-span-5 bg-zinc-950 rounded-xl border border-white/[0.08] p-2.5 sm:p-3 flex flex-col items-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={currentSlide.figurePath}
                    alt={currentSlide.title}
                    className="max-h-[220px] sm:max-h-[300px] w-auto object-contain rounded"
                  />
                  <span className="text-[10px] text-zinc-500 font-mono mt-1.5 sm:mt-2 text-center">
                    Research Figure Output &bull; High-Resolution 300 DPI
                  </span>
                </div>
              )}
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Slide Index Drawer */}
      {showSlideList && (
        <div className="h-32 bg-zinc-950 border-t border-white/[0.06] p-3 overflow-x-auto flex space-x-3 items-center">
          {PRESENTATION_SLIDES.map((s, idx) => (
            <button
              key={s.id}
              onClick={() => {
                setCurrentSlideIndex(idx);
                setShowSlideList(false);
              }}
              className={`shrink-0 w-44 h-24 rounded-lg p-2.5 text-left border flex flex-col justify-between transition-all ${
                idx === currentSlideIndex
                  ? "border-emerald-500 bg-emerald-950/20"
                  : "border-white/[0.06] bg-zinc-900/40 hover:border-white/[0.14]"
              }`}
            >
              <div className="flex justify-between items-center text-[10px] font-mono text-zinc-400">
                <span>Slide {String(s.id).padStart(2, "0")}</span>
                <span className="text-emerald-400">{s.category}</span>
              </div>
              <p className="text-[11px] font-sans font-medium text-zinc-200 line-clamp-2">
                {s.title}
              </p>
            </button>
          ))}
        </div>
      )}

      {/* Bottom Minimal Navigation Bar */}
      <div className="h-14 border-t border-white/[0.06] px-3 sm:px-6 flex items-center justify-between bg-zinc-950/80">
        <button
          onClick={() => setCurrentSlideIndex((prev) => Math.max(0, prev - 1))}
          disabled={currentSlideIndex === 0}
          className="inline-flex items-center space-x-1.5 sm:space-x-2 px-3 py-1.5 rounded-md text-xs font-mono text-zinc-300 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
          <span>Previous</span>
        </button>

        <div className="text-[11px] font-mono text-zinc-500 hidden sm:block">
          Navigate with <kbd className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700">←</kbd> and{" "}
          <kbd className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700">→</kbd> keys
        </div>

        <button
          onClick={() => setCurrentSlideIndex((prev) => Math.min(totalSlides - 1, prev + 1))}
          disabled={currentSlideIndex === totalSlides - 1}
          className="inline-flex items-center space-x-1.5 sm:space-x-2 px-3.5 py-1.5 rounded-md text-xs font-mono bg-zinc-100 hover:bg-white text-zinc-950 font-semibold disabled:opacity-30 disabled:cursor-not-allowed transition-all"
        >
          <span>Next</span>
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
