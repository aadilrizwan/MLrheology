"use client";

import React, { useState, useEffect } from "react";
import { Presentation, Menu, X, ChevronRight, FlaskConical, Gauge, Activity, Sparkles, Image as ImageIcon } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface HeaderProps {
  onOpenPresentation: () => void;
  activeSection: string;
}

export const Header: React.FC<HeaderProps> = ({ onOpenPresentation, activeSection }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Global shortcut 'p' or 'P' to open presentation
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (
        (e.key === "p" || e.key === "P") &&
        !["INPUT", "TEXTAREA"].includes((e.target as HTMLElement)?.tagName)
      ) {
        e.preventDefault();
        onOpenPresentation();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onOpenPresentation]);

  const navLinks = [
    { href: "#overview", label: "Overview" },
    { href: "#simulator", label: "Rheometer", icon: FlaskConical },
    { href: "#benchmarks", label: "Benchmarks", icon: Activity },
    { href: "#hydraulics", label: "Hydraulics", icon: Gauge },
    { href: "#shap", label: "Explainability", icon: Sparkles },
    { href: "#gallery", label: "Figures", icon: ImageIcon },
  ];

  return (
    <header className="sticky top-0 z-40 w-full bg-[#09090b]/90 backdrop-blur-md border-b border-white/[0.06] transition-all">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        {/* Left: Brand & Academic Status */}
        <div className="flex items-center space-x-3 sm:space-x-4">
          <a href="#overview" className="flex items-center space-x-2 group">
            <span className="font-mono text-xs sm:text-sm font-semibold tracking-wider text-zinc-100 uppercase">
              RHEO<span className="text-zinc-500">.</span>ML
            </span>
            <div className="hidden xs:flex items-center space-x-1.5 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-[10px] font-mono font-medium text-emerald-400 tracking-tight">
                MNNIT
              </span>
            </div>
          </a>
        </div>

        {/* Center: Desktop Nav Links */}
        <nav className="hidden md:flex items-center space-x-1 text-xs font-mono text-zinc-400">
          {navLinks.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="px-3 py-1.5 rounded-md hover:text-white hover:bg-white/[0.04] transition-colors"
            >
              {link.label}
            </a>
          ))}
        </nav>

        {/* Right: Keynote Presentation CTA & Mobile Menu Toggle */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          <button
            onClick={onOpenPresentation}
            className="group inline-flex items-center space-x-1.5 sm:space-x-2 px-2.5 sm:px-3 py-1.5 rounded-lg text-xs font-medium bg-zinc-100 hover:bg-white text-zinc-950 transition-all active:scale-[0.98] shadow-sm cursor-pointer"
          >
            <Presentation className="w-3.5 h-3.5 text-zinc-950 shrink-0" />
            <span className="font-sans font-semibold tracking-tight text-[11px] sm:text-xs">
              <span className="hidden sm:inline">Presentation Mode</span>
              <span className="sm:hidden">Keynote</span>
            </span>
            <kbd className="hidden sm:inline-block px-1 py-0.2 text-[9px] font-mono bg-zinc-200 text-zinc-700 rounded border border-zinc-300">
              P
            </kbd>
          </button>

          {/* Mobile Menu Hamburger */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-white/[0.06] transition-colors focus:outline-none"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="md:hidden border-b border-white/[0.08] bg-[#0c0d11] px-4 py-3 space-y-1 overflow-hidden"
          >
            <div className="grid grid-cols-2 gap-1.5 pt-1 pb-2">
              {navLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center space-x-2 px-3 py-2 rounded-lg text-xs font-mono text-zinc-300 hover:text-white hover:bg-white/[0.06] transition-colors"
                >
                  <ChevronRight className="w-3 h-3 text-emerald-400 shrink-0" />
                  <span>{link.label}</span>
                </a>
              ))}
            </div>
            <div className="pt-2 border-t border-white/[0.04] flex items-center justify-between text-[10px] font-mono text-zinc-500">
              <span>MNNIT Allahabad &bull; Chemical Eng.</span>
              <span className="text-emerald-400">78 Tests Verified</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};
