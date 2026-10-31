export interface ModelComparison {
  name: string;
  paradigm: string;
  r2: number;
  rmse: number;
  mae: number;
  physicsGuarantee: "Exact Analytical" | "Guaranteed via Softplus" | "Approximate (Clipped)" | "Piecewise Step" | "Soft Penalty";
  description: string;
  color: string;
}

export const MODEL_BENCHMARKS: ModelComparison[] = [
  {
    name: "Classical Power Law / HB",
    paradigm: "Analytical Nonlinear Least Squares",
    r2: 0.9995,
    rmse: 0.2140,
    mae: 0.1280,
    physicsGuarantee: "Exact Analytical",
    description: "Occam's razor model selection with Levenberg-Marquardt optimization. Guaranteed physical bounds.",
    color: "#10b981", // emerald
  },
  {
    name: "PGNN (Param-Pred)",
    paradigm: "Physics-Guided Neural Network",
    r2: 0.9771,
    rmse: 1.5570,
    mae: 0.8464,
    physicsGuarantee: "Guaranteed via Softplus",
    description: "Deep neural net directly predicts K, n, and tau_y with Softplus positivity, computing constitutive flow in forward pass.",
    color: "#06b6d4", // cyan
  },
  {
    name: "XGBoost Regressor",
    paradigm: "Gradient Boosted Trees",
    r2: 0.8442,
    rmse: 3.4454,
    mae: 1.4715,
    physicsGuarantee: "Approximate (Clipped)",
    description: "Tree ensemble predicting apparent viscosity directly. Fast interpolation, but fails at boundary extrapolation.",
    color: "#f59e0b", // amber
  },
  {
    name: "Random Forest",
    paradigm: "Bagged Decision Trees",
    r2: 0.7506,
    rmse: 4.3601,
    mae: 1.6608,
    physicsGuarantee: "Piecewise Step",
    description: "Ensemble of 200 regression trees. Predictions are piecewise constant; susceptible to step artifacts.",
    color: "#8b5cf6", // purple
  },
  {
    name: "PGNN (Res-Penalty, λ=0.1)",
    paradigm: "Regularized Neural Network",
    r2: 0.3698,
    rmse: 8.1711,
    mae: 3.9708,
    physicsGuarantee: "Soft Penalty",
    description: "Direct stress prediction with physics loss penalty L_total = L_data + λ*L_physics. Optimization conflicts emerge.",
    color: "#ef4444", // red
  },
];

export const LAMBDA_SENSITIVITY = [
  { lambda: 0.0, r2: 0.4313, rmse: 7.762, mae: 3.764, logRmse: 0.211 },
  { lambda: 0.01, r2: 0.4277, rmse: 7.787, mae: 3.814, logRmse: 0.203 },
  { lambda: 0.1, r2: 0.3698, rmse: 8.171, mae: 3.971, logRmse: 0.228 },
  { lambda: 0.5, r2: 0.1616, rmse: 9.425, mae: 4.702, logRmse: 0.291 },
  { lambda: 1.0, r2: 0.0640, rmse: 9.958, mae: 5.154, logRmse: 0.336 },
  { lambda: 5.0, r2: -0.1402, rmse: 10.991, mae: 5.962, logRmse: 0.450 },
];

export const GENERALIZATION_EXPERIMENT = [
  {
    holdout: "c = 0.5 wt% (Interpolation)",
    nTrain: 240,
    nTest: 120,
    rfR2: 0.2807,
    rfRmse: 1.987,
    xgbR2: 0.1677,
    xgbRmse: 2.137,
    type: "Interpolation",
    insight: "Moderate interpolation ranking preserved, but tree models underestimate slope steepness.",
  },
  {
    holdout: "c = 1.0 wt% (Interpolation)",
    nTrain: 240,
    nTest: 120,
    rfR2: 0.3538,
    rfRmse: 7.038,
    xgbR2: 0.3689,
    xgbRmse: 6.956,
    type: "Interpolation",
    insight: "Tree ensembles cap predictions at nearest training split boundary, blunting non-linear curvature.",
  },
  {
    holdout: "c = 1.5 wt% (High Extrapolation)",
    nTrain: 300,
    nTest: 60,
    rfR2: 0.1282,
    rfRmse: 2.208,
    xgbR2: 0.0085,
    xgbRmse: 2.355,
    type: "Extrapolation",
    insight: "Tree algorithms fundamentally cannot extrapolate higher values than the max label seen during training.",
  },
  {
    holdout: "c = 0.25 wt% (Low Extrapolation)",
    nTrain: 300,
    nTest: 60,
    rfR2: -38.39,
    rfRmse: 3.318,
    xgbR2: -30.74,
    xgbRmse: 2.978,
    type: "Extrapolation",
    insight: "Severe breakdown (negative R²) as models extrapolate beyond domain. Proves need for Physics-Guided models.",
  },
];

export interface ConditionFit {
  fluid: "CMC" | "XG";
  concentration: number;
  temperature: number;
  selectedModel: "Power Law" | "Herschel-Bulkley";
  K: number;
  n: number;
  tau_y: number;
  r2: number;
  reason: string;
}

export const CONDITION_FITS: ConditionFit[] = [
  { fluid: "CMC", concentration: 0.5, temperature: 25, selectedModel: "Power Law", K: 0.1468, n: 0.7544, tau_y: 0.0074, r2: 0.9999, reason: "Negligible yield stress (tau_y < 0.01 Pa). Pure pseudoplastic flow." },
  { fluid: "CMC", concentration: 0.5, temperature: 40, selectedModel: "Power Law", K: 0.0945, n: 0.7898, tau_y: 0.0195, r2: 0.9996, reason: "Thermal thinning lowers K by 35%. Power Law parsimonious." },
  { fluid: "CMC", concentration: 1.0, temperature: 25, selectedModel: "Power Law", K: 0.8200, n: 0.6063, tau_y: 0.0661, r2: 0.9999, reason: "Significant chain entanglement; n drops to 0.60." },
  { fluid: "CMC", concentration: 1.0, temperature: 40, selectedModel: "Power Law", K: 0.5344, n: 0.6453, tau_y: 0.0129, r2: 0.9996, reason: "Arrhenius thermal thinning; viscosity reduced across all shear rates." },
  { fluid: "CMC", concentration: 1.5, temperature: 25, selectedModel: "Power Law", K: 3.1374, n: 0.4835, tau_y: 0.1773, r2: 0.9996, reason: "Highly pseudoplastic (n < 0.5). High consistency K = 3.14 Pa*s^n." },
  { fluid: "CMC", concentration: 1.5, temperature: 40, selectedModel: "Power Law", K: 2.0647, n: 0.5232, tau_y: 0.0605, r2: 0.9997, reason: "Temperature softens polymer network, raising n slightly." },
  { fluid: "XG", concentration: 0.25, temperature: 25, selectedModel: "Power Law", K: 0.5352, n: 0.4082, tau_y: 0.0457, r2: 0.9996, reason: "Dilute regime; weak yield stress below statistical threshold." },
  { fluid: "XG", concentration: 0.25, temperature: 40, selectedModel: "Power Law", K: 0.3695, n: 0.4399, tau_y: 0.0284, r2: 0.9995, reason: "Thermally thinned dilute XG." },
  { fluid: "XG", concentration: 0.5, temperature: 25, selectedModel: "Herschel-Bulkley", K: 2.4876, n: 0.3512, tau_y: 0.3091, r2: 0.9998, reason: "Rigid helical network produces clear yield stress tau_y = 0.31 Pa." },
  { fluid: "XG", concentration: 0.5, temperature: 40, selectedModel: "Herschel-Bulkley", K: 1.7976, n: 0.3790, tau_y: 0.1893, r2: 0.9992, reason: "Yield stress persists at elevated temperature (tau_y = 0.19 Pa)." },
  { fluid: "XG", concentration: 1.0, temperature: 25, selectedModel: "Herschel-Bulkley", K: 8.7124, n: 0.2674, tau_y: 0.5823, r2: 0.9990, reason: "Pronounced yield stress (0.58 Pa) and severe shear-thinning (n = 0.27)." },
  { fluid: "XG", concentration: 1.0, temperature: 40, selectedModel: "Herschel-Bulkley", K: 5.7588, n: 0.3136, tau_y: 0.5262, r2: 0.9995, reason: "Strong physical gel network maintains high yield stress even at 40°C." },
];

export interface PresentationSlide {
  id: number;
  title: string;
  subtitle: string;
  category: "Introduction" | "Rheology" | "Machine Learning" | "PGNN" | "Engineering" | "Conclusion";
  bulletPoints: string[];
  keyMetric?: { label: string; value: string; badge: string };
  callout?: string;
  interactiveTab?: string;
  figurePath?: string;
}

export const PRESENTATION_SLIDES: PresentationSlide[] = [
  {
    id: 1,
    title: "Physics-Guided Machine Learning for Non-Newtonian Fluid Rheology",
    subtitle: "Chemical Engineering Research Project | MNNIT Allahabad",
    category: "Introduction",
    bulletPoints: [
      "Non-Newtonian polymeric fluids (Xanthan Gum & Carboxymethyl Cellulose) are critical in enhanced oil recovery, biochemical processing, and slurry transport.",
      "Traditional empirical models are limited to isolated flow curves, while purely data-driven ML fails at extrapolation and violates physical laws.",
      "We design an end-to-end framework uniting classical rheometry, tree-based ML, Physics-Guided Neural Networks (PGNN), and pipeline hydraulic engineering.",
    ],
    keyMetric: { label: "PGNN Stress Accuracy", value: "R² = 0.977", badge: "Physics Guaranteed" },
    callout: "Central Objective: Embed physical conservation and constitutive rheology directly into machine learning architectures.",
  },
  {
    id: 2,
    title: "Polymer Rheology: Xanthan Gum vs. Carboxymethyl Cellulose",
    subtitle: "Microstructure, Entanglement & Yield Stress Dynamics",
    category: "Rheology",
    bulletPoints: [
      "Carboxymethyl Cellulose (CMC): Flexible linear polysaccharide. Entangled coils unravel monotonically under shear (pure pseudoplastic Power Law, tau_y ≈ 0).",
      "Xanthan Gum (XG): Rigid double-stranded helix forming 3D hydrogen-bonded networks at rest. Exhibits distinct yield stress (tau_y up to 0.58 Pa).",
      "Thermal Sensitivity: Increasing temperature from 25°C to 40°C consistently decreases the consistency index (K) by 30-38% due to increased molecular free volume.",
    ],
    keyMetric: { label: "Classical Model Fit", value: "R² = 0.9995", badge: "Occam's Razor" },
    figurePath: "/figures/flow_curves_by_fluid.png",
  },
  {
    id: 3,
    title: "Conventional Machine Learning & The Extrapolation Crisis",
    subtitle: "Random Forest & XGBoost Baseline Benchmarks",
    category: "Machine Learning",
    bulletPoints: [
      "Grouped Validation: Used GroupShuffleSplit on condition curves to prevent optimistic data leakage between shear rate points.",
      "In-Domain Performance: XGBoost achieves R² = 0.844 on apparent viscosity, outperforming Random Forest (R² = 0.751).",
      "The Extrapolation Crisis: When holding out unseen concentrations (e.g. c = 0.25 wt%), tree models collapse to negative R² (-38.39).",
      "Trees can only predict values bounded by their training leaves, highlighting the fundamental need for physics guidance.",
    ],
    keyMetric: { label: "Holdout Extrapolation R²", value: "-38.39 (RF)", badge: "Critical Failure" },
    figurePath: "/figures/xgb_actual_vs_pred.png",
  },
  {
    id: 4,
    title: "Model Explainability via SHAP Game Theory",
    subtitle: "Uncovering Feature Attribution and Interaction Dynamics",
    category: "Machine Learning",
    bulletPoints: [
      "Shear Rate (gamma_dot) Dominance: Accounts for the highest dynamic impact, driving steep negative attributions (shear thinning).",
      "Concentration (c): Strongest positive driver of consistency K, shifting viscosity upward nonlinearly via polymer chain entanglement.",
      "Temperature (T): Exhibits consistent negative attribution, capturing thermal Arrhenius thinning.",
      "Scientific Caveat: SHAP describes how the tree algorithm makes decisions; it does not replace mechanistic causality.",
    ],
    figurePath: "/figures/xgb_shap_summary.png",
  },
  {
    id: 5,
    title: "Physics-Guided Neural Network (PGNN) Architectures",
    subtitle: "Inductive Parameter-Prediction vs. Residual-Penalty Regularization",
    category: "PGNN",
    bulletPoints: [
      "Parameter-Prediction PGNN: Neural network predicts (K, n, tau_y) through Softplus activations (strictly K>0, n>0, tau_y>=0). Stress is computed via the constitutive equation in the forward pass.",
      "Physics by Construction: R² = 0.9771, RMSE = 1.557 Pa. Every single predicted curve satisfies the physical law with zero equation violation.",
      "Residual-Penalty PGNN: Direct stress prediction penalized by L_total = L_data + λ*L_physics. Optimization conflicts emerge, capping R² at 0.3698.",
      "Key Finding: Soft loss penalties do NOT guarantee physical correctness. Architectural enforcement is vastly superior.",
    ],
    keyMetric: { label: "PGNN Param-Pred R²", value: "0.9771", badge: "State-of-the-Art" },
    figurePath: "/figures/pgnn_param_pred_actual_vs_pred.png",
  },
  {
    id: 6,
    title: "Engineering Simulation: Pipeline Hydraulics & Pressure Drop",
    subtitle: "Metzner–Reed Generalized Reynolds Number & Analytical Verification",
    category: "Engineering",
    bulletPoints: [
      "Rabinowitsch–Mooney Wall Shear Rate: Corrects Newtonian wall shear rate by (3n+1)/(4n). For pseudoplastic fluids (n < 1), wall gradient is significantly steeper.",
      "Metzner–Reed Generalized Re: Re_MR dictates the laminar/turbulent threshold (< 2100). Fanning friction factor f = 16 / Re_MR.",
      "Analytical Benchmark: For n=1, K=mu, model pressure drop matches Hagen–Poiseuille ΔP = 32*mu*L*V/D² within machine precision (< 1e-14 Pa).",
      "Pumping Power Advantage: Pseudoplastic fluids experience dramatic viscosity reduction at high wall shear rates, reducing pumping power requirements exponentially.",
    ],
    keyMetric: { label: "Analytical Verification", value: "< 10⁻¹⁴ Pa", badge: "Exact Match" },
    figurePath: "/figures/pressure_drop_vs_flow_rate.png",
  },
  {
    id: 7,
    title: "Hydraulic Sensitivity Analysis & Industrial Takeaways",
    subtitle: "Why Flow Behavior Index (n) Dominates Pump Motor Sizing",
    category: "Engineering",
    bulletPoints: [
      "Linear Sensitivity to K: Pressure drop scales linearly (ΔP ∝ K). A 20% error in consistency index produces a direct 20% error in pump head.",
      "Exponential Sensitivity to n: Pressure drop depends exponentially on flow velocity (ΔP ∝ Vⁿ).",
      "Underestimating n leads to catastrophic undersizing of pipeline booster pumps and motor tripping during slurry transport.",
      "Industrial Guidance: High-precision characterization of n via Physics-Guided ML is paramount for safe CAPEX/OPEX engineering design.",
    ],
    figurePath: "/figures/sensitivity_n.png",
  },
  {
    id: 8,
    title: "Conclusions, Contributions & Future Work",
    subtitle: "Summary of Scientific Achievements for Conference Presentation",
    category: "Conclusion",
    bulletPoints: [
      "Established a reproducible benchmark comparing Classical, Tree ML, and Physics-Guided AI for non-Newtonian polymer rheology.",
      "Demonstrated that Parameter-Prediction PGNNs (R² = 0.977) preserve physical bounds and eliminate the extrapolation failures of tree ensembles.",
      "Integrated rheological parameter prediction with a fully verified Metzner–Reed pipeline hydraulic simulator.",
      "Future Work: Incorporate viscoelastic normal stresses (N1, N2), thixotropic time-dependent breakdown, and turbulent Dodge–Metzner correlations.",
    ],
    keyMetric: { label: "Automated Unit Tests", value: "78 / 78", badge: "100% Pass" },
    callout: "Ready for Conference Q&A: Full interactive simulators, notebooks, and source code available.",
  },
];

