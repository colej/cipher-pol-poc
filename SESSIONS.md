<!-- Session-by-session log of work performed and findings. -->

## Session 2 — Auxiliary viscosity sweep dataset (2026-05-22 → 2026-05-25)

### What was done
- Written `scripts/gen_visc_sweep.py`: pseudo-spectral 2D incompressible NS solver
  (vorticity-streamfunction formulation, periodic BCs on [0, 2π]², Kolmogorov forcing
  F_x = A·sin(k_f·y) with A=1.0, k_f=4, Ekman drag α=0.01, RK4 for ω and passive tracer,
  2/3 dealiasing after each step).
- Generated 200 trajectories × 5 viscosity levels → `data/visc_sweep_v2/velocity_nu{0..4}.nc`.
  ν values: np.logspace(-4, -2, 5) = [1.00e-4, 3.16e-4, 1.00e-3, 3.16e-3, 1.00e-2].
  Holdout: index 2 (ν = 1.00e-3). Total dataset: ~2.4 GB.
- Written `scripts/check_energy_spectra.py`: spherically-averaged E(k) gate check plot.
  Output: `figures/energy_spectra_visc_sweep.png`.

### Key findings
- Low-ν (turbulent) cases show broad, shallow spectral tail (enstrophy cascade ~k^{-3}),
  energy reaching k~40. High-ν (laminar) cases drop 9+ decades for k > k_f. Holdout
  (ν=1e-3) is intermediate. Ordering is monotonic in ν at high k.
- U_rms ≈ 1.8–3.0 across ν values (dominated by large-scale energy, Ekman-limited).
- NaN-check infrastructure: per-channel error raised immediately on non-finite snapshot.

### Gate result (energy spectra, visual review)
High-ν curves (ν ≥ 3.16e-3): steep spectral decay for k > k_f — clearly laminar.
Low-ν curves (ν ≤ 3.16e-4): broad Kolmogorov-like tail — clearly turbulent.
**GATE PASSED.** Plot: `figures/energy_spectra_visc_sweep.png`.

### Warnings / gotchas discovered
- **Forward Euler unconditionally unstable for spectral advection.** Even at CFL ≈ 0.15,
  Euler for the passive tracer blows up after ~10 time units because stability requires
  dt ≤ 2κ/U² ≈ 6e-5 (given ν=1e-4, U~1.8), far below the working dt=0.002.
  Fixed by upgrading tracer to RK4 with explicit dealias after each step.
- Generation was slow for high-ν levels (some thermal throttling on M3; ν[4] took
  ~12 extra hours). No accuracy impact — the solver ran at correct step count.

### What failed
- First run: tracer NaN at first snapshot due to Forward Euler instability. Fixed.

---

## Session 1 — Data pipeline + ground-truth labels (2026-05-20)

### What was done
- Written `scripts/data_pipeline.py`: samples 50 trajectories per NC file (10 files),
  shuffles with seed=42, splits 400/50/50. Saves `data/splits.json`.
- Written `scripts/ground_truth_labels.py`: computes u, v, ω, ∇·u, Q, Re_local, regime
  for every (trajectory, timestep, pixel). Saves `data/labels.npz` (~3.8 GB compressed).
- Written `scripts/canary_r4.py`: plots vorticity + regime label side-by-side for 5 trajs.
- Created `SCIENCE.md` with label definitions and physical constant rationale.

### Key findings
- Channel 2 of velocity data is a **passive tracer**, NOT viscosity (identical for all trajs).
- Chosen ν = 5×10⁻⁴ gives global Re_rms ≈ 1000 and turbulent fraction = **0.455** at Re_c = 847.
- div_u std ≈ 11 (near-incompressible but not zero due to coarse finite differences — normal).
- Vorticity fields show coherent spatial structures consistent with 2D turbulence.
- Regime labels have connected regions (not noise), spatially correlated with high-|u| zones.

### Gate R4 result
Turbulent fraction = **0.455** (target 0.20–0.50). **GATE PASSED.**
Canary plot: `figures/canary_r4.png`

### Warnings / gotchas discovered
- `ds.dims["sample"]` deprecated → use `ds.sizes["sample"]` (xarray FutureWarning; fixed).
- `np.gradient` uses one-sided differences at grid edges — do not use outermost 2 pixels
  of gradient-based labels for quantitative statistics.
- labels.npz is 3.8 GB; ensure sufficient disk space before Session 3 (activations will be larger).

### What failed
Nothing failed. NaN checks passed across all 500 trajectories and all 7 computed labels.
