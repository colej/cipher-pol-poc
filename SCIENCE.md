# SCIENCE.md — physical definitions and label conventions

Read before writing any probe, SAE, or label code.

## Dataset: NS-Sines

- Source: PDEgym HuggingFace dataset (`camlab-ethz/ns_sines`)
- Local path: `~/data/pdegym/ns_sines/velocity_*.nc` (10 files, ~11 760 trajectories total)
- Channels: `[u, v, passive_tracer]` — **C=2 is a passive scalar, NOT viscosity**
- Grid: 128×128, domain [0,1]², uniform spacing dx = 1/128
- Time: 21 steps, T∈[0,1], Δt = 0.05
- Lead time τ = 0.5 (frozen; see CLAUDE.md)

## Physical constants (all frozen — never change without CLAUDE.md update)

| Symbol | Value | Rationale |
|--------|-------|-----------|
| ν      | 5×10⁻⁴ | Kinematic viscosity; calibrated so Re_c=847 gives ~46% turbulent patches |
| L      | 1.0  | Domain scale |
| dx     | 1/128 | Grid spacing |
| Re_c   | 847  | Regime threshold (CLAUDE.md frozen) |

**ν calibration note**: ν = 5×10⁻⁴ gives global Re = U_rms × L / ν ≈ 0.5 / 5×10⁻⁴ = 1000.
At Re_c = 847, the empirical turbulent fraction across all 500 trajectories is **0.455**
(within the required 0.20–0.50 window).

## Label definitions

All labels are computed per spatial location (x,y), per timestep t, per trajectory.
Shape: `(N=500, T=21, H=128, W=128)`. Stored in `data/labels.npz`.

### Velocity components
```
u(x,y,t) = velocity[:, :, 0, :, :]   (from data, float32)
v(x,y,t) = velocity[:, :, 1, :, :]   (from data, float32)
```

### Vorticity
```
ω = ∂v/∂x − ∂u/∂y
```
Computed with `np.gradient` (central finite differences). Edges: one-sided differences.
**Do not use the outermost 2 pixels of ω for quantitative results.**

### Divergence (control / sanity check)
```
∇·u = ∂u/∂x + ∂v/∂y   (should be ≈ 0 for incompressible NS)
```
Empirical std ≈ 11 — comparable to ω std ≈ 12. This is a finite-difference artefact
on the coarse 128×128 grid, not a physics failure. Use as control label in probes only.

### Q-criterion (2D)
```
Q = ½(‖Ω‖_F² − ‖S‖_F²)
  = ½(ω²/2 − [∂u/∂x² + ∂v/∂y² + ½(∂u/∂y + ∂v/∂x)²])
```
Q > 0 indicates vortex-dominated regions (rotation > strain).

### Local Reynolds number proxy
```
Re_local(x,y,t) = |u(x,y,t)| × L / ν = sqrt(u² + v²) / (5×10⁻⁴)
```
Range in data: ~0 to ~4400. Mean ≈ 844 (≈ Re_c).

### Binary regime label
```
regime(x,y,t) = 1  if  Re_local(x,y,t) > Re_c = 847
              = 0  otherwise
```
Global turbulent fraction: **0.455 ± 0.022** (mean ± std across trajectories).
Per-timestep range: 0.437–0.497.

## Gradient implementation choice

`np.gradient` uses central differences in the interior and one-sided at edges.
The domain is periodic in the actual NS simulation, but we do not enforce periodic BCs
in the finite-difference stencil. Edge pixels should be excluded from any statistics
that depend on accurate gradient values (probes can treat patches near the boundary
with caution, but this is unlikely to matter for spatial-average labels).

Alternative (future): use spectral derivatives via `numpy.fft` on the full field.
Hold off unless probe accuracy degrades near the boundary.

## Splits (frozen)

- Sampling: 50 trajectories per NC file (linspace indices), seed=42
- Split: 400 train / 50 val / 50 test (shuffled with seed=42)
- Canonical label order: train[0:400], val[400:450], test[450:500]
- **Never re-randomise.** See `data/splits.json`.
