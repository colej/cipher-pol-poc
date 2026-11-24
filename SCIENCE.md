# SCIENCE.md — Setup, physical definitions, and label conventions

For our experiments, we the NS-Sines dataset

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


## Equation & terms

The terms, from first principles

On the periodic unit square:

$$\partial_t \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} + \nabla p = B(\mathbf{u}), \qquad \nabla\cdot\mathbf{u} = 0.$$

In NS-Sines the dissipative term is spectral viscosity (Herde et al., 2024, App. B):

$$B(\mathbf{u}) = \varepsilon_N\,\Delta(Q_N * \mathbf{u}), \qquad \hat B(\mathbf{k}) = -\varepsilon_N\,|2\pi\mathbf{k}|^2\,\hat Q_N(\mathbf{k})\,\hat{\mathbf{u}}(\mathbf{k}),$$

with $\varepsilon_N = 0.05/N$, $N = 128$, and $\hat Q_N(\mathbf{k}) = 0$ for $|\mathbf{k}| \le m_N = \sqrt N$, smooth above. The exact form and parameters of $\hat Q_N$, and the wavenumber convention for the cutoff, are unverified and must come from the AZEBAN solver code.

Eliminating pressure

Write $N(\mathbf{u}) = -(\mathbf{u}\cdot\nabla)\mathbf{u}$, so $\partial_t\mathbf{u} = N - \nabla p + B$.

$B$ preserves incompressibility: $\mathbf{k}\cdot\hat B = -\varepsilon_N|2\pi\mathbf{k}|^2\hat Q_N\,(\mathbf{k}\cdot\hat{\mathbf{u}}) = 0$.
Take the divergence of the momentum equation. $\nabla\cdot\partial_t\mathbf{u} = 0$ and $\nabla\cdot B = 0$, so $\Delta p = \nabla\cdot N$.
In Fourier space, $-|\mathbf{k}|^2\hat p = i\mathbf{k}\cdot\hat N$, so $\hat p = -i\,\mathbf{k}\cdot\hat N/|\mathbf{k}|^2$ and $\widehat{\nabla p} = i\mathbf{k}\hat p = \frac{\mathbf{k}\mathbf{k}^\top}{|\mathbf{k}|^2}\hat N = (I - \hat P)\hat N$, where $\hat P(\mathbf{k}) = I - \mathbf{k}\mathbf{k}^\top/|\mathbf{k}|^2$ is the Leray projector.
Zero mode: $\hat N(\mathbf{0}) = -\langle \nabla\cdot(\mathbf{u}\otimes\mathbf{u})\rangle = 0$ on a periodic domain, so set $\hat P(\mathbf{0}) = I$.

Therefore

$$\partial_t\mathbf{u} = \underbrace{PN(\mathbf{u})}_{A:\ \text{projected advection}} + \underbrace{B(\mathbf{u})}_{B:\ \text{spectral dissipation}}.$$

Both are computable from $\mathbf{u}$ with FFTs. Agreement with the generating solver also depends on its truncation, dealiasing and conventions (unverified).

Short-time tendency

For the true solution operator $\mathcal S_t$: $\mathcal S_t(\mathbf{u}) = \mathbf{u} + t\,[A + B] + O(t^2)$. Write the model as $f_\theta = \mathcal S_t + \epsilon_\theta$. Then

$$g_\theta(\mathbf{u},t) := \frac{f_\theta(\mathbf{u},t) - \mathbf{u}}{t} = A + B + O(t) + \frac{\epsilon_\theta(\mathbf{u},t)}{t}.$$

Shorter $t$ shrinks the $O(t)$ term but amplifies model error. NS-Sines has 21 snapshots on $[0,1]$, so the shortest training lead time is 0.05. A rough estimate puts the fastest initial dynamical timescale at the same order, so the $O(t)$ term may not be small. Step 0 measures it.

Conventions
All diagnostics on denormalised velocity and physical lead time. $A$ is quadratic, so normalisation would otherwise be absorbed into the coefficients.
The loader normalises time as t / constants["time"]. Confirm that model time equals physical time before comparing coefficients with (1, 1).
Regressions use the velocity output channels only (the pressure channel is masked).
Split and bootstrap by trajectory, not by snapshot.


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
