# CIPHER-POL proof-of-concept

## Goal
Demonstrate that mech interp tools recover physically meaningful structure
from Poseidon-T. One result needed: layer-wise selectivity curves + at least
one Top-k SAE feature with r² > 0.5 against a computable physical diagnostic.
Target output: 4-page LaTeX workshop paper draft.

## Why this matters
This PoC must demonstrate one thing for the ERC application: that mech interp
tools recover physically meaningful structure from a pretrained neural operator.
Every session should be evaluated against this criterion. If a result is
ambiguous, flag it — do not paper over it.

## The one result we need
A layer-wise selectivity curve showing regime information (binary laminar/turbulent) is more decodable in the late decoder than the early encoder, AND at least one Top-k SAE feature with r² > 0.5 against a computable physical diagnostic.

## What "done" means for this project
A 4-page workshop paper draft with real numbers in it. Not a notebook. Not a
README. A LaTeX draft with figures that could be submitted to ICLR AI4Science.

## Known layer names
See LAYERS.txt — full named-module list for Poseidon-T.
Probe layers confirmed present:
- encoder.layers.0 (ScOTEncodeStage)
- encoder.layers.2.blocks.1 (ScOTLayer)
- decoder.layers.3 (ScOTDecodeStage)

## Canary result
R4 (Session 1): turbulent fraction = 0.455, Re_c = 847, ν = 5×10⁻⁴. GATE PASSED 2026-05-20.
Full canary (Session 4): [fill after Session 4]

## Active gotchas
- MPS backend: torch.roll patched in scOT/model.py — forward pass runs natively on MPS. CPU fallback only needed for SAE training (float stability).
- Lead time: fixed at τ = 0.5 throughout
- Trajectory splits: indices saved at data/splits.json — never re-randomise
- Channel 2 is passive tracer (NOT viscosity). ν = 5×10⁻⁴ is a fixed simulation parameter.
- labels.npz is ~3.8 GB. Activations (Session 3) will be larger — plan disk space.
- np.gradient edge pixels unreliable — exclude outermost 2 pixels from gradient-based stats.
- Aux dataset domain: [0, 2π]² (not [0,1]²). Velocity magnitudes differ from NS-Sines (U_rms~1.8–3). Probe uses ν as label, not Re_local, so this is fine.
- Spectral solver gotcha: Forward Euler is unconditionally unstable for spectral advection. Always use RK4 for tracer (and omega). Stability requires dt ≤ 2κ/U² which is far below working dt=0.002.
- Aux dataset generation speed: high-ν cases (ν≥3e-3) show M3 thermal throttling; may take 10–12 hours per ν level at full spinup. Plan accordingly.

## Current state
- [x] Session 0 — environment + model verified
- [x] Session 1 — data pipeline + labels (R4 canary PASSED: turb frac=0.455)
- [x] Session 2 — aux viscosity dataset (energy spectra gate PASSED 2026-05-25)
- [ ] Session 3 — activation collection
- [ ] Session 4 — probe infrastructure + canary
- [ ] Session 5 — full probe sweep
- [ ] Session 6 — SAE training
- [ ] Session 7 — physical recoverability
- [ ] Session 8 — causal ablation
- [ ] Session 9 — write-up

## Frozen decisions (never revisit without explicit instruction)
- Splits: data/splits.json — never re-randomise
- Regime threshold: Re_c = 847
- Lead time: τ = 0.5
- Probe layers: encoder.layers.0, encoder.layers.2.blocks.1, decoder.layers.3
- Viscosity holdout: index 2
- Aux dataset: data/visc_sweep_v2/ (v1 had wrong BCs — do not use)

## Environment
- Model weights: ~/models/poseidon_T.pt
- NS-Sines: ~/data/pdegym/ns_sines/
- Python venv: ~/envs/cipher-pol-poc/
- MPS available. Use float32. CPU fallback for SAE training if MPS gives NaNs.

## Environment
- Hardware: MacBook Pro, Apple M3, 16GB unified memory
- Python: 3.14
.3, venv at [path]
- PyTorch: [version], MPS available: True
- Poseidon weights: [exact path]
- PDEgym NS-Sines: [exact path]
- Aux dataset: data/visc_sweep_v2/ (1000 trajs, 5 ν levels, generated Session 2)

## MPS behaviour
- SAE training: use MPS, but wrap in try/except with CPU fallback
- Probe training: CPU is fine and more stable
- Always set torch.manual_seed(42) and numpy.random.seed(42) at script start
- MPS does not support float64 — use float32 throughout

## Hard constraints
- Never re-randomise splits or change Re_c
- Never silently discard NaNs — raise and log to SESSIONS.md
- Never skip the control-task selectivity step
- If a gate fails, write GATE FAILED to SESSIONS.md and stop
- Never change the regime label threshold.
- Never modify files in data/ns_sines/ or data/visc_sweep_v2/ — treat as read-only.
- Never report a result without its CI or a note explaining why CI was not computed.
- Never skip the selectivity control-task step to save time.
- If a gate criterion is not met, stop and write "GATE FAILED: [reason]" to
  SESSIONS.md. Do not proceed to the next session's work.


## How to use this file
Read this file at the start of every session before writing any code.
Update SESSIONS.md at the end of every session with: what was done, what
was found, what failed, what the gate result was.
Update this file if: a frozen decision changes (with reason), a new gotcha
is discovered, a gate fails.
If anything in this file contradicts an instruction given in the session
prompt, flag the contradiction before proceeding.

## Read before starting any session
- SESSIONS.md — what happened, warnings, dead ends
- SCIENCE.md — if writing probe, SAE, or label code
- OUTPUTS.md — if writing anything that produces a file
- FAILURES.md — if about to try a new approach