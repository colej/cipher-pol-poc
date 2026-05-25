#!/usr/bin/env python
"""Compute physical ground-truth labels for all 500 trajectories.

Labels computed per (trajectory, timestep, spatial location):
  u         - x-velocity (from data)
  v         - y-velocity (from data)
  omega     - vorticity: ∂v/∂x - ∂u/∂y
  div_u     - divergence: ∂u/∂x + ∂v/∂y  (≈0 for incompressible)
  Q         - Q-criterion: ½(‖Ω‖² - ‖S‖²), Q>0 = vortex core
  Re_local  - local Re proxy: |u| * L / ν, ν=5e-4, L=1
  regime    - binary: 1 if Re_local > Re_c=847, else 0

Outputs:
  data/labels.npz  — compressed npz with each label as a key,
                     arrays of shape (N=500, T=21, H=128, W=128)

Constants (frozen — see CLAUDE.md):
  Re_c = 847, ν = 5e-4 (calibrated: gives ~46% turbulent at Re_c)

Usage:
    uv run python scripts/ground_truth_labels.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_FILE = DATA_DIR / "splits.json"
LABELS_FILE = DATA_DIR / "labels.npz"

# Frozen physical constants (CLAUDE.md)
RE_C: float = 847.0
NU: float = 5e-4        # kinematic viscosity — calibrated for ~46% turbulent at Re_c
L: float = 1.0          # domain size [0,1]²
DX: float = L / 128.0   # grid spacing

N_TRAJ = 500
N_TIME = 21
H = W = 128


def compute_labels_field(u: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """Compute physical labels for a single (H, W) velocity field.

    Spatial derivatives use np.gradient (central finite differences).
    Edges use one-sided differences — not periodic. Acceptable for interior
    statistics; do not trust the outermost pixels for physical accuracy.
    """
    # axis 0 = rows (y-direction), axis 1 = columns (x-direction)
    du_dy = np.gradient(u, DX, axis=0)
    du_dx = np.gradient(u, DX, axis=1)
    dv_dy = np.gradient(v, DX, axis=0)
    dv_dx = np.gradient(v, DX, axis=1)

    omega = dv_dx - du_dy                   # vorticity ω = ∂v/∂x - ∂u/∂y

    div_u = du_dx + dv_dy                   # divergence ∇·u (≈ 0 for incompressible)

    # Q-criterion (2D): Q = ½(‖Ω‖_F² - ‖S‖_F²)
    # ‖Ω‖_F² = ω²/2   (anti-symmetric part)
    # ‖S‖_F² = du_dx² + dv_dy² + ½(du_dy + dv_dx)²   (symmetric part)
    S_norm_sq = du_dx**2 + dv_dy**2 + 0.5 * (du_dy + dv_dx) ** 2
    Q = 0.5 * (0.5 * omega**2 - S_norm_sq)

    u_mag = np.sqrt(u**2 + v**2)
    Re_local = (u_mag * L) / NU             # local Re proxy

    regime = (Re_local > RE_C).astype(np.uint8)

    return {
        "u":        u.astype(np.float32),
        "v":        v.astype(np.float32),
        "omega":    omega.astype(np.float32),
        "div_u":    div_u.astype(np.float32),
        "Q":        Q.astype(np.float32),
        "Re_local": Re_local.astype(np.float32),
        "regime":   regime,
    }


def main() -> None:
    if not SPLITS_FILE.exists():
        raise FileNotFoundError(
            f"{SPLITS_FILE} not found — run scripts/data_pipeline.py first"
        )

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    # Canonical order: train first, then val, then test (indices 0–499)
    all_refs = splits["train"] + splits["val"] + splits["test"]
    if len(all_refs) != N_TRAJ:
        raise ValueError(f"Expected {N_TRAJ} trajectory refs, got {len(all_refs)}")

    # Pre-allocate label arrays
    label_names = ["u", "v", "omega", "div_u", "Q", "Re_local"]
    labels: dict[str, np.ndarray] = {
        k: np.empty((N_TRAJ, N_TIME, H, W), dtype=np.float32) for k in label_names
    }
    labels["regime"] = np.empty((N_TRAJ, N_TIME, H, W), dtype=np.uint8)

    for traj_idx, (file_path, local_idx) in enumerate(
        tqdm(all_refs, desc="Computing labels", unit="traj")
    ):
        ds = xr.open_dataset(file_path)
        traj = ds["velocity"][int(local_idx)].values.astype(np.float32)  # (T, C, H, W)
        ds.close()

        nan_count = int(np.isnan(traj).sum())
        if nan_count > 0:
            msg = f"NaN in traj {traj_idx} ({file_path}, {local_idx}): {nan_count} NaNs"
            raise ValueError(msg)

        for t in range(N_TIME):
            lbl = compute_labels_field(traj[t, 0], traj[t, 1])
            for k in labels:
                labels[k][traj_idx, t] = lbl[k]

    # Post-hoc NaN check on computed labels
    for k, arr in labels.items():
        n_nan = int(np.isnan(arr.astype(np.float32)).sum())
        if n_nan > 0:
            raise ValueError(f"NaN in computed label '{k}': {n_nan} NaNs")

    # Gate check: fraction turbulent must be 20–50%
    frac_turb = float(labels["regime"].mean())
    print(f"\nLabel shapes    : {labels['u'].shape}  (N, T, H, W)")
    print(f"Turbulent frac  : {frac_turb:.3f}  (target 0.20–0.50)")

    if not (0.20 <= frac_turb <= 0.50):
        warnings.warn(
            f"GATE WARNING: turbulent fraction {frac_turb:.3f} outside [0.20, 0.50] — "
            "Re_c threshold may need recalibration before proceeding."
        )

    print("\nLabel statistics:")
    for k, arr in labels.items():
        a = arr.astype(np.float32)
        print(f"  {k:10s}  mean={a.mean():+.4f}  std={a.std():.4f}  "
              f"min={a.min():+.4f}  max={a.max():+.4f}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(LABELS_FILE), **labels)

    import os
    size_mb = os.path.getsize(LABELS_FILE) / 1e6
    print(f"\nSaved → {LABELS_FILE}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
