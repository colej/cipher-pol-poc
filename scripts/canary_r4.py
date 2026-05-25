#!/usr/bin/env python
"""R4 Canary: vorticity + binary regime label side-by-side for 5 trajectories.

Loads data/labels.npz, plots ω and regime at t=10 (mid-trajectory) for 5 randomly
chosen trajectories from the training set. Saves figures/canary_r4.png.

Gate criteria (visually checked):
  1. Vorticity field shows spatially coherent structure (not noise).
  2. Regime label has connected regions (not salt-and-pepper).
  3. Turbulent fraction reported ≈ 0.20–0.50.

Usage:
    uv run python scripts/canary_r4.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
LABELS_FILE = DATA_DIR / "labels.npz"
SPLITS_FILE = DATA_DIR / "splits.json"

N_SHOW = 5      # number of trajectories to plot
T_IDX = 10      # timestep to show (mid-trajectory)
CANARY_TRAJ_INDICES = [7, 42, 123, 256, 380]  # fixed sample from train set


def main() -> None:
    print("Loading labels...")
    lbl = np.load(str(LABELS_FILE))
    omega  = lbl["omega"]   # (500, 21, 128, 128)
    regime = lbl["regime"]  # (500, 21, 128, 128) uint8
    Re_loc = lbl["Re_local"]

    # Turbulent fraction statistics
    frac_turb_global = float(regime.mean())
    frac_turb_spatial = regime.mean(axis=(0, 2, 3))  # per timestep
    print(f"Global turbulent fraction : {frac_turb_global:.3f}")
    print(f"Per-timestep range        : {frac_turb_spatial.min():.3f}–{frac_turb_spatial.max():.3f}")

    # Per-trajectory turbulent fraction for selected samples
    fig, axes = plt.subplots(N_SHOW, 3, figsize=(13, 3 * N_SHOW), constrained_layout=True)
    fig.suptitle(
        f"R4 Canary — vorticity & regime label  (t={T_IDX}/20, ν=5×10⁻⁴, Re_c=847)\n"
        f"Global turbulent fraction: {frac_turb_global:.3f}  [target 0.20–0.50]",
        fontsize=12, fontweight="bold"
    )

    for row, traj_idx in enumerate(CANARY_TRAJ_INDICES):
        om  = omega [traj_idx, T_IDX]   # (128, 128)
        reg = regime[traj_idx, T_IDX]   # (128, 128)
        re  = Re_loc[traj_idx, T_IDX]   # (128, 128)
        traj_frac = float(reg.mean())

        # Vorticity
        vmax = float(np.abs(om).max()) * 0.8
        im0 = axes[row, 0].imshow(om, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axes[row, 0].set_title(f"traj {traj_idx} | ω  (std={om.std():.1f})", fontsize=9)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        # Binary regime label
        axes[row, 1].imshow(reg, cmap="Blues", vmin=0, vmax=1, origin="lower")
        axes[row, 1].set_title(
            f"regime  (turb frac={traj_frac:.2f})", fontsize=9
        )

        # Re_local field
        im2 = axes[row, 2].imshow(re, cmap="hot_r", vmin=0, vmax=2000, origin="lower")
        axes[row, 2].set_title(f"Re_local  (mean={re.mean():.0f})", fontsize=9)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "canary_r4.png"
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"\nSaved → {out_path}")

    # Gate verdict
    gate_pass = 0.20 <= frac_turb_global <= 0.50
    print(f"\nGATE R4: turbulent fraction = {frac_turb_global:.3f}  →  "
          f"{'PASS' if gate_pass else 'FAIL — threshold needs recalibration'}")


if __name__ == "__main__":
    main()
