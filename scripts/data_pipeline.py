#!/usr/bin/env python
"""Build trajectory splits for NS-Sines dataset and save to data/splits.json.

Samples 50 trajectories from each of the 10 velocity_*.nc files (500 total),
shuffles with seed=42, then splits 400/50/50 train/val/test.

Usage:
    uv run python scripts/data_pipeline.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

np.random.seed(42)

NS_SINES_DIR = Path.home() / "data" / "pdegym" / "ns_sines"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_FILE = DATA_DIR / "splits.json"

N_TOTAL = 500
N_TRAIN, N_VAL, N_TEST = 400, 50, 50
PER_FILE = 50   # trajectories sampled per NC file
SEED = 42


def build_splits() -> dict:
    """Sample PER_FILE trajectories per NC file, shuffle with SEED, split 400/50/50."""
    nc_files = sorted(NS_SINES_DIR.glob("velocity_*.nc"))
    if len(nc_files) != 10:
        raise RuntimeError(f"Expected 10 NC files in {NS_SINES_DIR}, got {len(nc_files)}")

    traj_refs = []  # list of [file_path_str, local_idx]
    for nc_f in nc_files:
        ds = xr.open_dataset(nc_f)
        n_samples = ds.sizes["sample"]
        ds.close()
        # Uniformly spaced indices for diversity across each file
        local_indices = np.linspace(0, n_samples - 1, PER_FILE, dtype=int).tolist()
        for idx in local_indices:
            traj_refs.append([str(nc_f), int(idx)])

    assert len(traj_refs) == N_TOTAL, f"Expected {N_TOTAL}, got {len(traj_refs)}"

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_TOTAL).tolist()

    splits = {
        "train": [traj_refs[i] for i in perm[:N_TRAIN]],
        "val":   [traj_refs[i] for i in perm[N_TRAIN:N_TRAIN + N_VAL]],
        "test":  [traj_refs[i] for i in perm[N_TRAIN + N_VAL:]],
        "metadata": {
            "seed": SEED,
            "per_file": PER_FILE,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "nc_files": [str(f) for f in nc_files],
        },
    }
    return splits


def load_splits() -> dict:
    with open(SPLITS_FILE) as f:
        return json.load(f)


def load_trajectory(file_path: str, local_idx: int) -> np.ndarray:
    """Load a single trajectory. Returns float32 array of shape (T=21, C=3, H=128, W=128)."""
    ds = xr.open_dataset(file_path)
    traj = ds["velocity"][int(local_idx)].values.astype(np.float32)
    ds.close()
    return traj


def main() -> None:
    if SPLITS_FILE.exists():
        print(f"Splits already exist at {SPLITS_FILE} — loading (never re-randomise).")
        splits = load_splits()
    else:
        print("Building splits...")
        splits = build_splits()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SPLITS_FILE, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"Saved → {SPLITS_FILE}")

    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {len(splits[split_name])} trajectories")

    # Verify: load first trajectory from each split and check shape / NaNs
    print("\nVerifying trajectory shapes and NaN checks...")
    for split_name in ("train", "val", "test"):
        file_p, idx = splits[split_name][0]
        traj = load_trajectory(file_p, int(idx))
        expected = (21, 3, 128, 128)
        if traj.shape != expected:
            raise ValueError(f"{split_name}[0] shape {traj.shape} != {expected}")
        nan_count = int(np.isnan(traj).sum())
        if nan_count > 0:
            raise ValueError(f"NaN in {split_name}[0] ({file_p}, {idx}): {nan_count} NaNs")
        print(
            f"  {split_name}[0]: shape={traj.shape}  "
            f"u∈[{traj[:,0].min():.3f}, {traj[:,0].max():.3f}]  "
            f"NaNs=0"
        )

    print("\nData pipeline complete.")
    print("Trajectory layout: (T=21, C=3, H=128, W=128)")
    print("  C=0: u (x-velocity)  C=1: v (y-velocity)  C=2: passive tracer")


if __name__ == "__main__":
    main()
