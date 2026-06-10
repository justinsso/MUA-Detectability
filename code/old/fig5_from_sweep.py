"""
Figure 5 — N* Sensitivity Analysis (from sweep data)
=====================================================
Loads sweep_results.npz and produces two heatmaps:

  (a) N* vs Distance × Jitter σ   (at a reference cell count that just
      crosses threshold — or shows the full detection-rate grid)
  (b) Detection probability heatmap at a fixed N (e.g., N = 25)

Usage:
    python fig5_from_sweep.py
    python fig5_from_sweep.py --fixed-n 50

Requires: numpy, matplotlib
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
SWEEP_FILE  = SCRIPT_DIR / "sweep_results" / "sweep_results.npz"
FIGURE_DIR  = SCRIPT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--fixed-n", type=int, default=25,
                    help="Fixed N for detection-probability heatmap")
args = parser.parse_args()

# ──────────────── load ────────────────
data = np.load(str(SWEEP_FILE))
N_arr   = data['N']
D_arr   = data['D']
jit_arr = data['jitter_std']
snr_arr = data['snr']
det_arr = data['detected']

N_values   = sorted(set(N_arr))
D_values   = sorted(set(D_arr))
jit_values = sorted(set(jit_arr))

noise_rms   = 5.0
threshold_k = 4.0
detection_snr = threshold_k / 2.0

# ──────────────── Heatmap 1: N* vs Distance × Jitter ────────────────
# For each (D, jitter) cell, find the smallest N where mean SNR ≥ detection_snr
nstar_grid = np.full((len(jit_values), len(D_values)), np.nan)

for i, jit in enumerate(jit_values):
    for j, d in enumerate(D_values):
        for n in N_values:
            mask = (N_arr == n) & (D_arr == d) & (jit_arr == jit)
            if mask.sum() == 0:
                continue
            mean_snr = snr_arr[mask].mean()
            if mean_snr >= detection_snr:
                nstar_grid[i, j] = n
                break
        else:
            nstar_grid[i, j] = np.nan   # never crossed

# ──────────────── Heatmap 2: detection rate at fixed N ────────────────
det_rate_grid = np.full((len(jit_values), len(D_values)), np.nan)
fixed_N = args.fixed_n

for i, jit in enumerate(jit_values):
    for j, d in enumerate(D_values):
        mask = (N_arr == fixed_N) & (D_arr == d) & (jit_arr == jit)
        if mask.sum() == 0:
            continue
        det_rate_grid[i, j] = det_arr[mask].mean() * 100   # percent

# ──────────────── plot ────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: N* heatmap
im1 = axes[0].imshow(nstar_grid, cmap='YlOrRd', aspect='auto', origin='lower')
axes[0].set_xticks(range(len(D_values)))
axes[0].set_xticklabels([f'{d:.0f}' for d in D_values])
axes[0].set_yticks(range(len(jit_values)))
axes[0].set_yticklabels([f'{j:.0f}' for j in jit_values])
axes[0].set_xlabel('Distance from electrode (µm)')
axes[0].set_ylabel('Spike-time jitter σ (ms)')
axes[0].set_title(f'N* (smallest N detected)\nk = {threshold_k}, noise = {noise_rms:.0f} µV')

for i in range(len(jit_values)):
    for j in range(len(D_values)):
        val = nstar_grid[i, j]
        if np.isnan(val):
            txt = '>100'
        else:
            txt = f'{int(val)}'
        max_val = np.nanmax(nstar_grid)
        axes[0].text(j, i, txt, ha='center', va='center',
                     fontsize=8, fontweight='bold',
                     color='white' if (not np.isnan(val) and val > max_val * 0.6) else 'black')
fig.colorbar(im1, ax=axes[0], label='N*', shrink=0.8)

# Right: detection rate at fixed N
im2 = axes[1].imshow(det_rate_grid, cmap='RdYlGn', aspect='auto', origin='lower',
                      vmin=0, vmax=100)
axes[1].set_xticks(range(len(D_values)))
axes[1].set_xticklabels([f'{d:.0f}' for d in D_values])
axes[1].set_yticks(range(len(jit_values)))
axes[1].set_yticklabels([f'{j:.0f}' for j in jit_values])
axes[1].set_xlabel('Distance from electrode (µm)')
axes[1].set_ylabel('Spike-time jitter σ (ms)')
axes[1].set_title(f'Detection rate (%) at N = {fixed_N}\nk = {threshold_k}, noise = {noise_rms:.0f} µV')

for i in range(len(jit_values)):
    for j in range(len(D_values)):
        val = det_rate_grid[i, j]
        if np.isnan(val):
            txt = '—'
        else:
            txt = f'{val:.0f}%'
        axes[1].text(j, i, txt, ha='center', va='center',
                     fontsize=8, fontweight='bold',
                     color='white' if (not np.isnan(val) and val < 50) else 'black')
fig.colorbar(im2, ax=axes[1], label='Detection %', shrink=0.8)

plt.suptitle('Figure 5 — N* Sensitivity Analysis', fontsize=14, y=1.02)
plt.tight_layout()
out = FIGURE_DIR / 'fig5_sensitivity.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight')
plt.show()
print(f"Saved → {out}")
