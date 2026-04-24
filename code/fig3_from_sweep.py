"""
Figure 3 — SNR vs Number of Synchronous Neurons (from sweep data)
=================================================================
Loads the sweep_results.npz produced by sweep.py and plots SNR vs N
at multiple distances, averaged over repeats.

Usage:
    python fig3_from_sweep.py                          # default jitter = 5 ms
    python fig3_from_sweep.py --jitter 10              # pick a jitter value
    python fig3_from_sweep.py --distances 75 100 150   # pick distances

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

# ──────────────── args ────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--jitter", type=float, default=5.0,
                    help="Jitter σ to plot (ms).  Default: 5")
parser.add_argument("--distances", type=float, nargs='+',
                    default=[50, 75, 100, 150, 200],
                    help="Shell distances to include (µm)")
args = parser.parse_args()

# ──────────────── load ────────────────
data = np.load(str(SWEEP_FILE))
N_arr      = data['N']
D_arr      = data['D']
jit_arr    = data['jitter_std']
snr_arr    = data['snr']
det_arr    = data['detected']

N_values   = sorted(set(N_arr))
noise_rms  = 5.0       # must match sweep
threshold_k = 4.0
detection_snr = threshold_k / 2.0

# ──────────────── plot ────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

cmap = plt.cm.viridis
colors = cmap(np.linspace(0.15, 0.85, len(args.distances)))

for d, color in zip(args.distances, colors):
    means, stds, ns = [], [], []
    for n in N_values:
        mask = (N_arr == n) & (D_arr == d) & (jit_arr == args.jitter)
        if mask.sum() == 0:
            continue
        vals = snr_arr[mask]
        means.append(vals.mean())
        stds.append(vals.std())
        ns.append(n)

    if not ns:
        continue
    means, stds, ns = np.array(means), np.array(stds), np.array(ns)
    ax.errorbar(ns, means, yerr=stds, fmt='o-', color=color,
                capsize=3, markersize=5, linewidth=1.4,
                label=f'D = {d:.0f} µm')

# √N reference from first distance
if len(args.distances) > 0:
    n_ref = np.linspace(1, max(N_values), 200)
    # Scale to pass through N=1 of the closest distance
    mask0 = (N_arr == N_values[0]) & (D_arr == args.distances[0]) & (jit_arr == args.jitter)
    if mask0.sum() > 0:
        s0 = snr_arr[mask0].mean()
        ax.plot(n_ref, s0 * np.sqrt(n_ref),
                'k--', alpha=0.35, lw=1.2, label=r'$\propto \sqrt{N}$')

# Detection threshold
ax.axhline(detection_snr, color='red', ls='--', lw=1.5,
           label=f'Detection threshold ({threshold_k:.0f}×σ)')

ax.set_xlabel('Number of synchronous neurons (N)')
ax.set_ylabel('SNR  (V$_{pp}$ / 2σ$_{noise}$)')
ax.set_title(f'Figure 3 — SNR vs N  (jitter σ = {args.jitter:.0f} ms, 3 reps)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=8, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = FIGURE_DIR / 'fig3_snr_vs_n.png'
fig.savefig(str(out), dpi=200, bbox_inches='tight')
plt.show()
print(f"Saved → {out}")
