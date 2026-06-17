"""
Figure 5 — N* Sensitivity Analysis (Heatmap)
=============================================
Varies two key parameters — distance (r) and noise floor (σ_noise) —
and computes the detection threshold N* at each combination.

The result is a heatmap showing how robust N* is to parameter choices,
as described on slide 8 of the proposal.

An additional secondary heatmap varies detection-threshold factor (k)
and spike-time jitter (σ_jitter) at a fixed distance of 175 µm.

Usage:
    python fig5_sensitivity.py

Requires: LFPy, NEURON, numpy, matplotlib, scipy
"""

import os
import LFPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ──────────────── paths ────────────────
figure_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(figure_dir, exist_ok=True)

morph_path = r"C:\Users\justi\OneDrive - Rice University\Desktop\ELEC 481\Final Project\LFPy\LFPy-2.3.6\examples\morphologies\L5_Mainen96_LFPy.hoc"

# ──────────────── simulation parameters ────────────────
v_init   = -65
dt       = 2**-4
tstart   = 0
tstop    = 100

syn_type   = 'Exp2Syn'
syn_weight = 0.05
tau1       = 0.5
tau2       = 5.0
e_syn      = 0
n_synapses = 20
syn_height_min = 0.3
syn_height_max = 0.9

sigma          = 0.3
method         = 'linesource'
contact_size   = 12.0
contact_shape  = 'square'
contact_normal = np.array([[1., 0., 0.]])
n_avg_points   = 50

mua_low    = 300.0
mua_high   = 5000.0
filt_order = 4
fs_hz      = 1000.0 / dt

base_spike_time = 20.0

# ──────────────── sweep axes ────────────────
# Heatmap 1: distance vs noise
distances_sweep  = np.array([100, 125, 150, 175, 200, 250, 300])   # µm
noise_rms_sweep  = np.array([3, 5, 7, 10, 15])                     # µV RMS
k_default        = 4.0
jitter_default   = 1.0   # ms

# Heatmap 2: threshold factor vs jitter  (at r = 175 µm, noise = 5 µV)
k_sweep      = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
jitter_sweep = np.array([0.5, 1.0, 2.0, 3.0, 5.0])                # ms
r_fixed      = 175.0
noise_fixed  = 5.0

# N values to test during binary search
N_max = 300
M_trials = 30   # trials per N value (kept moderate for speed)

# ──────────────── helpers (same as fig3) ────────────────
def _butter_filter(data, fs, low=None, high=None, order=4):
    nyq = 0.5 * fs
    if low is None and high is not None:
        b, a = butter(order, high / nyq, btype='low')
    elif high is None and low is not None:
        b, a = butter(order, low / nyq, btype='high')
    else:
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)


def simulate_template(dist_um, seed=42):
    """Simulate a single cell at `dist_um` and return the extracellular waveform (µV)."""
    np.random.seed(seed)
    cell = LFPy.Cell(morphology=morph_path, v_init=v_init, passive=False,
                      dt=dt, tstart=tstart, tstop=tstop)
    for sec in cell.allseclist:
        if 'soma' in sec.name() or 'axon' in sec.name():
            sec.insert('hh')
        else:
            sec.insert('pas')
            sec.g_pas = 1.0 / 30000
            sec.e_pas = -65
    cell.set_pos(x=dist_um, y=0., z=0.)

    seg_centers = np.column_stack([
        cell.x.mean(axis=1), cell.y.mean(axis=1), cell.z.mean(axis=1)])
    soma_center = seg_centers[cell.somaidx].mean(axis=0)
    dists_from_soma = np.linalg.norm(seg_centers - soma_center, axis=1)
    apex = seg_centers[np.argmax(dists_from_soma)]
    main_vec  = apex - soma_center
    main_len  = np.linalg.norm(main_vec)
    main_unit = main_vec / main_len

    dend_idxs = []
    for sec in cell.allseclist:
        if 'soma' not in sec.name() and 'axon' not in sec.name():
            dend_idxs.extend(cell.get_idx(section=sec.name()))
    dend_idxs = np.array(dend_idxs, dtype=int)
    dend_centers  = seg_centers[dend_idxs]
    projections   = np.dot(dend_centers - soma_center, main_unit)
    h_min, h_max  = main_len * syn_height_min, main_len * syn_height_max
    dend_idxs     = dend_idxs[(projections >= h_min) & (projections <= h_max)]

    for _ in range(n_synapses):
        idx = np.random.choice(dend_idxs)
        syn = LFPy.Synapse(cell, idx=idx, syntype=syn_type,
                            weight=syn_weight, record_current=True,
                            **{'tau1': tau1, 'tau2': tau2, 'e': e_syn})
        syn.set_spike_times(np.array([base_spike_time + np.random.uniform(-1, 1)]))

    electrode = LFPy.RecExtElectrode(
        cell, x=np.array([0.]), y=np.array([0.]), z=np.array([0.]),
        sigma=sigma, method=method,
        N=contact_normal, r=contact_size / 2.0,
        n=n_avg_points, contact_shape=contact_shape)
    cell.simulate(rec_imem=True, rec_vmem=True, probes=[electrode])
    return electrode.data[0] * 1000.0


def shift_template(template, shift_ms, dt_ms):
    shift_samples = shift_ms / dt_ms
    idx = np.arange(len(template), dtype=float)
    return np.interp(idx - shift_samples, idx, template, left=0., right=0.)


def find_n_star(template_uV, noise_rms, k_factor, jitter_std,
                n_max=N_max, m_trials=M_trials):
    """
    Binary-search for the smallest N where mean SNR crosses the detection
    threshold  (k_factor × noise_rms) / (2 × noise_rms)  =  k_factor / 2.
    Returns N* (int), or n_max if not reached.
    """
    detection_snr = k_factor / 2.0
    n_samples = len(template_uV)

    def mean_snr(N):
        snrs = np.zeros(m_trials)
        for m in range(m_trials):
            summed = np.zeros(n_samples)
            for _ in range(N):
                summed += shift_template(template_uV,
                                         np.random.normal(0, jitter_std), dt)
            mua = _butter_filter(summed, fs_hz, low=mua_low, high=mua_high,
                                 order=filt_order)
            vpp = np.max(mua) - np.min(mua)
            snrs[m] = vpp / (2.0 * noise_rms)
        return snrs.mean()

    # Coarse scan first
    coarse_N = [1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300]
    coarse_N = [n for n in coarse_N if n <= n_max]
    lo, hi = 1, n_max
    for cn in coarse_N:
        s = mean_snr(cn)
        if s >= detection_snr:
            hi = cn
            break
        else:
            lo = cn
    else:
        return n_max   # never crossed

    # Binary search between lo and hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mean_snr(mid) >= detection_snr:
            hi = mid
        else:
            lo = mid

    return hi


# ──────────────── Heatmap 1: distance × noise ────────────────
print("=== Heatmap 1: Distance × Noise RMS ===")
# Pre-compute templates at each distance
templates = {}
for d in distances_sweep:
    print(f"  Simulating template at r = {d} µm …")
    templates[d] = simulate_template(d)

nstar_grid1 = np.zeros((len(noise_rms_sweep), len(distances_sweep)))
for i, noise in enumerate(noise_rms_sweep):
    for j, dist in enumerate(distances_sweep):
        ns = find_n_star(templates[dist], noise, k_default, jitter_default)
        nstar_grid1[i, j] = ns
        print(f"  noise={noise:5.1f} µV, r={dist:4.0f} µm  →  N* = {ns}")

# ──────────────── Heatmap 2: k × jitter ────────────────
print("\n=== Heatmap 2: Threshold Factor × Jitter ===")
template_175 = templates.get(r_fixed, simulate_template(r_fixed))

nstar_grid2 = np.zeros((len(jitter_sweep), len(k_sweep)))
for i, jit in enumerate(jitter_sweep):
    for j, k in enumerate(k_sweep):
        ns = find_n_star(template_175, noise_fixed, k, jit)
        nstar_grid2[i, j] = ns
        print(f"  k={k:.1f}, jitter={jit:.1f} ms  →  N* = {ns}")

# ──────────────── plot ────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Heatmap 1
im1 = axes[0].imshow(nstar_grid1, cmap='YlOrRd', aspect='auto',
                      origin='lower')
axes[0].set_xticks(range(len(distances_sweep)))
axes[0].set_xticklabels([f'{d:.0f}' for d in distances_sweep])
axes[0].set_yticks(range(len(noise_rms_sweep)))
axes[0].set_yticklabels([f'{n:.0f}' for n in noise_rms_sweep])
axes[0].set_xlabel('Distance from electrode (µm)')
axes[0].set_ylabel('Noise floor (µV RMS)')
axes[0].set_title(f'N* — Distance × Noise\n(k = {k_default}, jitter = {jitter_default} ms)')
# Annotate cells
for i in range(len(noise_rms_sweep)):
    for j in range(len(distances_sweep)):
        val = int(nstar_grid1[i, j])
        txt = f'{val}' if val < N_max else f'>{N_max}'
        axes[0].text(j, i, txt, ha='center', va='center',
                     fontsize=9, fontweight='bold',
                     color='white' if val > nstar_grid1.max() * 0.6 else 'black')
fig.colorbar(im1, ax=axes[0], label='N*', shrink=0.8)

# Heatmap 2
im2 = axes[1].imshow(nstar_grid2, cmap='YlOrRd', aspect='auto',
                      origin='lower')
axes[1].set_xticks(range(len(k_sweep)))
axes[1].set_xticklabels([f'{k:.1f}' for k in k_sweep])
axes[1].set_yticks(range(len(jitter_sweep)))
axes[1].set_yticklabels([f'{j:.1f}' for j in jitter_sweep])
axes[1].set_xlabel('Detection threshold factor (k)')
axes[1].set_ylabel('Spike-time jitter σ (ms)')
axes[1].set_title(f'N* — Threshold × Jitter\n(r = {r_fixed:.0f} µm, noise = {noise_fixed:.0f} µV)')
for i in range(len(jitter_sweep)):
    for j in range(len(k_sweep)):
        val = int(nstar_grid2[i, j])
        txt = f'{val}' if val < N_max else f'>{N_max}'
        axes[1].text(j, i, txt, ha='center', va='center',
                     fontsize=9, fontweight='bold',
                     color='white' if val > nstar_grid2.max() * 0.6 else 'black')
fig.colorbar(im2, ax=axes[1], label='N*', shrink=0.8)

plt.suptitle('Figure 5 — N* Sensitivity Analysis', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(figure_dir, 'fig5_sensitivity.png'), dpi=200, bbox_inches='tight')
plt.show()
print(f"\nSaved → {os.path.join(figure_dir, 'fig5_sensitivity.png')}")
