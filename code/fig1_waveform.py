"""
Figure 1 — Single Neuron Extracellular Waveform at Multiple Distances
=====================================================================
Simulates one action potential in a L5 pyramidal cell and records the
extracellular potential at all sweep distances (25, 50, 75, 100, 150,
200 µm).  Uses a single simulation with a multi-contact electrode so
the cell and spike are identical across distances.

Expected output: biphasic waveforms that attenuate steeply with
distance, visually motivating the decay analysis of Figure 2.

Usage:
    python fig1_waveform.py

Requires: LFPy, NEURON, numpy, matplotlib, scipy
"""

import os
import LFPy
import numpy as np
import matplotlib.pyplot as plt

# ──────────────── paths ────────────────
figure_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(figure_dir, exist_ok=True)

# Morphology — update this path for your machine
morph_path = r"C:\Users\justi\OneDrive - Rice University\Desktop\ELEC 481\Final Project\LFPy\139653\L5bPCmodelsEH\morphologies\cell1.asc"

# ──────────────── simulation parameters ────────────────
v_init   = -65       # mV
dt       = 2**-4     # ms  (≈ 0.0625 ms → 16 kHz)
tstart   = 0         # ms
tstop    = 100       # ms

# Synapse (strong enough to elicit a single AP)
syn_type   = 'Exp2Syn'
syn_weight = 0.05
tau1       = 0.5     # ms
tau2       = 5.0     # ms
e_syn      = 0       # mV
n_synapses = 20
syn_height_min = 0.3
syn_height_max = 0.9

# Electrode geometry  (Neuropixels 1.0 contact: 12 × 12 µm square)
# Contacts placed at each sweep distance along the x-axis
distances = np.array([25, 50, 75, 100, 150, 200], dtype=float)  # µm
sigma          = 0.3            # S/m
method         = 'linesource'
contact_size   = 12.0           # µm
contact_shape  = 'square'
contact_normal = np.tile([[1., 0., 0.]], (len(distances), 1))
n_avg_points   = 50

# Spike time
spike_time = 20.0   # ms

# Reproducibility
np.random.seed(42)

# ──────────────── build cell ────────────────
cell = LFPy.Cell(
    morphology=morph_path,
    v_init=v_init, passive=False,
    dt=dt, tstart=tstart, tstop=tstop,
)

# Active channels: HH on soma/axon, passive dendrites
for sec in cell.allseclist:
    if 'soma' in sec.name() or 'axon' in sec.name():
        sec.insert('hh')
    else:
        sec.insert('pas')
        sec.g_pas = 1.0 / 30000
        sec.e_pas = -65

# ──────────────── place synapses ────────────────
seg_centers = np.column_stack([
    cell.x.mean(axis=1),
    cell.y.mean(axis=1),
    cell.z.mean(axis=1),
])
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

synapses = []
for _ in range(n_synapses):
    idx = np.random.choice(dend_idxs)
    syn = LFPy.Synapse(cell, idx=idx, syntype=syn_type,
                        weight=syn_weight, record_current=True,
                        **{'tau1': tau1, 'tau2': tau2, 'e': e_syn})
    syn.set_spike_times(np.array([spike_time + np.random.uniform(-1, 1)]))
    synapses.append(syn)

# ──────────────── multi-contact electrode ────────────────
# One contact per distance, all along the x-axis.
# Single simulation → identical cell/spike across all distances.
electrode = LFPy.RecExtElectrode(
    cell,
    x=distances,
    y=np.zeros(len(distances)),
    z=np.zeros(len(distances)),
    sigma=sigma, method=method,
    N=contact_normal,
    r=contact_size / 2.0,
    n=n_avg_points,
    contact_shape=contact_shape,
)

# ──────────────── simulate ────────────────
cell.simulate(rec_imem=True, rec_vmem=True, probes=[electrode])

tvec   = cell.tvec                       # ms
v_soma = cell.vmem[0]                    # mV

# ──────────────── plot ────────────────
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                         gridspec_kw={'height_ratios': [0.8, 1.2]})

# ── Top: soma membrane potential ──
axes[0].plot(tvec, v_soma, color='tab:blue', linewidth=1.5)
axes[0].axhline(-55, color='red', ls='--', lw=1, label='Threshold (−55 mV)')
axes[0].set_ylabel('V$_m$ (mV)')
axes[0].set_title('Soma membrane potential')
axes[0].legend(fontsize=8, loc='upper right')

# ── Bottom: extracellular waveforms at each distance ──
cmap = plt.cm.plasma
colors = cmap(np.linspace(0.1, 0.9, len(distances)))

for i, (d, color) in enumerate(zip(distances, colors)):
    v_ext = electrode.data[i] * 1000.0   # mV → µV
    vpp = np.max(v_ext) - np.min(v_ext)
    axes[1].plot(tvec, v_ext, color=color, linewidth=1.4,
                 label=f'{d:.0f} µm  (V$_{{pp}}$ = {vpp:.1f} µV)')

# Noise floor reference
noise_rms = 5.0
axes[1].axhline( noise_rms, color='gray', ls=':', lw=0.8, alpha=0.6)
axes[1].axhline(-noise_rms, color='gray', ls=':', lw=0.8, alpha=0.6)
axes[1].text(tstop * 0.98, noise_rms * 1.3,
             f'±{noise_rms:.0f} µV noise floor', fontsize=7,
             color='gray', ha='right', va='bottom')

axes[1].set_ylabel('V$_{ext}$ (µV)')
axes[1].set_xlabel('Time (ms)')
axes[1].set_title('Extracellular potential — single AP at multiple distances')
axes[1].legend(fontsize=8, loc='upper right')

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Figure 1 — Single Neuron Extracellular Waveform', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(figure_dir, 'fig1_waveform.png'), dpi=200, bbox_inches='tight')
plt.show()

# Print summary table
print(f"\n{'Distance':>10}  {'V_pp (µV)':>10}")
print(f"{'─'*10}  {'─'*10}")
for i, d in enumerate(distances):
    v_ext = electrode.data[i] * 1000.0
    vpp = np.max(v_ext) - np.min(v_ext)
    print(f"{d:>8.0f} µm  {vpp:>10.2f}")
print(f"\nSaved → {os.path.join(figure_dir, 'fig1_waveform.png')}")
