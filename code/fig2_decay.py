"""
Figure 2 — Peak Extracellular Amplitude vs Distance
====================================================
Sweeps electrode distance from 10–400 µm and measures the peak-to-peak
extracellular amplitude of a single AP.  Fits a power-law decay (1/r^α)
to characterise the transition between monopole (α≈1) and dipole (α≈2).

Expected output: log-log plot showing steep amplitude decay, with fitted
exponent α between 1 and 2.

Usage:
    python fig2_decay.py

Requires: LFPy, NEURON, numpy, matplotlib, scipy
"""

import os
import LFPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
spike_time     = 20.0

sigma          = 0.3
method         = 'linesource'
contact_size   = 12.0
contact_shape  = 'square'
contact_normal = np.array([[1., 0., 0.]])
n_avg_points   = 50

# Distance sweep
distances = np.array([10, 20, 30, 50, 75, 100, 140, 175, 200, 250, 300, 400])  # µm

# Fix the seed so the cell + synapse placement is identical across sweeps
np.random.seed(42)

# ──────────────── helper: build and simulate one cell ────────────────
def simulate_at_distance(dist_um):
    """Return peak-to-peak V_ext (µV) for electrode at `dist_um` from soma."""

    # Re-seed so every call builds the identical cell
    np.random.seed(42)

    cell = LFPy.Cell(
        morphology=morph_path,
        v_init=v_init, passive=False,
        dt=dt, tstart=tstart, tstop=tstop,
    )

    for sec in cell.allseclist:
        if 'soma' in sec.name() or 'axon' in sec.name():
            sec.insert('hh')
        else:
            sec.insert('pas')
            sec.g_pas = 1.0 / 30000
            sec.e_pas = -65

    # Dendritic compartment indices + height filter
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

    synapses = []
    for _ in range(n_synapses):
        idx = np.random.choice(dend_idxs)
        syn = LFPy.Synapse(cell, idx=idx, syntype=syn_type,
                            weight=syn_weight, record_current=True,
                            **{'tau1': tau1, 'tau2': tau2, 'e': e_syn})
        syn.set_spike_times(np.array([spike_time + np.random.uniform(-1, 1)]))
        synapses.append(syn)

    electrode = LFPy.RecExtElectrode(
        cell,
        x=np.array([dist_um]), y=np.array([0.]), z=np.array([0.]),
        sigma=sigma, method=method,
        N=contact_normal, r=contact_size / 2.0,
        n=n_avg_points, contact_shape=contact_shape,
    )

    cell.simulate(rec_imem=True, rec_vmem=True, probes=[electrode])
    v_ext_uV = electrode.data[0] * 1000.0
    vpp = np.max(v_ext_uV) - np.min(v_ext_uV)
    return vpp

# ──────────────── sweep ────────────────
print("Sweeping electrode distance …")
amplitudes = np.zeros(len(distances))
for i, d in enumerate(distances):
    amplitudes[i] = simulate_at_distance(d)
    print(f"  r = {d:4d} µm  →  V_pp = {amplitudes[i]:8.2f} µV")

# ──────────────── fit power law:  V = A / r^α  ────────────────
def power_law(r, A, alpha):
    return A / r**alpha

# Fit on data beyond 30 µm (close-field is dominated by nearby compartments)
mask = distances >= 30
popt, pcov = curve_fit(power_law, distances[mask], amplitudes[mask],
                       p0=[1e5, 1.5], maxfev=10000)
A_fit, alpha_fit = popt
alpha_err = np.sqrt(pcov[1, 1])

r_fit = np.linspace(distances[mask].min(), distances[mask].max(), 200)
v_fit = power_law(r_fit, A_fit, alpha_fit)

print(f"\nFitted decay exponent:  α = {alpha_fit:.2f} ± {alpha_err:.2f}")

# ──────────────── plot ────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.loglog(distances, amplitudes, 'ko', markersize=7, label='Data')
ax.loglog(r_fit, v_fit, 'r-', linewidth=2,
          label=f'Fit: $V_{{pp}} \\propto 1/r^{{{alpha_fit:.2f}}}$')

# Reference slopes
r_ref = np.array([50, 400], dtype=float)
v_ref_mono = amplitudes[distances == 50][0] if 50 in distances else power_law(50, A_fit, alpha_fit)
ax.loglog(r_ref, v_ref_mono * (50 / r_ref)**1,
          'b--', alpha=0.4, label='$1/r$ (monopole)')
ax.loglog(r_ref, v_ref_mono * (50 / r_ref)**2,
          'g--', alpha=0.4, label='$1/r^2$ (dipole)')

# Annotate noise floor (MUA-band reference; raw Vpp will be larger
# than MUA Vpp, so this line is a conservative lower bound on detectability)
noise_floor = 5.0   # µV RMS in MUA band
ax.axhline(noise_floor, color='gray', ls=':', lw=1.2)
ax.text(distances[-1] * 1.05, noise_floor * 1.15,
        f'~Noise floor ({noise_floor:.0f} µV RMS, MUA band)',
        fontsize=8, color='gray', va='bottom')

# SUA / MUA radius annotations
for boundary, label, color in [(140, 'SUA limit', 'tab:blue'),
                                (300, 'MUA limit', 'tab:orange')]:
    ax.axvline(boundary, color=color, ls='--', lw=1, alpha=0.6)
    ax.text(boundary * 1.05, ax.get_ylim()[1] * 0.5, label,
            fontsize=8, color=color, rotation=90, va='center')

ax.set_xlabel('Distance from electrode (µm)')
ax.set_ylabel('Peak-to-peak V$_{ext}$ (µV)')
ax.set_title(f'Figure 2 — Signal Decay with Distance  (α = {alpha_fit:.2f})')
ax.legend(fontsize=8, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(figure_dir, 'fig2_decay.png'), dpi=200, bbox_inches='tight')
plt.show()
print(f"Saved → {os.path.join(figure_dir, 'fig2_decay.png')}")
