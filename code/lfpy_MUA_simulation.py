

import os
import gc
import datetime
import LFPy
import neuron
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import butter, lfilter
from scipy.ndimage import gaussian_filter1d

# Interactive mode: plt.show() no longer blocks. Script continues to the next
# plot immediately; existing figure windows stay open for inspection.
plt.ion()

# Where to dump PNGs so they can be inspected outside Spyder
figure_dir = r"C:\Users\zachr\OneDrive\Skrivbord\claude_code\figures"
os.makedirs(figure_dir, exist_ok=True)

# Create a new subfolder per run (timestamped) so each run's figures stay
# grouped together and don't mix with previous runs. Format: YYYYMMDD_HHMMSS.
run_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir   = os.path.join(figure_dir, f"run_{run_stamp}")
os.makedirs(run_dir, exist_ok=True)


def fig_path(name):
    """Return run_dir/<name>.png — figures from one script run live in one folder."""
    base, _ = os.path.splitext(name)
    return os.path.join(run_dir, f"{base}.png")


import sys
_t0 = datetime.datetime.now()
def log(msg):
    """Print with elapsed-time prefix and flush immediately so progress appears
    in real time in VS Code's Jupyter interactive / terminal."""
    dt_s = (datetime.datetime.now() - _t0).total_seconds()
    print(f"[{dt_s:6.1f}s] {msg}", flush=True)
    sys.stdout.flush()

# Run flags — toggle what happens at the bottom of the script
run_main_plots = False   # signals / bands / 3D figures from the reference simulation
run_sweep      = True    # n_cells × outer_radius heatmap (slow)

# Sweep grid
sweep_n_cells   = [100]                                        # [5, 10, 20, 50, 100]
sweep_distances = [50, 100]                              # µm — shell sampling: ALL cells sit at exactly this distance
sweep_jitters   = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30, 50]       # ms — synchrony axis (low = synchronous)
sweep_repeats   = 3                                         # average over N realisations per grid point to tame variability

sweep_n_cells   = [1, 5, 10, 25, 50, 100]
sweep_distances = [25, 50, 75, 100, 150, 200]
sweep_jitters   = [0.0, 3.0, 5.0, 10.0, 20.0, 30, 50]
sweep_repeats   = 3

# ============ PARAMETERS ============
# NEURON's HOC interpreter treats "\" as an escape character, so use forward slashes
# in all morphology paths — even on Windows.
morph_dir = "C:/Users/zachr/Downloads"

# Ensure Python's CWD is the morphology directory so NEURON's xopen() always has a
# valid working directory to fall back on, regardless of where the script is run from.
os.chdir(morph_dir)

morphologies = [
    f"{morph_dir}/L5_Mainen96_LFPy.hoc",
    # Add more morphology files here:
    # f"{morph_dir}/some_other_cell.hoc",
    # f"{morph_dir}/another_cell.swc",
]

v_init = -65
dt = 2**-4
tstart = 0
tstop = 200   # ms — long enough to accommodate jitter_std up to ~50 ms without truncating spikes
channel = 'hh'

# Drive mode — how cells are made to fire
#   'synapses' = LFPy.Synapse objects on dendritic compartments (biophysically realistic)
#   'iclamp'   = neuron.h.IClamp at the soma (deterministic, cleaner for methodological work)
drive_mode = 'synapses'

# Synapse-mode parameters (ignored when drive_mode='iclamp')
syn_type = 'Exp2Syn'
syn_weight = 0.05
tau1 = 0.5
tau2 = 2.0
e_syn = 0
n_synapses = 20

# Synapse placement (fraction of cell height along principal axis)
syn_height_min = 0.5   # 30% up the cell
syn_height_max = 0.9   # 90% up the cell

# IClamp-mode parameters (ignored when drive_mode='synapses')
iclamp_amp = 10       # nA — well suprathreshold for fast, reproducible AP
iclamp_dur = 0.1       # ms — brief pulse puts most spike energy in the MUA band

# Electrode (Neuropixels 1.0 contact by default: 12 x 12 µm square)
elec_x = np.array([0.])
elec_y = np.array([0.])
elec_z = np.array([0.])
sigma = 0.3
method = 'linesource'

contact_size = 12.0                 # side length (µm). Neuropixels 1.0 = 12
contact_shape = 'square'            # 'square' or 'circle'
contact_normal = np.array([[1., 0., 0.]])  # outward normal of the contact face
n_avg_points = 50                   # integration points over the contact area

# Band-splitting (standard electrophysiology convention)
lfp_cutoff = 300.0     # Hz — lowpass for LFP band
mua_low    = 300.0     # Hz — bandpass low for MUA band
mua_high   = 5000.0    # Hz — bandpass high for MUA band
filt_order = 3        # 3rd-order Butterworth — Kilosort's convention

# Noise floor
noise_rms_uV         = 5.0    # µV RMS in MUA band — ~5 µV for Neuropixels, ~10 µV for conventional microelectrode
mua_threshold_factor = 5    # threshold = N × noise RMS (standard spike-detection convention)
refractory_ms        = 0.5    # ms — ignore further crossings within this window after a detection.
                              #      0.5 ms collapses filter ringing (sub-ms oscillations on a single AP)
                              #      without compressing jitter ≥ 1 ms. Set to 0 to count every crossing.

# Neuron population
n_cells = 10
# Shell mode: if fix_cell_distance=True, every cell sits at exactly cell_distance
# (matches the sweep's shell sampling). Otherwise cells are scattered uniformly
# in the annulus [inner_radius, outer_radius].
fix_cell_distance = True
cell_distance     = 25      # µm — used only when fix_cell_distance=True
inner_radius      = 50      # minimum distance from electrode (µm) — annulus mode
outer_radius      = 150     # maximum distance from electrode (µm) — annulus mode
base_spike_time   = 20.0    # mean spike time (ms)
jitter_std        = 0      # spike time jitter std (ms)

# Cell alignment
align_cells        = True    # True = all cells same orientation, False = random rotation
align_rot_x        = 0       # shared x rotation (degrees) when aligned
align_rot_y        = 0       # shared y rotation (degrees) when aligned
align_rot_z        = 0       # shared z rotation (degrees) when aligned
align_to_electrode = False   # True = rotate each cell's z axis to match its azimuth around
                             #        the electrode, so every cell has the SAME geometric
                             #        relationship with the contact → identical extracellular
                             #        amplitudes (on top of align_cells; overrides rot_z)

# Firing threshold (for display only)
threshold = -55            # approximate HH threshold (mV)

# Random seed for reproducibility
#np.random.seed(42)

# ============ GENERATE RANDOM NEURONS ============
neurons = []
for i in range(n_cells):
    angle = np.random.uniform(0, 2 * np.pi)
    if fix_cell_distance:
        # Shell sampling: every cell at exactly `cell_distance` from the electrode.
        dist = cell_distance
        z_   = 0.0
    else:
        # Annulus sampling: distance uniform in [inner_radius, outer_radius].
        dist = np.random.uniform(inner_radius, outer_radius)
        z_   = np.random.uniform(-20, 20)
    x = dist * np.cos(angle)
    y = dist * np.sin(angle)
    z = z_

    if align_cells:
        rot_x = align_rot_x
        rot_y = align_rot_y
        rot_z = align_rot_z
    else:
        rot_x = np.random.uniform(0, 10)
        rot_y = np.random.uniform(0, 10)
        rot_z = np.random.uniform(0, 360)

    # Optional: rotate each cell to match its azimuth around the electrode, so
    # every cell has the same geometric relationship with the contact. Overrides
    # rot_z set above. Only meaningful when cells are at a fixed distance.
    if align_to_electrode:
        rot_z = np.degrees(angle)

    spike_time = base_spike_time + np.random.normal(0, jitter_std)
    spike_time = max(1.0, spike_time)

    morph = morphologies[np.random.randint(len(morphologies))]

    neurons.append({
        'x': x, 'y': y, 'z': z,
        'rot_x': rot_x, 'rot_y': rot_y, 'rot_z': rot_z,
        'spike_time': spike_time,
        'morph': morph,
    })

# ============ SIMULATE ============
log(f"Starting main simulation: {n_cells} cells, drive_mode={drive_mode}")
all_lfps = []
all_cells = []
all_syns = []
all_contrib_soma_axon = []   # extracellular contribution from soma+axon compartments
all_contrib_dend      = []   # extracellular contribution from dendritic compartments
all_syn_currents      = []   # per-cell sum of synaptic currents (nA)

for i, n in enumerate(neurons):
    cell = LFPy.Cell(
        morphology=n['morph'],
        v_init=v_init, passive=False,
        dt=dt, tstart=tstart, tstop=tstop,
    )

    for sec in cell.allseclist:
        if 'soma' in sec.name() or 'axon' in sec.name():
            sec.insert('hh')
        else:
            sec.insert('pas')
            sec.g_pas = 1./30000
            sec.e_pas = -65

    # Rotate then translate FIRST
    cell.set_rotation(x=np.radians(n['rot_x']),
                      y=np.radians(n['rot_y']),
                      z=np.radians(n['rot_z']))
    cell.set_pos(x=n['x'], y=n['y'], z=n['z'])

    # Full index sets (soma+axon vs dendrites) — used for contribution decomposition
    # (computed regardless of drive mode; needed for the compartment-class plot).
    soma_axon_idxs_full = []
    dend_idxs_full = []
    for sec in cell.allseclist:
        sec_idxs = cell.get_idx(section=sec.name())
        if 'soma' in sec.name() or 'axon' in sec.name():
            soma_axon_idxs_full.extend(sec_idxs)
        else:
            dend_idxs_full.extend(sec_idxs)
    soma_axon_idxs_full = np.array(soma_axon_idxs_full, dtype=int)
    dend_idxs_full = np.array(dend_idxs_full, dtype=int)

    # Spike drive: either dendritic synapses or a somatic IClamp.
    cell_syns  = []   # LFPy.Synapse objects (synapse mode only)
    cell_stims = []   # NEURON IClamp objects (iclamp mode only) — kept alive to avoid GC

    if drive_mode == 'synapses':
        # --- Height-fraction filter for synapse placement ---
        dend_idxs = dend_idxs_full.copy()
        seg_centers = np.column_stack([
            cell.x.mean(axis=1),
            cell.y.mean(axis=1),
            cell.z.mean(axis=1),
        ])
        soma_center     = seg_centers[cell.somaidx].mean(axis=0)
        dists_from_soma = np.linalg.norm(seg_centers - soma_center, axis=1)
        apex            = seg_centers[np.argmax(dists_from_soma)]
        main_vec        = apex - soma_center
        main_len        = np.linalg.norm(main_vec)
        main_unit       = main_vec / main_len
        dend_centers    = seg_centers[dend_idxs]
        projections     = np.dot(dend_centers - soma_center, main_unit)
        h_min = main_len * syn_height_min
        h_max = main_len * syn_height_max
        dend_idxs = dend_idxs[(projections >= h_min) & (projections <= h_max)]

        assert len(dend_idxs) > 0, (
            f"Cell {i}: no dendritic compartments found between "
            f"{syn_height_min*100:.0f}%–{syn_height_max*100:.0f}% of cell height. "
            f"Try widening the range."
        )

        # Deterministic synapse placement — the same dedicated RNG drives every
        # cell's choices, so each cell gets an identical pattern of dendritic
        # indices (relative to its own dend_idxs list). Without this, different
        # cells get different placements → different path attenuation → different
        # AP latencies even when their spike_time inputs are identical.
        syn_rng = np.random.default_rng(seed=0)
        for s in range(n_synapses):
            syn_idx = int(syn_rng.choice(dend_idxs))
            syn = LFPy.Synapse(cell,
                idx=syn_idx,
                syntype=syn_type,
                weight=syn_weight,
                record_current=True,
                **{'tau1': tau1, 'tau2': tau2, 'e': e_syn})
            # All synapses on a cell fire at the cell's spike time (no extra per-synapse
            # jitter). Inter-cell jitter lives in n['spike_time'] itself, which is set
            # by jitter_std — making this the ONLY source of spike-timing variability.
            syn.set_spike_times(np.array([max(0.1, n['spike_time'])]))
            cell_syns.append(syn)

    elif drive_mode == 'iclamp':
        # Attach an IClamp to the soma. The brief, suprathreshold pulse triggers
        # a reproducible AP after ~0.2 ms. We keep the IClamp object alive in
        # cell_stims so Python doesn't GC it before simulate() runs.
        soma_sec = next(s for s in cell.allseclist if 'soma' in s.name())
        iclamp = neuron.h.IClamp(soma_sec(0.5))
        iclamp.delay = n['spike_time']
        iclamp.dur   = iclamp_dur
        iclamp.amp   = iclamp_amp
        cell_stims.append(iclamp)

    else:
        raise ValueError(f"Unknown drive_mode: {drive_mode!r} (use 'synapses' or 'iclamp')")

    # Finite-area electrode contact (Neuropixels-style)
    electrode = LFPy.RecExtElectrode(cell,
        x=elec_x, y=elec_y, z=elec_z,
        sigma=sigma, method=method,
        N=contact_normal,
        r=contact_size / 2.0,   # LFPy uses half side length for squares / radius for circles
        n=n_avg_points,
        contact_shape=contact_shape)

    cell.simulate(rec_imem=True, rec_vmem=True, probes=[electrode])

    # --- Decompose extracellular contribution by compartment class ---
    # V_e(t) = M @ I_mem(t), where M depends only on geometry. Split the row sum
    # by which compartments contributed, so we can see soma/axon (spike) vs
    # dendrite (synaptic-return) components at this single contact.
    M = electrode.get_transformation_matrix()      # shape (1, totnsegs)
    m_row = M[0]
    contrib_sa = m_row[soma_axon_idxs_full] @ cell.imem[soma_axon_idxs_full]
    contrib_dd = m_row[dend_idxs_full]      @ cell.imem[dend_idxs_full]

    # Sum of synaptic currents for this cell (ground-truth synaptic drive).
    # In iclamp mode there are no synapses, so use zeros with the right shape.
    if cell_syns:
        syn_i_total = np.sum([s.i for s in cell_syns], axis=0)
    else:
        syn_i_total = np.zeros_like(cell.tvec)

    all_lfps.append(electrode.data[0].copy())
    all_cells.append(cell)
    all_syns.append(cell_syns)
    all_contrib_soma_axon.append(contrib_sa)
    all_contrib_dend.append(contrib_dd)
    all_syn_currents.append(syn_i_total)
    drive_str = (f"synapses={n_synapses}" if drive_mode == 'synapses'
                 else f"IClamp={iclamp_amp}nA×{iclamp_dur}ms")
    log(f"Cell {i+1}/{n_cells}: pos=({n['x']:.0f},{n['y']:.0f},{n['z']:.0f}), "
        f"rot_z={n['rot_z']:.0f}°, spike={n['spike_time']:.1f} ms, "
        f"{drive_str}")

# Superposition
total_lfp = np.sum(all_lfps, axis=0)                       # mV
total_contrib_sa = np.sum(all_contrib_soma_axon, axis=0)   # mV
total_contrib_dd = np.sum(all_contrib_dend, axis=0)        # mV
total_syn_i = np.sum(all_syn_currents, axis=0)             # nA
tvec = all_cells[0].tvec

# ============ BAND-SPLIT THE RAW EXTRACELLULAR SIGNAL ============
fs_hz = 1000.0 / dt   # dt is in ms, so fs = samples/s

def _butter_filter(data, fs, low=None, high=None, order=4):
    """Butterworth filter applied CAUSALLY (one-way lfilter), matching Kilosort.
    Causal filtering preserves realism vs real-time hardware chains — ringing
    appears only AFTER a spike, not symmetrically around it like zero-phase."""
    nyq = 0.5 * fs
    if low is None and high is not None:
        b, a = butter(order, high / nyq, btype='low')
    elif high is None and low is not None:
        b, a = butter(order, low / nyq, btype='high')
    else:
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data)

raw_uV = total_lfp * 1000.0
lfp_uV = _butter_filter(raw_uV, fs_hz, high=lfp_cutoff, order=filt_order)
mua_uV = _butter_filter(raw_uV, fs_hz, low=mua_low, high=mua_high, order=filt_order)

# ============ NOISE FLOOR ============
# Add Gaussian noise directly to the MUA band signal.
# noise_rms_uV represents the electrode noise floor in the MUA band (post-filter).
noise_mua    = np.random.normal(0, noise_rms_uV, mua_uV.shape)
mua_noisy_uV = mua_uV + noise_mua

# Threshold = N × robust noise estimate (MAD) of the noisy MUA signal.
# MAD = median(|x|) / 0.6745 converges to the true noise std even when the signal
# contains large spikes, whereas plain np.std gets inflated by the spikes themselves.
# This is the Quiroga (2004) estimator used by Kilosort, Wave_clus, Spyking Circus.
mua_signal_std = np.median(np.abs(mua_noisy_uV)) / 0.6745
mua_threshold  = mua_threshold_factor * mua_signal_std  # µV

# Negative-going threshold crossings: extracellular spikes appear as negative deflections
below         = mua_noisy_uV < -mua_threshold
all_crossings = np.where(np.diff(below.astype(int)) > 0)[0]
# Apply refractory period: collapse multiple crossings within `refractory_ms`
# into a single event (standard in spike-detection pipelines).
refractory_samples = int(round(refractory_ms / dt))
crossings_idx = []
last = -refractory_samples   # so the first crossing is always accepted
for idx in all_crossings:
    if idx - last >= refractory_samples:
        crossings_idx.append(idx)
        last = idx
crossings_idx = np.array(crossings_idx, dtype=int)
n_crossings   = len(crossings_idx)
log(f"MUA threshold: {mua_threshold:.1f} µV  ({mua_threshold_factor:.0f} × {mua_signal_std:.1f} µV MAD)"
    f"  |  Threshold crossings: {n_crossings}")

# ============ SPIKING BAND POWER (SBP — Nason et al. 2020) ============
# SBP pipeline (Nason et al. 2020, Nature Biomed Eng):
#   1) 2nd-order Butterworth bandpass 300–1000 Hz on the raw signal
#   2) rectify (|x|)
#   3) (downsample to ~2 kSps — skipped here; for plotting we keep fs_hz)
#   4) smooth with a 50 ms Gaussian window
# Narrower than the standard 300–5000 Hz MUA band because the band <1 kHz
# captures the bulk of single-neuron AP energy while reducing high-frequency
# noise. Widely used as a cheap, continuous proxy for local firing rate in
# intracortical brain–machine interfaces.
sbp_low         = 300.0   # Hz
sbp_high        = 1000.0  # Hz
sbp_filt_order  = 2       # 2nd-order Butterworth, per the paper
# Gaussian σ for smoothing. Nason et al. describe a "50 ms window" in BCI work,
# which is reasonable when recording for many seconds — but at that σ the kernel
# support (~300 ms) swamps a 100 ms simulation. Set to 0 to disable smoothing
# entirely and see the raw rectified bandpass.
sbp_smooth_ms   = 0.0     # ms — Gaussian smoothing kernel σ (0 = no smoothing)

sbp_band_uV     = _butter_filter(
    mua_noisy_uV, fs_hz, low=sbp_low, high=sbp_high, order=sbp_filt_order,
)
sbp_mag_uV      = np.abs(sbp_band_uV)
if sbp_smooth_ms > 0:
    sbp_sigma_smps   = sbp_smooth_ms / dt   # convert σ ms → samples (dt is in ms)
    sbp_smoothed_uV  = gaussian_filter1d(sbp_mag_uV, sbp_sigma_smps)
else:
    sbp_smoothed_uV  = sbp_mag_uV            # no smoothing — pure rectified bandpass

# ============ PARAMETER INFO ============
if drive_mode == 'synapses':
    drive_info = f"Drive: synapses ({n_synapses}/cell, τ1/τ2={tau1}/{tau2}ms)"
else:
    drive_info = f"Drive: IClamp ({iclamp_amp}nA × {iclamp_dur}ms)"

distance_info = (f"Shell: {cell_distance} µm" if fix_cell_distance
                 else f"Annulus: {inner_radius}–{outer_radius} µm")
pop_info = (
    f"Cells: {n_cells}\n"
    f"{distance_info}\n"
    f"Spike: {base_spike_time:.1f}±{jitter_std:.1f} ms\n"
    f"{drive_info}\n"
    f"Aligned: {align_cells}"
    f"{' (→ electrode)' if align_to_electrode else ''}\n"
    f"Morphs: {len(morphologies)}\n"
    f"Contact: {contact_size:.0f} µm {contact_shape}\n"
    f"σ: {sigma}   method: {method}\n"
    f"dt: {dt:.4f} ms ({fs_hz:.0f} Hz)\n"
    f"LFP: <{lfp_cutoff:.0f} Hz\n"
    f"MUA: {mua_low:.0f}–{mua_high:.0f} Hz\n"
    f"Noise: {noise_rms_uV:.0f} µV RMS\n"
    f"Thr: {mua_threshold_factor:.0f}×MAD = {mua_threshold:.0f} µV\n"
    f"Crossings: {n_crossings}"
)

# ============ PLOT SIGNALS ============
if run_main_plots:
    log("Plotting signals figure (1/3)...")
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    for i, n in enumerate(neurons):
        dist = np.sqrt(n['x']**2 + n['y']**2 + n['z']**2)
        axes[0].plot(tvec, all_lfps[i] * 1000,
                     label=f'Cell {i} (d={dist:.0f}µm, t={n["spike_time"]:.1f}ms)')
    axes[0].set_ylabel('µV')
    axes[0].set_title('Individual contributions')
    axes[0].legend(fontsize=7, loc='upper right')

    for i in range(n_cells):
        axes[1].plot(tvec, all_cells[i].vmem[0], label=f'Cell {i}')
    axes[1].set_ylabel('Vm (mV)')
    axes[1].set_title('Soma membrane potentials')
    axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, zorder=10, label=f'Threshold ({threshold} mV)')
    axes[1].legend(fontsize=7, loc='upper right')

    axes[2].plot(tvec, total_lfp * 1000, 'k-', linewidth=2)
    axes[2].set_ylabel('µV')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_title('Summed extracellular potential')

    # Parameter box — anchored to the figure (not the axes) to match the pattern
    # used in the other two plots. Avoids tight_layout + axes-outside-text issues.
    fig.text(0.995, 0.5, pop_info, fontsize=8, ha='right', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Multi-neuron LFPy Simulation', fontsize=13)
    fig.subplots_adjust(right=0.78, hspace=0.45, top=0.94)
    log("  saving signals figure to disk...")
    fig.savefig(fig_path('signals'), dpi=130, bbox_inches='tight')
    log("  signals figure done.")
    plt.show()

# ============ BANDS + SYNAPTIC DECOMPOSITION ============
if run_main_plots:
    log("Plotting bands + decomposition figure (2/3)...")
    fig_b, axb = plt.subplots(7, 1, figsize=(11, 17), sharex=True)

    axb[0].plot(tvec, raw_uV, color='k', linewidth=1.2)
    axb[0].set_title('Raw extracellular (wideband)')
    axb[0].set_ylabel('µV')

    axb[1].plot(tvec, lfp_uV, color='C0', linewidth=1.2)
    axb[1].set_title(f'LFP band  (lowpass < {lfp_cutoff:.0f} Hz, Butter order {filt_order}, zero-phase)')
    axb[1].set_ylabel('µV')

    axb[2].plot(tvec, mua_uV, color='C3', linewidth=1.0)
    axb[2].set_title(f'MUA band — clean signal  (bandpass {mua_low:.0f}–{mua_high:.0f} Hz)')
    axb[2].set_ylabel('µV')

    axb[3].plot(tvec, mua_noisy_uV, color='C3', linewidth=0.7, alpha=0.75, label='Noisy MUA')
    axb[3].axhline(-mua_threshold, color='k', linestyle='--', linewidth=1.2,
                   label=f'Threshold  −{mua_threshold:.0f} µV  ({mua_threshold_factor:.0f}×{mua_signal_std:.1f} µV MAD)')
    if n_crossings > 0:
        axb[3].scatter(tvec[crossings_idx],
                       np.full(n_crossings, -mua_threshold),
                       color='k', s=50, marker='v', zorder=5,
                       label=f'{n_crossings} crossing(s)')
    axb[3].set_title(f'MUA band — {noise_rms_uV:.0f} µV RMS noise + threshold crossings')
    axb[3].set_ylabel('µV')
    axb[3].legend(loc='upper right', fontsize=8)

    # Spiking Band Power (Nason et al. 2020) — continuous firing-rate proxy.
    # Overlay the discrete threshold crossings (from axb[3]) as vertical ticks so
    # the two MUA conventions can be compared at a glance.
    _sbp_smooth_label = (f'σ={sbp_smooth_ms:.0f} ms' if sbp_smooth_ms > 0
                         else 'no smoothing')
    axb[4].plot(tvec, sbp_smoothed_uV, color='tab:olive', linewidth=1.2,
                label=f'SBP  (bp {sbp_low:.0f}–{sbp_high:.0f} Hz, '
                      f'order {sbp_filt_order}, {_sbp_smooth_label})')
    if n_crossings > 0:
        axb[4].scatter(tvec[crossings_idx],
                       np.full(n_crossings, sbp_smoothed_uV.max() * 1.05),
                       color='k', s=40, marker='v', zorder=5,
                       label=f'Threshold crossings ({n_crossings})')
    axb[4].set_title('Spiking Band Power (Nason et al. 2020)  —  '
                     'bandpass → rectify → Gaussian smooth')
    axb[4].set_ylabel('µV')
    axb[4].legend(loc='upper right', fontsize=8)

    axb[5].plot(tvec, total_contrib_dd * 1000.0,
                color='tab:green', linewidth=1.3,
                label='Dendritic compartments  (synaptic-return currents)')
    axb[5].plot(tvec, total_contrib_sa * 1000.0,
                color='tab:purple', linewidth=1.3,
                label='Soma + axon compartments  (spike currents)')
    axb[5].plot(tvec, raw_uV, color='k', linewidth=0.8, alpha=0.4, label='Total (sum)')
    axb[5].set_title('Extracellular contribution decomposed by compartment class')
    axb[5].set_ylabel('µV')
    axb[5].legend(loc='upper right', fontsize=8)

    axb[6].plot(tvec, total_syn_i, color='tab:orange', linewidth=1.3)
    axb[6].set_title('Sum of synaptic currents across all synapses (ground-truth synaptic drive)')
    axb[6].set_ylabel('nA')
    axb[6].set_xlabel('Time (ms)')

    # Parameter box — anchored to the right of the figure
    fig_b.text(0.995, 0.5, pop_info, fontsize=8, ha='right', va='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Band-split extracellular signal + synaptic decomposition', fontsize=13)
    fig_b.subplots_adjust(right=0.82, hspace=0.55, top=0.94)
    log("  saving bands figure to disk...")
    fig_b.savefig(fig_path('bands_decomp'), dpi=130, bbox_inches='tight')
    log("  bands figure done.")
    plt.show()

# ============ 3D PLOT ============
def _contact_face(center, normal, size, shape):
    """Return vertices of a contact face centred at `center` with outward `normal`."""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    # Build two in-plane basis vectors orthogonal to normal
    helper = np.array([0., 0., 1.]) if abs(normal[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(normal, helper); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    half = size / 2.0
    if shape == 'square':
        offsets = [(-half, -half), (half, -half), (half, half), (-half, half)]
    else:  # circle approximated by polygon
        offsets = [(half * np.cos(t), half * np.sin(t))
                   for t in np.linspace(0, 2 * np.pi, 32, endpoint=False)]
    return [center + a * u + b * v for a, b in offsets]

if run_main_plots:
    log("Plotting 3D scene (3/3) — can take a while for many segments...")
    fig3d = plt.figure(figsize=(10, 10))
    ax = fig3d.add_subplot(111, projection='3d')

    cmap = plt.cm.tab10
    for i, cell in enumerate(all_cells):
        log(f"  drawing cell {i+1}/{n_cells} ({cell.totnsegs} segments)...")
        color = cmap(i % 10)
        for j in range(cell.totnsegs):
            ax.plot(
                [cell.x[j, 0], cell.x[j, 1]],
                [cell.y[j, 0], cell.y[j, 1]],
                [cell.z[j, 0], cell.z[j, 1]],
                color=color, linewidth=max(cell.d[j] / 5, 0.5))

        # Synapse markers for all synapses on this cell
        for s in all_syns[i]:
            ax.scatter(cell.x[s.idx].mean(), cell.y[s.idx].mean(),
                       cell.z[s.idx].mean(),
                       c='green', s=100, marker='^', zorder=5)

    for ex, ey, ez in zip(elec_x, elec_y, elec_z):
        verts = _contact_face(np.array([ex, ey, ez]),
                              contact_normal[0], contact_size, contact_shape)
        face = Poly3DCollection([verts], facecolor='red', edgecolor='darkred',
                                alpha=0.85, linewidth=1.5)
        ax.add_collection3d(face)

    # Legend proxy for the electrode
    ax.scatter([], [], [], c='red', marker='s', s=80,
               label=f'Contact ({contact_size:.0f} µm {contact_shape})')

    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_zlabel('Z (µm)')
    ax.legend()
    ax.set_title('Cell population around electrode')
    fig3d.text(0.995, 0.5, pop_info, fontsize=8, ha='right', va='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig3d.subplots_adjust(right=0.80)
    log("  saving 3D scene to disk (rendering can take 5–30 s)...")
    fig3d.savefig(fig_path('scene_3d'), dpi=130, bbox_inches='tight')
    log("  3D scene done.")
    plt.show()

# ============ SWEEP: n_cells × outer_radius ============
# Each population is simulated in a fresh Python subprocess so NEURON's internal
# state (section names, load_file bookkeeping) starts clean every time — this
# is the only reliable way to run hundreds of cells in series on Windows.
# The worker lives in sweep_worker.py next to this script.

if run_sweep:
    import sys
    import subprocess
    import tempfile
    import pickle as _pickle

    worker_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'sweep_worker.py'
    )
    if not os.path.isfile(worker_script):
        raise FileNotFoundError(
            f"sweep_worker.py not found next to lfpy_sim.py (looked at {worker_script})"
        )

    # Everything the worker needs, bundled once (arrays get .tolist()'d so the
    # pickle is fully self-contained and survives numpy version skew).
    base_args = {
        'morph_dir':            morph_dir,
        'morphologies':         list(morphologies),
        'v_init':               v_init,
        'dt':                   dt,
        'tstart':               tstart,
        'tstop':                tstop,
        'inner_radius':         inner_radius,
        'align_cells':          align_cells,
        'align_rot_x':          align_rot_x,
        'align_rot_y':          align_rot_y,
        'align_rot_z':          align_rot_z,
        'base_spike_time':      base_spike_time,
        'jitter_std':           jitter_std,
        'drive_mode':           drive_mode,
        'syn_height_min':       syn_height_min,
        'syn_height_max':       syn_height_max,
        'n_synapses':           n_synapses,
        'syn_type':             syn_type,
        'syn_weight':           syn_weight,
        'tau1':                 tau1,
        'tau2':                 tau2,
        'e_syn':                e_syn,
        'iclamp_amp':           iclamp_amp,
        'iclamp_dur':           iclamp_dur,
        'elec_x':               elec_x.tolist(),
        'elec_y':               elec_y.tolist(),
        'elec_z':               elec_z.tolist(),
        'sigma':                sigma,
        'method':               method,
        'contact_size':         contact_size,
        'contact_shape':        contact_shape,
        'contact_normal':       contact_normal.tolist(),
        'n_avg_points':         n_avg_points,
        'fs_hz':                fs_hz,
        'mua_low':              mua_low,
        'mua_high':             mua_high,
        'filt_order':           filt_order,
        'noise_rms_uV':         noise_rms_uV,
        'mua_threshold_factor': mua_threshold_factor,
        'refractory_ms':        refractory_ms,
    }

    n_J = len(sweep_jitters)
    n_D = len(sweep_distances)
    n_N = len(sweep_n_cells)
    # Per-repeat arrays so we can derive means AND detection probability AND amplitudes
    crossings_all = np.zeros((n_J, n_D, n_N, sweep_repeats))
    peak_neg_all  = np.zeros((n_J, n_D, n_N, sweep_repeats))
    peak_sbp_all  = np.zeros((n_J, n_D, n_N, sweep_repeats))

    total_runs = n_J * n_D * n_N * sweep_repeats
    log(f"=== Subprocess sweep: {n_N} n_cells × {n_D} distances × {n_J} jitters × {sweep_repeats} reps "
        f"= {total_runs} populations ===")
    log(f"Worker: {worker_script}")

    run_idx = 0
    for ij, J in enumerate(sweep_jitters):
        log(f"--- jitter = {J:.1f} ms ---")
        for ir, D in enumerate(sweep_distances):
            for ic, Nc in enumerate(sweep_n_cells):
                for rep in range(sweep_repeats):
                    run_idx += 1

                    args = dict(base_args)
                    args['n_cells']       = int(Nc)
                    args['cell_distance'] = float(D)
                    args['jitter_std']    = float(J)
                    # Common random numbers: seed depends on (distance, n_cells, rep)
                    # but NOT on jitter, so jitter comparisons are paired.
                    args['seed']          = abs(hash((ir, ic, rep))) % (2**31 - 1)

                    args_fd, args_path = tempfile.mkstemp(suffix='_args.pkl')
                    os.close(args_fd)
                    res_fd,  result_path = tempfile.mkstemp(suffix='_res.pkl')
                    os.close(res_fd)

                    with open(args_path, 'wb') as f:
                        _pickle.dump(args, f)

                    try:
                        proc = subprocess.run(
                            [sys.executable, worker_script, args_path, result_path],
                            capture_output=True, text=True, timeout=600,
                        )
                        if proc.returncode != 0:
                            log(f"  [{run_idx:3d}/{total_runs}]  j={J:.1f}  N={Nc:3d}  d={D:3.0f} µm  "
                                f"SUBPROCESS FAILED (rc={proc.returncode})")
                            if proc.stderr:
                                log(f"  stderr: {proc.stderr.strip().splitlines()[-3:]}")
                        else:
                            with open(result_path, 'rb') as f:
                                res = _pickle.load(f)
                            crossings_all[ij, ir, ic, rep] = res['n_crossings']
                            peak_neg_all [ij, ir, ic, rep] = res['peak_neg_uV']
                            peak_sbp_all [ij, ir, ic, rep] = res['peak_sbp_uV']
                    except subprocess.TimeoutExpired:
                        log(f"  [{run_idx:3d}/{total_runs}]  j={J:.1f}  N={Nc:3d}  d={D:3.0f} µm  TIMEOUT")
                    finally:
                        for p in (args_path, result_path):
                            try:
                                os.unlink(p)
                            except OSError:
                                pass

                _mean_cross = float(crossings_all[ij, ir, ic].mean())
                _det_p      = float((crossings_all[ij, ir, ic] > 0).mean())
                _psbp       = float(peak_sbp_all[ij, ir, ic].mean())
                log(f"  j={J:.1f}  N={Nc:3d}  d={D:3.0f} µm  "
                    f"crossings={_mean_cross:.1f}  P(det)={_det_p:.2f}  "
                    f"peak_SBP={_psbp:.1f} µV")

    # ---- Derived summaries from the per-repeat data ----
    sweep_results  = crossings_all.mean(axis=-1)                       # mean # crossings (legacy)
    detection_prob = (crossings_all > 0).astype(float).mean(axis=-1)   # P(≥1 crossing)
    mean_peak_sbp  = peak_sbp_all.mean(axis=-1)                        # µV

    # Save raw arrays so post-hoc analysis is possible without rerunning.
    np.savez(os.path.join(run_dir, 'sweep_raw.npz'),
             crossings_all=crossings_all,
             peak_neg_all=peak_neg_all,
             peak_sbp_all=peak_sbp_all,
             sweep_n_cells=np.asarray(sweep_n_cells),
             sweep_distances=np.asarray(sweep_distances),
             sweep_jitters=np.asarray(sweep_jitters),
             sweep_repeats=sweep_repeats)
    log(f"Saved raw sweep arrays → {os.path.join(run_dir, 'sweep_raw.npz')}")

    # ---- Heatmap grid: one subplot per jitter value ----
    fig_sw, axes_sw = plt.subplots(1, n_J, figsize=(5 * n_J + 2, 5),
                                   squeeze=False)
    axes_sw = axes_sw[0]   # collapse the row dimension

    # Shared colour scale across all subplots
    vmin = 0
    vmax = max(sweep_results.max(), 1.0)

    for ij, J in enumerate(sweep_jitters):
        ax = axes_sw[ij]
        im = ax.imshow(sweep_results[ij], aspect='auto', origin='lower',
                       cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_N))
        ax.set_xticklabels(sweep_n_cells, fontsize=8)
        ax.set_yticks(range(n_D))
        ax.set_yticklabels(sweep_distances, fontsize=8)
        ax.set_xlabel('# cells', fontsize=9)
        if ij == 0:
            ax.set_ylabel('Distance from electrode (µm)', fontsize=9)
        ax.set_title(f'jitter = {J:.1f} ms', fontsize=10)

        # Overlay crossing counts (1 decimal — values are means over repeats,
        # so whole-number display would round 3.5 and 4.5 both to "4" and hide
        # the fact that they're on different bins of the colormap).
        for i in range(n_D):
            for j in range(n_N):
                val = sweep_results[ij, i, j]
                ax.text(j, i, f'{val:.1f}',
                        ha='center', va='center',
                        color='w' if val < vmax / 2 else 'k',
                        fontsize=9, fontweight='bold')

    # Shared colourbar — give it a dedicated axis strip on the far right so it
    # doesn't steal space from the last subplot and overlap the data.
    cbar_ax = fig_sw.add_axes([0.94, 0.18, 0.012, 0.65])   # [left, bottom, width, height]
    cb = fig_sw.colorbar(im, cax=cbar_ax)
    cb.set_label('# threshold crossings (mean)', fontsize=9)

    # Parameter info text
    sweep_info = (
        f"Sweep parameters\n"
        f"n_cells: {sweep_n_cells}\n"
        f"distances: {sweep_distances} µm (shell)\n"
        f"jitters: {sweep_jitters} ms\n"
        f"repeats: {sweep_repeats}\n"
        f"noise: {noise_rms_uV:.0f} µV RMS\n"
        f"thr: {mua_threshold_factor:.0f}×MAD\n"
        f"MUA: {mua_low:.0f}–{mua_high:.0f} Hz\n"
        f"syns/cell: {n_synapses}  w={syn_weight}\n"
        f"τ1/τ2: {tau1}/{tau2} ms\n"
        f"dt: {dt:.4f} ms ({fs_hz:.0f} Hz)\n"
        f"morphs: {len(morphologies)}\n"
        f"contact: {contact_size:.0f} µm {contact_shape}\n"
        f"σ: {sigma}  method: {method}"
    )
    fig_sw.text(0.005, 0.5, sweep_info, fontsize=7, va='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig_sw.suptitle('MUA threshold crossings:  n_cells × distance (shell) × jitter', fontsize=12)
    # Leave clear margins: left for the parameter box, right for the colorbar.
    fig_sw.subplots_adjust(left=0.18, right=0.92, wspace=0.35, top=0.88, bottom=0.15)
    log("Saving sweep heatmap...")
    fig_sw.savefig(fig_path('sweep_ncells_distance_jitter'), dpi=130)
    log("Sweep heatmap done.")
    plt.show()

    # =================================================================
    # Helper to draw a (n_D × n_N) × n_J grid heatmap for any metric
    # =================================================================
    def _plot_sweep_grid(data, title, cb_label, filename, fmt='{:.1f}',
                         vmin=None, vmax=None, cmap='viridis'):
        """data shape (n_J, n_D, n_N) — one subplot per jitter."""
        fig, axes = plt.subplots(1, n_J, figsize=(5 * n_J + 2, 5), squeeze=False)
        axes = axes[0]
        _vmin = 0.0 if vmin is None else vmin
        _vmax = max(data.max(), 1e-9) if vmax is None else vmax

        for ij, J in enumerate(sweep_jitters):
            ax = axes[ij]
            im_ = ax.imshow(data[ij], aspect='auto', origin='lower',
                            cmap=cmap, vmin=_vmin, vmax=_vmax)
            ax.set_xticks(range(n_N))
            ax.set_xticklabels(sweep_n_cells, fontsize=8)
            ax.set_yticks(range(n_D))
            ax.set_yticklabels(sweep_distances, fontsize=8)
            ax.set_xlabel('# cells', fontsize=9)
            if ij == 0:
                ax.set_ylabel('Distance from electrode (µm)', fontsize=9)
            ax.set_title(f'jitter = {J:.1f} ms', fontsize=10)
            for i in range(n_D):
                for j in range(n_N):
                    val = data[ij, i, j]
                    ax.text(j, i, fmt.format(val),
                            ha='center', va='center',
                            color='w' if val < (_vmax - _vmin) / 2 + _vmin else 'k',
                            fontsize=9, fontweight='bold')

        cbar_ax = fig.add_axes([0.94, 0.18, 0.012, 0.65])
        cb_ = fig.colorbar(im_, cax=cbar_ax)
        cb_.set_label(cb_label, fontsize=9)

        fig.text(0.005, 0.5, sweep_info, fontsize=7, va='center',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        fig.suptitle(title, fontsize=12)
        fig.subplots_adjust(left=0.18, right=0.92, wspace=0.35, top=0.88, bottom=0.15)
        log(f"Saving {filename}.png ...")
        fig.savefig(fig_path(filename), dpi=130)
        plt.show()

    # Detection probability (0–1)
    _plot_sweep_grid(
        detection_prob,
        'Detection probability  P(≥1 crossing)  —  n_cells × distance × jitter',
        'P(detection)',
        'sweep_detection_probability',
        fmt='{:.2f}',
        vmin=0.0, vmax=1.0, cmap='magma',
    )

    # Peak SBP amplitude (µV)
    _plot_sweep_grid(
        mean_peak_sbp,
        'Peak SBP amplitude (µV)  —  n_cells × distance × jitter',
        'Peak |SBP| (µV, mean)',
        'sweep_peak_sbp',
        fmt='{:.0f}',
        cmap='plasma',
    )
