"""
Standalone worker for the lfpy_sim.py n_cells × cell_distance × jitter sweep.

Each invocation runs ONE population simulation in a fresh Python process so
NEURON's internal state (section name table, file-load bookkeeping) starts
clean every time. This is the only reliable way to run hundreds of cells in
series on Windows without NEURON losing track of the morphology file.

Usage:
    python sweep_worker.py <args_pickle> <result_pickle>

The args pickle must contain a dict with all the parameters the simulation
needs (see lfpy_sim.py's sweep section for the fields it writes).
"""
import os
import sys
import pickle
import numpy as np
import LFPy
import neuron
from scipy.signal import butter, lfilter


def butter_filter(data, fs, low=None, high=None, order=4):
    """Causal Butterworth (lfilter) — matches Kilosort / real-time hardware.
    Must stay in sync with lfpy_sim.py's _butter_filter."""
    nyq = 0.5 * fs
    if low is None and high is not None:
        b, a = butter(order, high / nyq, btype='low')
    elif high is None and low is not None:
        b, a = butter(order, low / nyq, btype='high')
    else:
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data)


def simulate_population_mua(args):
    """Run one population simulation and return the clean MUA trace in µV."""
    # Ensure NEURON's CWD matches the morphology directory
    os.chdir(args['morph_dir'])
    neuron.h.chdir(args['morph_dir'])

    n_cells       = args['n_cells']
    cell_distance = args['cell_distance']   # µm — shell sampling: every cell at exactly this distance

    lfps = []
    for _ in range(n_cells):
        # Shell sampling: every cell sits on a sphere of radius `cell_distance`.
        # Only the azimuthal angle is randomised, so all cells contribute the
        # same ~1/r-scaled extracellular amplitude.
        angle = np.random.uniform(0, 2 * np.pi)
        x_    = cell_distance * np.cos(angle)
        y_    = cell_distance * np.sin(angle)
        z_    = 0.0   # keep cells on the horizontal plane of the electrode

        if args['align_cells']:
            rx_ = args['align_rot_x']
            ry_ = args['align_rot_y']
            rz_ = args['align_rot_z']
        else:
            rx_ = np.random.uniform(0, 10)
            ry_ = np.random.uniform(0, 10)
            rz_ = np.random.uniform(0, 360)

        st_ = max(1.0, args['base_spike_time'] +
                  np.random.normal(0, args['jitter_std']))
        morph_ = args['morphologies'][np.random.randint(len(args['morphologies']))]

        c_ = LFPy.Cell(
            morphology=morph_,
            v_init=args['v_init'],
            passive=False,
            dt=args['dt'],
            tstart=args['tstart'],
            tstop=args['tstop'],
        )

        for sec in c_.allseclist:
            if 'soma' in sec.name() or 'axon' in sec.name():
                sec.insert('hh')
            else:
                sec.insert('pas')
                sec.g_pas = 1. / 30000
                sec.e_pas = -65

        c_.set_rotation(x=np.radians(rx_), y=np.radians(ry_), z=np.radians(rz_))
        c_.set_pos(x=x_, y=y_, z=z_)

        drive_mode = args.get('drive_mode', 'synapses')
        stims = []   # IClamp objects must be kept alive until simulate()

        if drive_mode == 'synapses':
            # Dendritic compartments + height-fraction filter
            dend_idxs_ = []
            for sec in c_.allseclist:
                if 'soma' not in sec.name() and 'axon' not in sec.name():
                    dend_idxs_.extend(c_.get_idx(section=sec.name()))
            dend_idxs_ = np.array(dend_idxs_, dtype=int)

            seg_c_ = np.column_stack([c_.x.mean(axis=1),
                                      c_.y.mean(axis=1),
                                      c_.z.mean(axis=1)])
            soma_c_      = seg_c_[c_.somaidx].mean(axis=0)
            d_from_soma_ = np.linalg.norm(seg_c_ - soma_c_, axis=1)
            apex_        = seg_c_[np.argmax(d_from_soma_)]
            mvec_        = apex_ - soma_c_
            mlen_        = np.linalg.norm(mvec_)
            munit_       = mvec_ / mlen_
            proj_        = np.dot(seg_c_[dend_idxs_] - soma_c_, munit_)
            dend_idxs_   = dend_idxs_[(proj_ >= mlen_ * args['syn_height_min']) &
                                      (proj_ <= mlen_ * args['syn_height_max'])]
            if len(dend_idxs_) == 0:
                continue

            # Deterministic synapse placement (same pattern for every cell in this
            # population, matching the main script). Every cell gets the same set
            # of dendritic indices, so AP latency no longer varies between cells.
            syn_rng = np.random.default_rng(seed=0)
            for _ in range(args['n_synapses']):
                syn_ = LFPy.Synapse(
                    c_,
                    idx=int(syn_rng.choice(dend_idxs_)),
                    syntype=args['syn_type'],
                    weight=args['syn_weight'],
                    record_current=False,
                    **{'tau1': args['tau1'],
                       'tau2': args['tau2'],
                       'e':    args['e_syn']},
                )
                # No extra per-synapse jitter — the cell-level jitter (st_) is the
                # only timing variability, so the sweep's jitter_std variable is the
                # only thing controlling spike synchrony.
                syn_.set_spike_times(np.array([max(0.1, st_)]))

        elif drive_mode == 'iclamp':
            soma_sec = next(s for s in c_.allseclist if 'soma' in s.name())
            iclamp = neuron.h.IClamp(soma_sec(0.5))
            iclamp.delay = st_
            iclamp.dur   = args['iclamp_dur']
            iclamp.amp   = args['iclamp_amp']
            stims.append(iclamp)

        else:
            raise ValueError(f"Unknown drive_mode: {drive_mode!r}")

        elec_ = LFPy.RecExtElectrode(
            c_,
            x=np.asarray(args['elec_x']),
            y=np.asarray(args['elec_y']),
            z=np.asarray(args['elec_z']),
            sigma=args['sigma'],
            method=args['method'],
            N=np.asarray(args['contact_normal']),
            r=args['contact_size'] / 2.0,
            n=args['n_avg_points'],
            contact_shape=args['contact_shape'],
        )
        c_.simulate(rec_imem=True, probes=[elec_])
        lfps.append(elec_.data[0].copy())

    if len(lfps) == 0:
        # No valid cells — return a short zero trace to keep the caller happy.
        return np.zeros(int(round((args['tstop'] - args['tstart']) / args['dt'])) + 1)

    total = np.sum(lfps, axis=0) * 1000.0  # µV
    return butter_filter(total, args['fs_hz'],
                         low=args['mua_low'],
                         high=args['mua_high'],
                         order=args['filt_order'])


def main():
    if len(sys.argv) != 3:
        print("Usage: python sweep_worker.py <args_pickle> <result_pickle>",
              file=sys.stderr)
        sys.exit(1)

    args_path   = sys.argv[1]
    result_path = sys.argv[2]

    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    if args.get('seed') is not None:
        np.random.seed(int(args['seed']))

    mua_clean = simulate_population_mua(args)
    noise_    = np.random.normal(0, args['noise_rms_uV'], mua_clean.shape)
    mua_noisy = mua_clean + noise_
    # Robust MAD estimator (Quiroga 2004) — converges to the true noise std
    # even when spike content is large, so the threshold stays ~fixed across
    # the sweep and crossings become monotonically interpretable.
    mua_std   = float(np.median(np.abs(mua_noisy)) / 0.6745)
    thr_      = args['mua_threshold_factor'] * mua_std
    below_    = mua_noisy < -thr_
    all_cross = np.where(np.diff(below_.astype(int)) > 0)[0]

    # Refractory period (standard spike-detection convention) — collapse multiple
    # threshold crossings within the refractory window into one event so a single
    # AP's MUA-band wiggles aren't counted as several spikes.
    refractory_ms      = args.get('refractory_ms', 1.0)
    refractory_samples = int(round(refractory_ms / args['dt']))
    crossings_list = []
    last = -refractory_samples
    for idx in all_cross:
        if idx - last >= refractory_samples:
            crossings_list.append(idx)
            last = idx
    crossings = np.array(crossings_list, dtype=int)

    # Peak extracellular amplitude (max negative deflection) — useful for SNR plots.
    peak_neg_uV = float(-mua_noisy.min())      # magnitude of most-negative excursion

    # Analog SBP peak — monotonic proxy for firing energy.
    # Causal bandpass to match the rest of the chain.
    def _bp(x, fs, lo, hi, order):
        nyq = 0.5 * fs
        b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
        return lfilter(b, a, x)
    sbp_band    = _bp(mua_noisy, args['fs_hz'], 300.0, 1000.0, 2)
    peak_sbp_uV = float(np.abs(sbp_band).max())

    with open(result_path, 'wb') as f:
        pickle.dump({
            'n_crossings':   int(len(crossings)),
            'detected':      bool(len(crossings) > 0),
            'peak_neg_uV':   peak_neg_uV,
            'peak_sbp_uV':   peak_sbp_uV,
            'mua_std':       mua_std,
            'threshold':     float(thr_),
            'n_cells':       int(args['n_cells']),
            'cell_distance': float(args['cell_distance']),
        }, f)


if __name__ == '__main__':
    main()
