
"""Worker: simulate one (N, D, jitter_std, rep) point."""
import sys, json, os
import numpy as np

def main():
    params = json.loads(sys.argv[1])

    # ── unpack ──
    N            = params["N"]
    D            = params["D"]
    jitter_std   = params["jitter_std"]
    rep          = params["rep"]
    sim          = params["sim"]

    import LFPy
    from scipy.signal import butter, filtfilt

    morph_path     = sim["morph_path"]
    v_init         = sim["v_init"]
    dt             = sim["dt"]
    tstart         = sim["tstart"]
    tstop          = sim["tstop"]
    syn_type       = sim["syn_type"]
    syn_weight     = sim["syn_weight"]
    tau1           = sim["tau1"]
    tau2           = sim["tau2"]
    e_syn          = sim["e_syn"]
    n_synapses     = sim["n_synapses"]
    syn_height_min = sim["syn_height_min"]
    syn_height_max = sim["syn_height_max"]
    sigma          = sim["sigma"]
    method         = sim["method"]
    contact_size   = sim["contact_size"]
    contact_shape  = sim["contact_shape"]
    n_avg_points   = sim["n_avg_points"]
    base_spike_time = sim["base_spike_time"]
    mua_low        = sim["mua_low"]
    mua_high       = sim["mua_high"]
    filt_order     = sim["filt_order"]
    noise_rms_uV   = sim["noise_rms_uV"]
    threshold_k    = sim["threshold_k"]
    align_cells    = sim["align_cells"]

    contact_normal = np.array([[1., 0., 0.]])
    fs_hz = 1000.0 / dt

    # ── Common random seed for (N, D, rep) ──
    # Jitter is NOT part of the seed so that positions/orientations/noise
    # are identical across jitter values → paired comparison.
    base_seed = hash((N, D, rep)) % (2**31)
    rng = np.random.RandomState(base_seed)

    # ── Generate neuron positions on spherical shell ──
    positions = []
    orientations = []
    spike_times = []
    for _ in range(N):
        # Uniform on sphere: Marsaglia method
        phi   = rng.uniform(0, 2 * np.pi)
        cos_theta = rng.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        x = D * sin_theta * np.cos(phi)
        y = D * sin_theta * np.sin(phi)
        z = D * cos_theta
        positions.append((x, y, z))

        if align_cells:
            orientations.append((0, 0, 0))
        else:
            orientations.append((rng.uniform(0, 10),
                                 rng.uniform(0, 10),
                                 rng.uniform(0, 360)))

        # Spike time: jitter drawn from a SEPARATE rng stream seeded
        # with (base_seed, cell_index, jitter_std) so that changing
        # jitter_std actually changes timing (not just scales it).
        # But positions/orientations above are already fixed by base_seed.
    # Now draw spike times with jitter (this part DOES depend on jitter_std)
    jitter_rng = np.random.RandomState(hash((base_seed, int(jitter_std * 1000))) % (2**31))
    for _ in range(N):
        if jitter_std > 0:
            t = base_spike_time + jitter_rng.normal(0, jitter_std)
            t = max(1.0, t)
        else:
            t = base_spike_time
        spike_times.append(t)

    # ── Simulate each cell and accumulate extracellular signal ──
    all_lfps = []

    for i in range(N):
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

        rx, ry, rz = orientations[i]
        cell.set_rotation(x=np.radians(rx), y=np.radians(ry), z=np.radians(rz))
        px, py, pz = positions[i]
        cell.set_pos(x=px, y=py, z=pz)

        # Synapse placement (height filter along principal axis)
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
        h_min = main_len * syn_height_min
        h_max = main_len * syn_height_max
        dend_idxs = dend_idxs[(projections >= h_min) & (projections <= h_max)]

        # Use a per-cell seed for synapse placement (deterministic, independent of jitter)
        syn_rng = np.random.RandomState(hash((base_seed, i, "syn")) % (2**31))
        for _ in range(n_synapses):
            idx = syn_rng.choice(dend_idxs)
            syn = LFPy.Synapse(cell, idx=idx, syntype=syn_type,
                                weight=syn_weight, record_current=False,
                                **{'tau1': tau1, 'tau2': tau2, 'e': e_syn})
            syn.set_spike_times(np.array([max(0.1,
                spike_times[i] + syn_rng.uniform(-1, 1))]))

        electrode = LFPy.RecExtElectrode(
            cell,
            x=np.array([0.]), y=np.array([0.]), z=np.array([0.]),
            sigma=sigma, method=method,
            N=contact_normal, r=contact_size / 2.0,
            n=n_avg_points, contact_shape=contact_shape,
        )
        cell.simulate(rec_imem=True, rec_vmem=False, probes=[electrode])
        all_lfps.append(electrode.data[0].copy())

    # ── Superpose and analyse ──
    total_uV = np.sum(all_lfps, axis=0) * 1000.0   # mV → µV

    # Bandpass into MUA band
    nyq = 0.5 * fs_hz
    b, a = butter(filt_order, [mua_low / nyq, mua_high / nyq], btype='band')
    mua_clean = filtfilt(b, a, total_uV)

    # Add noise (seeded from base_seed so it's identical across jitter values)
    noise_rng = np.random.RandomState(hash((base_seed, "noise")) % (2**31))
    noise = noise_rng.normal(0, noise_rms_uV, mua_clean.shape)
    mua_noisy = mua_clean + noise

    # Metrics
    vpp_clean = float(np.max(mua_clean) - np.min(mua_clean))
    vpp_noisy = float(np.max(mua_noisy) - np.min(mua_noisy))
    snr       = vpp_clean / (2.0 * noise_rms_uV)
    threshold = threshold_k * noise_rms_uV

    # Negative-going threshold crossings
    below     = mua_noisy < -threshold
    crossings = int(np.sum(np.diff(below.astype(int)) > 0))
    detected  = bool(crossings > 0)

    # Peak negative amplitude (absolute value)
    peak_neg = float(np.min(mua_clean))

    result = dict(
        N=N, D=D, jitter_std=jitter_std, rep=rep,
        vpp_clean_uV=vpp_clean,
        vpp_noisy_uV=vpp_noisy,
        snr=snr,
        peak_neg_uV=peak_neg,
        crossings=crossings,
        detected=detected,
    )
    print(json.dumps(result))

if __name__ == "__main__":
    main()
