"""
Parameter Sweep — N × Distance × Jitter
========================================
Full 3D grid sweep as specified:

    Cell count  N:   1, 5, 10, 25, 50, 100
    Distance    D:   25, 50, 75, 100, 150, 200 µm  (shell radius)
    Jitter      σ:   0, 3, 5, 10, 20, 30, 50 ms
    Repeats:         3 per (N, D, σ) point

Implementation choices
----------------------
1. **Shell sampling** — all N cells sit at exactly distance D from the
   electrode (uniformly distributed on a spherical shell), so the
   distance axis is clean with no intra-population variance.

2. **Common random numbers** — for a given (N, D, rep) cell, the
   positions, orientations, and noise are drawn from the same seed
   across all jitter values.  This makes inter-jitter comparisons
   paired (variance reduction).

3. **Subprocess-per-population** — each (N, D, σ, rep) simulation runs
   in a fresh Python/NEURON process to avoid NEURON's global state
   accumulation.  The outer loop calls `_run_one` via
   `multiprocessing` / `subprocess`.

Output
------
Saves a .npz file with the full results grid + metadata, plus CSV
for easy inspection.  Downstream figure scripts (fig3, fig5, etc.)
can load this file instead of re-running.

Usage:
    python sweep.py                  # full sweep
    python sweep.py --dry-run        # print grid size and exit
    python sweep.py --workers 4      # parallel subprocesses

Requires: LFPy, NEURON, numpy, scipy
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import itertools
import numpy as np
from pathlib import Path

# ──────────────── paths ────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "sweep_results"
OUTPUT_DIR.mkdir(exist_ok=True)

MORPH_PATH = r"C:\Users\justi\OneDrive - Rice University\Desktop\ELEC 481\Final Project\LFPy\LFPy-2.3.6\examples\morphologies\L5_Mainen96_LFPy.hoc"

# ──────────────── sweep grid ────────────────
N_VALUES       = [1, 5, 10, 25, 50, 100]
D_VALUES       = [25, 50, 75, 100, 150, 200]       # µm
JITTER_VALUES  = [0, 3, 5, 10, 20, 30, 50]          # ms
N_REPEATS      = 3

# ──────────────── shared simulation parameters ────────────────
SIM_PARAMS = dict(
    morph_path     = MORPH_PATH,
    v_init         = -65,
    dt             = 2**-4,
    tstart         = 0,
    tstop          = 100,
    syn_type       = 'Exp2Syn',
    syn_weight     = 0.05,
    tau1           = 0.5,
    tau2           = 5.0,
    e_syn          = 0,
    n_synapses     = 20,
    syn_height_min = 0.3,
    syn_height_max = 0.9,
    sigma          = 0.3,
    method         = 'linesource',
    contact_size   = 12.0,
    contact_shape  = 'square',
    n_avg_points   = 50,
    base_spike_time = 20.0,
    mua_low        = 300.0,
    mua_high       = 5000.0,
    filt_order     = 4,
    noise_rms_uV   = 5.0,
    threshold_k    = 4.0,
    align_cells    = True,
)


# ====================================================================
# Worker script — runs ONE (N, D, σ, rep) point in a fresh process
# ====================================================================
WORKER_SCRIPT = r'''
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
'''


# ====================================================================
# Orchestrator
# ====================================================================

def run_one_point(N, D, jitter_std, rep, python_exe=sys.executable):
    """
    Spawn a subprocess for one (N, D, σ, rep) point.
    Returns the result dict or None on failure.
    """
    payload = json.dumps(dict(
        N=N, D=D, jitter_std=jitter_std, rep=rep,
        sim=SIM_PARAMS,
    ))

    # Write worker script to a temp file (avoids shell-quoting issues)
    worker_file = SCRIPT_DIR / "_sweep_worker.py"
    if not worker_file.exists():
        worker_file.write_text(WORKER_SCRIPT, encoding="utf-8")

    try:
        proc = subprocess.run(
            [python_exe, str(worker_file), payload],
            capture_output=True, text=True,
            timeout=600,   # 10 min per point (large N at close distance)
        )
        if proc.returncode != 0:
            print(f"  FAIL  N={N}, D={D}, σ={jitter_std}, rep={rep}",
                  file=sys.stderr)
            print(proc.stderr[:500], file=sys.stderr)
            return None
        # Last line of stdout is the JSON result
        lines = proc.stdout.strip().split('\n')
        return json.loads(lines[-1])
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT  N={N}, D={D}, σ={jitter_std}, rep={rep}",
              file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ERROR  N={N}, D={D}, σ={jitter_std}, rep={rep}: {e}",
              file=sys.stderr)
        return None


def run_sweep(workers=1):
    """Run the full 3D sweep, optionally in parallel."""
    grid = list(itertools.product(N_VALUES, D_VALUES, JITTER_VALUES,
                                  range(N_REPEATS)))
    total = len(grid)
    print(f"Sweep grid: {len(N_VALUES)} N × {len(D_VALUES)} D × "
          f"{len(JITTER_VALUES)} σ × {N_REPEATS} reps = {total} points")

    results = []

    if workers <= 1:
        # Sequential
        for idx, (N, D, jit, rep) in enumerate(grid):
            print(f"[{idx+1}/{total}]  N={N:3d}, D={D:3d} µm, "
                  f"σ={jit:2d} ms, rep={rep} …", end="  ", flush=True)
            r = run_one_point(N, D, jit, rep)
            if r is not None:
                results.append(r)
                print(f"SNR={r['snr']:.2f}  det={r['detected']}")
            else:
                print("FAILED")
    else:
        # Parallel via concurrent.futures
        from concurrent.futures import ProcessPoolExecutor, as_completed
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for N, D, jit, rep in grid:
                f = pool.submit(run_one_point, N, D, jit, rep)
                futures[f] = (N, D, jit, rep)
            done = 0
            for f in as_completed(futures):
                done += 1
                key = futures[f]
                r = f.result()
                if r is not None:
                    results.append(r)
                if done % 50 == 0 or done == total:
                    print(f"  {done}/{total} complete")

    return results


def save_results(results):
    """Save results as .npz (for numpy) and .csv (for inspection)."""
    if not results:
        print("No results to save.")
        return

    # Build structured arrays
    keys = ['N', 'D', 'jitter_std', 'rep',
            'vpp_clean_uV', 'vpp_noisy_uV', 'snr',
            'peak_neg_uV', 'crossings', 'detected']

    arrays = {k: np.array([r[k] for r in results]) for k in keys}

    npz_path = OUTPUT_DIR / "sweep_results.npz"
    np.savez(str(npz_path), **arrays,
             N_values=N_VALUES, D_values=D_VALUES,
             jitter_values=JITTER_VALUES, n_repeats=N_REPEATS)
    print(f"Saved {npz_path}  ({len(results)} points)")

    # CSV
    csv_path = OUTPUT_DIR / "sweep_results.csv"
    with open(csv_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in results:
            f.write(','.join(str(r[k]) for k in keys) + '\n')
    print(f"Saved {csv_path}")


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N × D × Jitter parameter sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print grid size and exit")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel subprocesses (default: 1)")
    args = parser.parse_args()

    total = len(N_VALUES) * len(D_VALUES) * len(JITTER_VALUES) * N_REPEATS
    print(f"Grid: {len(N_VALUES)} N × {len(D_VALUES)} D × "
          f"{len(JITTER_VALUES)} jitter × {N_REPEATS} reps = {total} points")

    if args.dry_run:
        print("Dry run — exiting.")
        sys.exit(0)

    results = run_sweep(workers=args.workers)
    save_results(results)
    print("Done.")
