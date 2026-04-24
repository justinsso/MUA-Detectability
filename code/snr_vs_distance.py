"""
SNR-vs-distance calibration for a single synchronous cell.

For each distance in `distances_um`, runs one sweep_worker subprocess with
n_cells=1, fix_cell_distance=True, jitter=0, align_to_electrode=True. Reads
back the peak negative extracellular amplitude and the MAD noise estimate,
then reports SNR = peak / noise and identifies the detection horizon
(distance where SNR drops below `detect_factor`, e.g. 4 × noise).

Use this as a single-cell calibration curve — it is the numerical analogue
of Henze et al. (2000)'s spike-amplitude-vs-distance measurement.

Output: figures/run_<stamp>/snr_vs_distance.png
        figures/run_<stamp>/snr_vs_distance.npz   (raw data)
"""
import os
import sys
import subprocess
import tempfile
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------- USER CONFIG
morph_dir     = "C:/Users/zachr/Downloads"
morphologies  = [f"{morph_dir}/L5_Mainen96_LFPy.hoc"]

# Distance grid (µm) — single-cell peak amplitude is measured at each
distances_um  = np.array([10, 20, 30, 40, 50, 60, 75, 100, 125, 150,
                          200, 250, 300, 400, 500])
n_repeats     = 3            # different azimuthal positions per distance
detect_factor = 4.0          # detection threshold (× noise MAD)

# Match your main simulation's electrophysiology parameters
v_init = -65
dt     = 2**-4
tstart = 0
tstop  = 100

drive_mode      = 'synapses'
syn_type        = 'Exp2Syn'
syn_weight      = 0.05
tau1            = 0.5
tau2            = 2.0
e_syn           = 0
n_synapses      = 20
syn_height_min  = 0.5
syn_height_max  = 0.9
iclamp_amp      = 10.0
iclamp_dur      = 0.1

elec_x = [0.]
elec_y = [0.]
elec_z = [0.]
sigma  = 0.3
method = 'linesource'
contact_size   = 12.0
contact_shape  = 'square'
contact_normal = [[1., 0., 0.]]
n_avg_points   = 50

fs_hz          = 1000.0 / dt
mua_low        = 300.0
mua_high       = 5000.0
filt_order     = 4
noise_rms_uV   = 5.0
mua_threshold_factor = 4.0
refractory_ms  = 0.5

base_spike_time = 20.0
# -------------------------------------------------------- /USER CONFIG


figure_dir = r"C:\Users\zachr\OneDrive\Skrivbord\claude_code\figures"
os.makedirs(figure_dir, exist_ok=True)
run_stamp  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir    = os.path.join(figure_dir, f"snr_run_{run_stamp}")
os.makedirs(run_dir, exist_ok=True)

worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'sweep_worker.py')
if not os.path.isfile(worker_script):
    raise FileNotFoundError(f"sweep_worker.py not found at {worker_script}")


def run_one(distance, seed):
    """Invoke sweep_worker.py for a single cell at the given distance."""
    args = dict(
        morph_dir=morph_dir,
        morphologies=morphologies,
        v_init=v_init, dt=dt, tstart=tstart, tstop=tstop,
        inner_radius=0, align_cells=True,
        align_rot_x=0, align_rot_y=0, align_rot_z=0,
        base_spike_time=base_spike_time,
        jitter_std=0.0,
        drive_mode=drive_mode,
        syn_height_min=syn_height_min,
        syn_height_max=syn_height_max,
        n_synapses=n_synapses,
        syn_type=syn_type, syn_weight=syn_weight,
        tau1=tau1, tau2=tau2, e_syn=e_syn,
        iclamp_amp=iclamp_amp, iclamp_dur=iclamp_dur,
        elec_x=elec_x, elec_y=elec_y, elec_z=elec_z,
        sigma=sigma, method=method,
        contact_size=contact_size, contact_shape=contact_shape,
        contact_normal=contact_normal,
        n_avg_points=n_avg_points,
        fs_hz=fs_hz, mua_low=mua_low, mua_high=mua_high,
        filt_order=filt_order,
        noise_rms_uV=noise_rms_uV,
        mua_threshold_factor=mua_threshold_factor,
        refractory_ms=refractory_ms,
        n_cells=1,
        cell_distance=float(distance),
        seed=int(seed),
    )

    args_fd, args_path = tempfile.mkstemp(suffix='_args.pkl')
    os.close(args_fd)
    res_fd,  result_path = tempfile.mkstemp(suffix='_res.pkl')
    os.close(res_fd)

    with open(args_path, 'wb') as f:
        pickle.dump(args, f)

    try:
        proc = subprocess.run(
            [sys.executable, worker_script, args_path, result_path],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            print(f"  FAIL at d={distance} µm (rc={proc.returncode})")
            if proc.stderr:
                print("  stderr:", proc.stderr.strip().splitlines()[-3:])
            return None
        with open(result_path, 'rb') as f:
            return pickle.load(f)
    finally:
        for p in (args_path, result_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def main():
    n_D = len(distances_um)
    peak_neg = np.zeros((n_D, n_repeats))
    peak_sbp = np.zeros((n_D, n_repeats))
    mua_std  = np.zeros((n_D, n_repeats))
    cross    = np.zeros((n_D, n_repeats))

    print(f"=== SNR vs distance:  {n_D} distances × {n_repeats} repeats "
          f"= {n_D * n_repeats} runs ===")

    for i, d in enumerate(distances_um):
        for r in range(n_repeats):
            seed = abs(hash((int(d), r))) % (2**31 - 1)
            res = run_one(d, seed)
            if res is None:
                continue
            peak_neg[i, r] = res['peak_neg_uV']
            peak_sbp[i, r] = res['peak_sbp_uV']
            mua_std [i, r] = res['mua_std']
            cross  [i, r] = res['n_crossings']
        print(f"  d={d:4.0f} µm  peak_neg={peak_neg[i].mean():6.1f} µV  "
              f"SBP={peak_sbp[i].mean():5.1f} µV  "
              f"SNR={peak_neg[i].mean()/mua_std[i].mean():5.2f}  "
              f"cross={cross[i].mean():.1f}")

    # Derived metrics
    snr = peak_neg / mua_std          # SNR = peak / noise MAD
    snr_mean = snr.mean(axis=1)
    snr_sd   = snr.std(axis=1)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # (1) Peak negative amplitude
    axes[0].errorbar(distances_um, peak_neg.mean(axis=1),
                     yerr=peak_neg.std(axis=1),
                     fmt='o-', color='tab:purple', capsize=3,
                     label='peak |neg. MUA|')
    axes[0].axhline(detect_factor * noise_rms_uV, color='k', linestyle='--',
                    linewidth=1.2,
                    label=f'detection threshold  ({detect_factor:.0f}×noise = '
                          f'{detect_factor*noise_rms_uV:.0f} µV)')
    axes[0].axhline(noise_rms_uV, color='grey', linestyle=':',
                    linewidth=1.0,
                    label=f'noise floor  ({noise_rms_uV:.0f} µV RMS)')
    axes[0].set_ylabel('Peak extracellular AP amplitude (µV)')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0.0)
    axes[0].set_title('Single-cell extracellular spike amplitude vs distance')
    axes[0].grid(True, alpha=0.3)

    # (2) SNR
    axes[1].errorbar(distances_um, snr_mean, yerr=snr_sd,
                     fmt='s-', color='tab:red', capsize=3, label='SNR')
    axes[1].axhline(detect_factor, color='k', linestyle='--',
                    linewidth=1.2,
                    label=f'detection line  ({detect_factor:.0f}×)')
    axes[1].set_xlabel('Distance from electrode (µm)')
    axes[1].set_ylabel('SNR  (peak / noise MAD)')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0.0)
    axes[1].set_title('SNR vs distance  —  detection horizon = '
                      'where curve crosses the dashed line')
    axes[1].grid(True, alpha=0.3)

    # Parameter info sidebar — kept narrow so it doesn't spill into the subplots
    info = (
        f"Single-cell SNR\n"
        f"distances:\n"
        f"  {len(distances_um)} pts,\n"
        f"  {int(distances_um.min())}–{int(distances_um.max())} µm\n"
        f"repeats: {n_repeats}\n"
        f"noise: {noise_rms_uV:.0f} µV RMS\n"
        f"thr: {detect_factor:.0f}×MAD\n"
        f"MUA: {mua_low:.0f}–{mua_high:.0f} Hz\n"
        f"drive: {drive_mode}\n"
        f"dt: {dt:.4f} ms\n"
        f"fs: {fs_hz:.0f} Hz\n"
        f"contact: {contact_size:.0f} µm\n"
        f"  {contact_shape}"
    )
    fig.text(0.02, 0.5, info, fontsize=7, va='center', ha='left',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('SNR vs distance (single cell)', fontsize=13)
    # Generous left margin for the parameter box, right margin for the external legends
    fig.subplots_adjust(left=0.22, right=0.72, top=0.93, hspace=0.3)

    out_png = os.path.join(run_dir, 'snr_vs_distance.png')
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")

    out_npz = os.path.join(run_dir, 'snr_vs_distance.npz')
    np.savez(out_npz,
             distances_um=distances_um,
             peak_neg=peak_neg, peak_sbp=peak_sbp,
             mua_std=mua_std, cross=cross,
             detect_factor=detect_factor,
             noise_rms_uV=noise_rms_uV)
    print(f"Saved {out_npz}")

    # Locate approximate detection horizon (first distance where mean SNR < threshold)
    below = np.where(snr_mean < detect_factor)[0]
    if len(below) > 0 and below[0] > 0:
        d_prev = distances_um[below[0] - 1]
        d_cross = distances_um[below[0]]
        print(f"\nDetection horizon  ≈ between {d_prev} and {d_cross} µm "
              f"(SNR crosses below {detect_factor:.0f}×)")
    else:
        print("\nDetection horizon not located within the swept range.")

    plt.show()


if __name__ == '__main__':
    main()
