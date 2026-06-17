"""Run the human L5 contact-approach sweep via isolated worker subprocesses.

This sweep *inverts* the reference sweep. The reference places cells on a shell
at a varying ``cell_distance`` around a fixed electrode; here one fixed
volumetric population of L5 pyramidal cells fills a 300³ µm voxel and the
**contact moves** through 13 approach distances, all captured in one population
simulation per repeat.

Like ``run_reference_sweep.py``, this runner only coordinates jobs: the
NEURON/LFPy work stays in ``l5_approach_worker.py`` (via
``mua_core.simulate_l5_population_mua``) so each population/repeat runs in a
fresh Python process. Importing this module does not import LFPy or NEURON.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as _dt
import hashlib
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from mua_config import default_morphology_path, default_output_dir, morphology_dir, neuron_path, repo_root
from mua_core import approach_contact_positions


SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "l5_approach_worker.py"
JOB_TIMEOUT_S = 3600
RESULT_NPZ = "l5_approach_sweep_results.npz"
RESULT_CSV = "l5_approach_sweep_results.csv"
SEED_MODULUS = 2**31 - 1
LAYOUT_NOISE_SEED_NAMESPACE = "l5_approach_sweep_layout_noise_v1"
SEED_FIELD_SEPARATOR = "\x1f"
SEED_ENCODING = (
    "UTF-8 bytes of ASCII Unit Separator (0x1F)-joined fields; "
    "first 8 bytes of SHA-256 digest interpreted as big-endian uint64, "
    "then reduced modulo seed_modulus"
)

# Locked geometry / population (see L5_APPROACH_IMPLEMENTATION_HANDOFF.md §5).
VOXEL_UM = (300.0, 300.0, 300.0)
APPROACH_FACE = "+x"
CONTACT_DISTANCES_UM = [400, 367, 334, 301, 268, 235, 202, 169, 136, 103, 70, 37, 4]

FULL_N_CELLS = 500
FULL_REPEATS = 1

SMOKE_N_CELLS = 10
SMOKE_REPEATS = 1


@dataclass(frozen=True)
class ApproachGrid:
    n_cells: int
    contact_distances_um: list
    repeats: int
    voxel_um: tuple = VOXEL_UM
    approach_face: str = APPROACH_FACE

    @property
    def total_jobs(self):
        return self.repeats


@dataclass(frozen=True)
class Job:
    repeat: int
    seed: int
    result_path: Path
    args_path: Path

    @property
    def job_id(self):
        return f"l5_pop_r{self.repeat:02d}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the human L5 contact-approach sweep (one population job, 13 contact distances)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned job(s) and output location without writing files or running workers.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the 10-cell smoke population (all 13 contact distances, 1 repeat).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help=(
            "Number of independent population realizations (each a distinct "
            "placement + noise seed). Overrides the default (1). Repeats are "
            "separate jobs, so --workers parallelizes them."
        ),
    )
    parser.add_argument(
        "--workers",
        default="1",
        help=(
            "Number of worker subprocesses to run concurrently, or 'auto' to let "
            "the script choose based on CPU count and available RAM. Default: 1."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed per-job result files in an existing output directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/l5_approach_sweep_<timestamp>/.",
    )
    return parser.parse_args()


def selected_grid(smoke):
    if smoke:
        return ApproachGrid(
            n_cells=SMOKE_N_CELLS,
            contact_distances_um=list(CONTACT_DISTANCES_UM),
            repeats=SMOKE_REPEATS,
        )
    return ApproachGrid(
        n_cells=FULL_N_CELLS,
        contact_distances_um=list(CONTACT_DISTANCES_UM),
        repeats=FULL_REPEATS,
    )


def default_run_dir():
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return default_output_dir(create=False) / f"l5_approach_sweep_{stamp}"


def resolve_out_dir(path):
    if path is None:
        return default_run_dir()
    return path.expanduser().resolve()


def validate_static_paths():
    missing = [
        path
        for path in (WORKER_SCRIPT, default_morphology_path())
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required worker/morphology paths are missing: "
            + ", ".join(str(path) for path in missing)
        )


def has_result_artifacts(out_dir):
    if not out_dir.exists():
        return False
    artifacts = [out_dir / "run_info.json", out_dir / RESULT_NPZ, out_dir / RESULT_CSV]
    if any(path.exists() for path in artifacts):
        return True
    jobs_dir = out_dir / "jobs"
    return jobs_dir.exists() and any(jobs_dir.glob("*.pkl"))


def ensure_output_dir(out_dir, resume):
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {out_dir}")
    if out_dir.exists() and has_result_artifacts(out_dir) and not resume:
        raise FileExistsError(
            f"Output directory already contains result artifacts: {out_dir}. "
            "Use --resume to continue completed per-job outputs, or choose a new --out-dir."
        )
    aggregate_npz = out_dir / RESULT_NPZ
    aggregate_csv = out_dir / RESULT_CSV
    if aggregate_npz.exists() or aggregate_csv.exists():
        raise FileExistsError(
            f"Aggregate output already exists in {out_dir}; refusing to overwrite .npz/.csv files."
        )
    (out_dir / "jobs").mkdir(parents=True, exist_ok=True)
    (out_dir / "job_args").mkdir(parents=True, exist_ok=True)


def stable_seed(*parts):
    """Return a process- and machine-stable NumPy seed for structured inputs.

    Mirrors ``run_reference_sweep.stable_seed`` so the two sweeps share one
    seeding convention; never use Python's ``hash()``.
    """
    fields = []
    for part in parts:
        if isinstance(part, str):
            fields.append(part)
        elif isinstance(part, (int, np.integer)):
            fields.append(str(int(part)))
        else:
            raise TypeError(
                "stable_seed parts must be strings or integer indices; "
                f"got {type(part).__name__}"
            )
    payload = SEED_FIELD_SEPARATOR.join(fields).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS


def layout_noise_seed(repeat):
    """Seed volumetric layout + post-hoc noise by repeat index."""
    return stable_seed(LAYOUT_NOISE_SEED_NAMESPACE, int(repeat))


def build_jobs(grid, out_dir):
    jobs = []
    jobs_dir = out_dir / "jobs"
    args_dir = out_dir / "job_args"
    for repeat in range(grid.repeats):
        seed = layout_noise_seed(repeat)
        job_stub = f"l5_pop_r{repeat:02d}"
        jobs.append(
            Job(
                repeat=repeat,
                seed=int(seed),
                result_path=jobs_dir / f"{job_stub}.pkl",
                args_path=args_dir / f"{job_stub}_args.pkl",
            )
        )
    return jobs


def base_worker_args(grid):
    """Copy the reference biophysics/signal block; change ONLY the geometry."""
    dt = 2**-4
    elec_x, elec_y, elec_z, contact_normal = approach_contact_positions(
        grid.contact_distances_um, grid.voxel_um, grid.approach_face
    )
    return {
        "morph_dir": neuron_path(morphology_dir()),
        "morphologies": [neuron_path(default_morphology_path())],
        "v_init": -65,
        "dt": dt,
        "tstart": 0,
        "tstop": 200,
        # --- L5 volumetric geometry (the only departure from the reference) ---
        "voxel_um": tuple(float(axis) for axis in grid.voxel_um),
        "approach_face": grid.approach_face,
        "contact_distances_um": [float(d) for d in grid.contact_distances_um],
        "elec_x": elec_x.tolist(),
        "elec_y": elec_y.tolist(),
        "elec_z": elec_z.tolist(),
        "contact_normal": contact_normal.tolist(),
        # --- biophysics + drive (identical to the reference) ---
        "base_spike_time": 20.0,
        "jitter_std": 0.0,
        "drive_mode": "synapses",
        "syn_height_min": 0.5,
        "syn_height_max": 0.9,
        "n_synapses": 20,
        "syn_type": "Exp2Syn",
        "syn_weight": 0.05,
        "tau1": 0.5,
        "tau2": 2.0,
        "e_syn": 0,
        "synapse_rng_seed": 0,
        # --- electrode + signal processing (identical to the reference) ---
        "sigma": 0.3,
        "method": "linesource",
        "contact_size": 12.0,
        "contact_shape": "square",
        "n_avg_points": 50,
        "fs_hz": 1000.0 / dt,
        "mua_low": 300.0,
        "mua_high": 5000.0,
        "filt_order": 3,
        "noise_rms_uV": 5.0,
        "mua_threshold_factor": 5,
        "refractory_ms": 0.5,
    }


def worker_args_for_job(job, grid):
    args = base_worker_args(grid)
    args.update({"n_cells": int(grid.n_cells), "seed": int(job.seed)})
    return args


def write_pickle_exclusive(path, data):
    with path.open("xb") as handle:
        pickle.dump(data, handle)


def read_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def run_job(job, grid):
    if job.result_path.exists():
        return {"job_id": job.job_id, "status": "skipped", "result_path": str(job.result_path)}

    if not job.args_path.exists():
        write_pickle_exclusive(job.args_path, worker_args_for_job(job, grid))

    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER_SCRIPT), str(job.args_path), str(job.result_path)],
            capture_output=True,
            text=True,
            timeout=JOB_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "job_id": job.job_id,
            "status": "failed",
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timed out after {JOB_TIMEOUT_S}s",
        }
    if proc.returncode != 0:
        return {
            "job_id": job.job_id,
            "status": "failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    if not job.result_path.exists():
        return {
            "job_id": job.job_id,
            "status": "failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": "worker exited successfully but did not write a result file",
        }
    return {"job_id": job.job_id, "status": "completed", "result_path": str(job.result_path)}


def git_value(args):
    try:
        return subprocess.check_output(
            ["git"] + args,
            cwd=repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def module_version(name):
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def geometry_info(grid):
    half_x = float(grid.voxel_um[0]) / 2.0
    return {
        "voxel_um": [float(axis) for axis in grid.voxel_um],
        "voxel_center_um": [0.0, 0.0, 0.0],
        "approach_face": grid.approach_face,
        "approach_axis": "+X (contact travels along X toward the +X face)",
        "apical_axis": "+Y (apical north)",
        "basal_axis": "-Y",
        "face_center_um": [half_x, 0.0, 0.0],
        "contact_distances_um": [float(d) for d in grid.contact_distances_um],
        "distance_definition": "contact-center to approached +X-face center",
        "soma_placement": "uniform-random soma positions in the voxel, centered on the origin",
        "contact_normal": "+x (sign sets only the finite-area contact-disk plane; matches the reference normal)",
        "n_cells": int(grid.n_cells),
        "n_cells_interpretation": (
            "fixed anatomical count of all L5 pyramidal cells in the voxel; "
            "n_cells sweep and f_active active-contributor split deferred"
        ),
        "voxel_interpretation": "cube treated as fully inside layer 5 for the first pass",
        "region": "generic L5 (region dependence deferred)",
    }


def run_info(grid, jobs, out_dir, workers, smoke, resume):
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "sweep": "l5_approach",
        "mode": "smoke" if smoke else "full",
        "out_dir": str(out_dir),
        "workers": int(workers),
        "resume": bool(resume),
        "total_jobs": int(len(jobs)),
        "job_model": "one job = one population/repeat; all 13 contact distances recorded together",
        "geometry": geometry_info(grid),
        "worker_script": str(WORKER_SCRIPT),
        "morphology": str(default_morphology_path()),
        "seed_scheme": (
            "seed = sha256(namespace, repeat) % (2**31 - 1); seeds both volumetric "
            "layout and post-hoc per-contact noise"
        ),
        "seed_namespace": LAYOUT_NOISE_SEED_NAMESPACE,
        "seed_encoding": SEED_ENCODING,
        "seed_modulus": SEED_MODULUS,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "versions": {name: module_version(name) for name in ("numpy", "scipy", "LFPy", "neuron")},
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "short": git_value(["rev-parse", "--short", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(git_value(["status", "--porcelain"])),
        },
    }


def write_run_info_once(out_dir, info, resume):
    path = out_dir / "run_info.json"
    if path.exists():
        if resume:
            return
        raise FileExistsError(f"run_info.json already exists: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
        handle.write("\n")


def write_aggregate_outputs(out_dir, grid, jobs):
    n_r = grid.repeats
    n_d = len(grid.contact_distances_um)
    shape = (n_r, n_d)

    crossings_all = np.full(shape, np.nan)
    detected_all = np.full(shape, np.nan)
    peak_neg_all = np.full(shape, np.nan)
    peak_sbp_all = np.full(shape, np.nan)
    peak_neg_clean_all = np.full(shape, np.nan)
    peak_sbp_clean_all = np.full(shape, np.nan)
    mua_std_all = np.full(shape, np.nan)
    threshold_all = np.full(shape, np.nan)

    csv_rows = []
    for job in jobs:
        result = read_pickle(job.result_path)
        per_contact = result["per_contact"]
        if len(per_contact) != n_d:
            raise ValueError(
                f"{job.job_id}: expected {n_d} contacts, got {len(per_contact)}"
            )
        for distance_index, contact in enumerate(per_contact):
            crossings_all[job.repeat, distance_index] = float(contact["n_crossings"])
            detected_all[job.repeat, distance_index] = 1.0 if contact["detected"] else 0.0
            peak_neg_all[job.repeat, distance_index] = float(contact["peak_neg_uV"])
            peak_sbp_all[job.repeat, distance_index] = float(contact["peak_sbp_uV"])
            peak_neg_clean_all[job.repeat, distance_index] = float(contact["peak_neg_clean_uV"])
            peak_sbp_clean_all[job.repeat, distance_index] = float(contact["peak_sbp_clean_uV"])
            mua_std_all[job.repeat, distance_index] = float(contact["mua_std"])
            threshold_all[job.repeat, distance_index] = float(contact["threshold"])
            csv_rows.append(
                {
                    "job_id": job.job_id,
                    "repeat": job.repeat,
                    "distance_index": distance_index,
                    "distance_um": float(contact["distance_um"]),
                    "n_cells": int(result["n_cells"]),
                    "seed": int(job.seed),
                    "n_crossings": int(contact["n_crossings"]),
                    "detected": bool(contact["detected"]),
                    "peak_neg_uV": float(contact["peak_neg_uV"]),
                    "peak_sbp_uV": float(contact["peak_sbp_uV"]),
                    "peak_neg_clean_uV": float(contact["peak_neg_clean_uV"]),
                    "peak_sbp_clean_uV": float(contact["peak_sbp_clean_uV"]),
                    "mua_std": float(contact["mua_std"]),
                    "threshold": float(contact["threshold"]),
                }
            )

    npz_path = out_dir / RESULT_NPZ
    with npz_path.open("xb") as handle:
        np.savez(
            handle,
            crossings_all=crossings_all,
            detected_all=detected_all,
            peak_neg_all=peak_neg_all,
            peak_sbp_all=peak_sbp_all,
            peak_neg_clean_all=peak_neg_clean_all,
            peak_sbp_clean_all=peak_sbp_clean_all,
            mua_std_all=mua_std_all,
            threshold_all=threshold_all,
            sweep_distances=np.asarray(grid.contact_distances_um, dtype=float),
            sweep_n_cells=int(grid.n_cells),
            sweep_repeats=int(grid.repeats),
            voxel_um=np.asarray(grid.voxel_um, dtype=float),
        )

    csv_path = out_dir / RESULT_CSV
    fieldnames = [
        "job_id",
        "repeat",
        "distance_index",
        "distance_um",
        "n_cells",
        "seed",
        "n_crossings",
        "detected",
        "peak_neg_uV",
        "peak_sbp_uV",
        "peak_neg_clean_uV",
        "peak_sbp_clean_uV",
        "mua_std",
        "threshold",
    ]
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    return npz_path, csv_path


def print_dry_run(grid, jobs, out_dir, workers, smoke):
    print("L5 approach sweep dry run")
    print(f"mode: {'smoke' if smoke else 'full'}")
    print(f"out_dir: {out_dir}")
    print(f"worker_script: {WORKER_SCRIPT}")
    print(f"workers: {workers}")
    print(f"total_jobs: {len(jobs)}")
    print(f"n_cells: {grid.n_cells}")
    print(f"voxel_um: {' x '.join(f'{axis:g}' for axis in grid.voxel_um)}")
    print(f"approach_face: {grid.approach_face}")
    print(f"contact_distances_um ({len(grid.contact_distances_um)}): {grid.contact_distances_um}")
    print(f"repeats: {grid.repeats}")
    print("job_model: one job = one population/repeat with all contact distances")
    print("planned_jobs:")
    for job in jobs:
        print(
            f"  {job.job_id}: n_cells={grid.n_cells}, "
            f"n_contact_distances={len(grid.contact_distances_um)}, "
            f"repeat={job.repeat}, seed={job.seed}"
        )


_RAM_PER_JOB_GB = 4.0  # conservative estimate for a 500-cell, 200 ms population sim


def auto_workers():
    """Return a worker count based on CPU cores and available RAM."""
    cpu_cores = os.cpu_count() or 1
    available_gb = None
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / 1024 ** 3
        ram_limit = max(1, int(available_gb / _RAM_PER_JOB_GB))
    except ImportError:
        ram_limit = cpu_cores  # no psutil — fall back to CPU count

    recommended = min(cpu_cores, ram_limit)
    ram_str = f"{available_gb:.1f} GB RAM available, " if available_gb is not None else ""
    print(
        f"auto-workers: {cpu_cores} CPU core(s), {ram_str}"
        f"{_RAM_PER_JOB_GB} GB/job estimate → {recommended} worker(s)"
    )
    return recommended


def resolve_workers(raw):
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return auto_workers()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"--workers must be a positive integer or 'auto', got: {raw!r}")
    if value < 1:
        raise ValueError(f"--workers must be >= 1, got {value}")
    return value


def run_jobs(jobs, grid, workers):
    failures = []
    completed = 0
    skipped = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(run_job, job, grid): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            result = future.result()
            status = result["status"]
            if status == "completed":
                completed += 1
            elif status == "skipped":
                skipped += 1
            else:
                failures.append(result)
            done = completed + skipped + len(failures)
            print(f"[{done}/{len(jobs)}] {result['job_id']} {status}", flush=True)

    if failures:
        first = failures[0]
        stderr_tail = "\n".join((first.get("stderr") or "").splitlines()[-10:])
        raise RuntimeError(
            f"{len(failures)} job(s) failed; first failed job {first['job_id']} "
            f"returned {first.get('returncode')}.\n{stderr_tail}"
        )
    return completed, skipped


def main():
    args = parse_args()

    validate_static_paths()
    grid = selected_grid(args.smoke)
    if args.repeats is not None:
        if args.repeats < 1:
            raise ValueError(f"--repeats must be >= 1, got {args.repeats}")
        grid = replace(grid, repeats=args.repeats)
    workers = resolve_workers(args.workers)
    out_dir = resolve_out_dir(args.out_dir)
    jobs = build_jobs(grid, out_dir)

    if args.dry_run:
        print_dry_run(grid, jobs, out_dir, workers, args.smoke)
        return 0

    ensure_output_dir(out_dir, args.resume)
    info = run_info(grid, jobs, out_dir, workers, args.smoke, args.resume)
    write_run_info_once(out_dir, info, args.resume)

    completed, skipped = run_jobs(jobs, grid, workers)
    missing = [job.result_path for job in jobs if not job.result_path.exists()]
    if missing:
        raise RuntimeError(
            "Cannot aggregate because result files are missing: "
            + ", ".join(str(path) for path in missing[:5])
        )
    npz_path, csv_path = write_aggregate_outputs(out_dir, grid, jobs)
    print(f"completed_jobs: {completed}")
    print(f"skipped_jobs: {skipped}")
    print(f"aggregate_npz: {npz_path}")
    print(f"aggregate_csv: {csv_path}")
    print(f"run_info: {out_dir / 'run_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
