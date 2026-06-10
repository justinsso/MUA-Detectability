"""Run the reference MUA detectability sweep via isolated worker subprocesses.

This runner coordinates jobs only. The NEURON/LFPy work stays in
``sweep_worker.py`` so each population/repeat runs in a fresh Python process.
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mua_config import default_morphology_path, default_output_dir, morphology_dir, neuron_path, repo_root


SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SCRIPT_DIR / "Reference"
WORKER_SCRIPT = SCRIPT_DIR / "sweep_worker.py"
JOB_TIMEOUT_S = 600
RESULT_NPZ = "reference_sweep_results.npz"
RESULT_CSV = "reference_sweep_results.csv"
SEED_MODULUS = 2**31 - 1
LAYOUT_NOISE_SEED_NAMESPACE = "reference_sweep_layout_noise_v1"
SEED_FIELD_SEPARATOR = "\x1f"
SEED_ENCODING = (
    "UTF-8 bytes of ASCII Unit Separator (0x1F)-joined fields; "
    "first 8 bytes of SHA-256 digest interpreted as big-endian uint64, "
    "then reduced modulo seed_modulus"
)


FULL_N_CELLS = [1, 5, 10, 25, 50, 100]
FULL_DISTANCES = [25, 50, 75, 100, 150, 200]
FULL_JITTERS = [0.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0]
FULL_REPEATS = 3

SMOKE_N_CELLS = [1, 5]
SMOKE_DISTANCES = [25, 50]
SMOKE_JITTERS = [0.0, 5.0]
SMOKE_REPEATS = 1


@dataclass(frozen=True)
class SweepGrid:
    n_cells: list
    distances: list
    jitters: list
    repeats: int

    @property
    def total_jobs(self):
        return len(self.n_cells) * len(self.distances) * len(self.jitters) * self.repeats


@dataclass(frozen=True)
class Job:
    jitter_index: int
    distance_index: int
    n_cell_index: int
    repeat: int
    jitter: float
    distance: float
    n_cells: int
    seed: int
    result_path: Path
    args_path: Path

    @property
    def job_id(self):
        return (
            f"j{self.jitter_index:02d}_d{self.distance_index:02d}_"
            f"n{self.n_cell_index:02d}_r{self.repeat:02d}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the reference n_cells x distance x jitter sweep."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned jobs and output location without writing files or running workers.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the tiny smoke grid: n_cells 1,5; distances 25,50; jitters 0,5; repeats 1.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker subprocesses to run concurrently. Default: 1.",
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
        help="Output directory. Defaults to outputs/reference_sweep_<timestamp>/.",
    )
    return parser.parse_args()


def selected_grid(smoke):
    if smoke:
        return SweepGrid(
            n_cells=SMOKE_N_CELLS,
            distances=SMOKE_DISTANCES,
            jitters=SMOKE_JITTERS,
            repeats=SMOKE_REPEATS,
        )
    return SweepGrid(
        n_cells=FULL_N_CELLS,
        distances=FULL_DISTANCES,
        jitters=FULL_JITTERS,
        repeats=FULL_REPEATS,
    )


def default_run_dir():
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return default_output_dir(create=False) / f"reference_sweep_{stamp}"


def resolve_out_dir(path):
    if path is None:
        return default_run_dir()
    return path.expanduser().resolve()


def validate_static_paths():
    missing = [
        path
        for path in (
            WORKER_SCRIPT,
            REFERENCE_DIR,
            REFERENCE_DIR / "lfpy_MUA_simulation.py",
            REFERENCE_DIR / "sweep_worker.py",
            default_morphology_path(),
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required reference/simulation paths are missing: "
            + ", ".join(str(path) for path in missing)
        )


def has_result_artifacts(out_dir):
    if not out_dir.exists():
        return False
    artifacts = [
        out_dir / "run_info.json",
        out_dir / RESULT_NPZ,
        out_dir / RESULT_CSV,
    ]
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
    """Return a process- and machine-stable NumPy seed for structured inputs."""
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


def layout_noise_seed(distance_index, n_cell_index, repeat):
    """Seed layout/noise by distance, cell-count, and repeat; jitter is excluded."""
    return stable_seed(
        LAYOUT_NOISE_SEED_NAMESPACE,
        int(distance_index),
        int(n_cell_index),
        int(repeat),
    )


def build_jobs(grid, out_dir):
    jobs = []
    jobs_dir = out_dir / "jobs"
    args_dir = out_dir / "job_args"
    for jitter_index, jitter in enumerate(grid.jitters):
        for distance_index, distance in enumerate(grid.distances):
            for n_cell_index, n_cells in enumerate(grid.n_cells):
                for repeat in range(grid.repeats):
                    seed = layout_noise_seed(distance_index, n_cell_index, repeat)
                    job_stub = (
                        f"j{jitter_index:02d}_d{distance_index:02d}_"
                        f"n{n_cell_index:02d}_r{repeat:02d}"
                    )
                    jobs.append(
                        Job(
                            jitter_index=jitter_index,
                            distance_index=distance_index,
                            n_cell_index=n_cell_index,
                            repeat=repeat,
                            jitter=float(jitter),
                            distance=float(distance),
                            n_cells=int(n_cells),
                            seed=int(seed),
                            result_path=jobs_dir / f"{job_stub}.pkl",
                            args_path=args_dir / f"{job_stub}_args.pkl",
                        )
                    )
    return jobs


def base_worker_args():
    dt = 2**-4
    morph_dir = neuron_path(morphology_dir())
    return {
        "morph_dir": morph_dir,
        "morphologies": [neuron_path(default_morphology_path())],
        "v_init": -65,
        "dt": dt,
        "tstart": 0,
        "tstop": 200,
        "inner_radius": 50,
        "align_cells": True,
        "align_rot_x": 0,
        "align_rot_y": 0,
        "align_rot_z": 0,
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
        "iclamp_amp": 10,
        "iclamp_dur": 0.1,
        "elec_x": [0.0],
        "elec_y": [0.0],
        "elec_z": [0.0],
        "sigma": 0.3,
        "method": "linesource",
        "contact_size": 12.0,
        "contact_shape": "square",
        "contact_normal": [[1.0, 0.0, 0.0]],
        "n_avg_points": 50,
        "fs_hz": 1000.0 / dt,
        "mua_low": 300.0,
        "mua_high": 5000.0,
        "filt_order": 3,
        "noise_rms_uV": 5.0,
        "mua_threshold_factor": 5,
        "refractory_ms": 0.5,
    }


def worker_args_for_job(job):
    args = base_worker_args()
    args.update(
        {
            "n_cells": job.n_cells,
            "cell_distance": job.distance,
            "jitter_std": job.jitter,
            "seed": job.seed,
        }
    )
    return args


def write_pickle_exclusive(path, data):
    with path.open("xb") as handle:
        pickle.dump(data, handle)


def read_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def run_job(job):
    if job.result_path.exists():
        return {"job_id": job.job_id, "status": "skipped", "result_path": str(job.result_path)}

    if not job.args_path.exists():
        write_pickle_exclusive(job.args_path, worker_args_for_job(job))

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


def run_info(grid, jobs, out_dir, workers, smoke, resume):
    return {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "mode": "smoke" if smoke else "full",
        "out_dir": str(out_dir),
        "workers": int(workers),
        "resume": bool(resume),
        "total_jobs": int(len(jobs)),
        "grid": {
            "n_cells": [int(value) for value in grid.n_cells],
            "distances": [float(value) for value in grid.distances],
            "jitters": [float(value) for value in grid.jitters],
            "repeats": int(grid.repeats),
        },
        "worker_script": str(WORKER_SCRIPT),
        "reference_dir": str(REFERENCE_DIR),
        "reference_snapshot": {
            "lfpy_MUA_simulation": str(REFERENCE_DIR / "lfpy_MUA_simulation.py"),
            "sweep_worker": str(REFERENCE_DIR / "sweep_worker.py"),
        },
        "active_source": str(SCRIPT_DIR / "lfpy_MUA_simulation.py"),
        "seed_scheme": (
            "stable Phase 4 scheme: seed = sha256(namespace, distance_index, "
            "n_cell_index, repeat) % (2**31 - 1); jitter is excluded for paired "
            "jitter comparisons"
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


def load_result_row(job):
    result = read_pickle(job.result_path)
    row = {
        "job_id": job.job_id,
        "jitter_index": job.jitter_index,
        "distance_index": job.distance_index,
        "n_cell_index": job.n_cell_index,
        "repeat": job.repeat,
        "jitter_std": job.jitter,
        "cell_distance": job.distance,
        "n_cells": job.n_cells,
        "seed": job.seed,
    }
    row.update(result)
    return row


def write_aggregate_outputs(out_dir, grid, jobs):
    rows = [load_result_row(job) for job in jobs]
    n_j = len(grid.jitters)
    n_d = len(grid.distances)
    n_n = len(grid.n_cells)
    shape = (n_j, n_d, n_n, grid.repeats)

    crossings_all = np.full(shape, np.nan)
    detected_all = np.full(shape, np.nan)
    peak_neg_all = np.full(shape, np.nan)
    peak_sbp_all = np.full(shape, np.nan)
    mua_std_all = np.full(shape, np.nan)
    threshold_all = np.full(shape, np.nan)

    for row in rows:
        idx = (
            int(row["jitter_index"]),
            int(row["distance_index"]),
            int(row["n_cell_index"]),
            int(row["repeat"]),
        )
        crossings_all[idx] = float(row["n_crossings"])
        detected_all[idx] = 1.0 if row["detected"] else 0.0
        peak_neg_all[idx] = float(row["peak_neg_uV"])
        peak_sbp_all[idx] = float(row["peak_sbp_uV"])
        mua_std_all[idx] = float(row["mua_std"])
        threshold_all[idx] = float(row["threshold"])

    npz_path = out_dir / RESULT_NPZ
    with npz_path.open("xb") as handle:
        np.savez(
            handle,
            crossings_all=crossings_all,
            detected_all=detected_all,
            peak_neg_all=peak_neg_all,
            peak_sbp_all=peak_sbp_all,
            mua_std_all=mua_std_all,
            threshold_all=threshold_all,
            sweep_n_cells=np.asarray(grid.n_cells),
            sweep_distances=np.asarray(grid.distances),
            sweep_jitters=np.asarray(grid.jitters),
            sweep_repeats=int(grid.repeats),
        )

    csv_path = out_dir / RESULT_CSV
    fieldnames = [
        "job_id",
        "jitter_index",
        "distance_index",
        "n_cell_index",
        "repeat",
        "jitter_std",
        "cell_distance",
        "n_cells",
        "seed",
        "n_crossings",
        "detected",
        "peak_neg_uV",
        "peak_sbp_uV",
        "mua_std",
        "threshold",
    ]
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})

    return npz_path, csv_path


def print_dry_run(grid, jobs, out_dir, workers, smoke):
    print("Reference sweep dry run")
    print(f"mode: {'smoke' if smoke else 'full'}")
    print(f"out_dir: {out_dir}")
    print(f"worker_script: {WORKER_SCRIPT}")
    print(f"reference_dir: {REFERENCE_DIR}")
    print(f"workers: {workers}")
    print(f"total_jobs: {len(jobs)}")
    print(f"n_cells: {grid.n_cells}")
    print(f"distances: {grid.distances}")
    print(f"jitters: {grid.jitters}")
    print(f"repeats: {grid.repeats}")
    print("first_jobs:")
    for job in jobs[: min(5, len(jobs))]:
        print(
            f"  {job.job_id}: n_cells={job.n_cells}, distance={job.distance:g}, "
            f"jitter={job.jitter:g}, repeat={job.repeat}, seed={job.seed}"
        )
    if len(jobs) > 5:
        print(f"  ... {len(jobs) - 5} more")


def run_jobs(jobs, workers):
    failures = []
    completed = 0
    skipped = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(run_job, job): job for job in jobs}
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
            print(
                f"[{done}/{len(jobs)}] {result['job_id']} {status}",
                flush=True,
            )

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
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    validate_static_paths()
    grid = selected_grid(args.smoke)
    out_dir = resolve_out_dir(args.out_dir)
    jobs = build_jobs(grid, out_dir)

    if args.dry_run:
        print_dry_run(grid, jobs, out_dir, args.workers, args.smoke)
        return 0

    ensure_output_dir(out_dir, args.resume)
    info = run_info(grid, jobs, out_dir, args.workers, args.smoke, args.resume)
    write_run_info_once(out_dir, info, args.resume)

    completed, skipped = run_jobs(jobs, args.workers)
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
