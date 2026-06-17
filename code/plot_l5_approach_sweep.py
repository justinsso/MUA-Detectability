"""Plot saved L5 approach sweep outputs without running simulations.

Reads an aggregate ``l5_approach_sweep_results.npz`` produced by
``run_l5_approach_sweep.py`` and writes detection-probability and peak-amplitude
figures versus contact distance. It intentionally does not import LFPy or NEURON.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_NPZ_NAME = "l5_approach_sweep_results.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot saved L5 approach sweep aggregate outputs (vs contact distance)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a sweep run directory or an l5_approach_sweep_results.npz file.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Figure output directory. Defaults to <run_dir>/figures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing figure files.",
    )
    return parser.parse_args()


def resolve_input(path):
    path = path.expanduser().resolve()
    if path.is_dir():
        npz_path = path / DEFAULT_NPZ_NAME
        run_dir = path
    else:
        npz_path = path
        run_dir = path.parent

    if not npz_path.is_file():
        raise FileNotFoundError(f"Saved sweep aggregate not found: {npz_path}")
    if npz_path.suffix.lower() != ".npz":
        raise ValueError(f"Expected a .npz file or run directory, got: {npz_path}")
    return run_dir, npz_path


def load_npz(npz_path):
    data = np.load(npz_path)
    required = {"sweep_distances", "peak_neg_all", "detected_all"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"Missing required arrays in {npz_path}: {', '.join(missing)}")
    return data


def mean_sem_over_repeats(array):
    """Return (mean, sem) over the repeat axis for a ``(repeats, distances)`` array."""
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        return values, np.zeros_like(values)
    if values.ndim != 2:
        raise ValueError(f"Expected metric shape (repeats, distances), got {values.shape}")
    mean = np.nanmean(values, axis=0)
    n = np.sum(np.isfinite(values), axis=0)
    std = np.nanstd(values, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sem = np.where(n > 1, std / np.sqrt(n), 0.0)
    return mean, sem


def sorted_by_distance(distances, *arrays):
    order = np.argsort(distances)
    return (np.asarray(distances)[order], *(np.asarray(a)[order] for a in arrays))


def ensure_fig_dir(fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_figure(fig, path, overwrite=False):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Figure already exists, refusing to overwrite: {path}")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_detection_probability(distances, detected_all, n_cells):
    prob, sem = mean_sem_over_repeats(detected_all)
    dist, prob, sem = sorted_by_distance(distances, prob, sem)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(dist, prob, yerr=sem, marker="o", color="C0", capsize=3)
    ax.set_xlabel("Contact distance to voxel face (µm)")
    ax.set_ylabel("P(detected)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"MUA detection probability vs contact distance (n_cells = {n_cells})")
    ax.grid(True, alpha=0.3)
    return fig


def plot_peak_amplitude(distances, signal_peak_all, signal_label, noisy_peak_all, threshold_all, n_cells):
    peak, sem = mean_sem_over_repeats(signal_peak_all)
    dist, peak, sem = sorted_by_distance(distances, peak, sem)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(dist, peak, yerr=sem, marker="o", color="C3", capsize=3, label=signal_label)
    if noisy_peak_all is not None:
        noisy, _ = mean_sem_over_repeats(noisy_peak_all)
        _, noisy = sorted_by_distance(distances, noisy)
        ax.plot(dist, noisy, marker=".", color="C0", alpha=0.6, label="peak negative |µV| (with noise)")
    if threshold_all is not None:
        thr, _ = mean_sem_over_repeats(threshold_all)
        _, thr = sorted_by_distance(distances, thr)
        ax.plot(dist, thr, linestyle="--", color="gray", label="MAD detection threshold")
    ax.set_xlabel("Contact distance to voxel face (µm)")
    ax.set_ylabel("Peak negative amplitude (µV)")
    ax.set_title(f"Peak MUA amplitude vs contact distance (n_cells = {n_cells})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def make_plots(data, fig_dir, overwrite=False):
    distances = np.asarray(data["sweep_distances"], dtype=float)
    n_cells = int(data["sweep_n_cells"]) if "sweep_n_cells" in data.files else "?"
    threshold_all = data["threshold_all"] if "threshold_all" in data.files else None

    created = []
    fig = plot_detection_probability(distances, data["detected_all"], n_cells)
    created.append(save_figure(fig, fig_dir / "detection_probability_vs_distance.png", overwrite=overwrite))

    # Prefer the noise-free signal peak (true amplitude-vs-distance falloff); the
    # noisy peak is overlaid for context and can dominate when the signal is weak.
    if "peak_neg_clean_all" in data.files:
        signal_peak_all = data["peak_neg_clean_all"]
        signal_label = "peak negative |µV| (clean signal)"
        noisy_peak_all = data["peak_neg_all"] if "peak_neg_all" in data.files else None
    else:
        signal_peak_all = data["peak_neg_all"]
        signal_label = "peak negative |µV|"
        noisy_peak_all = None
    fig = plot_peak_amplitude(
        distances, signal_peak_all, signal_label, noisy_peak_all, threshold_all, n_cells
    )
    created.append(save_figure(fig, fig_dir / "peak_amplitude_vs_distance.png", overwrite=overwrite))

    return created


def main():
    args = parse_args()
    run_dir, npz_path = resolve_input(args.input)
    fig_dir = args.fig_dir.expanduser().resolve() if args.fig_dir else run_dir / "figures"
    data = load_npz(npz_path)
    ensure_fig_dir(fig_dir)
    created = make_plots(data, fig_dir, overwrite=args.overwrite)
    print(f"input_npz: {npz_path}")
    for path in created:
        print(f"created: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
