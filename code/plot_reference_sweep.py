"""Plot saved reference sweep outputs without running simulations.

This script reads an aggregate ``reference_sweep_results.npz`` produced by
``run_reference_sweep.py`` and writes static figures. It intentionally does not
import LFPy or NEURON.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_NPZ_NAME = "reference_sweep_results.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot saved reference sweep aggregate outputs."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a sweep run directory or a reference_sweep_results.npz file.",
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
    required = {"crossings_all", "sweep_n_cells", "sweep_distances", "sweep_jitters"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"Missing required arrays in {npz_path}: {', '.join(missing)}")
    return data


def mean_over_repeats(array):
    values = np.asarray(array, dtype=float)
    if values.ndim == 4:
        return values.mean(axis=-1)
    if values.ndim == 3:
        return values
    raise ValueError(f"Expected metric shape (jitter, distance, n_cells[, repeat]), got {values.shape}")


def detection_probability(data):
    if "detected_all" in data.files:
        return mean_over_repeats(data["detected_all"])
    return mean_over_repeats(data["crossings_all"] > 0)


def ensure_fig_dir(fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_figure(fig, path, overwrite=False):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Figure already exists, refusing to overwrite: {path}")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def text_color(value, vmin, vmax):
    midpoint = vmin + (vmax - vmin) / 2.0
    return "white" if value < midpoint else "black"


def plot_jitter_grid(data, n_cells, distances, jitters, title, colorbar_label, value_format):
    metric = np.asarray(data, dtype=float)
    if metric.ndim != 3:
        raise ValueError(f"Expected metric shape (jitter, distance, n_cells), got {metric.shape}")

    n_jitters = metric.shape[0]
    width = max(4.0 * n_jitters, 5.0)
    fig, axes = plt.subplots(1, n_jitters, figsize=(width, 4.8), squeeze=False)
    axes = axes[0]

    finite = metric[np.isfinite(metric)]
    vmin = 0.0
    vmax = max(float(finite.max()) if finite.size else 0.0, 1e-9)

    last_image = None
    for jitter_index, jitter in enumerate(jitters):
        ax = axes[jitter_index]
        panel = metric[jitter_index]
        last_image = ax.imshow(panel, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"jitter = {float(jitter):g} ms")
        ax.set_xlabel("# cells")
        ax.set_xticks(np.arange(len(n_cells)))
        ax.set_xticklabels([str(int(value)) for value in n_cells])
        ax.set_yticks(np.arange(len(distances)))
        ax.set_yticklabels([f"{float(value):g}" for value in distances])
        if jitter_index == 0:
            ax.set_ylabel("Distance from electrode (um)")

        for distance_index in range(panel.shape[0]):
            for n_cell_index in range(panel.shape[1]):
                value = panel[distance_index, n_cell_index]
                label = "nan" if not np.isfinite(value) else value_format.format(value)
                ax.text(
                    n_cell_index,
                    distance_index,
                    label,
                    ha="center",
                    va="center",
                    color=text_color(value if np.isfinite(value) else 0.0, vmin, vmax),
                    fontsize=8,
                    fontweight="bold",
                )

    fig.suptitle(title)
    fig.subplots_adjust(right=0.90, wspace=0.35)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.62])
    colorbar = fig.colorbar(last_image, cax=cbar_ax)
    colorbar.set_label(colorbar_label)
    return fig


def make_plots(data, fig_dir, overwrite=False):
    n_cells = data["sweep_n_cells"]
    distances = data["sweep_distances"]
    jitters = data["sweep_jitters"]

    created = []
    crossings = mean_over_repeats(data["crossings_all"])
    fig = plot_jitter_grid(
        crossings,
        n_cells,
        distances,
        jitters,
        "MUA threshold crossings",
        "# threshold crossings (mean)",
        "{:.1f}",
    )
    created.append(save_figure(fig, fig_dir / "threshold_crossings_heatmap.png", overwrite=overwrite))

    det_prob = detection_probability(data)
    fig = plot_jitter_grid(
        det_prob,
        n_cells,
        distances,
        jitters,
        "MUA detection probability",
        "P(detected)",
        "{:.2f}",
    )
    created.append(save_figure(fig, fig_dir / "detection_probability_heatmap.png", overwrite=overwrite))

    if "peak_sbp_all" in data.files:
        peak_sbp = mean_over_repeats(data["peak_sbp_all"])
        fig = plot_jitter_grid(
            peak_sbp,
            n_cells,
            distances,
            jitters,
            "Peak spiking band power",
            "Peak SBP (uV, mean)",
            "{:.1f}",
        )
        created.append(save_figure(fig, fig_dir / "peak_sbp_heatmap.png", overwrite=overwrite))

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
