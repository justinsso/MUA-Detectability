"""Render the L5 approach geometry in 3D: voxel, population, and contacts.

Draws the same arrangement the L5 approach sweep simulates — a 300³ µm voxel
filled with volumetrically-placed, apical-north L5 pyramidal cells, and the 13
contact positions marching along +X toward the voxel face. By seeding with a
chosen repeat's layout seed, the rendered population matches that run exactly.

Unlike ``plot_l5_approach_sweep.py`` (load-only), this script *builds* a cell to
read its morphology geometry, so it imports LFPy/NEURON at run time. It does not
run any simulation.

Usage::

    python plot_l5_geometry.py [--repeat 0] [--n-cells 500] [--morph-cells 20]
                               [--out PATH] [--elev 18] [--azim -60]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from mua_config import default_figure_dir, default_morphology_path, morphology_dir, neuron_path
from mua_core import (
    apical_north_pose,
    approach_contact_positions,
    make_reference_cell,
    neuron_working_directory,
    set_reference_pose,
    volumetric_cell_positions,
)
from run_l5_approach_sweep import (
    APPROACH_FACE,
    CONTACT_DISTANCES_UM,
    FULL_N_CELLS,
    VOXEL_UM,
    layout_noise_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the L5 approach geometry (voxel + population + contacts) in 3D."
    )
    parser.add_argument("--repeat", type=int, default=0,
                        help="Repeat index whose layout seed is reproduced. Default: 0.")
    parser.add_argument("--n-cells", type=int, default=FULL_N_CELLS,
                        help=f"Population size to place (soma dots). Default: {FULL_N_CELLS}.")
    parser.add_argument("--morph-cells", type=int, default=20,
                        help="How many cells to draw with full morphology. Default: 20.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path. Default: <figures>/l5_geometry_3d.png.")
    parser.add_argument("--elev", type=float, default=18.0, help="3D view elevation.")
    parser.add_argument("--azim", type=float, default=-60.0, help="3D view azimuth.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing PNG.")
    return parser.parse_args()


def oriented_segments(cell):
    """Return ``(segments, linewidths)`` for an apical-north cell at the origin.

    ``segments`` is an ``(n_seg, 2, 3)`` array of start/end points (soma at the
    origin); ``linewidths`` scales with compartment diameter.
    """
    starts = np.column_stack([cell.x[:, 0], cell.y[:, 0], cell.z[:, 0]])
    ends = np.column_stack([cell.x[:, 1], cell.y[:, 1], cell.z[:, 1]])
    segments = np.stack([starts, ends], axis=1)
    linewidths = np.clip(cell.d / 4.0, 0.4, 3.0)
    return segments, linewidths


def cube_edges(voxel_um):
    """Return the 12 edges of the origin-centered voxel as (2, 3) point pairs."""
    hx, hy, hz = (np.asarray(voxel_um, dtype=float) / 2.0)
    corners = np.array([[sx * hx, sy * hy, sz * hz]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    edges = []
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            # corners share an edge when they differ in exactly one coordinate sign
            if np.sum(~np.isclose(corners[i], corners[j])) == 1:
                edges.append([corners[i], corners[j]])
    return edges


def contact_square(center, normal, size):
    """Vertices of a square contact face centered at ``center`` with ``normal``."""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    half = size / 2.0
    return [center + a * u + b * v for a, b in
            [(-half, -half), (half, -half), (half, half), (-half, half)]]


def build_scene_cell():
    """Build one apical-north cell at the origin and return its oriented geometry."""
    with neuron_working_directory(morphology_dir()):
        cell = make_reference_cell(neuron_path(default_morphology_path()))
        rot_x, rot_y, rot_z = apical_north_pose(cell)
        set_reference_pose(cell, x=0.0, y=0.0, z=0.0, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z)
        return oriented_segments(cell)


def make_figure(args):
    voxel_um = VOXEL_UM
    seed = layout_noise_seed(args.repeat)
    np.random.seed(int(seed))
    positions = volumetric_cell_positions(args.n_cells, voxel_um)

    segments, linewidths = build_scene_cell()
    elec_x, elec_y, elec_z, contact_normal = approach_contact_positions(
        CONTACT_DISTANCES_UM, voxel_um, APPROACH_FACE
    )

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # All soma positions first (volumetric fill / density), so morphology sits on top.
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               s=7, c="0.45", alpha=0.55, label=f"{args.n_cells} somas")

    # Full morphology for a readable subset, each translated to its soma.
    n_morph = min(args.morph_cells, args.n_cells)
    cmap = plt.cm.tab10
    thin = np.clip(linewidths * 0.7, 0.3, 1.8)
    for i in range(n_morph):
        translated = segments + positions[i][None, None, :]
        ax.add_collection3d(Line3DCollection(translated, colors=[cmap(i % 10)],
                                             linewidths=thin, alpha=0.5))

    # Voxel wireframe (bold, on top of the soma cloud).
    ax.add_collection3d(Line3DCollection(cube_edges(voxel_um), colors="black",
                                         linewidths=1.8, zorder=10))

    # The 13 contacts marching along +X toward the voxel face.
    for ex, ey, ez in zip(elec_x, elec_y, elec_z):
        verts = contact_square(np.array([ex, ey, ez]), contact_normal[0], 12.0)
        ax.add_collection3d(Poly3DCollection([verts], facecolor="red",
                                             edgecolor="darkred", alpha=0.95, zorder=12))
    ax.plot(elec_x, elec_y, elec_z, color="red", linewidth=1.2, linestyle=":", zorder=11)
    ax.scatter(elec_x, elec_y, elec_z, c="red", marker="s", s=22,
               label=f"{len(elec_x)} contacts (+X approach)")

    # Axis cue: apical north (+Y).
    ax.quiver(0, 0, 0, 0, 220, 0, color="navy", linewidth=2.5, arrow_length_ratio=0.08, zorder=13)
    ax.text(0, 245, 0, "apical north (+Y)", color="navy", fontsize=10, weight="bold")

    # Frame on the voxel + approach path; let apical tufts clip rather than dwarf it.
    half = np.asarray(voxel_um) / 2.0
    xlim = (-half[0] - 60, float(elec_x.max()) + 30)
    ylim = (-half[1] - 50, half[1] + 200)
    zlim = (-half[2] - 50, half[2] + 50)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    try:
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    except Exception:
        pass

    ax.set_xlabel("X (µm)  — approach axis")
    ax.set_ylabel("Y (µm)  — apical/basal")
    ax.set_zlabel("Z (µm)")
    ax.set_title(
        f"L5 approach geometry — {args.n_cells} cells in {int(voxel_um[0])}³ µm voxel "
        f"(morphology shown for {n_morph}), repeat {args.repeat}"
    )
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.legend(loc="upper left")
    return fig


def main():
    args = parse_args()
    out = args.out.expanduser().resolve() if args.out else default_figure_dir(create=True) / "l5_geometry_3d.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"Figure already exists, refusing to overwrite: {out}")
    fig = make_figure(args)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"created: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
