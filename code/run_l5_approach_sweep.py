"""Plan the future human L5 contact-approach sweep.

This is intentionally a scaffold only. It does not import NEURON, LFPy, or any
simulation code, and it does not run sweeps.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

PLANNED_POPULATION = "human L5 pyramidal cells"
PLANNED_VOXEL_UM = (300, 300, 300)
PLANNED_N_CELLS = 500
PLANNED_ORIENTATION = "apical north, basal south"
PLANNED_CONTACT_DISTANCES_UM = [
    400,
    367,
    334,
    301,
    268,
    235,
    202,
    169,
    136,
    103,
    70,
    37,
    4,
]
PLANNED_REPEATS = 1

SMOKE_N_CELLS = 10
SMOKE_CONTACT_DISTANCES_UM = [400, 202, 4]
SMOKE_REPEATS = 1


@dataclass(frozen=True)
class ApproachGeometry:
    population: str
    voxel_um: tuple[int, int, int]
    n_cells: int
    orientation: str
    contact_distances_um: list[int]
    repeats: int

    @property
    def total_jobs(self) -> int:
        return self.repeats


@dataclass(frozen=True)
class PlannedJob:
    repeat: int
    population: str
    voxel_um: tuple[int, int, int]
    n_cells: int
    orientation: str
    contact_distances_um: list[int]

    @property
    def job_id(self) -> str:
        return f"l5_population_r{self.repeat:02d}"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan the future human L5 contact-approach sweep scaffold."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs and geometry without writing files or running simulations.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use the first tiny L5 smoke plan: 10 cells, contact distances "
            "400/202/4 um, 1 repeat."
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        help="Planned worker count for future execution. Default: 1.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Planned output directory. Defaults to outputs/l5_approach_sweep_<timestamp>/.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / f"l5_approach_sweep_{stamp}"


def resolve_out_dir(path: Path | None) -> Path:
    if path is None:
        return default_run_dir()
    return path.expanduser().resolve()


def selected_geometry(smoke: bool) -> ApproachGeometry:
    if smoke:
        return ApproachGeometry(
            population=PLANNED_POPULATION,
            voxel_um=PLANNED_VOXEL_UM,
            n_cells=SMOKE_N_CELLS,
            orientation=PLANNED_ORIENTATION,
            contact_distances_um=SMOKE_CONTACT_DISTANCES_UM,
            repeats=SMOKE_REPEATS,
        )
    return ApproachGeometry(
        population=PLANNED_POPULATION,
        voxel_um=PLANNED_VOXEL_UM,
        n_cells=PLANNED_N_CELLS,
        orientation=PLANNED_ORIENTATION,
        contact_distances_um=PLANNED_CONTACT_DISTANCES_UM,
        repeats=PLANNED_REPEATS,
    )


def build_jobs(geometry: ApproachGeometry) -> list[PlannedJob]:
    return [
        PlannedJob(
            repeat=repeat,
            population=geometry.population,
            voxel_um=geometry.voxel_um,
            n_cells=geometry.n_cells,
            orientation=geometry.orientation,
            contact_distances_um=geometry.contact_distances_um,
        )
        for repeat in range(geometry.repeats)
    ]


def format_voxel(voxel_um: tuple[int, int, int]) -> str:
    return " x ".join(str(axis_um) for axis_um in voxel_um)


def print_dry_run(
    geometry: ApproachGeometry,
    jobs: list[PlannedJob],
    out_dir: Path,
    workers: int,
    smoke: bool,
) -> None:
    print("L5 approach sweep dry run")
    print(f"mode: {'smoke' if smoke else 'full'}")
    print(f"out_dir: {out_dir}")
    print(f"workers: {workers}")
    print(f"total_jobs: {len(jobs)}")
    print("planned_target:")
    print(f"  population: {PLANNED_POPULATION}")
    print(f"  voxel_um: {format_voxel(PLANNED_VOXEL_UM)}")
    print(f"  n_cells: {PLANNED_N_CELLS}")
    print(f"  orientation: {PLANNED_ORIENTATION}")
    print(f"  contact_distances_um: {PLANNED_CONTACT_DISTANCES_UM}")
    print("selected_geometry:")
    print(f"  population: {geometry.population}")
    print(f"  voxel_um: {format_voxel(geometry.voxel_um)}")
    print(f"  n_cells: {geometry.n_cells}")
    print(f"  orientation: {geometry.orientation}")
    print(f"  contact_distances_um: {geometry.contact_distances_um}")
    print(f"  repeats: {geometry.repeats}")
    print("job_model:")
    print("  one_job_represents: one population/repeat with all contact distances")
    print("planned_jobs:")
    for job in jobs:
        print(
            f"  {job.job_id}: population={job.population}, "
            f"voxel_um={format_voxel(job.voxel_um)}, n_cells={job.n_cells}, "
            f"orientation={job.orientation}, "
            f"contact_distances_um={job.contact_distances_um}"
        )


def main() -> int:
    args = parse_args()
    geometry = selected_geometry(args.smoke)
    out_dir = resolve_out_dir(args.out_dir)
    jobs = build_jobs(geometry)

    if args.dry_run:
        print_dry_run(geometry, jobs, out_dir, args.workers, args.smoke)
        return 0

    print(
        "NotImplementedError: L5 approach sweep execution is not implemented yet. "
        "This scaffold only plans jobs; rerun with --dry-run to inspect geometry.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
