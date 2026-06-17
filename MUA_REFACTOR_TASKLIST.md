# MUA Refactor Tasklist for Codex

This file is a handoff checklist for a future Codex implementation pass. It is
intended to reorganize the project safely while preserving the current working
simulation behavior.

## Non-Negotiable Guardrails

- Do not delete files.
- Do not use `rm`, `mv`, `git clean`, `git reset`, or `git checkout --`.
- Do not modify files inside `Reference/` after creating them.
- Do not modify files inside `code/old/`.
- Do not run full production sweeps.
- Do not move generated figures or data unless explicitly asked.
- Do not overwrite existing `.npz`, `.csv`, `.png`, `.pptx`, or manuscript files.
- Do not overwrite any existing file unless the task explicitly says to update it.
- Keep `code/lfpy_MUA_simulation.py` as the behavioral source of truth.
- Prefer copying reference files before refactoring active files.
- Use `git status --short` before and after work.
- If a file already exists, inspect it before editing.
- If behavior is ambiguous, preserve existing behavior and document the assumption.

## Goal

Separate the project into:

- a frozen reference implementation,
- reusable simulation/analysis modules,
- sweep runners that save reusable outputs,
- plotting scripts that load saved outputs without rerunning NEURON.

The immediate goal is safer project structure and sweep/plot separation, not a
full scientific redesign.

## Current Working Assumption

The canonical active reference script is:

```text
code/lfpy_MUA_simulation.py
```

The environment validation script is:

```text
code/check_env.py
```

The current isolated NEURON subprocess pattern is:

```text
code/sweep_worker.py
```

The archived files are in:

```text
code/old/
```

Do not assume the file is named `lfpy_MUA_stimulation.py` unless such a file is
actually present.

## Target File Layout

Create or evolve toward this layout:

```text
Reference/
  README.md
  lfpy_MUA_simulation.py
  check_env.py
  sweep_worker.py
  METHODS.md

code/
  check_env.py
  lfpy_MUA_simulation.py
  sweep_worker.py

  mua_config.py
  mua_metrics.py
  mua_core.py

  run_reference_sweep.py
  plot_reference_sweep.py

  run_l5_approach_sweep.py
  plot_l5_approach_sweep.py

  old/
```

## Phase 0: Preflight

- [ ] Run `git status --short`.
- [ ] Confirm `code/lfpy_MUA_simulation.py` exists.
- [ ] Confirm `code/check_env.py` exists.
- [ ] Confirm `code/sweep_worker.py` exists.
- [ ] Confirm `code/old/` exists.
- [ ] Record any dirty files in the final summary.
- [ ] Do not clean, reset, remove, or revert user changes.

## Phase 1: Create Frozen Reference Snapshot

- [ ] Create `Reference/` if it does not exist.
- [ ] If `Reference/` already exists, inspect it before adding files.
- [ ] Copy `code/lfpy_MUA_simulation.py` to `Reference/lfpy_MUA_simulation.py` only if the target file does not already exist.
- [ ] Copy `code/check_env.py` to `Reference/check_env.py` only if the target file does not already exist.
- [ ] Copy `code/sweep_worker.py` to `Reference/sweep_worker.py` only if the target file does not already exist.
- [ ] Copy `code/METHODS.md` to `Reference/METHODS.md` if present and only if the target file does not already exist.
- [ ] Create `Reference/README.md` only if it does not already exist.
- [ ] In `Reference/README.md`, state that these files are preserved as the validated reference implementation and should not be edited during refactoring.
- [ ] Verify the copied files exist.
- [ ] Do not edit reference files after this phase.
- [ ] If any `Reference/` target already exists and differs from the active source, stop and ask the user before overwriting or refreshing it.

## Phase 2: Extract Non-Controversial Shared Helpers

Create `code/mua_config.py` first.

- [ ] Add repo-relative path resolution.
- [ ] Add morphology directory resolution.
- [ ] Add default figure/output directory resolution.
- [ ] Add default morphology path.
- [ ] Keep paths portable across macOS, Linux, and Windows.
- [ ] Avoid hardcoded user-specific paths.

Create `code/mua_metrics.py`.

- [ ] Copy/extract the causal Butterworth filter helper from the reference script without deleting it from the reference script.
- [ ] Add MUA-band filtering helper.
- [ ] Add MAD noise estimator.
- [ ] Add negative threshold crossing detector.
- [ ] Add refractory-window crossing collapse.
- [ ] Add SBP calculation helper.
- [ ] Add peak negative amplitude helper.
- [ ] Add small pure-Python or NumPy-only tests/examples in docstrings if useful.
- [ ] Do not import `LFPy` or `neuron` in `mua_metrics.py`.

Create `code/mua_core.py`.

- [ ] Do not run simulations at import time.
- [ ] Do not change the process working directory at import time.
- [ ] Avoid top-level `LFPy.Cell` construction.
- [ ] Add cell biophysics setup matching `lfpy_MUA_simulation.py`.
- [ ] Add synapse placement helper matching `lfpy_MUA_simulation.py`.
- [ ] Add electrode construction helper matching `lfpy_MUA_simulation.py`.
- [ ] Add population simulation helper only after the smaller helpers are stable.
- [ ] Keep all behavior aligned with `Reference/lfpy_MUA_simulation.py`.
- [ ] Avoid broad rewrites of `code/lfpy_MUA_simulation.py` in this phase.

## Phase 3: Stabilize Worker And Sweep Runner

Create or update `code/run_reference_sweep.py`.

- [ ] Use `argparse`.
- [ ] Add `--dry-run`.
- [ ] Add `--smoke`.
- [ ] Add `--workers`.
- [ ] Add `--resume`.
- [ ] Add `--out-dir`.
- [ ] Use CPU subprocess parallelism across independent jobs.
- [ ] Keep each NEURON population/repeat isolated in a fresh subprocess.
- [ ] Save one result file per job before aggregating.
- [ ] Write a final aggregate `.npz`.
- [ ] Write a final aggregate `.csv`.
- [ ] Write `run_info.json`.
- [ ] Do not embed a worker script string inside the runner.
- [ ] Prefer calling `code/sweep_worker.py` or a small dedicated worker file.
- [ ] Make `--smoke` tiny enough to run quickly.
- [ ] Default outputs should go to a new timestamped run directory under `outputs/`, not into existing result folders.
- [ ] `--resume` should skip completed job files and should not overwrite them.
- [ ] If an output path already contains results and `--resume` is not set, stop with a clear error.

Recommended smoke grid:

```text
n_cells: 1, 5
distances: 25, 50
jitters: 0, 5
repeats: 1
workers: 1
```

Parallel verification:

- [ ] Run the smoke grid with `--workers 1`.
- [ ] Run the same smoke grid with `--workers 2`.
- [ ] Confirm both complete.
- [ ] Confirm output schema matches.
- [ ] Do not require exact numerical equality until deterministic seeds are implemented.

## Phase 4: Deterministic Seed Handling

- [ ] Add a stable seed helper using `hashlib`.
- [ ] Stop using Python's built-in `hash(...)` for reproducibility-critical seeds.
- [ ] Seed should be stable across processes, machines, and Python sessions.
- [ ] Preserve common-random-number design across jitter values where intended.
- [ ] Document the seed scheme in `run_info.json`.

Example intent:

```text
layout/noise seed depends on distance, n_cells, repeat
jitter seed additionally depends on jitter value
```

## Phase 5: Plotting-Only Scripts

Create `code/plot_reference_sweep.py`.

- [ ] Use `argparse`.
- [ ] Accept a path to a saved sweep output directory or `.npz`.
- [ ] Load saved results only.
- [ ] Do not import `LFPy`.
- [ ] Do not import `neuron`.
- [ ] Do not run simulations.
- [ ] Produce figures from saved `.npz`/`.csv`.
- [ ] Save figures into a subfolder of the run output directory by default.

Initial figure targets:

- [ ] Threshold crossings heatmap.
- [ ] Detection probability heatmap.
- [ ] Peak SBP or peak MUA heatmap if available.
- [ ] SNR vs number of cells if available.

## Phase 6: Future L5 Approach Sweep Scaffold

Create `code/run_l5_approach_sweep.py` as a scaffold only unless explicitly
asked to implement the full biology.

The intended scientific target:

```text
Population: human L5 pyramidal cells
Voxel: 300 x 300 x 300 um
Cell count: approximately 500
Orientation: apical dendrites north, basal dendrites south
Contact path: starts 400 um away and approaches in 33 um steps
```

Initial contact distances:

```python
[400, 367, 334, 301, 268, 235, 202, 169, 136, 103, 70, 37, 4]
```

Implementation notes for later:

- [ ] One job should represent one population/repeat.
- [ ] Record all contact approach distances in the same NEURON simulation when possible.
- [ ] Do not parallelize contact distances separately unless memory forces it.
- [ ] Keep the first smoke test much smaller than 500 cells.

Recommended first L5 smoke test:

```text
cells: 10
voxel: 300 um
contact distances: 400, 202, 4 um
repeats: 1
workers: 1
```

## Phase 7: Methods Notes

Update `code/METHODS.md` only after code structure exists.

- [ ] Add a short section on code organization.
- [ ] Add a short section on CPU subprocess parallelization.
- [ ] Add a short section on resumable per-job outputs.
- [ ] Add a planned section for the L5 density/contact approach sweep.
- [ ] Clearly label planned work as planned if not yet implemented.

## Validation Commands

Use lightweight validation only.

```bash
python code/check_env.py
python -m py_compile code/mua_config.py code/mua_metrics.py code/mua_core.py
python code/run_reference_sweep.py --dry-run
python code/run_reference_sweep.py --smoke --workers 1 --out-dir outputs/smoke_workers1
python code/run_reference_sweep.py --smoke --workers 2 --out-dir outputs/smoke_workers2
python code/plot_reference_sweep.py --help
```

Do not run full sweeps unless explicitly requested.

## Final Summary Requirements

At the end of an implementation pass, report:

- Files created.
- Files modified.
- Files intentionally not touched.
- Whether `Reference/` was created.
- Whether `code/old/` was untouched.
- Validation commands run.
- Validation commands skipped and why.
- Any assumptions made.
- Any remaining risks.

## Stop Conditions

Stop and ask the user before proceeding if:

- `code/lfpy_MUA_simulation.py` is missing.
- A file named `lfpy_MUA_stimulation.py` exists and conflicts with `lfpy_MUA_simulation.py`.
- Reference files already exist but differ from active files.
- A command would overwrite an existing reference, output, figure, or data file.
- Running a smoke test would require installing packages.
- The code requires editing files outside the repo.
- Any command would delete, reset, or overwrite existing outputs.
