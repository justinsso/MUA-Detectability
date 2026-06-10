# Methods

Full methodological description of the MUA detectability simulation.


## 1. Simulation framework

The simulation is built on **LFPy 2.3** (Einevoll et al., 2013), a Python interface to the **NEURON** simulator (Hines & Carnevale, 1997) that computes extracellular potentials from compartmental cell models under the quasi-static approximation. Transmembrane currents I_mem,i(t) at each cell compartment i generate the extracellular field at an electrode position r_e as:

    V_ext(r_e, t) = Σ_i  I_mem,i(t) / (4π σ |r_e − r_i|)

with tissue conductivity σ = 0.3 S/m. A line-source approximation is used for extended cylindrical compartments (rather than point-source), following Holt and Koch (1999).


## 2. Biophysical cell model

All cells in the simulation use the L5 pyramidal cell morphology of **Mainen & Sejnowski (1996)**, distributed with LFPy as `L5_Mainen96_LFPy.hoc`. The morphology contains a soma, 80 dendritic sections, 83 apical sections, and an axon initial segment (164 total compartments).

Channel distribution (simplified from the original Mainen model):

- **Soma and axon**: classic Hodgkin–Huxley Na+ and K+ channels (`hh` mechanism with default NEURON parameters, gNa = 0.12 S/cm², gK = 0.036 S/cm², gL = 3 × 10⁻⁴ S/cm²)
- **Dendrites (basal and apical)**: passive only (g_pas = 1/30000 S/cm², E_pas = −65 mV)

The passive-dendrite choice trades off biological realism (active dendrites produce larger extracellular AP amplitudes through back-propagating APs) for reproducibility and simpler interpretation. Restoring active dendrites via the full Mainen mechanism set (Na, Kv, Ca, KCa, KM, CaD, Ih) would scale the extracellular AP amplitude by ~1.5–2× but is not essential for the scientific question investigated here.

**Numerical parameters:**

- Integration time step: Δt = 2⁻⁴ = 0.0625 ms (sampling rate 16 kHz)
- Simulation duration: t_stop = 200 ms, covering a pre-stimulus baseline, the spiking event, and post-AP filter settling
- Initial membrane potential: V_init = −65 mV


## 3. Spike generation

Two alternative modes are implemented, selectable via a flag (`drive_mode`):

### 3.1 Synaptic drive (default, biologically motivated)

- **20 Exp2Syn synapses** per cell, placed on dendritic compartments with a height-fraction filter restricting placement to the upper 50–90 % of the principal cell axis
- Synapse placement is **deterministic across cells** (fixed RNG seed) to ensure that every cell in a population sees the same synaptic input pattern, removing placement-driven variability between cells
- Synaptic kinetics: τ₁ = 0.5 ms rise, τ₂ = 2.0 ms decay, reversal potential E_syn = 0 mV, peak conductance g = 0.05 µS — representative of fast AMPAergic excitation
- Every synapse on one cell fires at that cell's assigned spike time; no intra-cell release jitter is applied

### 3.2 Current clamp (alternative, deterministic)

- `neuron.h.IClamp` placed at the soma centre (`soma(0.5)`)
- 0.1 ms × 10 nA pulse, well suprathreshold, triggers a reproducible AP ~0.2 ms after pulse onset
- Used when temporal precision of spike timing is more important than biological realism; produces visible "square-wave" artifact at the injection site due to the capacitive return current

### 3.3 Inter-cell spike timing

Cell k's spike time t_k is drawn from a Gaussian distribution N(t₀, σ_jitter²), where:

- t₀ = 20 ms is the population mean spike time
- σ_jitter is the **primary independent variable**, controlling population synchrony, swept over {0, 3, 5, 10, 20, 30, 50} ms
- Spike times are clamped to ≥ 1 ms to avoid NEURON initialization transients


## 4. Electrode model

A single recording site is modeled with **Neuropixels 1.0 geometry**:

- 12 × 12 µm square contact
- Outward normal along the +x axis (n = (1, 0, 0))
- Tissue conductivity σ = 0.3 S/m
- Extracellular potential computed by finite-area integration with `LFPy.RecExtElectrode` using 50 quadrature points across the contact face, following Ness et al. (2015)'s correction for non-point contacts


## 5. Cell population geometry

Cells are placed around the electrode with two alternative distance-sampling modes:

- **Shell mode** (used throughout the main sweep): every cell in a population sits at exactly d = D µm from the electrode center, parameterized across the sweep for D in {25, 50, 75, 100, 150, 200} µm. Only azimuthal angle is randomized; z-coordinate is fixed at 0. This removes intra-population distance variance so each sweep cell answers a clean "if all N cells fire at distance D" question.
- **Annulus mode** (used for the reference single-run simulation): distance drawn uniformly from [r_inner, r_outer] = [50, 150] µm, z uniform in [−20, 20] µm.

Cells share the same global orientation (`align_cells = True`, rotation angles all zero), so the population forms a ring of morphologically identical cells around the electrode. An optional mode (`align_to_electrode`) can rotate each cell so its frame is oriented radially toward the contact, producing a cell-by-cell identical extracellular signature at the cost of biological realism.


## 6. Signal processing pipeline

### 6.1 Bandpass filtering

The summed extracellular signal V_ext(t) is filtered into two bands following standard electrophysiology convention (Buzsáki, 2004):

- **LFP band**: 0 – 300 Hz lowpass (not used in MUA analysis but retained for visualization)
- **MUA band**: 300 – 5000 Hz bandpass, 3rd-order Butterworth

Filtering is **causal** (`scipy.signal.lfilter`, one-way), matching the Kilosort pipeline (Pachitariu et al., 2016) and real hardware acquisition. Filter order (3) also matches Kilosort's default. Causal filtering introduces a ~1 ms group delay but produces realistic post-spike ringing rather than the symmetric pre/post ringing of zero-phase filtering.

### 6.2 Additive noise

Electrode noise is modeled as Gaussian white noise, ε(t) ~ N(0, σ_n²), added post-filter directly to the MUA-band signal. The noise RMS σ_n = 5 µV matches the Neuropixels 1.0 AP-band noise floor (5–6 µV).

### 6.3 Spike detection

Detection follows the Quiroga et al. (2004) MAD-based convention used by all modern spike sorters (Kilosort, Wave_clus, Spyking Circus):

1. Compute the median absolute deviation of the noisy signal:

        σ_MAD = median(|x(t)|) / 0.6745

   Unlike plain standard deviation, this robust estimator converges to the true noise σ even when the signal contains large spikes (spikes contaminate the std but not the median).
2. Set threshold at θ = k × σ_MAD, where k = 5 (matching Kilosort's default for single-unit isolation; k = 4 is an alternative with slightly higher sensitivity).
3. Detect negative-going threshold crossings (extracellular APs deflect negative).
4. Apply a **refractory period** of 0.5 ms: any crossing within 0.5 ms of a previous detection is discarded. This collapses filter-ringing-induced multi-crossings of a single AP, without compressing population jitter ≥ 1 ms (since pairs of spikes more than 0.5 ms apart remain separable).

### 6.4 Analog MUA — Spiking Band Power (SBP)

A complementary continuous metric is computed following **Nason et al. (2020)**:

1. 2nd-order Butterworth bandpass 300–1000 Hz on the noisy MUA signal (narrower than MUA band; tuned to the peak spectral energy of a single AP while rejecting high-frequency noise).
2. Rectify: |x(t)|.
3. Optional Gaussian smoothing (σ set to 0 for raw rectified output, or up to 50 ms for typical BCI-style firing-rate estimation).

Unlike discrete threshold crossings, the SBP envelope scales monotonically with both population size and synchrony — synchronous APs stack into one tall envelope peak while asynchronous APs produce several smaller peaks, but the integrated area under the envelope is preserved.


## 7. Parameter sweep

The sweep quantifies MUA detectability as a function of three independent variables:

| Variable | Symbol | Values |
|---|---|---|
| Cell count (synchronous population) | N | 1, 5, 10, 25, 50, 100 |
| Cell distance (shell) | D | 25, 50, 75, 100, 150, 200 µm |
| Spike-timing jitter standard deviation | σ_jitter | 0, 3, 5, 10, 20, 30, 50 ms |
| Independent realizations | R | 3 |

Total: 6 × 6 × 7 × 3 = 756 populations (each containing N individual cells).

### 7.1 Variance reduction: common random numbers

To increase the statistical efficiency of the jitter comparison, the pseudorandom seed used to generate cell positions, orientations, synapse placements, and noise realizations depends on the tuple (D, N, repeat) but **not** on σ_jitter. Every jitter value at a given grid point therefore simulates the identical cell layout and noise realization — only the jitter parameter changes.

This is a paired-comparison (common random numbers) design familiar from Monte Carlo variance reduction: Var(X_highjitter − X_lowjitter) is dramatically smaller than Var(X_high) + Var(X_low) when the variates share their random inputs, so jitter-specific effects are detectable at far fewer repeats. The stable `hashlib`-based seed scheme that realizes this design in the refactored runner is described in Section 12.4.

### 7.2 NEURON state isolation

Running hundreds of populations inside a single Python process failed reliably at ~440 cells due to NEURON's internal state (section name table, load_file bookkeeping on Windows) corrupting across `LFPy.Cell` instantiations. The fix implemented here is **one subprocess per population**: a worker script (`sweep_worker.py`) receives a pickled parameter dict, constructs N cells in a fresh Python + NEURON process, records the extracellular trace, applies the full detection pipeline, and writes a result pickle back before exiting. In the refactored runner (`run_reference_sweep.py`, Section 12.2) the parent process dispatches these worker subprocesses concurrently up to a configurable `--workers` count, while each population/repeat still executes in its own clean process.

This incurs ~1–2 s subprocess-startup overhead per population but guarantees that NEURON state cannot drift. Run serially (`--workers 1`), the full sweep costs approximately 8–10 hours on a typical workstation; because the jobs are independent, additional workers can reduce wall-clock time when multiple cores are available.


## 8. Outcome metrics

Each of the 756 populations yields three independent metrics, collected as (n_J × n_D × n_N × R) arrays:

1. **Threshold crossings** (discrete): number of post-refractory negative crossings
2. **Detection probability** (binary summary): P(crossings ≥ 1), computed across the R repeats — the empirical probability that the population is detected at all
3. **Peak SBP amplitude** (continuous): maximum of the rectified 300–1000 Hz envelope — a monotonic firing-energy proxy

Heatmaps (one subplot per jitter value, axes N × D) are produced for each metric. Raw per-repeat arrays are saved as `sweep_raw.npz` for post-hoc analysis.


## 9. SNR calibration

Independently of the main sweep, a single-cell SNR-vs-distance calibration is run via `snr_vs_distance.py`. For each distance d in {10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500} µm, a single cell is simulated (3 independent azimuthal placements) and the following are measured:

- Peak negative extracellular amplitude in the MUA band
- SBP peak amplitude
- MAD-based noise estimate
- SNR = peak amplitude / MAD

This provides a direct numerical analogue to Henze et al.'s (2000) empirical spike-amplitude-vs-distance measurements, and anchors the sweep results to known biophysics: the detection horizon (distance at which SNR drops below the 5× threshold) can be read off as a single number and compared to the literature's ~50 µm estimate for single-unit isolation.


## 10. Reference simulation

A separate "reference" simulation runs a small population (N = 10 cells) at user-specified parameters, producing three diagnostic figures:

1. `signals.png` — individual cell contributions, soma membrane potentials, summed extracellular LFP
2. `bands_decomp.png` — 7-panel band decomposition showing the raw wideband, LFP band, clean MUA, noisy MUA with threshold crossings, SBP, compartment-class contribution (soma+axon vs dendrites), and total synaptic currents
3. `scene_3d.png` — 3D visualization of the cell population and contact geometry

Outputs are time-stamped into a new run folder (`figures/run_<YYYYMMDD_HHMMSS>/`) per invocation, preserving history.


## 11. Software and reproducibility

- Python 3.11, LFPy 2.3.6, NEURON 8.2, NumPy 1.26, SciPy 1.12, Matplotlib 3.8
- Single-threaded (NEURON); sweep parallelism is at the subprocess level
- Run on Windows 11, 10-core CPU
- Active source: `lfpy_MUA_simulation.py` (reference single-run simulation, the behavioral source of truth), `sweep_worker.py` (per-population worker), `snr_vs_distance.py` (calibration); the refactored module/runner layout is described in Section 12
- Each run produces a self-contained `run_<timestamp>/` folder with figures, raw arrays, and a reproducible parameter record

### 11.1 Portable paths and per-run provenance

The morphology and figure-output directories are resolved **relative to the script's location in the repository** rather than hardcoded to one machine: `morph_dir` points at the bundled `LFPy-2.3.6/examples/morphologies/` and `figure_dir` defaults to `<repo>/figures` (overridable via the `MUA_FIGURE_DIR` environment variable). Paths use forward slashes so NEURON's HOC interpreter accepts them on Windows. A startup check raises a clear error if the expected morphology file is missing, so a broken or partial clone fails immediately instead of silently inside each NEURON subprocess. This lets the sweep run on any machine (e.g. the lab workstation) after a plain `git clone`, with no path editing.

Each sweep run additionally writes a `run_info.json` into its `run_<timestamp>/` folder, recording the git commit (and dirty flag), hostname, OS/platform, and the LFPy / NEURON / NumPy / SciPy / Matplotlib versions, alongside the sweep grid and seed scheme. Because the sweep is fully seeded, identical code and environment should reproduce identical results; this provenance record makes any run-to-run difference attributable to a specific code, package, or platform change.


## 12. Refactored pipeline: code organization and reproducible workflow

The simulation and analysis code was reorganized into a layered structure that separates a frozen reference implementation, reusable modules, sweep runners that persist their outputs, and plotting scripts that never re-run NEURON. This section documents the **implemented** infrastructure; the human-L5 contact-approach study in Section 12.6 is a planning scaffold only and has not been run.

### 12.1 Code organization

- **`code/Reference/`** — a frozen snapshot of the validated reference implementation (`lfpy_MUA_simulation.py`, `sweep_worker.py`, `check_env.py`, and a copy of these methods). These files are preserved as the behavioral source of truth and are not edited during refactoring.
- **`code/old/`** — archived earlier scripts, retained for provenance and not part of the active pipeline.
- **`code/mua_config.py`** — portable, repo-relative path and configuration helpers (morphology directory, default output directory) that avoid machine-specific hardcoded paths across macOS, Linux, and Windows.
- **`code/mua_metrics.py`** — array-level signal metrics only (causal Butterworth filtering, MAD noise estimate, negative threshold-crossing detection with refractory collapse, SBP envelope, peak negative amplitude). This module does **not** import LFPy or NEURON, so the detection pipeline can be exercised on plain NumPy arrays.
- **`code/mua_core.py`** — reusable simulation helpers (cell biophysics, synapse placement, electrode construction) aligned with the reference script. Importing it is side-effect free: it does not construct cells, run simulations, or change the working directory at import time.
- **`code/run_reference_sweep.py`** — the reference-sweep runner (Section 12.2).
- **`code/plot_reference_sweep.py`** — figure generation from saved outputs only (Section 12.4).
- **`code/run_l5_approach_sweep.py`** — a planning scaffold for the future human-L5 contact-approach sweep (Section 12.6).

### 12.2 Sweep execution and CPU subprocess parallelism

`run_reference_sweep.py` coordinates jobs only; all NEURON/LFPy work remains in `sweep_worker.py`. Each job — one (jitter, distance, cell-count, repeat) grid point — is dispatched to a **fresh worker subprocess**, preserving the per-population NEURON state isolation described in Section 7.2. Independent jobs run concurrently up to a configurable worker count (`--workers`), so the sweep can use multiple CPU cores while each population/repeat still executes in its own clean Python + NEURON process. A `--dry-run` mode prints the planned jobs, grid, and output location without launching workers, and `--smoke` selects a tiny grid for quick end-to-end checks.

### 12.3 Resumable per-job outputs

Each worker writes its result to a dedicated per-job file before any aggregation step, and the runner builds the aggregate arrays only once every job result is present. This makes the sweep **resumable**: with `--resume`, jobs whose result files already exist are skipped rather than recomputed or overwritten, and the runner refuses to clobber an existing aggregate. Final results are saved as a combined `.npz` (per-metric arrays indexed jitter × distance × cell-count × repeat) and a flat `.csv` (one row per job). A `run_info.json` is written once per run (see Section 11.1), additionally recording the sweep mode, grid, and seed scheme as provenance.

### 12.4 Deterministic seeds

Reproducibility-critical seeds in `run_reference_sweep.py` no longer use Python's built-in `hash(...)`, whose per-process salting makes it unstable across processes and machines. Instead, seeds are derived with a stable `hashlib` (SHA-256) helper: structured fields are joined, hashed, and the leading digest bytes are reduced into a NumPy-compatible seed. The layout/noise seed for a job depends on `distance_index`, `n_cell_index`, and `repeat`, and **excludes** the jitter value. All jitter values at a given (distance, cell-count, repeat) grid point therefore share the identical cell layout and noise realization, preserving exactly the common-random-numbers paired-comparison design of Section 7.1 while keeping seeds stable across processes, machines, and Python sessions.

### 12.5 Plotting workflow

`plot_reference_sweep.py` loads a saved aggregate `.npz` (or a run directory containing one) and writes static figures — currently threshold-crossings and detection-probability heatmaps, plus a peak-SBP heatmap when that field is present — into a `figures/` subfolder of the run directory. It does **not** import LFPy or NEURON and runs no simulations, so saved sweeps can be re-plotted without repeating the expensive NEURON work.

### 12.6 Planned (scaffolded) human-L5 contact-approach sweep

> **Status: planned / scaffold only — not yet implemented and not run.** `code/run_l5_approach_sweep.py` currently defines the intended target geometry and prints a job plan under `--dry-run`; any non-dry-run invocation exits without simulating. It does not import LFPy or NEURON and performs no biophysical simulation.

The intended scientific target, as encoded in the scaffold, is a contact-approach study in which a recording site is advanced toward a dense cortical population:

- **Population:** human L5 pyramidal cells
- **Voxel:** 300 × 300 × 300 µm
- **Cell count:** approximately 500 (500 in the current scaffold constant)
- **Orientation:** apical dendrites oriented "north", basal dendrites "south"
- **Contact approach:** beginning at 400 µm and stepping inward by 33 µm, i.e. {400, 367, 334, 301, 268, 235, 202, 169, 136, 103, 70, 37, 4} µm

A reduced smoke configuration (10 cells; contact distances 400/202/4 µm; one repeat) is defined for future quick checks. No biophysical model, full 500-cell simulation, validation, or result for this target has been implemented or produced; the methods and any results in the preceding sections pertain to the reference sweep, not to this planned study.


## 13. Key references

- **Buzsáki (2004)**, *Nat Neurosci*, "Large-scale recording of neuronal ensembles" — MUA detection radius concepts
- **Buzsáki, Anastassiou & Koch (2012)**, *Nat Rev Neurosci*, "The origin of extracellular fields and currents" — biophysical theory
- **Henze et al. (2000)**, *J Neurophysiol*, "Intracellular features predicted by extracellular recordings in the hippocampus in vivo" — SNR-vs-distance empirical anchor
- **Shoham et al. (2006)**, *J Neurophysiol*, "How silent is the brain" — MUA radius and the "dark matter" problem
- **Einevoll et al. (2013)**, *Nat Rev Neurosci*, "Modelling and analysis of local field potentials for studying the function of cortical circuits" — LFPy framework
- **Pettersen & Einevoll (2008)**, *Biophys J*, "Amplitude variability and extracellular low-pass filtering of neuronal spikes" — modeling template
- **Mainen & Sejnowski (1996)**, *Nature*, "Influence of dendritic structure on firing pattern in model neocortical neurons" — cell morphology
- **Holt & Koch (1999)**, *J Comput Neurosci*, "Electrical interactions via the extracellular potential near cell bodies" — line-source theory
- **Quiroga et al. (2004)**, *Neural Computation*, "Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering" — MAD threshold
- **Nason et al. (2020)**, *Nat Biomed Eng*, "A low-power band of neuronal spiking activity dominated by local single units..." — Spiking Band Power pipeline
- **Pachitariu et al. (2016)**, *bioRxiv*, "Kilosort: realtime spike-sorting for extracellular electrophysiology..." — filter convention, detection threshold
- **Ness et al. (2015)**, *PLoS Comput Biol*, "Modelling and analysis of electrical potentials recorded in microelectrode arrays (MEAs)" — finite-contact correction
