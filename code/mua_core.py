"""Core simulation helpers aligned with the reference MUA script.

Importing this module must be side-effect free: it does not create cells, run
simulations, or change the process working directory at import time.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np


def _import_lfpy():
    import LFPy

    return LFPy


def _import_neuron():
    import neuron

    return neuron


@contextlib.contextmanager
def neuron_working_directory(path):
    """Temporarily set Python and NEURON working directories for morphology loads."""
    if path is None:
        yield
        return

    neuron = _import_neuron()
    previous = os.getcwd()
    target = Path(path).expanduser().resolve()
    os.chdir(target)
    neuron.h.chdir(target.as_posix())
    try:
        yield
    finally:
        os.chdir(previous)
        try:
            neuron.h.chdir(Path(previous).as_posix())
        except Exception:
            pass


def make_reference_cell(
    morphology,
    v_init=-65,
    passive=False,
    dt=2**-4,
    tstart=0,
    tstop=200,
):
    """Create an LFPy cell and apply the reference biophysics."""
    morphology_path = Path(morphology).expanduser()
    if not morphology_path.is_file():
        raise FileNotFoundError(
            f"Expected morphology not found at {str(morphology)!r}. "
            "Is the repo fully cloned, and does morph_dir point to "
            "LFPy-2.3.6/examples/morphologies?"
        )

    LFPy = _import_lfpy()
    cell = LFPy.Cell(
        morphology=morphology,
        v_init=v_init,
        passive=passive,
        dt=dt,
        tstart=tstart,
        tstop=tstop,
    )
    apply_reference_biophysics(cell)
    return cell


def apply_reference_biophysics(
    cell,
    dendrite_g_pas=1.0 / 30000,
    dendrite_e_pas=-65,
):
    """Insert HH into soma/axon and passive channels into dendrites."""
    for sec in cell.allseclist:
        if "soma" in sec.name() or "axon" in sec.name():
            sec.insert("hh")
        else:
            sec.insert("pas")
            sec.g_pas = dendrite_g_pas
            sec.e_pas = dendrite_e_pas
    return cell


def set_reference_pose(cell, x=0.0, y=0.0, z=0.0, rot_x=0.0, rot_y=0.0, rot_z=0.0):
    """Apply the reference rotate-then-translate pose convention."""
    cell.set_rotation(x=np.radians(rot_x), y=np.radians(rot_y), z=np.radians(rot_z))
    cell.set_pos(x=x, y=y, z=z)
    return cell


def compartment_class_indices(cell):
    """Return ``(soma_axon_indices, dendrite_indices)`` for a cell."""
    soma_axon_idxs = []
    dend_idxs = []
    for sec in cell.allseclist:
        sec_idxs = cell.get_idx(section=sec.name())
        if "soma" in sec.name() or "axon" in sec.name():
            soma_axon_idxs.extend(sec_idxs)
        else:
            dend_idxs.extend(sec_idxs)
    return np.asarray(soma_axon_idxs, dtype=int), np.asarray(dend_idxs, dtype=int)


def segment_centers(cell):
    """Return compartment center coordinates as an ``(n_segments, 3)`` array."""
    return np.column_stack(
        [
            cell.x.mean(axis=1),
            cell.y.mean(axis=1),
            cell.z.mean(axis=1),
        ]
    )


def dendritic_indices_by_height(cell, syn_height_min=0.5, syn_height_max=0.9):
    """Return dendritic indices passing the reference principal-axis height filter."""
    _, dend_idxs = compartment_class_indices(cell)
    if dend_idxs.size == 0:
        return dend_idxs

    seg_centers = segment_centers(cell)
    soma_center = seg_centers[cell.somaidx].mean(axis=0)
    dists_from_soma = np.linalg.norm(seg_centers - soma_center, axis=1)
    apex = seg_centers[np.argmax(dists_from_soma)]
    main_vec = apex - soma_center
    main_len = np.linalg.norm(main_vec)
    if main_len == 0:
        return np.asarray([], dtype=int)

    main_unit = main_vec / main_len
    dend_centers = seg_centers[dend_idxs]
    projections = np.dot(dend_centers - soma_center, main_unit)
    h_min = main_len * syn_height_min
    h_max = main_len * syn_height_max
    return dend_idxs[(projections >= h_min) & (projections <= h_max)]


def place_reference_synapses(
    cell,
    spike_time,
    n_synapses=20,
    syn_type="Exp2Syn",
    syn_weight=0.05,
    tau1=0.5,
    tau2=2.0,
    e_syn=0,
    syn_height_min=0.5,
    syn_height_max=0.9,
    record_current=True,
    rng_seed=0,
    cell_label=None,
):
    """Place deterministic dendritic synapses as in the reference script."""
    LFPy = _import_lfpy()
    dend_idxs = dendritic_indices_by_height(cell, syn_height_min, syn_height_max)
    if len(dend_idxs) == 0 and n_synapses:
        label = "" if cell_label is None else f"Cell {cell_label}: "
        raise AssertionError(
            f"{label}no dendritic compartments found between "
            f"{syn_height_min * 100:.0f}%-{syn_height_max * 100:.0f}% of cell height. "
            f"Try widening the range."
        )

    synapses = []
    syn_rng = np.random.default_rng(seed=rng_seed)
    for _ in range(n_synapses):
        syn = LFPy.Synapse(
            cell,
            idx=int(syn_rng.choice(dend_idxs)),
            syntype=syn_type,
            weight=syn_weight,
            record_current=record_current,
            **{"tau1": tau1, "tau2": tau2, "e": e_syn},
        )
        syn.set_spike_times(np.array([max(0.1, spike_time)]))
        synapses.append(syn)
    return synapses


def place_reference_iclamp(cell, spike_time, iclamp_amp=10, iclamp_dur=0.1):
    """Attach the reference somatic IClamp and return it so callers keep it alive."""
    neuron = _import_neuron()
    soma_sec = next(sec for sec in cell.allseclist if "soma" in sec.name())
    iclamp = neuron.h.IClamp(soma_sec(0.5))
    iclamp.delay = spike_time
    iclamp.dur = iclamp_dur
    iclamp.amp = iclamp_amp
    return iclamp


def attach_reference_drive(cell, drive_mode, spike_time, **kwargs):
    """Attach either reference synapses or reference IClamp drive."""
    if drive_mode == "synapses":
        return place_reference_synapses(cell, spike_time, **kwargs)
    if drive_mode == "iclamp":
        return [place_reference_iclamp(cell, spike_time, **kwargs)]
    raise ValueError(f"Unknown drive_mode: {drive_mode!r} (use 'synapses' or 'iclamp')")


def make_reference_electrode(
    cell,
    elec_x=None,
    elec_y=None,
    elec_z=None,
    sigma=0.3,
    method="linesource",
    contact_normal=None,
    contact_size=12.0,
    n_avg_points=50,
    contact_shape="square",
):
    """Construct the finite-area electrode used by the reference script."""
    LFPy = _import_lfpy()
    normal = np.asarray([[1.0, 0.0, 0.0]] if contact_normal is None else contact_normal, dtype=float)
    if normal.ndim == 1:
        normal = normal[np.newaxis, :]

    return LFPy.RecExtElectrode(
        cell,
        x=np.asarray([0.0] if elec_x is None else elec_x),
        y=np.asarray([0.0] if elec_y is None else elec_y),
        z=np.asarray([0.0] if elec_z is None else elec_z),
        sigma=sigma,
        method=method,
        N=normal,
        r=contact_size / 2.0,
        n=n_avg_points,
        contact_shape=contact_shape,
    )


def random_reference_cell_spec(args):
    """Draw one cell's position, rotation, morphology, and spike time."""
    angle = np.random.uniform(0, 2 * np.pi)
    if args.get("fix_cell_distance", True):
        if "cell_distance" not in args:
            raise ValueError("cell_distance is required when fix_cell_distance=True")
        dist = args["cell_distance"]
        z = 0.0
    else:
        dist = np.random.uniform(args["inner_radius"], args["outer_radius"])
        z = np.random.uniform(-20, 20)

    x = dist * np.cos(angle)
    y = dist * np.sin(angle)

    if args.get("align_cells", True):
        rot_x = args.get("align_rot_x", 0)
        rot_y = args.get("align_rot_y", 0)
        rot_z = args.get("align_rot_z", 0)
    else:
        rot_x = np.random.uniform(0, 10)
        rot_y = np.random.uniform(0, 10)
        rot_z = np.random.uniform(0, 360)

    if args.get("align_to_electrode", False):
        rot_z = np.degrees(angle)

    spike_time = max(1.0, args.get("base_spike_time", 20.0) + np.random.normal(0, args.get("jitter_std", 0.0)))
    morphologies = args["morphologies"]
    morphology = morphologies[np.random.randint(len(morphologies))]
    return {
        "x": x,
        "y": y,
        "z": z,
        "rot_x": rot_x,
        "rot_y": rot_y,
        "rot_z": rot_z,
        "spike_time": spike_time,
        "morph": morphology,
    }


def simulate_population_mua(args):
    """Run one population simulation and return the clean MUA trace in uV.

    The returned trace is noise-free. Callers must add Gaussian noise using the
    configured ``noise_rms_uV`` before threshold detection; see
    ``sweep_worker.py::main()`` for the canonical usage pattern.

    This mirrors the current subprocess worker's behavior, but lives behind a
    callable helper so importing ``mua_core`` stays side-effect free.
    """
    from mua_metrics import mua_band_filter

    dt = args.get("dt", 2**-4)
    fs_hz = args.get("fs_hz", 1000.0 / dt)
    lfps = []
    with neuron_working_directory(args.get("morph_dir")):
        for cell_index in range(args["n_cells"]):
            spec = random_reference_cell_spec(args)
            cell = make_reference_cell(
                morphology=spec["morph"],
                v_init=args.get("v_init", -65),
                passive=False,
                dt=dt,
                tstart=args.get("tstart", 0),
                tstop=args.get("tstop", 200),
            )
            set_reference_pose(
                cell,
                x=spec["x"],
                y=spec["y"],
                z=spec["z"],
                rot_x=spec["rot_x"],
                rot_y=spec["rot_y"],
                rot_z=spec["rot_z"],
            )

            drive_mode = args.get("drive_mode", "synapses")
            if drive_mode == "synapses":
                keepalive = place_reference_synapses(
                    cell,
                    spike_time=spec["spike_time"],
                    n_synapses=args.get("n_synapses", 20),
                    syn_type=args.get("syn_type", "Exp2Syn"),
                    syn_weight=args.get("syn_weight", 0.05),
                    tau1=args.get("tau1", 0.5),
                    tau2=args.get("tau2", 2.0),
                    e_syn=args.get("e_syn", 0),
                    syn_height_min=args.get("syn_height_min", 0.5),
                    syn_height_max=args.get("syn_height_max", 0.9),
                    record_current=args.get("record_synapse_current", False),
                    rng_seed=args.get("synapse_rng_seed", 0),
                    cell_label=cell_index,
                )
            elif drive_mode == "iclamp":
                keepalive = [
                    place_reference_iclamp(
                        cell,
                        spike_time=spec["spike_time"],
                        iclamp_amp=args.get("iclamp_amp", 10),
                        iclamp_dur=args.get("iclamp_dur", 0.1),
                    )
                ]
            else:
                raise ValueError(f"Unknown drive_mode: {drive_mode!r} (use 'synapses' or 'iclamp')")

            electrode = make_reference_electrode(
                cell,
                elec_x=args.get("elec_x", [0.0]),
                elec_y=args.get("elec_y", [0.0]),
                elec_z=args.get("elec_z", [0.0]),
                sigma=args.get("sigma", 0.3),
                method=args.get("method", "linesource"),
                contact_normal=args.get("contact_normal", [[1.0, 0.0, 0.0]]),
                contact_size=args.get("contact_size", 12.0),
                n_avg_points=args.get("n_avg_points", 50),
                contact_shape=args.get("contact_shape", "square"),
            )
            cell.simulate(rec_imem=True, probes=[electrode])
            lfps.append(electrode.data[0].copy())
            del keepalive

    if len(lfps) == 0:
        n_samples = int(round((args.get("tstop", 200) - args.get("tstart", 0)) / args.get("dt", 2**-4))) + 1
        return np.zeros(n_samples)

    raw_uV = np.sum(lfps, axis=0) * 1000.0
    return mua_band_filter(
        raw_uV,
        fs_hz,
        low=args.get("mua_low", 300.0),
        high=args.get("mua_high", 5000.0),
        order=args.get("filt_order", 3),
    )
