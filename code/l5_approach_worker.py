"""Standalone worker for the human L5 contact-approach sweep.

This inverts the reference sweep: instead of cells on a shell around a fixed
electrode, one fixed volumetric population of L5 pyramidal cells is recorded by a
single multi-contact electrode whose contacts sit at every approach distance, so
all distances come from ONE population simulation.

Each invocation runs ONE population/repeat in a fresh Python process so NEURON's
internal state starts clean, exactly like ``sweep_worker.py``. The simulation
itself lives in :func:`mua_core.simulate_l5_population_mua`; this file only owns
the pickle-in/pickle-out CLI contract and the post-hoc noise + MAD detection.

Usage::

    python l5_approach_worker.py <args_pickle> <result_pickle>

The args pickle is a dict written by ``run_l5_approach_sweep.py`` (see its
``base_worker_args``); the contact distances and matching electrode positions
must already be present.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

import mua_core
import mua_metrics


def detect_per_contact(clean_traces, args):
    """Return one detection-result dict per contact (post-hoc noise + MAD).

    ``clean_traces`` is the ``(n_contacts, n_samples)`` noise-free µV array from
    :func:`mua_core.simulate_l5_population_mua`. Each contact gets an independent
    Gaussian-noise realization, then the reference MAD threshold / refractory
    crossing convention from ``sweep_worker.py``.
    """
    dt = args.get("dt", 2**-4)
    fs_hz = args.get("fs_hz", 1000.0 / dt)
    distances = list(args["contact_distances_um"])
    noise_rms = args["noise_rms_uV"]
    threshold_factor = args["mua_threshold_factor"]
    refractory_ms = args.get("refractory_ms", 0.5)

    if clean_traces.shape[0] != len(distances):
        raise ValueError(
            f"contact count mismatch: {clean_traces.shape[0]} traces "
            f"but {len(distances)} contact distances"
        )

    per_contact = []
    for contact_index, distance_um in enumerate(distances):
        clean = clean_traces[contact_index]
        noisy = clean + np.random.normal(0, noise_rms, clean.shape)
        metrics = mua_metrics.threshold_metrics(
            noisy,
            threshold_factor=threshold_factor,
            refractory_ms=refractory_ms,
            dt_ms=dt,
        )
        per_contact.append(
            {
                "distance_um": float(distance_um),
                "n_crossings": int(metrics["n_crossings"]),
                "detected": bool(metrics["detected"]),
                # Reference convention: peak of the NOISY trace (detection-time).
                # When the signal is below the noise floor (few cells), this is
                # dominated by noise; the clean peaks below carry the true
                # amplitude-vs-distance falloff.
                "peak_neg_uV": mua_metrics.peak_negative_amplitude(noisy),
                "peak_sbp_uV": mua_metrics.peak_sbp_amplitude(noisy, fs_hz),
                # Noise-free signal peaks — the physically meaningful amplitude.
                "peak_neg_clean_uV": mua_metrics.peak_negative_amplitude(clean),
                "peak_sbp_clean_uV": mua_metrics.peak_sbp_amplitude(clean, fs_hz),
                "mua_std": float(metrics["mua_std"]),
                "threshold": float(metrics["threshold"]),
            }
        )
    return per_contact


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python l5_approach_worker.py <args_pickle> <result_pickle>",
            file=sys.stderr,
        )
        sys.exit(1)

    args_path = sys.argv[1]
    result_path = sys.argv[2]

    with open(args_path, "rb") as handle:
        args = pickle.load(handle)

    if args.get("seed") is not None:
        np.random.seed(int(args["seed"]))

    clean_traces = mua_core.simulate_l5_population_mua(args)
    per_contact = detect_per_contact(clean_traces, args)

    with open(result_path, "wb") as handle:
        pickle.dump(
            {
                "n_cells": int(args["n_cells"]),
                "contact_distances_um": [float(d) for d in args["contact_distances_um"]],
                "per_contact": per_contact,
            },
            handle,
        )


if __name__ == "__main__":
    main()
