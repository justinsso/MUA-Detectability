"""Signal-processing helpers for MUA detectability analyses.

This module intentionally has no LFPy or NEURON dependency. It contains only
array-level metrics shared by simulation workers and plotting/analysis code.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, lfilter


def butter_filter(data, fs_hz, low=None, high=None, order=4):
    """Apply a causal Butterworth filter with ``scipy.signal.lfilter``.

    This is extracted from ``lfpy_MUA_simulation.py`` and preserves the one-way
    filtering convention used by the reference script.

    Examples
    --------
    >>> x = np.ones(8)
    >>> butter_filter(x, fs_hz=1000.0, high=100.0).shape
    (8,)
    """
    if low is None and high is None:
        raise ValueError("at least one of low or high must be provided")

    nyq = 0.5 * fs_hz
    if low is None and high is not None:
        b, a = butter(order, high / nyq, btype="low")
    elif high is None and low is not None:
        b, a = butter(order, low / nyq, btype="high")
    else:
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, data)


causal_butter_filter = butter_filter


def mua_band_filter(data, fs_hz, low=300.0, high=5000.0, order=3):
    """Return the standard reference MUA-band filtered signal."""
    return butter_filter(data, fs_hz, low=low, high=high, order=order)


def mad_noise_std(data, scale=0.6745):
    """Estimate noise standard deviation using the MAD convention.

    The reference uses ``median(abs(x)) / 0.6745`` for zero-centered MUA traces.

    Examples
    --------
    >>> round(mad_noise_std(np.array([-1.0, 0.0, 1.0])), 3)
    1.483
    """
    return float(np.median(np.abs(data)) / scale)


def negative_threshold_crossings(data, threshold):
    """Return indices where ``data`` crosses downward below ``-threshold``.

    The returned index matches the reference script: it is the sample just before
    the boolean below-threshold state turns true.
    """
    below = np.asarray(data) < -float(threshold)
    return np.where(np.diff(below.astype(int)) > 0)[0].astype(int)


def collapse_refractory_crossings(crossings, refractory_ms, dt_ms):
    """Collapse crossings that occur within a refractory window.

    Examples
    --------
    >>> collapse_refractory_crossings([10, 12, 25], refractory_ms=0.5, dt_ms=0.1).tolist()
    [10, 25]
    """
    crossings = np.asarray(crossings, dtype=int)
    if crossings.size == 0:
        return crossings

    refractory_samples = int(round(float(refractory_ms) / float(dt_ms)))
    if refractory_samples <= 0:
        return crossings.copy()

    kept = []
    last = -refractory_samples
    for idx in crossings:
        if idx - last >= refractory_samples:
            kept.append(int(idx))
            last = int(idx)
    return np.asarray(kept, dtype=int)


def threshold_crossings(data, threshold, refractory_ms=0.5, dt_ms=2**-4):
    """Detect negative threshold crossings and apply refractory collapse."""
    all_crossings = negative_threshold_crossings(data, threshold)
    return collapse_refractory_crossings(all_crossings, refractory_ms, dt_ms)


def peak_negative_amplitude(data):
    """Return the magnitude of the most-negative excursion."""
    return float(-np.min(data))


def spiking_band_power(
    data,
    fs_hz,
    low=300.0,
    high=1000.0,
    order=2,
    smooth_ms=0.0,
    dt_ms=None,
):
    """Return the rectified SBP envelope used by the reference script.

    The reference defaults to no Gaussian smoothing for short simulations.
    """
    band = butter_filter(data, fs_hz, low=low, high=high, order=order)
    magnitude = np.abs(band)
    if smooth_ms <= 0:
        return magnitude

    sample_dt_ms = (1000.0 / fs_hz) if dt_ms is None else float(dt_ms)
    sigma_samples = float(smooth_ms) / sample_dt_ms
    return gaussian_filter1d(magnitude, sigma_samples)


def peak_sbp_amplitude(
    data,
    fs_hz,
    low=300.0,
    high=1000.0,
    order=2,
    smooth_ms=0.0,
    dt_ms=None,
):
    """Return the peak rectified SBP amplitude."""
    return float(
        spiking_band_power(
            data,
            fs_hz,
            low=low,
            high=high,
            order=order,
            smooth_ms=smooth_ms,
            dt_ms=dt_ms,
        ).max()
    )


def threshold_metrics(
    data,
    threshold_factor=5.0,
    refractory_ms=0.5,
    dt_ms=2**-4,
):
    """Return the reference MAD threshold and collapsed crossing indices."""
    noise_std = mad_noise_std(data)
    threshold = float(threshold_factor) * noise_std
    crossings = threshold_crossings(data, threshold, refractory_ms, dt_ms)
    return {
        "mua_std": noise_std,
        "threshold": threshold,
        "crossings": crossings,
        "n_crossings": int(len(crossings)),
        "detected": bool(len(crossings) > 0),
    }
