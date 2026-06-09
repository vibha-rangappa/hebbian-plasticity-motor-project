# tests/test_analysis.py

import numpy as np
import pytest

from part1.run_part1 import compute_cv_isi, compute_pairwise_corr, compute_power_spectrum


def _make_poisson_trains(n_neurons: int, rate_hz: float, t_end: float, seed: int) -> dict:
    """Generate independent Poisson spike trains for testing."""
    rng = np.random.default_rng(seed)
    trains = {}
    for i in range(n_neurons):
        n = rng.poisson(rate_hz * t_end)
        trains[i] = np.sort(rng.uniform(0, t_end, n))
    return trains


def _make_regular_trains(n_neurons: int, rate_hz: float, t_end: float) -> dict:
    """Generate perfectly regular (clock-like) spike trains for testing."""
    isi = 1.0 / rate_hz
    times = np.arange(isi, t_end, isi)
    return {i: times.copy() for i in range(n_neurons)}


# ---- compute_cv_isi ----

def test_cv_isi_poisson_near_one():
    """Independent Poisson trains → CV-ISI ≈ 1.0."""
    trains = _make_poisson_trains(n_neurons=200, rate_hz=10.0, t_end=10.0, seed=0)
    _, mean_cv = compute_cv_isi(trains, t_start=0.0, t_end=10.0, min_spikes=20)
    assert abs(mean_cv - 1.0) < 0.15, f"Expected CV≈1 for Poisson, got {mean_cv:.3f}"


def test_cv_isi_regular_near_zero():
    """Regular spike trains → CV-ISI ≈ 0."""
    trains = _make_regular_trains(n_neurons=20, rate_hz=10.0, t_end=10.0)
    _, mean_cv = compute_cv_isi(trains, t_start=0.0, t_end=10.0, min_spikes=20)
    assert mean_cv < 0.05, f"Expected CV≈0 for regular trains, got {mean_cv:.3f}"


def test_cv_isi_excludes_low_spike_neurons():
    """Neurons with fewer than min_spikes spikes should be excluded."""
    trains = {0: np.array([0.1, 0.2, 0.3])}  # only 3 spikes
    per_neuron, mean_cv = compute_cv_isi(trains, t_start=0.0, t_end=5.0, min_spikes=10)
    assert len(per_neuron) == 0
    assert np.isnan(mean_cv)


# ---- compute_pairwise_corr ----

def test_pairwise_corr_uncorrelated_near_zero():
    """Independent Poisson trains → mean pairwise correlation ≈ 0."""
    trains = _make_poisson_trains(n_neurons=100, rate_hz=10.0, t_end=10.0, seed=1)
    r = compute_pairwise_corr(trains, t_start=0.0, t_end=10.0,
                               bin_ms=10.0, n_pairs=50, seed=42)
    assert abs(r) < 0.08, f"Expected r≈0 for independent trains, got {r:.4f}"


def test_pairwise_corr_identical_trains():
    """Identical trains → correlation = 1."""
    times = np.array([0.1, 0.3, 0.5, 0.9, 1.4, 2.0])
    trains = {0: times, 1: times}
    r = compute_pairwise_corr(trains, t_start=0.0, t_end=3.0,
                               bin_ms=10.0, n_pairs=1, seed=0)
    assert abs(r - 1.0) < 1e-6, f"Expected r=1 for identical trains, got {r:.6f}"


# ---- compute_power_spectrum ----

def test_power_spectrum_output_shapes():
    """freqs and power must have equal length; freqs[0] = 0; power >= 0."""
    trains = _make_poisson_trains(n_neurons=50, rate_hz=5.0, t_end=5.0, seed=2)
    freqs, power = compute_power_spectrum(trains, t_start=0.0, t_end=5.0, smooth_sigma_ms=5.0)
    assert len(freqs) == len(power)
    assert freqs[0] == pytest.approx(0.0)
    assert np.all(power >= 0)
    assert len(freqs) > 10
