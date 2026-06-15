# tests/test_analysis.py
#
# Tests for the basic spike-train analysis helpers in circuit/run_baseline.py:
# CV-ISI (how regular vs. irregular each neuron's spiking is), pairwise spike
# count correlation between neurons, and the population power spectrum. Each
# function is checked against spike trains where we already know the right
# answer (e.g. plain Poisson noise, or perfectly regular clock-like firing),
# so we can confirm the math gives sensible numbers before trusting it on
# real simulated data.

import numpy as np
import pytest

from circuit.run_baseline import compute_cv_isi, compute_pairwise_corr, compute_power_spectrum


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
    """Random (Poisson) spike trains should give a CV-ISI close to 1.0, since that is the
    known value for a totally random process with no regularity or burstiness."""
    trains = _make_poisson_trains(n_neurons=200, rate_hz=10.0, t_end=10.0, seed=0)
    _, mean_cv = compute_cv_isi(trains, t_start=0.0, t_end=10.0, min_spikes=20)
    assert abs(mean_cv - 1.0) < 0.15, f"Expected CV≈1 for Poisson, got {mean_cv:.3f}"


def test_cv_isi_regular_near_zero():
    """Perfectly regular, clock-like spike trains should give a CV-ISI close to 0, since
    there is no variability in the time between spikes."""
    trains = _make_regular_trains(n_neurons=20, rate_hz=10.0, t_end=10.0)
    _, mean_cv = compute_cv_isi(trains, t_start=0.0, t_end=10.0, min_spikes=20)
    assert mean_cv < 0.05, f"Expected CV≈0 for regular trains, got {mean_cv:.3f}"


def test_cv_isi_excludes_low_spike_neurons():
    """A neuron with too few spikes to compute a reliable CV-ISI should be dropped from
    the result. Here the neuron has only 3 spikes but min_spikes is set to 10, so it
    should be excluded and the overall mean should come back as NaN (not a number)."""
    trains = {0: np.array([0.1, 0.2, 0.3])}  # only 3 spikes
    per_neuron, mean_cv = compute_cv_isi(trains, t_start=0.0, t_end=5.0, min_spikes=10)
    assert len(per_neuron) == 0
    assert np.isnan(mean_cv)


# ---- compute_pairwise_corr ----

def test_pairwise_corr_uncorrelated_near_zero():
    """Neurons that fire independently (no shared input or coupling) should show a mean
    pairwise correlation close to 0, since there is nothing linking their spike counts."""
    trains = _make_poisson_trains(n_neurons=100, rate_hz=10.0, t_end=10.0, seed=1)
    r = compute_pairwise_corr(trains, t_start=0.0, t_end=10.0,
                               bin_ms=10.0, n_pairs=50, seed=42)
    assert abs(r) < 0.08, f"Expected r≈0 for independent trains, got {r:.4f}"


def test_pairwise_corr_identical_trains():
    """Two neurons that fire at exactly the same times should have a correlation of 1,
    the maximum possible value, since their spike counts move together perfectly."""
    times = np.array([0.1, 0.3, 0.5, 0.9, 1.4, 2.0])
    trains = {0: times, 1: times}
    r = compute_pairwise_corr(trains, t_start=0.0, t_end=3.0,
                               bin_ms=10.0, n_pairs=1, seed=0)
    assert abs(r - 1.0) < 1e-6, f"Expected r=1 for identical trains, got {r:.6f}"


# ---- compute_power_spectrum ----

def test_power_spectrum_output_shapes():
    """Basic sanity checks on the power spectrum output: the frequency and power arrays
    must be the same length, the first frequency bin must be 0 Hz, power values can
    never be negative, and there should be more than just a handful of frequency bins."""
    trains = _make_poisson_trains(n_neurons=50, rate_hz=5.0, t_end=5.0, seed=2)
    freqs, power = compute_power_spectrum(trains, t_start=0.0, t_end=5.0, smooth_sigma_ms=5.0)
    assert len(freqs) == len(power)
    assert freqs[0] == pytest.approx(0.0)
    assert np.all(power >= 0)
    assert len(freqs) > 10
