# tests/test_validate_training.py
#
# Tests for plasticity/validate_training.py: a set of "sanity check"
# functions that run after training to catch obvious problems. These checks
# look for NaNs in the saved weights, make sure the monitored firing rate and
# weight statistics stayed within a healthy band, confirm that weights
# actually changed over training (learning happened), and confirm that the
# seeded cross-pool weight pattern (pool rescaling) matches what we expect
# compared to a control network. Each test builds a small fake snapshot file
# with made-up numbers, then checks that the validation function either
# passes quietly on good data or raises a clear error on bad data.

import numpy as np
import pytest

from plasticity.snapshot import save_snapshot, load_snapshot, load_monitoring
from plasticity.validate_training import (
    check_no_nans,
    check_monitoring_band,
    check_weight_movement,
    check_pool_rescaling,
)


def _make_snapshot_h5(tmp_path, name, epochs_data):
    """Build a small HDF5 snapshot file for testing. epochs_data is a dict
    mapping each epoch number to its (W_EE_coo, spike_data, trial_labels,
    metrics) tuple, which gets written to the file with save_snapshot()."""
    h5_path = str(tmp_path / name)
    for epoch, (W_EE_coo, spike_data, trial_labels, metrics) in epochs_data.items():
        save_snapshot(h5_path, epoch, W_EE_coo, spike_data, trial_labels, metrics)
    return h5_path


def _basic_coo(row, col, data, n=4):
    return {'data': np.array(data, dtype=np.float32),
            'row': np.array(row, dtype=np.int32),
            'col': np.array(col, dtype=np.int32),
            'shape': np.array([n, n], dtype=np.int32)}


def _basic_spikes():
    return {'spike_times_ms': np.array([1.0, 2.0], dtype=np.float32),
            'spike_neuron_idx': np.array([0, 1], dtype=np.int32),
            'spike_trial_idx': np.array([0, 0], dtype=np.int32)}


def _basic_metrics(rate=5.0, w=0.06e-9, frac=0.05, cv=0.9):
    return {'mean_rate_E': rate, 'mean_w_EE': w, 'frac_w_max': frac, 'mean_cv_isi': cv}


def test_check_no_nans_passes_on_clean_snapshot(tmp_path):
    # A snapshot with normal, finite weight values should pass check_no_nans()
    # without raising anything.
    coo = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5",
                                  {0: (coo, _basic_spikes(), [0, 1], _basic_metrics())})
    snap = load_snapshot(h5_path, epoch=0)
    check_no_nans(snap, epoch=0)  # should not raise


def test_check_no_nans_raises_on_nan_weight(tmp_path):
    # If one of the saved weights is NaN ("not a number", i.e. the simulation
    # produced an invalid value), check_no_nans() should catch it and raise
    # an error that mentions "NaN in W_EE".
    coo = _basic_coo([0, 1], [1, 0], [np.nan, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5",
                                  {0: (coo, _basic_spikes(), [0, 1], _basic_metrics())})
    snap = load_snapshot(h5_path, epoch=0)
    with pytest.raises(AssertionError, match="NaN in W_EE"):
        check_no_nans(snap, epoch=0)


def test_check_monitoring_band_passes_in_range(tmp_path):
    # Two epochs with firing rates (3 Hz and 8 Hz) and weight fractions
    # (0.02 and 0.10) that are both within the expected healthy range.
    # check_monitoring_band() should pass without raising.
    coo = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5", {
        0:  (coo, _basic_spikes(), [0, 1], _basic_metrics(rate=3.0, frac=0.02)),
        50: (coo, _basic_spikes(), [0, 1], _basic_metrics(rate=8.0, frac=0.10)),
    })
    monitoring = load_monitoring(h5_path)
    check_monitoring_band(monitoring, "test")  # should not raise


def test_check_monitoring_band_raises_on_runaway_rate(tmp_path):
    # A mean E firing rate of 50 Hz is way too high, a sign of runaway
    # activity. check_monitoring_band() should catch this and raise an error
    # mentioning "mean_rate_E".
    coo = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5", {
        0: (coo, _basic_spikes(), [0, 1], _basic_metrics(rate=50.0)),
    })
    monitoring = load_monitoring(h5_path)
    with pytest.raises(AssertionError, match="mean_rate_E"):
        check_monitoring_band(monitoring, "test")


def test_check_weight_movement_raises_if_unchanged(tmp_path):
    # Both snapshots (epoch 0 and epoch 100) use the exact same weight
    # values. If the weights never moved at all over 100 epochs, that means
    # learning isn't happening, so check_weight_movement() should raise an
    # error mentioning "identical".
    coo = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5", {
        0:   (coo, _basic_spikes(), [0, 1], _basic_metrics()),
        100: (coo, _basic_spikes(), [0, 1], _basic_metrics()),
    })
    snap0 = load_snapshot(h5_path, epoch=0)
    snap100 = load_snapshot(h5_path, epoch=100)
    with pytest.raises(AssertionError, match="identical"):
        check_weight_movement(snap0, snap100, epoch_n=100)


def test_check_weight_movement_passes_if_changed(tmp_path):
    # Here the weights at epoch 100 are different from epoch 0 (0.07e-9 and
    # 0.04e-9 instead of 0.05e-9 and 0.06e-9), so check_weight_movement()
    # should pass without raising: learning did happen.
    coo0 = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    coo100 = _basic_coo([0, 1], [1, 0], [0.07e-9, 0.04e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5", {
        0:   (coo0, _basic_spikes(), [0, 1], _basic_metrics()),
        100: (coo100, _basic_spikes(), [0, 1], _basic_metrics()),
    })
    snap0 = load_snapshot(h5_path, epoch=0)
    snap100 = load_snapshot(h5_path, epoch=100)
    check_weight_movement(snap0, snap100, epoch_n=100)  # should not raise


def test_check_pool_rescaling_passes_for_correctly_scaled_weights(tmp_path):
    # Two pools of size 2: P = {0, 1} and X = {2, 3}.
    # In the saved COO format, "row" is the postsynaptic neuron and "col" is
    # the presynaptic neuron. The four synapses here are:
    #   synapse 0: pre=0, post=1 -> P->P (within-pool, not cross)
    #   synapse 1: pre=0, post=2 -> P->X (cross-pool)
    #   synapse 2: pre=2, post=0 -> X->P (cross-pool)
    #   synapse 3: pre=2, post=3 -> X->X (within-pool, not cross)
    row = [1, 2, 0, 3]
    col = [0, 0, 2, 2]
    w_control = [1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9]
    # In the "seeded" network, the cross-pool synapses (1 and 2) were scaled
    # down by p_cross=0.2 (1.0e-9 -> 0.2e-9), while the within-pool synapses
    # (0 and 3) are unchanged.
    w_seeded  = [1.0e-9, 0.2e-9, 0.2e-9, 1.0e-9]

    h5_seeded = _make_snapshot_h5(
        tmp_path, "seeded.h5",
        {0: (_basic_coo(row, col, w_seeded), _basic_spikes(), [0, 1], _basic_metrics())})
    h5_control = _make_snapshot_h5(
        tmp_path, "control.h5",
        {0: (_basic_coo(row, col, w_control), _basic_spikes(), [0, 1], _basic_metrics())})

    snap_seeded = load_snapshot(h5_seeded, epoch=0)
    snap_control = load_snapshot(h5_control, epoch=0)

    # Comparing the seeded network to the control, the cross-pool weights
    # should be exactly p_cross=0.2 times the control weights. Since that's
    # true here, check_pool_rescaling() should pass without raising.
    check_pool_rescaling(snap_seeded, snap_control, p_cross=0.2, P_size=2, X_size=2)


def test_check_pool_rescaling_raises_on_connectivity_mismatch(tmp_path):
    # The "seeded" and "control" networks are supposed to have the same
    # connectivity (same synapses), just with different cross-pool weights.
    # Here the second synapse's postsynaptic neuron (row[1]) differs between
    # the two files (2 vs 3), so the connectivity doesn't match. This should
    # make check_pool_rescaling() raise an error mentioning "connectivity".
    coo_seeded = _basic_coo([1, 2], [0, 0], [1.0e-9, 0.2e-9])
    coo_control = _basic_coo([1, 3], [0, 0], [1.0e-9, 1.0e-9])  # row[1] differs

    h5_seeded = _make_snapshot_h5(
        tmp_path, "seeded.h5",
        {0: (coo_seeded, _basic_spikes(), [0, 1], _basic_metrics())})
    h5_control = _make_snapshot_h5(
        tmp_path, "control.h5",
        {0: (coo_control, _basic_spikes(), [0, 1], _basic_metrics())})

    snap_seeded = load_snapshot(h5_seeded, epoch=0)
    snap_control = load_snapshot(h5_control, epoch=0)

    with pytest.raises(AssertionError, match="connectivity"):
        check_pool_rescaling(snap_seeded, snap_control, p_cross=0.2, P_size=2, X_size=2)
