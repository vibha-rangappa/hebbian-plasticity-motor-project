# tests/test_validate_training.py

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
    """epochs_data: dict {epoch: (W_EE_coo, spike_data, trial_labels, metrics)}"""
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
    coo = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5",
                                  {0: (coo, _basic_spikes(), [0, 1], _basic_metrics())})
    snap = load_snapshot(h5_path, epoch=0)
    check_no_nans(snap, epoch=0)  # should not raise


def test_check_no_nans_raises_on_nan_weight(tmp_path):
    coo = _basic_coo([0, 1], [1, 0], [np.nan, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5",
                                  {0: (coo, _basic_spikes(), [0, 1], _basic_metrics())})
    snap = load_snapshot(h5_path, epoch=0)
    with pytest.raises(AssertionError, match="NaN in W_EE"):
        check_no_nans(snap, epoch=0)


def test_check_monitoring_band_passes_in_range(tmp_path):
    coo = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5", {
        0:  (coo, _basic_spikes(), [0, 1], _basic_metrics(rate=3.0, frac=0.02)),
        50: (coo, _basic_spikes(), [0, 1], _basic_metrics(rate=8.0, frac=0.10)),
    })
    monitoring = load_monitoring(h5_path)
    check_monitoring_band(monitoring, "test")  # should not raise


def test_check_monitoring_band_raises_on_runaway_rate(tmp_path):
    coo = _basic_coo([0, 1], [1, 0], [0.05e-9, 0.06e-9])
    h5_path = _make_snapshot_h5(tmp_path, "a.h5", {
        0: (coo, _basic_spikes(), [0, 1], _basic_metrics(rate=50.0)),
    })
    monitoring = load_monitoring(h5_path)
    with pytest.raises(AssertionError, match="mean_rate_E"):
        check_monitoring_band(monitoring, "test")


def test_check_weight_movement_raises_if_unchanged(tmp_path):
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
    # P_size=2, X_size=2. row=postsynaptic, col=presynaptic.
    # synapse 0: pre=0,post=1 -> P->P (not cross)
    # synapse 1: pre=0,post=2 -> P->X (cross)
    # synapse 2: pre=2,post=0 -> X->P (cross)
    # synapse 3: pre=2,post=3 -> X->X (not cross)
    row = [1, 2, 0, 3]
    col = [0, 0, 2, 2]
    w_control = [1.0e-9, 1.0e-9, 1.0e-9, 1.0e-9]
    w_seeded  = [1.0e-9, 0.2e-9, 0.2e-9, 1.0e-9]  # cross terms x0.2

    h5_seeded = _make_snapshot_h5(
        tmp_path, "seeded.h5",
        {0: (_basic_coo(row, col, w_seeded), _basic_spikes(), [0, 1], _basic_metrics())})
    h5_control = _make_snapshot_h5(
        tmp_path, "control.h5",
        {0: (_basic_coo(row, col, w_control), _basic_spikes(), [0, 1], _basic_metrics())})

    snap_seeded = load_snapshot(h5_seeded, epoch=0)
    snap_control = load_snapshot(h5_control, epoch=0)

    check_pool_rescaling(snap_seeded, snap_control, p_cross=0.2, P_size=2, X_size=2)


def test_check_pool_rescaling_raises_on_connectivity_mismatch(tmp_path):
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
