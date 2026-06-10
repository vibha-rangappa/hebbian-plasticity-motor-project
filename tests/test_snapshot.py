# tests/test_snapshot.py

import h5py
import numpy as np
import pytest

from part2.network_part2 import DEFAULT_PARAMS_PART2
from part2.snapshot import (
    save_snapshot, load_snapshot, load_monitoring,
    copy_part1_provenance, save_part2_params,
)


def _dummy_snapshot_data(epoch, n_exc=20):
    rng = np.random.default_rng(epoch + 1)
    n_syn = 30
    W_EE_coo = {
        'data': rng.uniform(0, 0.24e-9, size=n_syn).astype(np.float32),
        'row': rng.integers(0, n_exc, size=n_syn).astype(np.int32),
        'col': rng.integers(0, n_exc, size=n_syn).astype(np.int32),
        'shape': np.array([n_exc, n_exc], dtype=np.int32),
    }
    spike_data = {
        'spike_times_ms': np.array([1.0, 2.5, 100.0], dtype=np.float32),
        'spike_neuron_idx': np.array([0, 5, 21], dtype=np.int32),
        'spike_trial_idx': np.array([0, 0, 1], dtype=np.int32),
    }
    trial_labels = np.arange(8) % 8
    monitoring_metrics = {
        'mean_rate_E': 2.5 + epoch * 0.01,
        'mean_w_EE': 0.06e-9,
        'frac_w_max': 0.01,
        'mean_cv_isi': 0.9,
    }
    return W_EE_coo, spike_data, trial_labels, monitoring_metrics


def test_save_and_load_snapshot_round_trip(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    W_EE_coo, spike_data, trial_labels, metrics = _dummy_snapshot_data(epoch=0)

    save_snapshot(h5_path, epoch=0, W_EE_coo=W_EE_coo, spike_data=spike_data,
                   trial_labels=trial_labels, monitoring_metrics=metrics)

    loaded = load_snapshot(h5_path, epoch=0)
    np.testing.assert_array_equal(loaded['W_EE_coo']['data'], W_EE_coo['data'])
    np.testing.assert_array_equal(loaded['W_EE_coo']['row'], W_EE_coo['row'])
    np.testing.assert_array_equal(loaded['W_EE_coo']['col'], W_EE_coo['col'])
    np.testing.assert_array_equal(loaded['W_EE_coo']['shape'], W_EE_coo['shape'])
    np.testing.assert_array_equal(loaded['spike_times_ms'], spike_data['spike_times_ms'])
    np.testing.assert_array_equal(loaded['spike_neuron_idx'], spike_data['spike_neuron_idx'])
    np.testing.assert_array_equal(loaded['spike_trial_idx'], spike_data['spike_trial_idx'])
    np.testing.assert_array_equal(loaded['trial_labels'], trial_labels)


def test_save_snapshot_creates_monitoring_row(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    W_EE_coo, spike_data, trial_labels, metrics = _dummy_snapshot_data(epoch=0)
    save_snapshot(h5_path, epoch=0, W_EE_coo=W_EE_coo, spike_data=spike_data,
                   trial_labels=trial_labels, monitoring_metrics=metrics)

    mon = load_monitoring(h5_path)
    np.testing.assert_array_equal(mon['epochs'], [0])
    np.testing.assert_allclose(mon['mean_rate_E'], [metrics['mean_rate_E']])
    np.testing.assert_allclose(mon['mean_w_EE'], [metrics['mean_w_EE']])
    np.testing.assert_allclose(mon['frac_w_max'], [metrics['frac_w_max']])
    np.testing.assert_allclose(mon['mean_cv_isi'], [metrics['mean_cv_isi']])


def test_save_snapshot_appends_monitoring_across_epochs(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    for epoch in (0, 50):
        W_EE_coo, spike_data, trial_labels, metrics = _dummy_snapshot_data(epoch=epoch)
        save_snapshot(h5_path, epoch=epoch, W_EE_coo=W_EE_coo, spike_data=spike_data,
                       trial_labels=trial_labels, monitoring_metrics=metrics)

    mon = load_monitoring(h5_path)
    np.testing.assert_array_equal(mon['epochs'], [0, 50])
    assert mon['mean_rate_E'].shape == (2,)

    snap0 = load_snapshot(h5_path, epoch=0)
    snap50 = load_snapshot(h5_path, epoch=50)
    assert not np.array_equal(snap0['W_EE_coo']['data'], snap50['W_EE_coo']['data'])


def test_load_snapshot_missing_epoch_raises_keyerror(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    W_EE_coo, spike_data, trial_labels, metrics = _dummy_snapshot_data(epoch=0)
    save_snapshot(h5_path, epoch=0, W_EE_coo=W_EE_coo, spike_data=spike_data,
                   trial_labels=trial_labels, monitoring_metrics=metrics)

    with pytest.raises(KeyError):
        load_snapshot(h5_path, epoch=999)


def test_copy_part1_provenance_copies_groups(tmp_path):
    baseline_path = str(tmp_path / "baseline.h5")
    with h5py.File(baseline_path, 'w') as f:
        ng = f.create_group('network')
        ng.create_dataset('N_exc', data=20)
        wg = f.create_group('weights')
        eeg = wg.create_group('W_EE')
        eeg.create_dataset('data', data=np.array([1.0, 2.0], dtype=np.float32))
        vg = f.create_group('validation')
        vg.create_dataset('mean_rate_E', data=2.5)

    h5_path = str(tmp_path / "test.h5")
    copy_part1_provenance(h5_path, baseline_path)

    with h5py.File(h5_path, 'r') as f:
        assert f['network/N_exc'][()] == 20
        np.testing.assert_array_equal(f['weights/W_EE/data'][:], [1.0, 2.0])
        assert f['validation/mean_rate_E'][()] == pytest.approx(2.5)


def test_save_part2_params_writes_attrs(tmp_path):
    h5_path = str(tmp_path / "test.h5")
    save_part2_params(h5_path, DEFAULT_PARAMS_PART2, p_cross=0.2, seed=42)

    with h5py.File(h5_path, 'r') as f:
        attrs = f['part2_params'].attrs
        assert attrs['p_cross'] == pytest.approx(0.2)
        assert attrs['seed'] == 42
        assert attrs['tau_plus'] == pytest.approx(DEFAULT_PARAMS_PART2['tau_plus'])
        assert attrs['w_max'] == pytest.approx(DEFAULT_PARAMS_PART2['w_max'])
        assert attrs['t_burn_in'] == pytest.approx(DEFAULT_PARAMS_PART2['t_burn_in'])
