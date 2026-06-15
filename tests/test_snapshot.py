# tests/test_snapshot.py
#
# Tests for plasticity/snapshot.py, which handles saving and loading "snapshots" of
# the network during training: the current weight matrix (in sparse COO format),
# recent spike data, trial labels, and training-progress metrics, all stored in an
# HDF5 file. These tests check that data saved for one epoch can be loaded back
# exactly, that monitoring metrics accumulate correctly across multiple epochs, that
# asking for a missing epoch fails the right way, and that provenance info (from a
# baseline file) and training parameters get copied/written correctly.

import h5py
import numpy as np
import pytest

from plasticity.stdp_network import DEFAULT_PARAMS_PLASTICITY
from plasticity.snapshot import (
    save_snapshot, load_snapshot, load_monitoring,
    copy_baseline_provenance, save_training_params,
)


def _dummy_snapshot_data(epoch, n_exc=20):
    """Build a fake but realistically-shaped set of snapshot data for one training
    epoch: a small sparse weight matrix (W_EE_coo), some spike data, trial labels for
    8 directions, and a few monitoring metrics. Used as input for the save/load tests
    below, so we don't need a real training run."""
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
    """Save a snapshot (weights, spikes, trial labels, metrics) for epoch 0 and load it
    straight back. Every piece of data that comes back out should be exactly identical
    to what was saved, i.e. nothing gets dropped, reordered, or corrupted when it goes
    into and back out of the HDF5 file."""
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
    """Besides saving the snapshot itself, save_snapshot should also add a row to the
    "monitoring" table that tracks training progress over epochs. After saving epoch
    0, that table should have one entry (epoch 0), and its values should match the
    metrics dictionary we passed in (mean_rate_E, mean_w_EE, frac_w_max,
    mean_cv_isi)."""
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
    """Save snapshots for two different epochs (0 and 50). The monitoring table should
    now have two rows, one per epoch (so mean_rate_E should have shape (2,)), and the
    snapshots for the two epochs should have different weight data, since each epoch
    gets its own randomly generated dummy weights (_dummy_snapshot_data uses the epoch
    number as part of its random seed)."""
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
    """Only epoch 0 is saved here. Asking load_snapshot for an epoch that was never
    saved (epoch 999) should raise a KeyError, rather than returning empty or
    incorrect data."""
    h5_path = str(tmp_path / "test.h5")
    W_EE_coo, spike_data, trial_labels, metrics = _dummy_snapshot_data(epoch=0)
    save_snapshot(h5_path, epoch=0, W_EE_coo=W_EE_coo, spike_data=spike_data,
                   trial_labels=trial_labels, monitoring_metrics=metrics)

    with pytest.raises(KeyError):
        load_snapshot(h5_path, epoch=999)


def test_copy_baseline_provenance_copies_groups(tmp_path):
    """copy_baseline_provenance should take the 'network', 'weights', and
    'validation' groups from a baseline HDF5 file and copy them into a new training
    HDF5 file, so the training file keeps a record of what baseline network it started
    from. This builds a small fake baseline file with one entry in each of those
    groups, copies it, and checks the new file has the same values."""
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
    copy_baseline_provenance(h5_path, baseline_path)

    with h5py.File(h5_path, 'r') as f:
        assert f['network/N_exc'][()] == 20
        np.testing.assert_array_equal(f['weights/W_EE/data'][:], [1.0, 2.0])
        assert f['validation/mean_rate_E'][()] == pytest.approx(2.5)


def test_save_training_params_writes_attrs(tmp_path):
    """save_training_params should write out the training configuration (the
    plasticity parameters, the fraction of cross-condition trials p_cross, and the
    random seed) as HDF5 attributes on a 'training_params' group, so we can always
    check later exactly what settings a given training run used. This checks a few key
    values (p_cross, seed, tau_plus, w_max, t_burn_in) come back correctly."""
    h5_path = str(tmp_path / "test.h5")
    save_training_params(h5_path, DEFAULT_PARAMS_PLASTICITY, p_cross=0.2, seed=42)

    with h5py.File(h5_path, 'r') as f:
        attrs = f['training_params'].attrs
        assert attrs['p_cross'] == pytest.approx(0.2)
        assert attrs['seed'] == 42
        assert attrs['tau_plus'] == pytest.approx(DEFAULT_PARAMS_PLASTICITY['tau_plus'])
        assert attrs['w_max'] == pytest.approx(DEFAULT_PARAMS_PLASTICITY['w_max'])
        assert attrs['t_burn_in'] == pytest.approx(DEFAULT_PARAMS_PLASTICITY['t_burn_in'])
