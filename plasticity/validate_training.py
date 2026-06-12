# plasticity/validate_training.py

"""
Post-training validation checks (spec section 7). Run after both conditions
have completed:

    PYTHONPATH=. python plasticity/train.py --condition seeded
    PYTHONPATH=. python plasticity/train.py --condition control
    PYTHONPATH=. python plasticity/validate_training.py

Each check_* function takes loaded snapshot/monitoring dicts (from
plasticity.snapshot) and either returns None (pass) or raises AssertionError
with a descriptive message.
"""

import sys

import numpy as np

from plasticity.stdp_network import DEFAULT_PARAMS_PLASTICITY
from plasticity.snapshot import load_snapshot, load_monitoring


def check_no_nans(snapshot, epoch):
    """W_EE weights and spike times must not contain NaNs."""
    w = snapshot['W_EE_coo']['data']
    assert not np.any(np.isnan(w)), f"epoch {epoch}: NaN in W_EE data"
    assert not np.any(np.isnan(snapshot['spike_times_ms'])), \
        f"epoch {epoch}: NaN in spike_times_ms"


def check_monitoring_band(monitoring, condition_name, rate_max=30.0, frac_w_max_max=0.5):
    """Abort criteria (spec 2.5) should hold at every recorded epoch."""
    for epoch, rate, frac in zip(monitoring['epochs'],
                                  monitoring['mean_rate_E'],
                                  monitoring['frac_w_max']):
        assert rate <= rate_max, \
            f"{condition_name} epoch {epoch}: mean_rate_E={rate:.2f} > {rate_max}"
        assert frac <= frac_w_max_max, \
            f"{condition_name} epoch {epoch}: frac_w_max={frac:.3f} > {frac_w_max_max}"
    assert not np.any(np.isnan(monitoring['mean_cv_isi'])), \
        f"{condition_name}: NaN in mean_cv_isi (no neuron had >=20 spikes in a snapshot window)"


def check_weight_movement(snap_epoch0, snap_epoch_n, epoch_n, atol=1e-12):
    """STDP should have changed at least some E->E weights by epoch_n.

    atol=1e-12 A (1 pA) is well below the smallest meaningful STDP step
    (A_plus/A_minus ~ 1e-12 A) but far above float32 rounding noise at the
    ~1e-10 A weight scale -- np.allclose's default atol=1e-8 would swamp
    real differences at this scale and always report "unchanged".
    """
    w0 = snap_epoch0['W_EE_coo']['data']
    wn = snap_epoch_n['W_EE_coo']['data']
    assert w0.shape == wn.shape, "W_EE sparsity pattern changed between snapshots"
    assert not np.allclose(w0, wn, atol=atol), \
        f"W_EE identical between epoch 0 and epoch {epoch_n} -- STDP had no effect"


def check_pool_rescaling(snap_seeded_epoch0, snap_control_epoch0, p_cross, P_size, X_size,
                          atol=1e-6):
    """
    At epoch 0, seeded and control W_EE should have identical connectivity
    (row, col) -- both come from load_baseline(seed=7) -- and differ
    only on P<->X cross-pool synapses, by exactly a factor of p_cross
    (spec 2.2).
    """
    row_s, col_s, w_s = (snap_seeded_epoch0['W_EE_coo'][k] for k in ('row', 'col', 'data'))
    row_c, col_c, w_c = (snap_control_epoch0['W_EE_coo'][k] for k in ('row', 'col', 'data'))

    assert np.array_equal(row_s, row_c) and np.array_equal(col_s, col_c), \
        "seeded and control W_EE have different connectivity at epoch 0 -- " \
        "did both conditions use the same seed?"

    # row = postsynaptic, col = presynaptic (circuit/run_baseline.py's save_baseline convention)
    pre, post = col_s, row_s
    in_P_pre = pre < P_size
    in_X_pre = (pre >= P_size) & (pre < P_size + X_size)
    in_P_post = post < P_size
    in_X_post = (post >= P_size) & (post < P_size + X_size)
    cross = (in_P_pre & in_X_post) | (in_X_pre & in_P_post)

    ratio = w_s[cross] / w_c[cross]
    np.testing.assert_allclose(
        ratio, p_cross, atol=atol,
        err_msg="P<->X cross-pool weights are not scaled by p_cross")
    np.testing.assert_allclose(
        w_s[~cross], w_c[~cross], atol=atol,
        err_msg="non-cross-pool weights differ between seeded and control")


def main():
    params = DEFAULT_PARAMS_PLASTICITY
    snapshot_epochs = (0, 50, 100)

    results = {}
    for condition in ('seeded', 'control'):
        h5_path = f'plasticity/results/training_{condition}.h5'
        monitoring = load_monitoring(h5_path)
        snapshots = {epoch: load_snapshot(h5_path, epoch) for epoch in snapshot_epochs}
        results[condition] = (monitoring, snapshots)

    all_ok = True

    for condition, (monitoring, snapshots) in results.items():
        for epoch, snap in snapshots.items():
            try:
                check_no_nans(snap, epoch)
            except AssertionError as e:
                print(f"FAIL [{condition}] {e}")
                all_ok = False

        try:
            check_monitoring_band(monitoring, condition)
        except AssertionError as e:
            print(f"FAIL [{condition}] {e}")
            all_ok = False

        try:
            check_weight_movement(snapshots[0], snapshots[100], epoch_n=100)
        except AssertionError as e:
            print(f"FAIL [{condition}] {e}")
            all_ok = False

    try:
        check_pool_rescaling(
            results['seeded'][1][0], results['control'][1][0],
            p_cross=params['p_cross_seeded'],
            P_size=params['P_size'], X_size=params['X_size'])
    except AssertionError as e:
        print(f"FAIL [pool rescaling] {e}")
        all_ok = False

    print("\n=== Monitoring summary ===")
    for condition, (monitoring, _) in results.items():
        print(f"\n{condition}:")
        for epoch, rate, w, frac, cv in zip(
                monitoring['epochs'], monitoring['mean_rate_E'],
                monitoring['mean_w_EE'], monitoring['frac_w_max'],
                monitoring['mean_cv_isi']):
            print(f"  epoch {epoch:5d}: rate={rate:6.2f} Hz  "
                  f"w_EE={w * 1e9:7.4f} nA  frac_w_max={frac:.3f}  cv_isi={cv:.3f}")

    if all_ok:
        print("\nAll checks PASSED.")
        return 0
    else:
        print("\nSome checks FAILED -- see above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
