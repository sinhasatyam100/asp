"""Part 3/4 mechanics: schedule, exact sparsity, honest zeros, regrowth invariants, sparse path."""
import numpy as np
import pytest

from engine import Tensor, softmax_cross_entropy
from nn import MLP, Adam
from prune import SelfPruner, PruneConfig, SparseMLP, cubic_sparsity, dense_masked_forward, flops_dense, flops_sparse, sparsity_report
from prune.pruner import _select_top_k


# ------------------------------------------------------------------------ schedule
def test_cubic_schedule_endpoints_monotone_and_front_loaded():
    s = [cubic_sparsity(t, 100, 500, 0.9) for t in range(0, 700)]
    assert s[0] == 0.0 and s[100] == 0.0 and s[500] == 0.9 and s[-1] == 0.9
    assert all(b >= a for a, b in zip(s, s[1:]))
    assert s[300] > 0.9 * 0.5            # more than half the sparsity is reached by the midpoint
    assert (s[101] - s[100]) > (s[499] - s[498])  # steep at the start, flat at the end


# ----------------------------------------------------------------- selection logic
def test_select_top_k_global_with_floors():
    a = np.array([[5.0, 4.0], [3.0, 2.0]])
    b = np.array([1.0, 0.5])
    keep = _select_top_k([a, b], k=3, floors=[0, 1])
    assert keep[1].sum() == 1 and keep[1][0]                # floor forces best of layer b
    assert keep[0].sum() == 2 and keep[0][0, 0] and keep[0][0, 1]  # then global best fill


def test_select_top_k_ignores_ineligible():
    a = np.array([3.0, -np.inf, 2.0])
    keep = _select_top_k([a], k=5, floors=[0])
    assert keep[0].tolist() == [True, False, True]


# ---------------------------------------------------------- pruner on a real (tiny) run
def tiny_run(criterion, target, regrow=0.0, seed=0, steps=300):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(128, 10))
    y = (x[:, 0] + x[:, 1] > 0).astype(int) + (x[:, 2] > 1).astype(int)
    model = MLP([10, 32, 32, 3], rng)
    opt = Adam(model.parameters(), lr=1e-2)
    pr = SelfPruner(model.prunable(), PruneConfig(target, criterion, start_step=50, end_step=200,
                                                 update_every=10, regrow_frac=regrow))
    for step in range(steps):
        loss = softmax_cross_entropy(model(Tensor(x)), y)
        opt.zero_grad(); loss.backward()
        if criterion != "magnitude" or regrow > 0:
            pr.observe()
        opt.step()
        pr.maybe_update(step + 1)
        assert np.isfinite(loss.data)
    return model, opt, pr


@pytest.mark.parametrize("criterion", ["magnitude", "saliency", "fisher"])
def test_pruner_hits_target_sparsity_and_zeros_are_exact(criterion):
    model, opt, pr = tiny_run(criterion, 0.9)
    rep = sparsity_report(model)  # also asserts pruned weights are exact zeros
    assert abs(rep["weight_sparsity"] - 0.9) < 0.005
    for p, m, v, t in zip(model.parameters(), opt.m, opt.v, opt.t):  # optimizer state is clean too
        dead = p.mask == 0
        assert np.all(m[dead] == 0) and np.all(v[dead] == 0) and np.all(t[dead] == 0)


def test_no_layer_is_severed_at_extreme_sparsity():
    model, _, _ = tiny_run("saliency", 0.98)
    assert all(l["active"] >= 1 for l in sparsity_report(model)["layers"])


def test_regrowth_keeps_exact_budget_and_revived_weights_start_at_zero():
    rng = np.random.default_rng(0)
    model, opt, pr = tiny_run("saliency", 0.9, regrow=0.3)
    assert any(h[3] > 0 for h in pr.history), "regrowth never fired"
    assert abs(sparsity_report(model)["weight_sparsity"] - 0.9) < 0.01
    # replay a single update and check the revived entries explicitly
    before = [p.mask.copy() for p in model.prunable()]
    pr._update(0.9, step=60)
    for p, b, rev in zip(model.prunable(), before, pr.recent_revived):
        assert np.all(b[rev] == 0) and np.all(p.mask[rev] == 1)   # were pruned, now live
        assert np.all(p.data[rev] == 0.0)                           # come back as exact zeros


# ------------------------------------------------------------------- sparse path
def random_model(sparsity=0.9, seed=0):
    rng = np.random.default_rng(seed)
    m = MLP([20, 30, 25, 4], rng)
    for p in m.prunable():
        p.mask = (rng.random(p.data.shape) > sparsity).astype(float)
        p.data *= p.mask
    # force an entirely-empty output column to exercise the reduceat edge case
    m.linears[1].W.mask[:, 3] = 0.0
    m.linears[1].W.data[:, 3] = 0.0
    return m, rng


def test_sparse_forward_equals_dense_masked_forward():
    m, rng = random_model()
    x = rng.normal(size=(7, 20))
    dense = dense_masked_forward(m, x)
    sparse = SparseMLP.from_model(m).forward(x)
    np.testing.assert_allclose(sparse, dense, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dense, m(Tensor(x)).data, rtol=1e-12, atol=1e-12)  # and the autodiff model


def test_flops_scale_with_active_weights():
    m, _ = random_model(0.9)
    rep = sparsity_report(m)
    assert flops_sparse(m) == 2 * rep["active_weights"]
    assert flops_dense(m) == 2 * rep["total_weights"]
    assert flops_sparse(m) < 0.15 * flops_dense(m)


# -------------------------------------------------------------------- reproducibility
def test_same_seed_same_result():
    a, _, _ = tiny_run("saliency", 0.8, seed=3)
    b, _, _ = tiny_run("saliency", 0.8, seed=3)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert np.array_equal(pa.data, pb.data) and np.array_equal(pa.mask, pb.mask)
