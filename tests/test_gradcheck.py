"""Part 1: gradient-checking suite. One command: python -m pytest tests/test_gradcheck.py"""
import numpy as np
import pytest

from engine import Tensor, softmax_cross_entropy
from engine.gradcheck import check_op

rng = np.random.default_rng(123)


def rnd(*shape):
    return rng.normal(size=shape)


def away_from_zero(*shape, margin=0.2):
    """Random values with |x| > margin (finite differences straddle kinks otherwise)."""
    x = rng.normal(size=shape)
    return np.where(np.abs(x) < margin, np.sign(x + 1e-12) * margin, x)


# ------------------------------------------------------------ element-wise ops
@pytest.mark.parametrize("name,fn,arrs", [
    ("add same shape",        lambda a, b: a + b,  (rnd(3, 4), rnd(3, 4))),
    ("sub same shape",        lambda a, b: a - b,  (rnd(3, 4), rnd(3, 4))),
    ("mul same shape",        lambda a, b: a * b,  (rnd(3, 4), rnd(3, 4))),
    ("div same shape",        lambda a, b: a / b,  (rnd(3, 4), away_from_zero(3, 4, margin=0.5))),
    ("neg",                   lambda a: -a,        (rnd(3, 4),)),
])
def test_elementwise(name, fn, arrs):
    check_op(fn, *arrs)


# ----------------------------------------------------------------- broadcasting
@pytest.mark.parametrize("name,fn,arrs", [
    ("(N,D)+(D,)   <- the bias case", lambda a, b: a + b, (rnd(5, 3), rnd(3))),
    ("(D,)+(N,D)",                    lambda a, b: a + b, (rnd(3), rnd(5, 3))),
    ("(N,D)*(D,)",                    lambda a, b: a * b, (rnd(5, 3), rnd(3))),
    ("(N,1)-(1,D) outer",             lambda a, b: a - b, (rnd(5, 1), rnd(1, 3))),
    ("(N,D)/(D,)",                    lambda a, b: a / b, (rnd(5, 3), away_from_zero(3, margin=0.5))),
    ("(N,D)+(1,)",                    lambda a, b: a + b, (rnd(5, 3), rnd(1))),
    ("(N,D)*scalar-tensor ()",        lambda a, b: a * b, (rnd(5, 3), np.array(1.7))),
])
def test_broadcasting(name, fn, arrs):
    check_op(fn, *arrs)


def test_python_scalars_and_reflected_ops():
    check_op(lambda a: 3.0 + a, rnd(4))
    check_op(lambda a: 2.0 * a, rnd(4))
    check_op(lambda a: 1.0 - a, rnd(4))
    check_op(lambda a: 1.0 / a, away_from_zero(4, margin=0.5))


# ----------------------------------------------------------------------- matmul
def test_matmul():
    check_op(lambda a, b: a @ b, rnd(4, 3), rnd(3, 5))


def test_linear_layer_expression():
    # x @ W + b : matmul and bias-broadcast composed
    check_op(lambda x, W, b: x @ W + b, rnd(6, 4), rnd(4, 3), rnd(3))


# ------------------------------------------------------------------- reductions
@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, False), (0, True), (1, True), (-1, False)])
def test_sum(axis, keepdims):
    check_op(lambda a: a.sum(axis=axis, keepdims=keepdims), rnd(4, 5))


@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, False), (1, True)])
def test_mean(axis, keepdims):
    check_op(lambda a: a.mean(axis=axis, keepdims=keepdims), rnd(4, 5))


# -------------------------------------------------------------- non-linearities
def test_relu():
    check_op(lambda a: a.relu(), away_from_zero(4, 5))


def test_tanh():
    check_op(lambda a: a.tanh(), rnd(4, 5))


def test_sigmoid():
    check_op(lambda a: a.sigmoid(), rnd(4, 5))


def test_sigmoid_extreme_inputs_are_finite():
    out = Tensor(np.array([-1e4, 0.0, 1e4])).sigmoid().data
    assert np.all(np.isfinite(out)) and out[0] == 0.0 and out[1] == 0.5 and out[2] == 1.0


# ------------------------------------------------------- softmax cross-entropy
def test_softmax_cross_entropy_gradient():
    y = np.array([0, 2, 1, 2, 0])
    check_op(lambda z: softmax_cross_entropy(z, y), rnd(5, 3))


def test_softmax_cross_entropy_matches_naive_formula():
    z = rnd(6, 4)
    y = rng.integers(0, 4, size=6)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    expected = -np.log(p[np.arange(6), y]).mean()
    assert np.isclose(softmax_cross_entropy(Tensor(z), y).data, expected, rtol=1e-12)


def test_softmax_cross_entropy_is_stable_for_huge_logits():
    z = np.array([[1000.0, 0.0, -1000.0], [-1000.0, 1000.0, 0.0]])
    y = np.array([0, 1])
    t = Tensor(z, requires_grad=True)
    loss = softmax_cross_entropy(t, y)
    loss.backward()
    assert np.isfinite(loss.data) and np.all(np.isfinite(t.grad))
    assert loss.data < 1e-6  # confidently correct => ~0 loss


# ------------------------------------------------- graph mechanics (topo + accum)
def test_gradient_accumulates_when_a_tensor_is_reused():
    x = Tensor(np.array([2.0, -3.0]), requires_grad=True)
    y = (x * x + x).sum()  # dy/dx = 2x + 1
    y.backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1)


def test_diamond_graph():
    x = Tensor(np.array([1.5]), requires_grad=True)
    a, b = x * 2.0, x * 3.0
    (a * b).sum().backward()  # (2x)(3x) = 6x^2 -> 12x
    np.testing.assert_allclose(x.grad, 12 * x.data)


def test_deep_chain_has_no_recursion_limit():
    x = Tensor(np.array([1.0]), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + x  # y = 5001 * x
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [5001.0])


def test_full_mlp_gradcheck():
    y = np.array([0, 1, 2, 1])

    def net(x, W1, b1, W2, b2):
        h = (x @ W1 + b1).tanh()
        return softmax_cross_entropy(h @ W2 + b2, y)

    check_op(net, rnd(4, 3), rnd(3, 5), rnd(5), rnd(5, 3), rnd(3))
