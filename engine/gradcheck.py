"""Finite-difference gradient checking.

Idea: the analytic gradient (our backward pass) must agree with the definition
of the derivative, estimated numerically with central differences:

    dL/dx_i  ~  [ L(x + eps*e_i) - L(x - eps*e_i) ] / (2*eps)

Central differences have O(eps^2) truncation error, so with eps=1e-6 in
float64 we can demand ~1e-7 agreement. Slow (2 forward passes per scalar), so
it's for tests only.
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of scalar function f at array x."""
    x = np.array(x, dtype=np.float64)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f(x)
        x[idx] = old - eps
        f_minus = f(x)
        x[idx] = old
        g[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return g


def check_op(fn, *arrays, seed: int = 0, eps: float = 1e-6, rtol: float = 1e-5, atol: float = 1e-7):
    """Check d(sum(R * fn(*inputs)))/d(inputs) for a random fixed weighting R.

    Weighting the output by random R means every output element gets a
    different upstream gradient, so a bug that only shows up for non-uniform
    upstream gradients (e.g. a wrong transpose) cannot hide behind a plain sum.
    Raises AssertionError on mismatch.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]

    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = fn(*tensors)
    R = rng.normal(size=out.data.shape)
    (out * Tensor(R)).sum().backward()

    for i, (t, a) in enumerate(zip(tensors, arrays)):
        def f_i(x, i=i):
            xs = [Tensor(b) for b in arrays]
            xs[i] = Tensor(x)
            return float((fn(*xs).data * R).sum())

        numeric = numerical_grad(f_i, a, eps)
        np.testing.assert_allclose(t.grad, numeric, rtol=rtol, atol=atol, err_msg=f"gradient of input #{i}")
