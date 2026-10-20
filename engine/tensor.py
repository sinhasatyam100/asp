"""Minimal reverse-mode autodiff on top of NumPy.

Mental model
------------
Every operation returns a NEW Tensor that remembers (a) its parents and (b) a
closure `_backward` that knows how to push the output's gradient to those
parents. `loss.backward()` walks the graph from the output back to the inputs
in reverse topological order and calls each closure once.

Two rules carry almost all of the correctness:

1. ACCUMULATE, never assign. A tensor used in two places (x*x, a diamond in
   the graph, a weight used twice) receives gradient from each use; the
   contributions add. Every closure therefore does `parent.grad += ...`.

2. UNBROADCAST. If an op silently broadcast an input (e.g. (N,D) + (D,)), the
   upstream gradient has the *broadcast* shape. The input only owns D numbers,
   so its gradient is the upstream gradient SUMMED over the broadcast axes.
   (Intuition: the same b[j] was copied to N rows; every copy contributed.)

Masking (the part this challenge is really about) lives at the bottom:
`Parameter` + `masked_weight`.
"""
from __future__ import annotations

import numpy as np


def _noop():
    pass


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` down to `shape` (the inverse of NumPy broadcasting)."""
    # (a) NumPy prepends size-1 axes when ranks differ: (D,) acts like (1, D).
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # (b) axes of size 1 that were stretched to a bigger size.
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _acc(t: "Tensor", g: np.ndarray) -> None:
    """Accumulate gradient g into tensor t (no-op if t doesn't need grad)."""
    if t.requires_grad:
        t.grad += _unbroadcast(g, t.data.shape)


def _as_tensor(x) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    __array_priority__ = 1000  # make `ndarray + Tensor` defer to Tensor.__radd__

    def __init__(self, data, requires_grad: bool = False, _parents=(), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self._parents = tuple(_parents)
        # A node needs a gradient if it is a trainable leaf OR depends on one.
        self.requires_grad = bool(requires_grad) or any(p.requires_grad for p in self._parents)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._backward = _noop
        self._op = _op

    # ------------------------------------------------------------------ basics
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.grad is not None:
            self.grad[...] = 0.0

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op or 'leaf'})"

    # -------------------------------------------------------------- arithmetic
    def __add__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data + other.data, _parents=(self, other), _op="add")

        def _backward():
            _acc(self, out.grad)
            _acc(other, out.grad)

        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, _parents=(self,), _op="neg")

        def _backward():
            _acc(self, -out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data - other.data, _parents=(self, other), _op="sub")

        def _backward():
            _acc(self, out.grad)
            _acc(other, -out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data * other.data, _parents=(self, other), _op="mul")

        def _backward():
            _acc(self, out.grad * other.data)
            _acc(other, out.grad * self.data)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = _as_tensor(other)
        out = Tensor(self.data / other.data, _parents=(self, other), _op="div")

        def _backward():
            _acc(self, out.grad / other.data)
            _acc(other, -out.grad * self.data / (other.data ** 2))

        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return _as_tensor(other) - self

    def __rtruediv__(self, other):
        return _as_tensor(other) / self

    # ------------------------------------------------------------------ matmul
    def __matmul__(self, other):
        other = _as_tensor(other)
        assert self.data.ndim == 2 and other.data.ndim == 2, "matmul supports 2-D only"
        out = Tensor(self.data @ other.data, _parents=(self, other), _op="matmul")

        def _backward():
            # C = A @ B  =>  dA = dC @ B^T,  dB = A^T @ dC
            if self.requires_grad:
                _acc(self, out.grad @ other.data.T)
            if other.requires_grad:
                _acc(other, self.data.T @ out.grad)

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return _as_tensor(other) @ self

    # -------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims: bool = False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _parents=(self,), _op="sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)  # put the reduced axis back (size 1)
            _acc(self, np.broadcast_to(g, self.data.shape))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False):
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            n = int(np.prod([self.data.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / float(n)

    # ---------------------------------------------------------- non-linearities
    def relu(self):
        pos = self.data > 0.0  # subgradient convention: relu'(0) = 0
        out = Tensor(self.data * pos, _parents=(self,), _op="relu")

        def _backward():
            _acc(self, out.grad * pos)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _parents=(self,), _op="tanh")

        def _backward():
            _acc(self, out.grad * (1.0 - t * t))

        out._backward = _backward
        return out

    def sigmoid(self):
        # sigmoid(x) = 0.5 * (1 + tanh(x/2)): exact identity, no overflow in exp.
        s = 0.5 * (1.0 + np.tanh(0.5 * self.data))
        out = Tensor(s, _parents=(self,), _op="sigmoid")

        def _backward():
            _acc(self, out.grad * s * (1.0 - s))

        out._backward = _backward
        return out

    # --------------------------------------------------------------- backward
    def backward(self):
        """Backpropagate from this (scalar) tensor.

        Call once per forward graph: intermediate nodes are created fresh with
        zero gradients on every forward pass, so a second backward() on the
        same graph would double-count.
        """
        if self.data.size != 1:
            raise ValueError("backward() needs a scalar output (e.g. a loss)")
        if not self.requires_grad:
            raise ValueError("output does not depend on any tensor that requires grad")

        # Iterative post-order DFS => `topo` lists every node AFTER all of its
        # parents. (Iterative, not recursive, so deep graphs can't blow the stack.)
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))

        self.grad = np.ones_like(self.data)  # d(out)/d(out) = 1
        for node in reversed(topo):  # children before parents
            node._backward()


# ----------------------------------------------------------------------------
# Softmax + cross-entropy, fused and numerically stable
# ----------------------------------------------------------------------------
def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy of softmax(logits) against integer class targets.

    Stability: softmax is invariant to subtracting a per-row constant, so we
    subtract the row max before exp (no overflow), and compute log-probs as
    `shifted - log(sum(exp(shifted)))` (no log(0)).

    Backward (derived once, used forever): for one sample,
        dL/dz_k = softmax_k - 1[k == y]
    and the batch mean contributes a factor 1/N.
    """
    z = logits.data
    n = z.shape[0]
    rows = np.arange(n)
    shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    s = e.sum(axis=1, keepdims=True)
    log_probs = shifted - np.log(s)
    loss = -log_probs[rows, targets].mean()
    out = Tensor(loss, _parents=(logits,), _op="softmax_ce")

    def _backward():
        g = e / s
        g[rows, targets] -= 1.0
        _acc(logits, g * (out.grad / n))

    out._backward = _backward
    return out


# ----------------------------------------------------------------------------
# Masking: the correctness requirement
# ----------------------------------------------------------------------------
class Parameter(Tensor):
    """A trainable leaf with a binary connectivity mask.

    Stored value  : self.data   (the optimizer updates this)
    Mask          : self.mask   (1.0 = connection exists, 0.0 = pruned)
    Forward uses  : W_eff = data * mask   (see `effective()`)

    Invariant maintained by the whole codebase: where mask == 0, data == 0
    EXACTLY (a real zero, not a small number).

    Two different gradients exist, and keeping them apart is the whole trick:

      dense_grad = dL/dW_eff   defined at EVERY position, including pruned ones.
                               At a pruned position it answers "what would the
                               loss do if this connection existed with value 0
                               and I nudged it?" -> used to decide REGROWTH.

      grad       = dL/dW       = dense_grad * mask   (chain rule through W*M).
                               Exactly 0 at pruned positions. This is the true
                               derivative w.r.t. the stored parameter, and the
                               one the optimizer sees.
    """

    def __init__(self, data):
        super().__init__(np.array(data, dtype=np.float64), requires_grad=True)
        self.mask = np.ones_like(self.data)
        self.dense_grad = np.zeros_like(self.data)

    def effective(self) -> Tensor:
        return masked_weight(self, self.mask)

    def zero_grad(self):
        self.grad[...] = 0.0
        self.dense_grad[...] = 0.0


def masked_weight(w: Parameter, mask: np.ndarray) -> Tensor:
    """W_eff = W * mask, recording the dense gradient on the way back."""
    out = Tensor(w.data * mask, _parents=(w,), _op="mask")

    def _backward():
        # out.grad is dL/dW_eff: dense, nonzero even where mask == 0.
        w.dense_grad += out.grad
        # Chain rule through the multiply-by-mask: pruned entries get exactly 0.
        w.grad += out.grad * mask

    out._backward = _backward
    return out
