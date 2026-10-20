"""Sparse-aware inference path (own implementation, NumPy only) + honest cost accounting.

Why this file exists: the challenge says a pruned model that "multiplies by zeros" is not
cheaper. `dense_masked_forward` is that shortcut (kept ONLY as a baseline to compare against).
`SparseMLP` stores just the surviving weights and does work proportional to their number.

Storage format: CSC (compressed sparse COLUMN) of W with shape (in, out):
    rows[k], vals[k] : row index and value of the k-th stored weight, sorted by output column
    col_ptr[j]       : weights of output unit j live in [col_ptr[j], col_ptr[j+1])
Forward for a batch X (B,in):   Y[:, j] = sum_{k in column j} X[:, rows[k]] * vals[k]
implemented as one gather  X[:, rows] * vals  -> (B, nnz)   (nnz multiplies per sample)
followed by a segmented sum with np.add.reduceat over column boundaries (nnz adds).
"""
from __future__ import annotations

import numpy as np


class SparseLinear:
    def __init__(self, W_eff: np.ndarray, mask: np.ndarray, b: np.ndarray):
        in_f, out_f = W_eff.shape
        rows, cols = np.nonzero(mask)                 # structure comes from the MASK, not the values
        order = np.lexsort((rows, cols))              # sort by output column, then row
        self.rows = rows[order]
        self.vals = W_eff[rows[order], cols[order]].copy()
        counts = np.bincount(cols, minlength=out_f)
        self.col_ptr = np.concatenate([[0], np.cumsum(counts)])
        self.nonempty = np.nonzero(counts)[0]         # reduceat can't handle empty segments
        self.starts = self.col_ptr[self.nonempty]
        self.b = b.copy()
        self.in_f, self.out_f, self.nnz = in_f, out_f, int(len(self.vals))

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros((x.shape[0], self.out_f))
        if self.nnz:
            prod = x[:, self.rows] * self.vals        # (B, nnz): the only multiplications performed
            y[:, self.nonempty] = np.add.reduceat(prod, self.starts, axis=1)
        return y + self.b


class SparseMLP:
    def __init__(self, layers, acts):
        self.layers, self.acts = layers, acts

    @classmethod
    def from_model(cls, model):
        layers = [SparseLinear(l.W.data * l.W.mask, l.W.mask, l.b.data) for l in model.linears]
        return cls(layers, model.activation_name)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0) if self.acts == "relu" else np.tanh(x)
        return x


def dense_masked_forward(model, x: np.ndarray) -> np.ndarray:
    """The dense-times-zero shortcut: full (in*out) multiplies per layer regardless of sparsity.
    NOT a real speed-up -- zeros still cost a multiply-add on dense hardware. Baseline only."""
    for i, lin in enumerate(model.linears):
        x = x @ (lin.W.data * lin.W.mask) + lin.b.data
        if i < len(model.linears) - 1:
            x = np.maximum(x, 0.0) if model.activation_name == "relu" else np.tanh(x)
    return x


# ------------------------------------------------------------------ cost accounting
# Convention: a multiply-add = 2 FLOPs; bias adds and activations are excluded because they are
# identical for the dense and sparse paths. FLOPs are PER SAMPLE.
def flops_dense(model) -> int:
    return int(sum(2 * l.W.data.size for l in model.linears))


def flops_sparse(model) -> int:
    return int(sum(2 * int(l.W.mask.sum()) for l in model.linears))


def sparsity_report(model) -> dict:
    layers, active, total = [], 0, 0
    for i, l in enumerate(model.linears):
        n, a = l.W.data.size, int(l.W.mask.sum())
        # "honest masking": every pruned weight must be an EXACT zero in storage
        assert np.all(l.W.data[l.W.mask == 0] == 0.0), "pruned weight is not an exact zero"
        layers.append({"layer": i, "shape": list(l.W.data.shape), "active": a, "total": n, "sparsity": 1 - a / n})
        active += a
        total += n
    n_bias = int(sum(l.b.data.size for l in model.linears))
    return {"layers": layers, "active_weights": active, "total_weights": total,
            "weight_sparsity": 1 - active / total,
            "active_params_incl_bias": active + n_bias, "total_params_incl_bias": total + n_bias}
