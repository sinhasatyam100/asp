"""Optimizers that understand connectivity masks.

THE PROBLEM
-----------
Pruning sets mask[i,j] = 0. The engine then gives that weight an exactly-zero
gradient (chain rule through W*M). A naive optimizer would think "zero
gradient, nothing to do" -- but Adam and momentum are *stateful*:

  * Adam keeps m (EMA of gradients) and v (EMA of squared gradients). A weight
    that had a big gradient before it was pruned still has big m, v. Even with
    a zero gradient now, update = lr * m_hat / (sqrt(v_hat)+eps) is NONZERO:
    the stored weight keeps drifting while "pruned" (momentum coasting).
  * When a connection is later REVIVED, it inherits that stale m, v, so its
    first steps point in an old, irrelevant direction with a stale step size.
  * Adam's bias correction uses a step counter t. With one GLOBAL t (e.g.
    t=1000), a freshly revived weight (m=v=0) is "bias-corrected" as if it had
    1000 steps of history: its first step is ~2.5x lr at t=1000 (-> 3.16x as t grows) instead of lr.

THE FIX (three invariants, enforced at the top of every step)
-------------------------------------------------------------
  1. masked  =>  m = v = 0           (no history survives pruning)
  2. masked  =>  per-element step count t = 0   (revived weights start fresh,
                                                 bias correction is per-element)
  3. masked  =>  stored weight == 0 exactly     (a real zero, not "small")
"""
from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class SGD(Optimizer):
    """SGD with (heavy-ball) momentum:  v <- mu*v + g ;  w <- w - lr*v."""

    def __init__(self, params, lr=0.1, momentum=0.9):
        super().__init__(params)
        self.lr, self.mu = lr, momentum
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.v):
            mask = p.mask
            v *= mask                 # invariant 1: no velocity survives pruning
            g = p.grad * mask         # defence in depth (grad is already 0 there)
            v *= self.mu
            v += g
            p.data -= self.lr * v
            p.data *= mask            # invariant 3


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = [np.zeros_like(p.data) for p in self.params]  # PER-ELEMENT step counts

    def step(self):
        for p, m, v, t in zip(self.params, self.m, self.v, self.t):
            mask = p.mask
            # Invariants 1 & 2: wipe state of anything masked BEFORE using it.
            # (Newly pruned entries still hold old state at this point.)
            m *= mask
            v *= mask
            t *= mask
            g = p.grad * mask

            t += mask  # only live connections advance their own clock
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g

            tt = np.maximum(t, 1.0)  # avoid 0/0 on masked entries (their m=0 anyway)
            m_hat = m / (1.0 - self.b1 ** tt)
            v_hat = v / (1.0 - self.b2 ** tt)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data *= mask  # invariant 3
