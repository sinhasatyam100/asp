"""Part 1 correctness requirement: masked weights, their gradient, and optimizer state.

One command: python -m pytest tests/test_masked.py -v

Each test states the claim it demonstrates. The NaiveAdam below is a deliberately
wrong optimizer (global step counter, ignores the mask, keeps stale state) so we can
SHOW the failure modes, not just assert the absence of them.
"""
import numpy as np

from engine import Parameter, Tensor, softmax_cross_entropy
from engine.gradcheck import numerical_grad
from nn import Adam


# ------------------------------------------------------------------------- fixtures
def make_problem(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(8, 5))
    y = rng.integers(0, 3, size=8)
    W1 = Parameter(rng.normal(size=(5, 6)) * 0.5)
    b1 = Parameter(np.zeros(6))
    W2 = Parameter(rng.normal(size=(6, 3)) * 0.5)
    b2 = Parameter(np.zeros(3))
    W1.mask = (rng.random(W1.data.shape) > 0.5).astype(float)  # ~50% pruned
    W1.data *= W1.mask
    return x, y, W1, b1, W2, b2


def forward_loss(x, y, W1, b1, W2, b2):
    h = (x @ W1.effective() + b1).tanh()
    return softmax_cross_entropy(h @ W2.effective() + b2, y)


class NaiveAdam:
    """What a quick implementation does: global t, trusts p.grad, never touches state."""

    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = list(params), lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * p.grad
            v *= self.b2; v += (1 - self.b2) * p.grad ** 2
            p.data -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


# ------------------------------------------------------- 1. forward: real zeros
def test_masked_weight_contributes_exactly_zero_to_forward():
    x, y, W1, b1, W2, b2 = make_problem()
    before = forward_loss(x, y, W1, b1, W2, b2).data.copy()
    W1.data[W1.mask == 0] = 123.0  # garbage hidden under the mask
    after = forward_loss(x, y, W1, b1, W2, b2).data
    assert before == after  # bit-for-bit: the pruned weight is not "near zero", it is absent


# ------------------------------------------------------- 2. gradient w.r.t. stored W
def test_gradient_wrt_stored_weight_is_exactly_zero_where_masked_and_correct_elsewhere():
    x, y, W1, b1, W2, b2 = make_problem()
    forward_loss(x, y, W1, b1, W2, b2).backward()

    assert np.all(W1.grad[W1.mask == 0] == 0.0)  # exactly, not approximately

    def loss_of_stored(arr):
        old = W1.data.copy()
        W1.data[...] = arr
        val = float(forward_loss(x, y, W1, b1, W2, b2).data)
        W1.data[...] = old
        return val

    numeric = numerical_grad(loss_of_stored, W1.data.copy())
    np.testing.assert_allclose(W1.grad, numeric, rtol=1e-5, atol=1e-8)
    # (numeric is 0 at masked entries too: wiggling a hidden value changes nothing.)


# ------------------------------------------------------- 3. dense gradient is well defined
def test_dense_gradient_at_masked_positions_equals_true_derivative_wrt_effective_weight():
    """dL/dW_eff exists at pruned positions: it is what the loss would do if that connection
    were switched on at value 0. This is the quantity regrowth needs."""
    x, y, W1, b1, W2, b2 = make_problem()
    forward_loss(x, y, W1, b1, W2, b2).backward()

    def loss_of_effective(Weff):
        h = (Tensor(x) @ Tensor(Weff) + Tensor(b1.data)).tanh()
        return float(softmax_cross_entropy(h @ Tensor(W2.data * W2.mask) + Tensor(b2.data), y).data)

    numeric = numerical_grad(loss_of_effective, W1.data * W1.mask)
    np.testing.assert_allclose(W1.dense_grad, numeric, rtol=1e-5, atol=1e-8)
    masked = W1.mask == 0
    assert np.any(np.abs(W1.dense_grad[masked]) > 1e-6)  # genuinely informative, not all zero
    np.testing.assert_allclose(W1.grad, W1.dense_grad * W1.mask)  # the relationship between the two


# ------------------------------------------------------- 4. optimizer: pruned weights stay dead
def _train_then_prune_then_continue(opt_cls, steps_before=30, steps_after=30):
    x, y, W1, b1, W2, b2 = make_problem()
    W1.mask[...] = 1.0  # start dense
    params = [W1, b1, W2, b2]
    opt = opt_cls(params, lr=1e-2)
    for _ in range(steps_before):
        loss = forward_loss(x, y, W1, b1, W2, b2)
        opt.zero_grad(); loss.backward(); opt.step()
    # --- prune ~50% of W1 mid-training (what the pruner does: new mask, real zeros)
    rng = np.random.default_rng(1)
    W1.mask = (rng.random(W1.data.shape) > 0.5).astype(float)
    W1.data *= W1.mask
    for _ in range(steps_after):
        loss = forward_loss(x, y, W1, b1, W2, b2)
        opt.zero_grad(); loss.backward(); opt.step()
    return W1, opt


def test_adam_keeps_pruned_weights_exactly_zero_with_clean_state():
    W1, opt = _train_then_prune_then_continue(Adam)
    dead = W1.mask == 0
    assert np.all(W1.data[dead] == 0.0)
    assert np.all(opt.m[0][dead] == 0.0) and np.all(opt.v[0][dead] == 0.0) and np.all(opt.t[0][dead] == 0.0)


def test_naive_adam_lets_pruned_weights_drift_off_zero():
    """The bug this challenge is about: zero gradient != zero update when the optimizer has memory."""
    W1, _ = _train_then_prune_then_continue(NaiveAdam)
    dead = W1.mask == 0
    assert np.any(W1.data[dead] != 0.0)  # stale momentum kept moving the 'pruned' weights
    assert np.max(np.abs(W1.data[dead])) > 1e-3


# ------------------------------------------------------- 5. revived connection restarts fresh
def _revival_first_step(opt_cls, lr=1e-2):
    w = Parameter(np.array([[0.7]]))
    opt = opt_cls([w], lr=lr)
    for _ in range(200):                      # long history: global t = 200
        w.grad[...] = 1.0; opt.step()
    w.mask[...] = 0.0; w.data *= w.mask       # prune it
    for _ in range(5):                        # pruned for a while (grads zero, as the engine gives)
        w.grad[...] = 0.0; opt.step()
    w.mask[...] = 1.0                         # REVIVE: the pruner re-activates it at value exactly 0
    w.data[...] = 0.0                         # (already true for Adam; forced for Naive to isolate the
                                              #  moment/step-counter bug from the drift bug tested above)
    w.grad[...] = 1.0
    before = w.data.copy()
    opt.step()
    return abs((w.data - before)[0, 0]), lr


def test_revived_connection_takes_a_fresh_adam_step_of_size_lr():
    delta, lr = _revival_first_step(Adam)
    # fresh element, t=1: m_hat = g, v_hat = g^2  =>  step = lr * g/(|g|+eps) ~= lr
    np.testing.assert_allclose(delta, lr, rtol=1e-6)


def test_naive_adam_revival_step_is_wrong():
    delta, lr = _revival_first_step(NaiveAdam)
    assert abs(delta - lr) / lr > 0.2  # stale m/v from before the pruning leak into the revived weight


def test_global_step_counter_alone_would_be_wrong():
    """Isolates the bias-correction bug: wipe m,v on revival but keep a global t=1000.
    m_hat ~ 0.1*g (b1^t ~ 0) but v_hat = 0.001*g^2 / (1 - 0.999^1000) ~ 0.00158*g^2,
    so the step is ~2.5*lr (-> 3.16*lr as t -> infinity), not lr."""
    lr, b1, b2, t = 1e-2, 0.9, 0.999, 1000
    g = 1.0
    m, v = (1 - b1) * g, (1 - b2) * g * g
    step = lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
    assert 2.4 < step / lr < 2.6
