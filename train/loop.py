"""Mini-batch training loop (shared by Part 2 and Part 3)."""
from __future__ import annotations

import numpy as np

from engine import Tensor, softmax_cross_entropy


def evaluate(model, x, y):
    logits = model(Tensor(x))
    loss = float(softmax_cross_entropy(logits, y).data)
    acc = float((logits.data.argmax(axis=1) == y).mean())
    return loss, acc


def train(model, optimizer, data, epochs, batch_size, rng, pruner=None, verbose=False):
    """Returns a history dict. Order inside a step matters:

        forward -> backward -> pruner.observe() -> optimizer.step() -> pruner.maybe_update()

    observe() runs BEFORE the step so (weight, gradient) are from the same point; the mask only
    changes AFTER the step, so the next forward pass already sees the new connectivity.
    """
    x_tr, y_tr = data["x_train"], data["y_train"]
    n = len(x_tr)
    hist = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_acc", "sparsity"]}
    step = 0
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = softmax_cross_entropy(model(Tensor(x_tr[idx])), y_tr[idx])
            if not np.isfinite(loss.data):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad()
            loss.backward()
            if pruner is not None:
                pruner.observe()
            optimizer.step()
            step += 1
            if pruner is not None:
                pruner.maybe_update(step)
        tl, ta = evaluate(model, x_tr, y_tr)
        vl, va = evaluate(model, data["x_val"], data["y_val"])
        _, te = evaluate(model, data["x_test"], data["y_test"])
        sp = 1.0 - sum(p.mask.sum() for p in model.prunable()) / sum(p.data.size for p in model.prunable())
        for k, v in zip(hist, [epoch, tl, ta, vl, va, te, float(sp)]):
            hist[k].append(v)
        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(f"epoch {epoch:3d}  train {tl:.4f}/{ta:.3f}  val {vl:.4f}/{va:.3f}  sparsity {sp:.3f}")
    return hist
