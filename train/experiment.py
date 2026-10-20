"""One reproducible experiment = (seed, criterion, target sparsity, regrowth)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nn import MLP, Adam
from prune import PruneConfig, SelfPruner, flops_dense, flops_sparse, sparsity_report
from .data import load_digits_split
from .loop import evaluate, train


@dataclass(frozen=True)
class Hyper:
    sizes: tuple = (64, 128, 128, 10)
    epochs: int = 60
    batch_size: int = 64
    lr: float = 3e-3
    prune_start_epoch: int = 6      # warm-up: learn a reasonable function before any cutting
    prune_end_epoch: int = 40       # ramp ends; remaining epochs fine-tune with the mask frozen
    update_every: int = 10          # steps between pruning updates


def run_experiment(seed: int, criterion: str = "saliency", target_sparsity: float = 0.0,
                   regrow_frac: float = 0.0, hyper: Hyper = Hyper(), verbose: bool = False) -> dict:
    data = load_digits_split(seed)
    # Three independent, explicitly-seeded streams: no global RNG state anywhere.
    rng_init = np.random.default_rng(10_000 + seed)
    rng_batch = np.random.default_rng(20_000 + seed)

    model = MLP(list(hyper.sizes), rng_init)
    opt = Adam(model.parameters(), lr=hyper.lr)

    spe = -(-len(data["x_train"]) // hyper.batch_size)  # steps per epoch (ceil)
    pruner = None
    if target_sparsity > 0:
        start = hyper.prune_start_epoch * spe
        n_updates = round((hyper.prune_end_epoch - hyper.prune_start_epoch) * spe / hyper.update_every)
        pruner = SelfPruner(model.prunable(), PruneConfig(
            target_sparsity=target_sparsity, criterion=criterion, start_step=start,
            end_step=start + n_updates * hyper.update_every, update_every=hyper.update_every,
            regrow_frac=regrow_frac))

    hist = train(model, opt, data, hyper.epochs, hyper.batch_size, rng_batch, pruner, verbose)

    _, tr_acc = evaluate(model, data["x_train"], data["y_train"])
    _, va_acc = evaluate(model, data["x_val"], data["y_val"])
    _, te_acc = evaluate(model, data["x_test"], data["y_test"])
    rep = sparsity_report(model)  # asserts pruned weights are exact zeros
    return {
        "seed": seed, "criterion": criterion if target_sparsity > 0 else "dense",
        "target_sparsity": target_sparsity, "regrow_frac": regrow_frac,
        "achieved_sparsity": rep["weight_sparsity"], "active_weights": rep["active_weights"],
        "total_weights": rep["total_weights"], "layers": rep["layers"],
        "train_acc": tr_acc, "val_acc": va_acc, "test_acc": te_acc,
        "flops_dense": flops_dense(model), "flops_sparse": flops_sparse(model),
        "history": hist, "_model": model, "_data": data,
    }
