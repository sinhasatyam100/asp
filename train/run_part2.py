"""Part 2: train the dense MLP with our Adam, report the learning curve.

    python -m train.run_part2
"""
import json
import os

import numpy as np

from .experiment import Hyper, run_experiment


def main():
    hyper = Hyper()
    r = run_experiment(seed=0, target_sparsity=0.0, hyper=hyper, verbose=True)
    h = r["history"]
    assert all(np.isfinite(h["train_loss"])) and all(np.isfinite(h["val_loss"])), "NaN/inf in training"
    print(f"\nfinal: train acc {r['train_acc']:.4f}  val acc {r['val_acc']:.4f}  test acc {r['test_acc']:.4f}  (no NaNs)")

    os.makedirs("results", exist_ok=True)
    json.dump(h, open("results/part2_curve.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["epoch"], h["train_loss"], label="train"); ax[0].plot(h["epoch"], h["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy"); ax[0].set_yscale("log"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(h["epoch"], h["train_acc"], label="train"); ax[1].plot(h["epoch"], h["val_acc"], label="val")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle("Part 2: dense MLP 64-128-128-10, our autodiff + Adam (sklearn digits)")
    plt.tight_layout(); plt.savefig("results/part2_curve.png", dpi=130)
    print("saved results/part2_curve.png and results/part2_curve.json")


if __name__ == "__main__":
    main()
