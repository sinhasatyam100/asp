"""Part 3: one self-pruning run.

    python -m train.run_part3 --sparsity 0.9 --criterion saliency
    python -m train.run_part3 --sparsity 0.95 --criterion saliency --regrow 0.2
"""
import argparse
import json
import os

from .experiment import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--criterion", choices=["magnitude", "saliency", "fisher"], default="saliency")
    ap.add_argument("--regrow", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    r = run_experiment(a.seed, a.criterion, a.sparsity, a.regrow, verbose=True)
    print(f"\ncriterion={a.criterion} regrow={a.regrow} seed={a.seed}")
    print(f"target sparsity {a.sparsity:.3f} -> achieved {r['achieved_sparsity']:.4f}  "
          f"({r['active_weights']}/{r['total_weights']} weights active)")
    print(f"test accuracy {r['test_acc']:.4f}   inference FLOPs/sample {r['flops_sparse']} vs dense {r['flops_dense']} "
          f"({r['flops_dense'] / r['flops_sparse']:.1f}x fewer)")
    for l in r["layers"]:
        print(f"  layer {l['layer']} {str(l['shape']):>12}: {l['active']:>5}/{l['total']:<5} active  (sparsity {l['sparsity']:.3f})")

    os.makedirs("results", exist_ok=True)
    out = {k: v for k, v in r.items() if not k.startswith("_")}
    json.dump(out, open(f"results/part3_{a.criterion}_{a.sparsity}.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = r["history"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["epoch"], h["val_acc"], label="val accuracy")
    ax.plot(h["epoch"], h["train_acc"], label="train accuracy", alpha=.6)
    ax2 = ax.twinx(); ax2.plot(h["epoch"], h["sparsity"], "r--", label="weight sparsity"); ax2.set_ylabel("sparsity")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.grid(alpha=.3)
    ax.legend(loc="lower left"); ax2.legend(loc="lower right")
    ax.set_title(f"Self-pruning run: {a.criterion}, target {a.sparsity:.0%}")
    plt.tight_layout(); plt.savefig(f"results/part3_{a.criterion}_{a.sparsity}.png", dpi=130)


if __name__ == "__main__":
    main()
