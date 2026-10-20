"""Part 4: sparsity-accuracy Pareto sweep with paired statistics.

    python -m train.sweep                # full sweep (10 seeds)  ~10 min on 1 CPU
    python -m train.sweep --seeds 3      # quick look

Design of the comparison (why the numbers can be trusted):
  * PAIRED: for a given seed, every criterion gets the same data split, the same initial weights and
    the same batch order. The only thing that differs is which weights get pruned. So we compare
    criteria on per-seed DIFFERENCES, which cancels most seed-to-seed noise.
  * Reported accuracy is test accuracy of the FINAL epoch (no best-epoch cherry-picking);
    hyper-parameters were chosen once and not tuned on test.
  * The primary claim's sparsity level (90%) was fixed before running the sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from .experiment import run_experiment

SPARSITIES = [0.5, 0.75, 0.9, 0.95, 0.98]
CONFIGS = [  # (label, criterion, regrow_frac)
    ("magnitude", "magnitude", 0.0),
    ("saliency", "saliency", 0.0),
    ("fisher", "fisher", 0.0),
    ("saliency+regrow", "saliency", 0.2),
]
CLAIM_SPARSITY = 0.9
OUT = "results"


def bootstrap_ci(x, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = list(range(args.seeds))
    runs = []

    t0 = time.time()
    for seed in seeds:
        r = run_experiment(seed, "saliency", 0.0)
        runs.append({k: r[k] for k in ["seed", "criterion", "target_sparsity", "achieved_sparsity", "train_acc", "val_acc",
                                       "test_acc", "active_weights", "total_weights", "flops_dense", "flops_sparse"]}
                    | {"label": "dense"})
    print(f"dense baselines done ({time.time() - t0:.0f}s)", flush=True)
    for s in SPARSITIES:
        for label, crit, regrow in CONFIGS:
            for seed in seeds:
                r = run_experiment(seed, crit, s, regrow)
                runs.append({k: r[k] for k in ["seed", "criterion", "target_sparsity", "achieved_sparsity", "train_acc",
                                               "val_acc", "test_acc", "active_weights", "total_weights", "flops_dense",
                                               "flops_sparse"]} | {"label": label})
            accs = [x["test_acc"] for x in runs if x["label"] == label and x["target_sparsity"] == s]
            print(f"sparsity {s:.2f}  {label:16s} test {np.mean(accs):.4f} +- {np.std(accs, ddof=1):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(runs, open(f"{OUT}/raw_runs.json", "w"), indent=1)

    analyse(runs, seeds)


def analyse(runs, seeds):
    labels = ["dense"] + [c[0] for c in CONFIGS]
    def accs(label, s):
        sel = sorted([r for r in runs if r["label"] == label and r["target_sparsity"] == s], key=lambda r: r["seed"])
        return np.array([r["test_acc"] for r in sel]), np.array([r["achieved_sparsity"] for r in sel]), sel

    dense_acc, _, _ = accs("dense", 0.0)

    # ---------- table (CSV)
    rows = []
    for label in labels[1:]:
        for s in SPARSITIES:
            a, ach, sel = accs(label, s)
            rows.append({"criterion": label, "target_sparsity": s, "achieved_sparsity": ach.mean(),
                         "test_acc_mean": a.mean(), "test_acc_std": a.std(ddof=1), "n_seeds": len(a),
                         "flops_sparse": sel[0]["flops_sparse"], "flops_dense": sel[0]["flops_dense"],
                         "flop_reduction_x": sel[0]["flops_dense"] / np.mean([r["flops_sparse"] for r in sel])})
    rows.insert(0, {"criterion": "dense", "target_sparsity": 0.0, "achieved_sparsity": 0.0, "test_acc_mean": dense_acc.mean(),
                    "test_acc_std": dense_acc.std(ddof=1), "n_seeds": len(dense_acc),
                    "flops_sparse": runs[0]["flops_dense"], "flops_dense": runs[0]["flops_dense"], "flop_reduction_x": 1.0})
    with open(f"{OUT}/pareto.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---------- paired statistics vs magnitude
    stats = {}
    for s in SPARSITIES:
        mag, _, _ = accs("magnitude", s)
        for label in labels[2:]:
            a, _, _ = accs(label, s)
            d = a - mag
            lo, hi = bootstrap_ci(d)
            stats[f"{label}@{s}"] = {"mean_diff_vs_magnitude": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                                    "ci95_lo": lo, "ci95_hi": hi, "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                                    "losses": int((d < 0).sum()), "n": len(d), "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    json.dump(stats, open(f"{OUT}/paired_stats.json", "w"), indent=1)

    # ---------- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = {"magnitude": "tab:gray", "saliency": "tab:blue", "fisher": "tab:green", "saliency+regrow": "tab:red"}
    ax[0].axhline(dense_acc.mean(), color="k", ls="--", lw=1, label=f"dense ({dense_acc.mean():.3f})")
    ax[0].axhspan(dense_acc.mean() - dense_acc.std(ddof=1), dense_acc.mean() + dense_acc.std(ddof=1), color="k", alpha=0.07)
    for label in labels[1:]:
        r = [x for x in rows if x["criterion"] == label]
        ax[0].errorbar([x["achieved_sparsity"] * 100 for x in r], [x["test_acc_mean"] for x in r],
                       yerr=[x["test_acc_std"] for x in r], marker="o", capsize=3, label=label, color=colors[label])
    ax[0].set_xlabel("achieved weight sparsity (%)"); ax[0].set_ylabel(f"test accuracy (mean +- sd, {len(seeds)} seeds)")
    ax[0].set_title("Sparsity-accuracy Pareto curve"); ax[0].legend(loc="lower left"); ax[0].grid(alpha=0.3)
    for label in labels[2:]:
        m = [stats[f"{label}@{s}"] for s in SPARSITIES]
        x = np.array(SPARSITIES) * 100
        mid = np.array([v["mean_diff_vs_magnitude"] for v in m]) * 100
        lo = np.array([v["ci95_lo"] for v in m]) * 100; hi = np.array([v["ci95_hi"] for v in m]) * 100
        ax[1].errorbar(x, mid, yerr=[mid - lo, hi - mid], marker="o", capsize=3, label=label, color=colors[label])
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xlabel("target sparsity (%)"); ax[1].set_ylabel("accuracy gain over magnitude (pct points)")
    ax[1].set_title("Paired difference vs magnitude (95% bootstrap CI)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/pareto.png", dpi=140); plt.close()

    # ---------- the falsifiable claim (pre-declared sparsity; text generated from the numbers)
    s = CLAIM_SPARSITY
    sal, _, sel = accs("saliency", s); mag, _, _ = accs("magnitude", s)
    st = stats[f"saliency@{s}"]
    verdict = ("the paired difference is distinguishable from zero at 95%" if st["ci_excludes_zero"]
               else "the 95% CI includes zero, so we do NOT claim saliency is better at this sparsity")
    claim = (f"At {s:.0%} target sparsity (achieved {np.mean([r['achieved_sparsity'] for r in sel]):.1%}), saliency pruning "
             f"retains {sal.mean():.2%} test accuracy (sd {sal.std(ddof=1):.2%}) vs {mag.mean():.2%} (sd {mag.std(ddof=1):.2%}) "
             f"for magnitude pruning, mean of {len(seeds)} paired seeds; dense is {dense_acc.mean():.2%}. "
             f"Paired difference {st['mean_diff_vs_magnitude'] * 100:+.2f} pp, 95% bootstrap CI "
             f"[{st['ci95_lo'] * 100:+.2f}, {st['ci95_hi'] * 100:+.2f}] pp; saliency won {st['wins']}, tied {st['ties']}, "
             f"lost {st['losses']} of {st['n']} seeds -- {verdict}. "
             f"Inference cost: {sel[0]['flops_dense'] / np.mean([r['flops_sparse'] for r in sel]):.1f}x fewer FLOPs "
             f"({int(np.mean([r['active_weights'] for r in sel]))} of {sel[0]['total_weights']} weights active).")
    open(f"{OUT}/claim.md", "w").write(claim + "\n")
    print("\nCLAIM:", claim)


if __name__ == "__main__":
    main()
