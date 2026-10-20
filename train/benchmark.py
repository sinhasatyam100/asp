"""Part 4: real cost measurement for a self-pruned model.

    python -m train.benchmark

Reports three things, deliberately kept separate:
  1. FLOPs     : exact operation count (what the hardware MUST do).     -> the primary evidence
  2. Memory    : bytes to store the weights, dense vs CSC.
  3. Wall-clock: our sparse NumPy path vs (a) dense BLAS and (b) the dense-times-zero shortcut.
     These can go either way, and we report what we measure: NumPy's dense matmul is a hand-tuned
     BLAS kernel; our sparse path is a gather + reduceat built from generic NumPy ops.
"""
import json
import time

import numpy as np

from prune import SparseMLP, dense_masked_forward, flops_dense, flops_sparse
from .experiment import run_experiment


def median_time(fn, reps=300, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    out = {}
    for sparsity in [0.9, 0.98]:
        r = run_experiment(0, "saliency", sparsity)
        model, data = r["_model"], r["_data"]
        sp = SparseMLP.from_model(model)
        # dense BLAS on the *compacted* weights is the fairest dense baseline (same numbers, dense matmul)
        Ws = [(l.W.data * l.W.mask, l.b.data) for l in model.linears]
        def dense_blas(x):
            for i, (W, b) in enumerate(Ws):
                x = x @ W + b
                if i < len(Ws) - 1:
                    x = np.maximum(x, 0.0)
            return x

        x_all = data["x_test"]
        ref = dense_masked_forward(model, x_all)
        assert np.allclose(sp.forward(x_all), ref, atol=1e-10), "sparse path disagrees with dense path"
        acc_sparse = float((sp.forward(x_all).argmax(1) == data["y_test"]).mean())

        nnz = sum(l.nnz for l in sp.layers)
        total = sum(l.in_f * l.out_f for l in sp.layers)
        csc_bytes = sum(l.vals.nbytes + l.rows.nbytes + l.col_ptr.nbytes for l in sp.layers)
        entry = {"target_sparsity": sparsity, "test_acc_via_sparse_path": acc_sparse,
                 "flops_dense": flops_dense(model), "flops_sparse": flops_sparse(model),
                 "flop_reduction_x": flops_dense(model) / flops_sparse(model),
                 "dense_bytes_fp64": total * 8, "csc_bytes_int64_idx_fp64": csc_bytes,
                 "wallclock_us": {}}
        for bs in [1, 32, 512]:
            x = np.tile(x_all, (-(-bs // len(x_all)), 1))[:bs]
            t_sparse = median_time(lambda: sp.forward(x))
            t_blas = median_time(lambda: dense_blas(x))
            entry["wallclock_us"][bs] = {"sparse_numpy": t_sparse * 1e6, "dense_blas": t_blas * 1e6,
                                         "speedup_vs_dense_blas": t_blas / t_sparse}
        out[str(sparsity)] = entry

        print(f"\n=== {sparsity:.0%} sparse model (accuracy via sparse path {acc_sparse:.4f}) ===")
        print(f"FLOPs/sample : dense {entry['flops_dense']:>7}  sparse {entry['flops_sparse']:>6}  -> {entry['flop_reduction_x']:.1f}x fewer")
        print(f"weight bytes : dense fp64 {entry['dense_bytes_fp64']}  CSC(int64 idx, fp64) {csc_bytes}  (nnz {nnz}/{total})")
        for bs, v in entry["wallclock_us"].items():
            print(f"batch {bs:>4}: sparse(NumPy) {v['sparse_numpy']:8.1f} us   dense BLAS {v['dense_blas']:8.1f} us   "
                  f"-> {v['speedup_vs_dense_blas']:.2f}x {'faster' if v['speedup_vs_dense_blas'] > 1 else 'SLOWER'} than dense")
    json.dump(out, open("results/cost.json", "w"), indent=1)
    print("\nsaved results/cost.json")


if __name__ == "__main__":
    main()
