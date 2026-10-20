# DESIGN.md

## 1. The importance criterion, and why it approximates "loss increase if removed"

Removing connection *i* means moving the weight vector by `delta = -w_i * e_i` (set `w_i` to 0).
Taylor-expand the loss around the current weights:

```
DeltaL_i = L(w - w_i e_i) - L(w)
         = -g_i * w_i  +  (1/2) * H_ii * w_i^2  +  O(w_i^3)
           ^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^
           first order      second order (diagonal of the Hessian)
```

Each criterion in `prune/pruner.py` is an approximation of this quantity:

| criterion | score | what it keeps from the expansion |
|---|---|---|
| `magnitude` | `abs(w_i)` | nothing: exact only if every weight had the same gradient and curvature |
| `saliency` | `EMA abs(w_i * g_i)` | the **first-order** term: size of the predicted loss change (Molchanov et al.) |
| `fisher` | `1/2 * w_i^2 * EMA g_i^2` | the **second-order** term, with `H_ii` replaced by the diagonal empirical Fisher `E[g_i^2]` (Optimal Brain Damage) |

Why the Fisher stands in for the Hessian: for softmax cross-entropy, the Hessian of the loss w.r.t.
the weights is (Gauss-Newton term) + (term proportional to the residual). Near a minimum with a
reasonably calibrated model the second term is small and the Gauss-Newton term equals the Fisher
information `E[grad l grad l^T]`, whose diagonal is `E[g_i^2]`.

Where each approximation is weak (these are the things I would expect to be asked):

* **First-order term vanishes at a minimum.** There `g -> 0`, so `|g w|` is dominated by minibatch
  noise: it ends up measuring (gradient noise scale) x |w|. That is why the EMA over many steps is
  used, and why the second-order `fisher` variant exists. We pre-declared saliency as the primary
  criterion because it is the one the task names; the sweep tests fisher as an ablation.
* **`|g w|` is a magnitude of change, not a signed one.** The true `DeltaL` can be negative
  (removing a weight can *help*). We rank by `|.|`, i.e. by sensitivity.
* **Scores are additive-independent.** Removing many weights at once ignores off-diagonal Hessian
  terms. This is the reason to prune *gradually* (cubic schedule) and fine-tune between cuts, rather
  than one-shot.
* **Staleness.** The EMA averages over weights that have since changed.

Layers are compared after dividing each layer's scores by that layer's mean score, with a 1% floor
per layer, so one layer's weight scale cannot decide how the sparsity budget is split. The same
normalisation is applied to the magnitude baseline, so the baseline is not handicapped.

**Why cubic schedule:** `s(t) = s_f * (1 - (1 - (t-t0)/(t_end-t0))^3)`. Steep early (the network is
massively redundant), flat late (every remaining weight matters, and the optimizer needs many steps
between cuts to re-route function through survivors).

## 2. What is "the gradient of a masked weight"?

Forward: `W_eff = W (.) M` (element-wise), `y = x W_eff + b`.
There are **two** gradients, and the engine keeps both (`engine/tensor.py: masked_weight`):

```
dense_grad = dL/dW_eff = x^T (dL/dy)            defined at EVERY position, masked or not
grad       = dL/dW     = dense_grad (.) M        chain rule through (.) M: exactly 0 where M = 0
```

* The optimizer uses `grad`. It is the true derivative of the function the network computes w.r.t.
  the stored parameter, so autodiff stays honest: perturbing a hidden value under the mask changes
  nothing, hence derivative 0 (`test_gradient_wrt_stored_weight_is_exactly_zero...`, checked against
  finite differences).
* `dense_grad` is *also* well-defined at pruned positions: it is the derivative w.r.t. the effective
  weight at `W_eff = 0`, i.e. how the loss would respond if that connection were switched on. It is
  verified against finite differences too (`test_dense_gradient_at_masked_positions...`) and is what
  regrowth ranks by. If we zeroed it, a pruned connection could never earn its way back.

**Optimizer.** "Zero gradient" does not mean "zero update" for a stateful optimizer. Three bugs a
quick implementation ships, each demonstrated in `tests/test_masked.py` against a deliberately naive Adam:

1. *Drift*: stale `m, v` keep moving a pruned weight even though its gradient is 0, so the stored
   weight is no longer a real zero (`test_naive_adam_lets_pruned_weights_drift_off_zero`).
2. *Stale revival*: a revived connection inherits old `m, v`, so its first step points in an
   irrelevant direction with a stale size (`test_naive_adam_revival_step_is_wrong`).
3. *Global bias-correction counter*: with `m = v = 0` but a global `t = 1000`, the first step is
   about 2.5x `lr` (-> 3.16x as `t` grows) instead of `lr`
   (`test_global_step_counter_alone_would_be_wrong`).

Our `Adam` enforces, at the top of every step: masked => `m = v = 0`, per-element step count
`t = 0`, stored weight `= 0` exactly. A revived connection therefore starts at exactly 0 with a fresh
state and takes a first step of exactly `lr` (`test_revived_connection_takes_a_fresh_adam_step_of_size_lr`).

## 3. Where the engine bottlenecks, and how I would optimize it

Measured on one training step (batch 64, MLP 64-128-128-10, one CPU core; note the profile ran while
a background job shared the core, so trust the proportions more than the absolute milliseconds):
forward ~0.8 ms, backward ~1.1 ms, **Adam step ~1.6 ms**. The largest single cost is the optimizer:
the per-element-state Adam makes ~15 full-array passes with fresh temporaries per parameter. The
matmuls themselves (BLAS) are a minor share; the rest is Python-object overhead (a `Tensor`, a
closure and a `np.zeros_like` gradient per op) and allocation.

What I would do, in order of payoff:

1. **Fuse and preallocate the optimizer**: in-place updates with reusable scratch buffers (or a
   compiled kernel). Same math, far fewer passes and allocations.
2. **Lazy gradient allocation**: allocate `grad` on first accumulation, not `zeros_like` in every
   `Tensor.__init__`.
3. **Fused `linear` op** (`x @ (W*M) + b` as one node): removes two graph nodes per layer.
4. **Static graph / tape reuse**: the MLP graph is identical every step; build it once and replay.
5. **float32** instead of float64 (kept float64 here so finite-difference tests are tight).
6. **Training does not get cheaper from sparsity here.** We compute the dense gradient every step
   (needed for `dense_grad`). RigL-style training computes the dense gradient only on mask-update
   steps and the masked gradient otherwise. Only *inference* is claimed to be cheaper in this repo.

## 4. Serving a self-pruned model in a multi-tenant service at thousands of requests per second

What our own measurements say (Part 4): the pruned model does 10x (90%) / 50x (98%) fewer FLOPs, and
the CSC weights are 4.75x smaller (fp64/int64 here; fp32 + int32 indices is 8 bytes per nonzero versus
4 per dense weight, so break-even is at 50% density). But my NumPy sparse path was **slower in
wall-clock** than dense BLAS at every batch size. Unstructured sparsity reduces *work*, not
automatically *time*: dense hardware is built for dense matmuls. The serving plan follows from that:

* **Make the sparsity hardware-friendly.** Change the pruner's selection step to N:M (e.g. 2:4)
  or block sparsity, which sparse tensor cores / vendor kernels can accelerate, or prune whole
  neurons (structured) so the result is a smaller *dense* model that every runtime already
  accelerates. The selection function `_select_top_k` is the only piece that has to change.
* **Use sparse kernels, not a Python path**, and keep a dense fallback behind a flag. Gate each
  artifact with the equivalence check we already test (`test_sparse_forward_equals_dense_masked_forward`).
* **Batching**: dynamic micro-batching per model amortises gather/launch overhead; sparse paths
  tend to want larger batches than dense ones, so the batching window is a latency/throughput knob.
* **Multi-tenancy**: the memory saving is the real win, as more tenant models fit per host. Keep a
  versioned model registry, load models on demand into an LRU cache, shard tenants across replicas
  (stateless workers behind a router), isolate tenants by process/container with per-tenant quotas
  so one tenant's model cannot starve others, and report p50/p99 latency, nnz and sparsity per model.
* **Safe rollout**: before a self-pruned model replaces a dense one, evaluate it on that tenant's
  held-out data (our sweep shows accuracy loss is task- and seed-dependent), shadow it against the
  dense model on live traffic, then canary.
* Quantization (int8) stacks on top of sparsity but changes the accuracy picture; evaluate jointly.

## 5. Honest limitations

* **Small, easy dataset** (sklearn digits, 1,078 training samples, a 359-sample test set). The dense
  model fits the training set perfectly by epoch 5. At 90% sparsity accuracy is unchanged, which says
  more about the dataset's redundancy than about the pruner. A harder dataset (e.g. Fashion-MNIST)
  would separate criteria better; the pipeline accepts any arrays.
* **No evidence that saliency beats magnitude here** (see `results/paired_stats.json`): gains at 50%
  (+0.50 pp, CI [+0.22, +0.78]) but the differences at 75-98% are within noise. 15 paired comparisons
  were made, so about one 95% CI excluding zero is expected by chance; 4 did. I report them as
  observed, not explained. The magnitude baseline is already strong because of per-layer
  normalisation plus gradual pruning plus fine-tuning, which probably does most of the work.
* **Regrowth** did not help (it is significantly *worse* at 95%, -0.31 pp, CI [-0.58, -0.03]). Its
  stability risks (revived weights are pruned straight back; flip-flopping masks) are mitigated by
  one-update protection and a cosine anneal of the regrow fraction to 0, but I did not tune it.
* **Wall-clock**: the sparse NumPy path is slower than dense BLAS (see Section 4 and `results/cost.json`).
* Hyper-parameters (lr 3e-3, 60 epochs, ramp epochs 6-40) were set once, not searched.
