"""Self-pruning: importance scores, global selection, honest masking, optional regrowth.

IMPORTANCE CRITERIA  (all answer: "how much would the loss rise if I removed w_i?")
---------------------------------------------------------------------------------
Let L(w) be the loss and delta = -w_i * e_i the change that removes weight i.
Taylor-expand around the current point:

    L(w + delta) - L(w) = g_i * (-w_i) + 1/2 * H_ii * w_i^2 + O(w_i^3)

    "magnitude"  : |w_i|                 Ignores both terms' curvature/sensitivity.
                                         Right only if every weight had the same g_i, H_ii.
    "saliency"   : EMA |w_i * g_i|        FIRST-order term. |g_i * w_i| is the size of the
                                         predicted loss change (Molchanov et al.).
                                         g_i is the minibatch gradient; EMA over steps
                                         smooths batch noise.
    "fisher"     : 1/2 * w_i^2 * EMA g_i^2  SECOND-order term, with H_ii replaced by the
                                         diagonal empirical Fisher E[g_i^2] (Optimal Brain
                                         Damage). Near a minimum the first-order term
                                         vanishes (g -> 0) and this term dominates.

HONEST MASKING   a pruned weight gets mask 0 AND stored value 0 (see engine.Parameter);
                 the optimizer wipes its state (see nn.optim).

GLOBAL SELECTION  Scores are normalised per layer (divide by the layer's mean score) so
                  layers with different weight scales compete fairly, ranked globally,
                  with a floor of `min_keep_frac` per layer so no layer can be severed.

REGROWTH (RigL-style, optional)  at each update drop `regrow_frac * n_keep` extra of the
                  weakest survivors, and revive the same number of PRUNED connections whose
                  dense gradient magnitude sqrt(EMA g^2) is largest. Revived weights start at
                  exactly 0 with fresh optimizer state. Newly revived connections are
                  protected for one update so they are not immediately re-pruned before
                  they have had time to grow.  `regrow_frac` is cosine-annealed to 0 over the
                  ramp so the final mask is stable (no flip-flopping during fine-tuning).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schedule import cubic_sparsity

_PROTECTED = 1e30  # score given to weights that must survive this update
_NEG = -np.inf     # score given to ineligible weights


@dataclass
class PruneConfig:
    target_sparsity: float
    criterion: str = "saliency"      # "magnitude" | "saliency" | "fisher"
    start_step: int = 0
    end_step: int = 1000
    update_every: int = 10
    regrow_frac: float = 0.0
    ema_beta: float = 0.9
    min_keep_frac: float = 0.01


def _select_top_k(scores, k, floors):
    """Boolean masks selecting k entries globally by score, forcing `floors[i]` best entries per layer.

    Entries with score -inf are ineligible. Returns a list of bool arrays shaped like `scores`.
    """
    keep = [np.zeros(s.shape, dtype=bool) for s in scores]
    for sel, s, f in zip(keep, scores, floors):
        flat = s.ravel()
        n_ok = int(np.isfinite(flat).sum())
        f = min(f, n_ok)
        if f > 0:
            idx = np.argpartition(-flat, f - 1)[:f]
            sel.reshape(-1)[idx] = True
    n_forced = sum(int(sel.sum()) for sel in keep)
    remaining = max(k - n_forced, 0)
    if remaining > 0:
        flat_all = np.concatenate([np.where(sel, _NEG, s).ravel() for sel, s in zip(keep, scores)])
        n_ok = int(np.isfinite(flat_all).sum())
        remaining = min(remaining, n_ok)
        if remaining > 0:
            idx = np.argpartition(-flat_all, remaining - 1)[:remaining]
            chosen = np.zeros(flat_all.shape, dtype=bool)
            chosen[idx] = True
            off = 0
            for sel in keep:
                n = sel.size
                sel |= chosen[off:off + n].reshape(sel.shape)
                off += n
    return keep


class SelfPruner:
    def __init__(self, params, cfg: PruneConfig):
        assert (cfg.end_step - cfg.start_step) % cfg.update_every == 0, "end-start must be a multiple of update_every"
        assert cfg.criterion in ("magnitude", "saliency", "fisher")
        self.params, self.cfg = list(params), cfg
        self.n_total = sum(p.data.size for p in self.params)
        self.ema_abs_gw = [np.zeros_like(p.data) for p in self.params]
        self.ema_g2 = [np.zeros_like(p.data) for p in self.params]
        self.recent_revived = [np.zeros(p.data.shape, dtype=bool) for p in self.params]
        self._n_obs = 0
        self.history = []  # (step, target_sparsity, achieved_sparsity, n_revived)

    # ---------------------------------------------------------------- statistics
    def observe(self):
        """Call after backward(), before optimizer.step(): weights and gradients are consistent."""
        c, beta = self.cfg, self.cfg.ema_beta
        need_gw = c.criterion == "saliency"
        need_g2 = c.criterion == "fisher" or c.regrow_frac > 0
        if not (need_gw or need_g2):
            return
        for i, p in enumerate(self.params):
            g = p.dense_grad  # dL/dW_eff: defined for pruned positions too
            if need_gw:
                x = np.abs(p.data * g)
                self.ema_abs_gw[i] = x if self._n_obs == 0 else beta * self.ema_abs_gw[i] + (1 - beta) * x
            if need_g2:
                x = g * g
                self.ema_g2[i] = x if self._n_obs == 0 else beta * self.ema_g2[i] + (1 - beta) * x
        self._n_obs += 1

    # ------------------------------------------------------------------- scores
    def _importance(self, i):
        p, c = self.params[i], self.cfg
        if c.criterion == "magnitude":
            return np.abs(p.data)
        if self._n_obs == 0:
            raise RuntimeError("call observe() before maybe_update() for gradient-based criteria")
        if c.criterion == "saliency":
            return self.ema_abs_gw[i]
        return 0.5 * self.ema_g2[i] * p.data ** 2

    @staticmethod
    def _normalise(score, eligible):
        mean = score[eligible].mean() if eligible.any() else 1.0
        return np.where(eligible, score / (mean + 1e-12), _NEG)

    # ------------------------------------------------------------------ schedule
    def maybe_update(self, step: int) -> bool:
        c = self.cfg
        if step < c.start_step or step > c.end_step or (step - c.start_step) % c.update_every:
            return False
        target = cubic_sparsity(step, c.start_step, c.end_step, c.target_sparsity)
        self._update(target, step)
        return True

    def _regrow_fraction(self, step):
        c = self.cfg
        if c.regrow_frac <= 0:
            return 0.0
        progress = np.clip((step - c.start_step) / max(c.end_step - c.start_step, 1), 0.0, 1.0)
        return c.regrow_frac * 0.5 * (1.0 + np.cos(np.pi * progress))  # 1 -> 0 over the ramp

    # -------------------------------------------------------------------- update
    def _update(self, target: float, step: int):
        c = self.cfg
        floors = [int(np.ceil(c.min_keep_frac * p.data.size)) for p in self.params]
        n_keep = max(int(round((1.0 - target) * self.n_total)), sum(floors))
        old_active = [p.mask > 0 for p in self.params]

        r = int(self._regrow_fraction(step) * n_keep)
        r = max(0, min(r, n_keep - sum(floors)))
        n_retain = n_keep - r

        # 1) who survives?  (rank ACTIVE weights by importance)
        scores = []
        for i, p in enumerate(self.params):
            s = self._normalise(self._importance(i), old_active[i])
            s = np.where(self.recent_revived[i] & old_active[i], _PROTECTED, s)
            scores.append(s)
        retained = _select_top_k(scores, n_retain, floors)

        # 2) who comes back?  (rank PRUNED weights by dense-gradient magnitude)
        revived = [np.zeros_like(a) for a in retained]
        if r > 0:
            gscores = []
            for i in range(len(self.params)):
                cand = ~old_active[i]
                gscores.append(self._normalise(np.sqrt(self.ema_g2[i]), cand))
            revived = _select_top_k(gscores, r, [0] * len(self.params))

        # 3) apply: real zeros, fresh start
        for i, p in enumerate(self.params):
            new_active = retained[i] | revived[i]
            newly_pruned = old_active[i] & ~new_active
            p.mask = new_active.astype(np.float64)
            p.data *= p.mask                      # honest zero (revived weights are already 0)
            self.ema_abs_gw[i][newly_pruned] = 0.0
        self.recent_revived = revived
        n_rev = int(sum(int(x.sum()) for x in revived))
        self.history.append((step, target, self.sparsity(), n_rev))

    # ------------------------------------------------------------------- queries
    def sparsity(self) -> float:
        active = sum(int(p.mask.sum()) for p in self.params)
        return 1.0 - active / self.n_total
