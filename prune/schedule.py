"""Sparsity schedule.

Cubic ramp (Zhu & Gupta, 2017, "To prune, or not to prune"):

    s(t) = s_f + (s_i - s_f) * (1 - (t - t0) / (t_end - t0)) ** 3     for t0 <= t <= t_end

Why cubic: the cubic is steep at the start and flat at the end. Early in
training the network is hugely redundant (most weights are still near their
random init and carry little signal), so we can remove a lot quickly. Late,
each remaining weight matters, so we remove slowly and give the optimizer many
steps to recover between cuts. One-shot pruning at the end skips that recovery:
the network never gets to re-route function through the survivors.
"""


def cubic_sparsity(step: int, start: int, end: int, final: float, initial: float = 0.0) -> float:
    if step <= start:
        return initial
    if step >= end:
        return final
    frac = (step - start) / (end - start)
    return final + (initial - final) * (1.0 - frac) ** 3
