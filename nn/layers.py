"""Composable layers on top of the autodiff engine."""
from __future__ import annotations

import numpy as np

from engine import Parameter, Tensor


class Module:
    def parameters(self):
        return []

    def __call__(self, x):
        return self.forward(x if isinstance(x, Tensor) else Tensor(x))


class Linear(Module):
    """y = x @ W_eff + b, where W_eff = W * mask (see engine.Parameter).

    Weight layout is (in_features, out_features) so that a batch (N, in) maps
    to (N, out) with a plain matmul. Only W is prunable; biases are 1.5% of the
    parameters and pruning them saves nothing meaningful.

    Initialization (derivation):
      Pre-activation z = sum_i w_i x_i over `fan_in` inputs. With independent
      zero-mean weights, Var(z) = fan_in * Var(w) * E[x^2]. To keep Var(z) the
      same from layer to layer (no vanishing/exploding signal) we want
      fan_in * Var(w) ~ 1 ... but ReLU zeroes half of its inputs, halving
      E[x^2], so we need Var(w) = 2 / fan_in  ("He" init).
      For the output layer (no ReLU after it) we use Glorot/Xavier,
      Var(w) = 2 / (fan_in + fan_out), which balances forward and backward
      variance for a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator, init: str = "he"):
        if init == "he":
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
        else:
            raise ValueError(init)
        self.W = Parameter(rng.normal(0.0, std, size=(in_features, out_features)))
        self.b = Parameter(np.zeros(out_features))

    def parameters(self):
        return [self.W, self.b]

    def forward(self, x: Tensor) -> Tensor:
        # (N,in) @ (in,out) + (out,)  -> the (N,D)+(D,) broadcast case
        return x @ self.W.effective() + self.b


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class MLP(Module):
    def __init__(self, sizes, rng: np.random.Generator, activation: str = "relu"):
        act = {"relu": ReLU, "tanh": Tanh}[activation]
        self.activation_name = activation
        self.layers: list[Module] = []
        self.linears: list[Linear] = []
        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            init = "xavier" if (last or activation != "relu") else "he"
            lin = Linear(sizes[i], sizes[i + 1], rng, init=init)
            self.linears.append(lin)
            self.layers.append(lin)
            if not last:
                self.layers.append(act())

    def parameters(self):
        return [p for lin in self.linears for p in lin.parameters()]

    def prunable(self):
        """The weight matrices (not biases) -- what the pruner may touch."""
        return [lin.W for lin in self.linears]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
