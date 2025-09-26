import numpy as np
import math
from typing import List

class SLP:
    def __init__(self, input_size: int, output_size: int, seed: int = 0):
        self.input_size = input_size
        self.output_size = output_size
        self.rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(max(1, input_size))
        self.W = self.rng.uniform(-1.0, 1.0, size=(output_size, input_size)).astype(float) * scale
        self.b = np.zeros((output_size,), dtype=float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: [N, in]
        self.x = x
        self.a = self.sigmoid(self.linear(x))     # [N, out]
        return self.a

    def backward(self, g_a_loss: np.ndarray, lr: float):
        # g_a_loss: dL/da, [N, out]
        g_a = g_a_loss * self.d_sigmoid(self.a)   # dL/dz = dL/da * da/dz, [N, out]
        # Gradients (sum over batch)
        g_W = g_a.T @ self.x                      # [out, in]
        g_b = np.sum(g_a, axis=0)                 # [out]
        g_x = g_a @ self.W                        # [N, in]
        # Update weights and biases
        self.W -= lr * g_W
        self.b -= lr * g_b
        return g_x

    def linear(self, x: np.ndarray) -> np.ndarray:
        # x[N, in] @ W[out, in]^T -> [N, out]
        return x @ self.W.T + self.b      # [N, out]

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    def d_sigmoid(self, s: np.ndarray) -> np.ndarray:
        # s should be sigmoid(z)
        return s * (1.0 - s)

class MLP:
    def __init__(self, layer_sizes: List[int], seed: int = 0):
        assert len(layer_sizes) >= 2, "Must provide at least input and output sizes"
        self.layer_sizes = layer_sizes
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(SLP(layer_sizes[i], layer_sizes[i + 1], seed + i))

    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        self.p = A
        return self.p

    def backward(self, Y: np.ndarray, lr: float):
        g = self.d_l2_loss(self.p, Y)
        for layer in reversed(self.layers):
            g = layer.backward(g, lr)

    def l2_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        diff = pred - target
        per_sample = np.sum(diff * diff, axis=1)
        return float(np.mean(per_sample))

    def d_l2_loss(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        N = pred.shape[0] if pred.ndim == 2 else 1
        return 2.0 * (pred - target) / float(max(1, N))

    def fit_batch(self, X: np.ndarray, Y: np.ndarray, lr: float, iters: int, print_every: int = 0):
        for it in range(1, iters + 1):
            self.forward(X)
            self.backward(Y, lr)
            if print_every > 0 and (it % print_every == 0 or it == 1 or it == iters):
                print(f"[iter {it:5d}] train_loss={self.l2_loss(self.p, Y):.6f}")
    
    def evaluate_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        self.forward(X)
        return self.l2_loss(self.p, Y)