"""
Dataset generators for XOR and two-bit binary adder tasks using NumPy arrays.
"""
import numpy as np
import random
from typing import Tuple

def _int_to_bits(n: int, width: int):
    return [(n >> i) & 1 for i in reversed(range(width))]

def make_xor() -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
    Y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)
    return X, Y

def make_two_bit_adder() -> Tuple[np.ndarray, np.ndarray]:
    rows_x = []
    rows_y = []
    for a in range(4):
        for b in range(4):
            for cin in range(2):
                a_bits = _int_to_bits(a, 2)
                b_bits = _int_to_bits(b, 2)
                x = [float(a_bits[0]), float(a_bits[1]), float(b_bits[0]), float(b_bits[1]), float(cin)]
                total = a + b + cin
                s_bits = _int_to_bits(total % 4, 2)
                cout = 1.0 if total >= 4 else 0.0
                y = [float(s_bits[0]), float(s_bits[1]), cout]
                rows_x.append(x)
                rows_y.append(y)
    return np.array(rows_x, dtype=float), np.array(rows_y, dtype=float)

def train_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    test_ratio: float = 0.25,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # X_tr, Y_tr, X_te, Y_te
    n = X.shape[0]
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_test = int(round(n * test_ratio))
    test_idxs = np.array(idxs[:n_test], dtype=int)
    train_idxs = np.array(idxs[n_test:], dtype=int)
    return X[train_idxs], Y[train_idxs], X[test_idxs], Y[test_idxs]

