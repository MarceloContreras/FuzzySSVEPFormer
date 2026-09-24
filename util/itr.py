import numpy as np


def itr(acc_pct, n_clases, T_seg, gaze=0.5):
    A = np.clip(acc_pct, 1e-9, 1 - 1e-9)
    N, T = n_clases, T_seg + gaze
    bits = np.log2(N) + A * np.log2(A) + (1 - A) * np.log2((1 - A) / (N - 1))
    return (60.0 / T) * bits
