"""Unit witness: continuation engine vs numpy eigh, including the eps*I cold start
(the degenerate all-equal spectrum) and long sequences of rank-1 injections."""
import numpy as np
from secular import rank1_update, rank_r_update, apply_invroot

rng = np.random.default_rng(0)


def check(n=40, steps=30, eps=1e-4, cold=True):
    if cold:
        d = np.full(n, eps)
        U = np.eye(n)
        L = eps * np.eye(n)
    else:
        A = rng.standard_normal((n, n)) / np.sqrt(n)
        L = A @ A.T + eps * np.eye(n)
        d, U = np.linalg.eigh(L)
    worst = dict(orth=0.0, recon=0.0, eig=0.0, app=0.0)
    for t in range(steps):
        v = rng.standard_normal(n)
        d, U = rank1_update(d, U, v, 1.0)
        L = L + np.outer(v, v)
        worst["orth"] = max(worst["orth"], np.abs(U.T @ U - np.eye(n)).max())
        worst["recon"] = max(worst["recon"],
                             np.abs(U @ (d[:, None] * U.T) - L).max() / np.abs(L).max())
        dref = np.linalg.eigvalsh(L)
        worst["eig"] = max(worst["eig"], np.abs(dref - d).max() / dref.max())
        # application accuracy: L^{-1/4} X vs eigh route
        X = rng.standard_normal((n, 3))
        dr, Ur = np.linalg.eigh(L)
        ref = Ur @ ((dr[:, None] ** -0.25) * (Ur.T @ X))
        got = apply_invroot(d, U, X)
        worst["app"] = max(worst["app"], np.abs(ref - got).max() / np.abs(ref).max())
    return worst


print("cold start (L0 = eps*I, fully degenerate):", check(cold=True))
print("warm start (random PSD):                 ", check(cold=False))
# rank-r batch
n = 48
d = np.full(n, 1e-4); U = np.eye(n); L = 1e-4 * np.eye(n)
for _ in range(10):
    V = rng.standard_normal((n, 5))
    d, U = rank_r_update(d, U, V)
    L += V @ V.T
print("rank-5 x10 recon:", np.abs(U @ (d[:, None] * U.T) - L).max() / np.abs(L).max(),
      "orth:", np.abs(U.T @ U - np.eye(n)).max())
