"""C2 trajectory-equality witness: full two-sided Shampoo on a quadratic model.
Three runs: (a) 50-digit mpmath reference, (b) fp64 eigh-per-step baseline,
(c) fp64 secular continuation (zero eigendecompositions ever).
Success: (c) deviates from (a) no more than (b) does — i.e., the continuation
trajectory is 'as exact as exact gets' at machine precision.
"""
import numpy as np
import mpmath as mp
from secular import rank1_update, rank_r_update, apply_invroot

mp.mp.dps = 50
rng = np.random.default_rng(42)
m, n, pa, pb = 32, 24, 16, 20
STEPS, ETA, EPS = 150, 0.05, 1e-3
A = rng.standard_normal((pa, m)) / np.sqrt(m)
B = rng.standard_normal((n, pb)) / np.sqrt(n)
C = rng.standard_normal((pa, pb))
W0 = rng.standard_normal((m, n)) / np.sqrt(m)

def grad_np(W):
    return A.T @ (A @ W @ B - C) @ B.T

def invroot_eig_np(M):
    d, Q = np.linalg.eigh(M)
    return Q @ ((d[:, None] ** -0.25) * Q.T)

# ---------- (b) fp64 eigh-per-step ----------
W = W0.copy(); L = EPS * np.eye(m); R = EPS * np.eye(n)
traj_eigh = []
for t in range(STEPS):
    G = grad_np(W)
    L += G @ G.T; R += G.T @ G
    W = W - ETA * invroot_eig_np(L) @ G @ invroot_eig_np(R)
    traj_eigh.append(W.copy())

# ---------- (c) fp64 continuation ----------
W = W0.copy()
dL = np.full(m, EPS); UL = np.eye(m)
dR = np.full(n, EPS); UR = np.eye(n)
traj_cont = []
for t in range(STEPS):
    G = grad_np(W)
    # factored PSD injections: G G^T = sum of columns? use SVD-free exact factors:
    # G G^T = (G) (G)^T -> feed columns of G as rank-1s? columns of G are not the
    # right factors (G G^T = sum_j g_j g_j^T over COLUMNS j of G). Yes they are.
    dL, UL = rank_r_update(dL, UL, G, 1.0)          # adds sum_j G[:,j] G[:,j]^T = G G^T
    dR, UR = rank_r_update(dR, UR, G.T, 1.0)        # adds G^T G
    PG = apply_invroot(dL, UL, G)                    # L^{-1/4} G
    W = W - ETA * apply_invroot(dR, UR, PG.T).T      # (R^{-1/4} (L^{-1/4}G)^T)^T
    traj_cont.append(W.copy())

# ---------- (a) mpmath reference ----------
Am, Bm, Cm = mp.matrix(A.tolist()), mp.matrix(B.tolist()), mp.matrix(C.tolist())
Wm = mp.matrix(W0.tolist())
Lm = mp.eye(m) * mp.mpf(EPS); Rm = mp.eye(n) * mp.mpf(EPS)
def invroot_mp(M, sz):
    E, Q = mp.eigsy(M)
    D = mp.zeros(sz, sz)
    for i in range(sz):
        D[i, i] = E[i] ** mp.mpf(-0.25)
    return Q * D * Q.T
ref_idx = sorted(set([0, 1, 4, 9, 24, 49, 99, STEPS - 1]))
refs = {}
for t in range(STEPS):
    Gm = Am.T * (Am * Wm * Bm - Cm) * Bm.T
    Lm = Lm + Gm * Gm.T; Rm = Rm + Gm.T * Gm
    Wm = Wm - mp.mpf(ETA) * invroot_mp(Lm, m) * Gm * invroot_mp(Rm, n)
    if t in ref_idx:
        refs[t] = np.array([[float(Wm[i, j]) for j in range(n)] for i in range(m)])
        e_eigh = np.abs(traj_eigh[t] - refs[t]).max() / np.abs(refs[t]).max()
        e_cont = np.abs(traj_cont[t] - refs[t]).max() / np.abs(refs[t]).max()
        print(f"t={t+1:4d}  |W_eigh - W_ref| = {e_eigh:.3e}   |W_cont - W_ref| = {e_cont:.3e}"
              f"   ratio cont/eigh = {e_cont/max(e_eigh,1e-300):.2f}", flush=True)
print("DONE")
