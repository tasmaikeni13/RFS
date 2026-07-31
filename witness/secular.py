"""Spectral continuation engine: maintain (d, U) with L = U diag(d) U^T under
additive rank-1 PSD injections L <- L + rho v v^T, without ever recomputing an
eigendecomposition. Exact algebraic characterization (secular equation) + Loewner
(Gu-Eisenstat) eigenvector reconstruction.

This is the correctness witness implementation (dense GEMM for U @ Qhat; the
O(N^2 polylog) structured product is a performance concern handled separately).
"""

import numpy as np

EPS = np.finfo(np.float64).eps


def _solve_secular_interval(delta, w, lo, hi, maxit=100):
    """Solve 1 + sum_j w_j/(delta_j - t) = 0 for t in (lo, hi).

    delta: shifted pole locations (d_j - d_i), w: rho*z_j^2 (>0).
    f is strictly increasing on (lo,hi) with f(lo+)=-inf, f(hi-)=+inf.
    Safeguarded Newton with bisection fallback; works in the shifted variable.
    """
    a, b = lo, hi
    t = 0.5 * (a + b)
    for _ in range(maxit):
        diff = delta - t
        # guard exact pole hits (can only happen from rounding)
        tiny = np.abs(diff) < 1e-300
        if tiny.any():
            diff = np.where(tiny, np.copysign(1e-300, diff), diff)
        r = w / diff
        f = 1.0 + r.sum()
        if f < 0:
            a = t
        else:
            b = t
        fp = (r / diff).sum()  # f'(t) = sum w/(delta-t)^2 > 0
        step = -f / fp if fp > 0 else 0.0
        t_new = t + step
        if not (a < t_new < b):
            t_new = 0.5 * (a + b)
        if abs(t_new - t) <= 4 * EPS * max(abs(t_new), abs(lo), abs(hi)) + 1e-300:
            t = t_new
            break
        t = t_new
    return t


def secular_eigenvalues(d, z2rho):
    """Eigenvalues of diag(d) + rank-1 with weights z2rho = rho*z_i^2 > 0,
    d strictly increasing. Returns mu (ascending) via per-interval solves in
    shifted coordinates (t_i = mu_i - d_i)."""
    n = d.size
    mu = np.empty(n)
    t_shift = np.empty(n)
    rho_zz = z2rho.sum()
    for i in range(n):
        hi = (d[i + 1] - d[i]) if i + 1 < n else rho_zz
        delta = d - d[i]
        t = _solve_secular_interval(delta, z2rho, 0.0, hi)
        t_shift[i] = t
        mu[i] = d[i] + t
    return mu, t_shift


def gu_eisenstat_z(d, mu, t_shift, rho):
    """Recompute zhat_i^2 from computed roots (Loewner formula):
    zhat_i^2 = prod_j (mu_j - d_i) / (rho * prod_{j != i} (d_j - d_i)).
    Computed in log-free stable form using shifted quantities:
    mu_j - d_i = (d_j - d_i) + t_j.
    """
    n = d.size
    ddM = d[None, :] - d[:, None]              # ddM[i,j] = d_j - d_i
    numM = ddM + t_shift[None, :]              # mu_j - d_i
    ratio = numM / np.where(np.eye(n, dtype=bool), 1.0, ddM + np.eye(n))
    np.fill_diagonal(ratio, 1.0)
    z2 = t_shift * np.prod(ratio, axis=1) / rho  # numM[i,i] = t_i
    return np.maximum(z2, 0.0)


def rank1_update(d, U, v, rho, deflate_tol=64.0):
    """One exact continuation step: (d, U) of L  ->  (d', U') of L + rho v v^T.

    Handles deflation: (a) negligible z components, (b) clustered eigenvalues
    (Householder rotation within each cluster to concentrate z).
    Cost: O(N^2) scalar work + one N x k GEMM for the non-deflated block
    (+ O(N * cluster) for cluster rotations).
    """
    n = d.size
    z = U.T @ v
    znorm2 = z @ z
    if znorm2 * rho <= 0:
        return d.copy(), U
    # --- RELATIVE cluster grouping on d (d assumed ascending, all > 0) ---
    # The consumed operator lambda^{-1/4} conditions RELATIVELY, so clustering must be
    # relative (LAPACK D&C's absolute rule is not enough for inverse-root use).
    clusters = []
    start = 0
    for i in range(1, n + 1):
        if i == n or (d[i] - d[i - 1]) > 4 * EPS * d[i]:
            clusters.append((start, i))
            start = i
    U = U.copy()
    z = z.copy()
    # --- within-cluster Householder to zero all but last z-component ---
    for (a, b) in clusters:
        if b - a >= 2:
            zc = z[a:b]
            nz = np.linalg.norm(zc)
            if nz > 0:
                e = np.zeros(b - a)
                e[-1] = 1.0
                w = zc - nz * e
                wn = np.linalg.norm(w)
                if wn > 1e-300:
                    w /= wn
                    # apply reflection H = I - 2 w w^T to z-block and U-columns
                    z[a:b] = zc - 2 * w * (w @ zc)
                    Ub = U[:, a:b]
                    U[:, a:b] = Ub - 2 * np.outer(Ub @ w, w)
                z[a:b][:-1] = 0.0
                z[a:b][-1] = nz
    # --- deflate tiny z entries: RELATIVE backward-error rule ---
    # zeroing z_i moves eigenvalue d_i by ~rho*z_i^2 to first order; keep the induced
    # RELATIVE eigenvalue perturbation below ~eps so lambda^{-1/4} stays eps-accurate.
    z2 = rho * z * z
    keep = z2 > 0.125 * EPS * d
    idx = np.where(keep)[0]
    if idx.size == 0:
        return d.copy(), U
    dk = d[idx]
    # secular needs strictly increasing dk; clusters were rotated so kept entries
    # within a cluster are single representatives -> enforce strict order by tiny jitter-free check
    mu_k, t_shift = secular_eigenvalues(dk, z2[idx])
    z2hat = gu_eisenstat_z(dk, mu_k, t_shift, rho)
    zhat = np.sqrt(z2hat) * np.sign(z[idx])
    # eigenvectors of the small (k x k) diagonal+rank1 problem, columns q_j
    # stable: d_i - mu_j = (d_i - d_j) - t_j
    dd = (dk[:, None] - dk[None, :]) - t_shift[None, :]
    Q = zhat[:, None] / dd
    Q /= np.linalg.norm(Q, axis=0, keepdims=True)
    # assemble
    d_new = d.copy()
    d_new[idx] = mu_k
    U[:, idx] = U[:, idx] @ Q
    # restore ascending order
    order = np.argsort(d_new, kind="stable")
    return d_new[order], U[:, order]


def rank_r_update(d, U, V, rho=1.0):
    """L <- L + rho * V V^T  (V: n x r), as r sequential exact rank-1 continuations."""
    for j in range(V.shape[1]):
        d, U = rank1_update(d, U, np.ascontiguousarray(V[:, j]), rho)
    return d, U


def apply_invroot(d, U, X, p=4):
    """L^{-1/p} @ X via the maintained spectral pair."""
    return U @ ((d[:, None] ** (-1.0 / p)) * (U.T @ X))
