"""RootFree-Shampoo engines (torch): the three exact eigendecomposition-free
representations of L^{-1/4}, one per rank regime.

GRAM  — cumulative rank < N: exact pushdown identity
        (eps*I + Y Y^T)^{-1/4} = eps^{-1/4} I + Y g(Y^T Y) Y^T   (Lean: Intertwine.lean)
CONT  — secular/Loewner spectral continuation under rank-1 injections
        (Lean: Secular.lean + Equivalence.lean)
TRACK — stable cold coupled Newton–Schulz for the inverse 4th root, matmul-
        only, online residual certificate (dense full-rank regime). Warm
        one-sided tracking is provably unstable — see NOTES §3.

All three produce the SAME mathematical operator (trajectory_equivalence);
they differ only in cost profile.
"""
import torch


# ---------------------------------------------------------------- CONT engine
class SpectralState:
    """Maintains L = U diag(d) U^T exactly by algebraic continuation."""

    def __init__(self, n, eps, device, dtype=torch.float64):
        self.n = n
        self.U = torch.eye(n, device=device, dtype=dtype)
        self.d = torch.full((n,), float(eps), device=device, dtype=dtype)
        self.dtype = dtype
        self.device = device

    @torch.no_grad()
    def scale(self, beta):
        """L <- beta * L (EMA support): exact, O(N)."""
        self.d.mul_(beta)

    @torch.no_grad()
    def rank1(self, v):
        """L <- L + v v^T by exact secular continuation. O(N^2) scalar work +
        one N x k GEMM (k = non-deflated count)."""
        EPS = torch.finfo(self.dtype).eps
        d, U = self.d, self.U
        z = U.t().mv(v.to(self.dtype))
        zn2 = float(z.dot(z))
        if zn2 <= 0.0:
            return
        # ---- relative clustering on ascending d (invariant: d sorted asc) ---
        gaps = d[1:] - d[:-1]
        new_cluster = torch.ones_like(d, dtype=torch.bool)
        new_cluster[1:] = gaps > 4 * EPS * d[1:]
        cid = torch.cumsum(new_cluster.long(), 0) - 1     # cluster index per entry
        ncl = int(cid[-1]) + 1
        if ncl < self.n:
            # Householder within clusters: concentrate z into last member.
            # done per cluster on CPU-side loop over clusters with >1 member
            # (cheap: touches only clustered columns)
            counts = torch.bincount(cid, minlength=ncl)
            for c in torch.nonzero(counts > 1).flatten().tolist():
                idx = torch.nonzero(cid == c).flatten()
                zc = z[idx]
                nzc = torch.linalg.vector_norm(zc)
                if float(nzc) == 0.0:
                    continue
                w = zc.clone()
                w[-1] -= nzc
                wn = torch.linalg.vector_norm(w)
                if float(wn) > 0:
                    w /= wn
                    z[idx] = zc - 2 * w * (w.dot(zc))
                    Ub = U[:, idx]
                    U[:, idx] = Ub - 2 * torch.outer(Ub.mv(w), w)
                z[idx[:-1]] = 0.0
                z[idx[-1]] = nzc
        # ---- relative deflation --------------------------------------------
        z2 = z * z
        keep = z2 > 0.125 * EPS * d
        idx = torch.nonzero(keep).flatten()
        kk = idx.numel()
        if kk == 0:
            return
        dk = d[idx].contiguous()
        z2k = z2[idx].contiguous()
        # ---- batched secular solve (all kk roots in parallel) --------------
        t = _secular_batched(dk, z2k)
        # ---- Gu–Eisenstat z reconstruction (Loewner product identity) ------
        ddM = dk.unsqueeze(0) - dk.unsqueeze(1)             # [i,j] = d_j - d_i
        numM = ddM + t.unsqueeze(0)                          # mu_j - d_i
        ratio = numM / (ddM + torch.eye(kk, device=self.device, dtype=self.dtype))
        ratio.fill_diagonal_(1.0)
        z2hat = t * torch.prod(ratio, dim=1)
        z2hat.clamp_(min=0.0)
        zhat = torch.sqrt(z2hat) * torch.sign(z[idx])
        # ---- Loewner eigenvectors (exact) ----------------------------------
        dd = (dk.unsqueeze(1) - dk.unsqueeze(0)) - t.unsqueeze(0)  # d_i - mu_j
        Q = zhat.unsqueeze(1) / dd
        Q /= torch.linalg.vector_norm(Q, dim=0, keepdim=True)
        # ---- assemble -------------------------------------------------------
        d[idx] = dk + t
        U[:, idx] = U[:, idx] @ Q
        order = torch.argsort(d, stable=True)
        if not torch.equal(order, torch.arange(self.n, device=self.device)):
            self.d = d[order].contiguous()
            self.U = U[:, order].contiguous()

    @torch.no_grad()
    def add_gram(self, G):
        """L <- L + G G^T via per-column continuations."""
        for j in range(G.shape[1]):
            self.rank1(G[:, j])

    @torch.no_grad()
    def apply_invroot(self, X, p=4):
        """L^{-1/p} @ X through the maintained pair."""
        Xd = X.to(self.dtype)
        return (self.U @ ((self.d.clamp_min(0).pow(-1.0 / p)).unsqueeze(1)
                          * (self.U.t() @ Xd))).to(X.dtype)

    @torch.no_grad()
    def diagnostics(self):
        orth = (self.U.t() @ self.U - torch.eye(self.n, device=self.device,
                                                dtype=self.dtype)).abs().max()
        return dict(orth=float(orth), dmin=float(self.d.min()),
                    dmax=float(self.d.max()))


@torch.no_grad()
def _secular_batched(d, z2, iters=90):
    """Roots of 1 + sum_j z2_j/(d_j - mu) in each interval, shifted variable
    t_i = mu_i - d_i in (0, width_i); safeguarded Newton, fully vectorized."""
    EPS = torch.finfo(d.dtype).eps
    k = d.shape[0]
    width = torch.empty_like(d)
    if k > 1:
        width[:-1] = d[1:] - d[:-1]
    width[-1] = z2.sum()
    lo = torch.zeros_like(d)
    hi = width.clone()
    t = 0.5 * width
    delta = d.unsqueeze(0) - d.unsqueeze(1)   # [i,j] = d_j - d_i
    w = z2.unsqueeze(0).expand(k, k)
    for _ in range(iters):
        diff = delta - t.unsqueeze(1)
        r = w / diff
        f = 1.0 + r.sum(dim=1)
        neg = f < 0
        lo = torch.where(neg, t, lo)
        hi = torch.where(neg, hi, t)
        fp = (r / diff).sum(dim=1)
        tn = t - f / fp
        bad = ~torch.isfinite(tn) | (tn <= lo) | (tn >= hi)
        tn = torch.where(bad, 0.5 * (lo + hi), tn)
        done = (hi - lo) <= 2 * EPS * hi
        t = torch.where(done, t, tn)
    return t


# --------------------------------------------------------------- GRAM engine
class GramState:
    """Exact low-rank phase: L = eps*I + Y Y^T with Y the collected gradient
    columns; applies L^{-1/4} X in O(N r^2 + r^3) by the pushdown identity."""

    def __init__(self, n, eps, device, dtype=torch.float64, max_rank=None):
        self.n, self.eps = n, float(eps)
        self.cols = []
        self.rank = 0
        self.device, self.dtype = device, dtype
        self.max_rank = max_rank or n

    @torch.no_grad()
    def add_gram(self, G):
        self.cols.append(G.to(self.dtype).clone())
        self.rank += G.shape[1]

    def full(self):
        return self.rank >= self.max_rank

    @torch.no_grad()
    def _Y(self):
        return torch.cat(self.cols, dim=1)

    @torch.no_grad()
    def apply_invroot(self, X, p=4):
        Y = self._Y()
        Gm = (Y.t() @ Y).double()
        sd, Vd = torch.linalg.eigh(Gm)          # r x r only, fp64
        s_, V = sd.to(self.dtype), Vd.to(self.dtype)
        s = s_.clamp_min(0)
        e = self.eps
        g = ((e + s).pow(-1.0 / p) - e ** (-1.0 / p)) / s.clamp_min(1e-300)
        g = torch.where(s > 0, g, torch.zeros_like(g))
        Xd = X.to(self.dtype)
        out = e ** (-1.0 / p) * Xd + Y @ (V @ (g.unsqueeze(1) * (V.t() @ (Y.t() @ Xd))))
        return out.to(X.dtype)

    @torch.no_grad()
    def to_spectral(self):
        """Exact handoff GRAM -> CONT: build the full spectral pair."""
        S = SpectralState(self.n, self.eps, self.device, self.dtype)
        S.add_gram(self._Y())
        return S

    @torch.no_grad()
    def to_dense_L(self):
        Y = self._Y()
        return self.eps * torch.eye(self.n, device=self.device, dtype=self.dtype) + Y @ Y.t()


# -------------------------------------------------------------- TRACK engine
class TrackState:
    """Matmul-only certified refresh of X = L^{-1/4} via stable coupled
    Newton–Schulz (invsqrt twice). Fresh at every step; the exact residual
    ||X^4 L - I||_F/sqrt(N) is the online certificate."""

    def __init__(self, n, eps, device, dtype=torch.float32):
        self.n = n
        self.device, self.dtype = device, dtype
        self.L = eps * torch.eye(n, device=device, dtype=dtype)
        self.X = (eps ** -0.25) * torch.eye(n, device=device, dtype=dtype)
        self.I = torch.eye(n, device=device, dtype=dtype)
        self.warm_iters = 30
        self.cold_iters = 120
        self.tol = 50 * torch.finfo(dtype).eps
        self.last_residual = 0.0
        self.last_iters = 0

    @torch.no_grad()
    def scale(self, beta):
        self.L.mul_(beta)
        self.X.mul_(beta ** -0.25)

    @torch.no_grad()
    def add_gram(self, G):
        Gd = G.to(self.dtype)
        self.L.add_(Gd @ Gd.t())
        self._retrack()

    @torch.no_grad()
    def _residual_M(self, X):
        X2 = X @ X
        M = (X2 @ X2) @ self.L
        return M, float((M - self.I).norm() / self.n ** 0.5)

    @torch.no_grad()
    def _ns_invsqrt(self, A):
        """Stable coupled Newton–Schulz: for spec(A) in (0,1], returns
        (A^{1/2}, A^{-1/2}). Y <- Y T, Z <- T Z with T = (3I - Z Y)/2.
        Symmetrized each iteration; matmul-only. Stall detection: at finite
        precision the T-residual floors near eps*sqrt(kappa); once progress
        stops below 1e-2 the floor is reached — stop and certify."""
        Y = A.clone()
        Z = self.I.clone()
        it = 0
        best = float('inf')
        since_best = 0
        while it < self.cold_iters:
            T = 1.5 * self.I - 0.5 * (Z @ Y)
            r = float((T - self.I).norm() / self.n ** 0.5)
            if r < 0.5 * self.tol:
                break
            # floor detection: once past the lift plateau (r < 0.1), demand
            # genuine progress; the quadratic phase clears (0.1 -> tol) in a
            # few iterations, so 4 iterations without a 5% new best = floor.
            if r < 0.1:
                if r < 0.95 * best:
                    best = r
                    since_best = 0
                else:
                    since_best += 1
                    if since_best >= 4:
                        break
            Y = Y @ T
            Z = T @ Z
            Y = 0.5 * (Y + Y.t())
            Z = 0.5 * (Z + Z.t())
            it += 1
        return Y, Z, it

    @torch.no_grad()
    def _retrack(self):
        """Cold, stable, matmul-only refresh: L^{-1/4} = (A^{1/2})^{-1/2}·s^{-1/4}
        with A = L/s. (Warm one-sided Newton on X alone is provably unstable:
        the linearized map has modes 1 - (r^{-1}+...+r^{-4})/4, unbounded for
        small eigenvalue ratios — see NOTES 'tracking instability'. Stability
        under warm carry-over requires auxiliary spectral state, i.e. CONT.)"""
        s = float(torch.linalg.matrix_norm(self.L, ord='fro'))
        A = self.L / s
        B, _, it1 = self._ns_invsqrt(A)              # B = A^{1/2}
        sB = float(torch.linalg.matrix_norm(B, ord='fro'))
        _, W, it2 = self._ns_invsqrt(B / sB)          # W = (B/sB)^{-1/2}
        X = (sB ** -0.5) * (s ** -0.25) * W           # = L^{-1/4}
        Mf, resf = self._residual_M(X)
        self.last_residual = resf
        self.last_iters = it1 + it2
        self.X = 0.5 * (X + X.t())

    @torch.no_grad()
    def apply_invroot(self, Xin, p=4):
        assert p == 4
        return (self.X.to(Xin.dtype) @ Xin)
