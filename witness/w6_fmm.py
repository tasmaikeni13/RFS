"""E5/w6: the Cauchy-kernel fast apply behind the O(N^2 log N) continuation claim.

The Loewner eigenvector matrix of a rank-1 update is Cauchy-structured:
Q[i,j] = zhat_i / (d_i - mu_j) (columns normalized). Applying U <- U @ Q row-wise
is, per row, a Cauchy matvec  y_j = sum_i c_i / (d_i - mu_j)  — computable to
machine precision in O((N+M) log(1/eps) log N) by a Chebyshev treecode
(degenerate-kernel expansion on well-separated interval pairs; direct near
field). Interlacing (d_i < mu_i < d_{i+1}) makes the geometry benign.

This witness measures accuracy vs dense evaluation and the scaling in N.
"""
import numpy as np, time

P = 18  # Chebyshev nodes per box (~1e-14 for eta=0.5 separation)


def cheb_nodes(a, b, p=P):
    k = np.arange(p)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) * np.pi / (2 * p))


def lagrange_matrix(nodes, pts):
    """L[i,k] = ell_k(pts_i) for Lagrange basis on nodes."""
    M = pts[:, None] - nodes[None, :]
    w = np.ones_like(nodes)
    for k in range(len(nodes)):
        w[k] = 1.0 / np.prod(nodes[k] - np.delete(nodes, k))
    with np.errstate(divide='ignore', invalid='ignore'):
        num = np.prod(M, axis=1)[:, None] * w[None, :] / M
    exact = np.abs(M) < 1e-300
    if exact.any():
        num[exact.any(axis=1)] = 0.0
        num[exact] = 1.0
    return num


class Box:
    __slots__ = ('lo', 'hi', 'src_idx', 'tgt_idx', 'children', 'nodes', 'up')
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.children = []


def build_tree(x, y, lo, hi, leaf=64):
    root = Box(lo, hi)
    root.src_idx = np.argsort(x)
    root.tgt_idx = np.argsort(y)
    stack = [(root, x[root.src_idx], y[root.tgt_idx], root.src_idx, root.tgt_idx)]
    boxes = []
    while stack:
        b, xs, ys, si, ti = stack.pop()
        boxes.append(b)
        if len(si) <= leaf and len(ti) <= leaf:
            b.src_idx, b.tgt_idx = si, ti
            continue
        mid = 0.5 * (b.lo + b.hi)
        c1, c2 = Box(b.lo, mid), Box(mid, b.hi)
        m1 = xs <= mid
        n1 = ys <= mid
        b.children = [c1, c2]
        b.src_idx, b.tgt_idx = si, ti
        stack.append((c1, xs[m1], ys[n1], si[m1], ti[n1]))
        stack.append((c2, xs[~m1], ys[~n1], si[~m1], ti[~n1]))
    return root


def fmm_cauchy(d, mu, c, leaf=64):
    """y_j = sum_i c_i / (d_i - mu_j), treecode with Chebyshev outgoing
    expansions; O((N+M) log N log(1/eps))."""
    lo = min(d.min(), mu.min()) - 1e-9
    hi = max(d.max(), mu.max()) + 1e-9
    root = build_tree(d, mu, lo, hi, leaf)
    out = np.zeros_like(mu)

    def upward(b):
        b.nodes = cheb_nodes(b.lo, b.hi)
        if not b.children:
            if len(b.src_idx):
                Lm = lagrange_matrix(b.nodes, d[b.src_idx])
                b.up = Lm.T @ c[b.src_idx]
            else:
                b.up = np.zeros(P)
        else:
            b.up = np.zeros(P)
            for ch in b.children:
                upward(ch)
                if len(ch.src_idx):
                    Lm = lagrange_matrix(b.nodes, ch.nodes)
                    b.up += Lm.T @ ch.up

    def interact(bs, bt):
        """source box bs -> target box bt"""
        if len(bs.src_idx) == 0 or len(bt.tgt_idx) == 0:
            return
        ws, wt = bs.hi - bs.lo, bt.hi - bt.lo
        dist = max(bt.lo - bs.hi, bs.lo - bt.hi, 0.0)
        if dist > 0.6 * max(ws, wt):
            K = 1.0 / (bs.nodes[:, None] - mu[bt.tgt_idx][None, :])
            out[bt.tgt_idx] += K.T @ bs.up
        elif not bs.children and not bt.children:
            K = 1.0 / (d[bs.src_idx][:, None] - mu[bt.tgt_idx][None, :])
            out[bt.tgt_idx] += K.T @ c[bs.src_idx]
        elif (ws >= wt and bs.children) or not bt.children:
            for ch in bs.children:
                interact(ch, bt)
        else:
            for ch in bt.children:
                interact(bs, ch)

    upward(root)
    interact(root, root)
    return out


def main():
    rng = np.random.default_rng(0)
    print(f"{'N':>7} {'dense_ms':>9} {'fmm_ms':>8} {'speedup':>8} {'rel_err':>10}")
    for N in [2000, 8000, 32000, 128000]:
        d = np.sort(rng.uniform(0, 1, N))
        gaps = np.diff(np.concatenate([d, [1 + 1e-3]]))
        mu = d + gaps * rng.uniform(0.2, 0.8, N)   # interlaced targets
        cvec = rng.standard_normal(N)
        t0 = time.time()
        if N <= 32000:
            dense = (cvec[:, None] / (d[:, None] - mu[None, :])).sum(axis=0)
            t_dense = time.time() - t0
        else:
            idx = rng.choice(N, 200, replace=False)
            dense_s = np.array([(cvec / (d - mu[j])).sum() for j in idx])
            t_dense = (time.time() - t0) * N / 200
        t0 = time.time()
        fast = fmm_cauchy(d, mu, cvec)
        t_fmm = time.time() - t0
        if N <= 32000:
            err = np.abs(fast - dense).max() / np.abs(dense).max()
        else:
            err = np.abs(fast[idx] - dense_s).max() / np.abs(dense_s).max()
        print(f"{N:>7} {1e3*t_dense:>9.1f} {1e3*t_fmm:>8.1f} "
              f"{t_dense/t_fmm:>8.1f} {err:>10.2e}")


if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(100000)
    main()
