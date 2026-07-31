"""Custom Triton kernel for the batched secular solve — the O(N^2) scalar core
of the continuation engine, one GPU program per root.

Each root i lives in its own interlacing interval (0, width_i) in the shifted
variable t = mu_i - d_i; the kernel runs an independent safeguarded Newton
bisection per root with early exit, reading the poles d and weights z^2 in
blocks. Work: O(N * iters_i) per root, fully parallel across roots — vs the
vectorized torch version which materializes O(N^2) tensors for a fixed 90
iterations for ALL roots.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _secular_kernel(d_ptr, z2_ptr, width_ptr, t_ptr, N,
                    EPS: tl.constexpr, MAXIT: tl.constexpr,
                    BLOCK: tl.constexpr):
    i = tl.program_id(0)
    d_i = tl.load(d_ptr + i)
    width = tl.load(width_ptr + i)
    lo = tl.full((), 0.0, dtype=tl.float64)
    hi = width + lo
    t = 0.5 * width + lo
    for _ in range(MAXIT):
        f = tl.full((), 1.0, dtype=tl.float64)
        fp = tl.full((), 0.0, dtype=tl.float64)
        for start in range(0, N, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < N
            dj = tl.load(d_ptr + offs, mask=mask, other=0.0)
            wj = tl.load(z2_ptr + offs, mask=mask, other=0.0)
            diff = (dj - d_i) - t
            r = tl.where(mask, wj / diff, 0.0)
            f += tl.sum(r, axis=0)
            fp += tl.sum(tl.where(mask, r / diff, 0.0), axis=0)
        neg = f < 0
        lo = tl.where(neg, t, lo)
        hi = tl.where(neg, hi, t)
        tn = t - f / fp
        bad = (tn <= lo) | (tn >= hi) | (tn != tn)
        tn = tl.where(bad, 0.5 * (lo + hi), tn)
        done = (hi - lo) <= 2.0 * EPS * hi
        t = tl.where(done, t, tn)
    tl.store(t_ptr + i, t)


def secular_triton(d, z2, maxit=70, block=256):
    """d ascending (N,) fp64, z2 (N,) fp64 -> shifted roots t (N,)."""
    N = d.shape[0]
    width = torch.empty_like(d)
    if N > 1:
        width[:-1] = d[1:] - d[:-1]
    width[-1] = z2.sum()
    t = torch.empty_like(d)
    EPS = torch.finfo(d.dtype).eps
    _secular_kernel[(N,)](d, z2, width, t, N, EPS=float(EPS), MAXIT=maxit,
                          BLOCK=block)
    return t


if __name__ == '__main__':
    import time, sys
    sys.path.insert(0, '/home/navin/shampoo/research')
    from rfshampoo.engine import _secular_batched
    dev = 'cuda:1'
    torch.manual_seed(0)
    torch.cuda.set_device(dev)
    for N in [1024, 2048, 4096, 8192]:
        d = torch.sort(torch.rand(N, device=dev, dtype=torch.float64) * 10)[0]
        z = torch.randn(N, device=dev, dtype=torch.float64) / N ** 0.5
        z2 = z * z
        t_ref = _secular_batched(d, z2)
        t_tri = secular_triton(d, z2)
        err = float((t_tri - t_ref).abs().max() / t_ref.abs().max())
        # eigenvalue check vs eigh
        L = torch.diag(d) + torch.outer(z, z)
        ev = torch.linalg.eigvalsh(L)
        err_eig = float((torch.sort(d + t_tri)[0] - ev).abs().max() / ev.max())
        for _ in range(2):
            secular_triton(d, z2); _secular_batched(d, z2)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            secular_triton(d, z2)
        torch.cuda.synchronize()
        t_k = (time.time() - t0) / 10
        t0 = time.time()
        for _ in range(10):
            _secular_batched(d, z2)
        torch.cuda.synchronize()
        t_v = (time.time() - t0) / 10
        print(f"N={N:5d} triton {1e3*t_k:7.2f}ms  torch {1e3*t_v:7.2f}ms  "
              f"speedup {t_v/t_k:5.1f}x  vs_torch_err {err:.2e}  "
              f"vs_eigh_err {err_eig:.2e}", flush=True)
