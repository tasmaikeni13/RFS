"""GPU benchmark: cost of the eigendecomposition Shampoo actually pays
(torch.linalg.eigh, cuSOLVER) vs the matmul-only work our continuation needs
per refresh (GEMMs + batched scalar secular solves, which are bandwidth-trivial).

Torch secular solver: fully vectorized batched bisection-Newton on GPU —
N independent roots solved in parallel (this is the custom-kernel-friendly path;
here expressed in torch ops, which is already fast enough at these sizes).
"""
import torch, time, json

torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device('cuda:0')

def bench(f, warmup=2, reps=5):
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps

def secular_batched(d, z2, rho, iters=60):
    """All N roots at once: root i in (d_i, next), shifted variable t in (0, width).
    d ascending (N,), z2 = z_i^2 (N,). Returns t_shift (N,)."""
    N = d.shape[0]
    width = torch.empty_like(d)
    width[:-1] = d[1:] - d[:-1]
    width[-1] = rho * z2.sum()
    lo = torch.zeros_like(d)
    hi = width.clone()
    t = 0.5 * hi
    delta = d.unsqueeze(0) - d.unsqueeze(1)     # delta[i,j] = d_j - d_i
    w = (rho * z2).unsqueeze(0).expand(N, N)
    for _ in range(iters):
        diff = delta - t.unsqueeze(1)           # (i,j): d_j - d_i - t_i
        r = w / diff
        fval = 1.0 + r.sum(dim=1)
        neg = fval < 0
        lo = torch.where(neg, t, lo)
        hi = torch.where(neg, hi, t)
        fp = (r / diff).sum(dim=1)
        tn = t - fval / fp
        bad = ~((tn > lo) & (tn < hi)) | ~torch.isfinite(tn)
        tn = torch.where(bad, 0.5 * (lo + hi), tn)
        t = tn
    return t

for N in [1024, 2048, 4096]:
    A = torch.randn(N, N, device=dev, dtype=torch.float32)
    L = A @ A.T + 1e-3 * torch.eye(N, device=dev)
    L64 = L.double()
    G = torch.randn(N, N, device=dev)

    t_eigh32 = bench(lambda: torch.linalg.eigh(L))
    t_eigh64 = bench(lambda: torch.linalg.eigh(L64))
    t_gemm = bench(lambda: A @ A)
    # continuation refresh cost model, rank-r batch: r x (z = U^T v [GEMV->GEMM],
    # secular batched, Loewner scaling [elementwise], U <- U @ Qhat [GEMM])
    U = torch.linalg.qr(A)[0].contiguous()
    d = torch.sort(torch.rand(N, device=dev) + 0.1)[0]
    v = torch.randn(N, device=dev)
    def refresh_rank1():
        z = U.t() @ v
        t = secular_batched(d, z * z, 1.0)
        dd = (d.unsqueeze(0) - d.unsqueeze(1)) - t.unsqueeze(0)
        Q = (z.unsqueeze(1) / dd)
        Q = Q / Q.norm(dim=0, keepdim=True)
        return U @ Q
    t_r1 = bench(refresh_rank1)
    print(json.dumps(dict(N=N, eigh_fp32_ms=1e3*t_eigh32, eigh_fp64_ms=1e3*t_eigh64,
                          gemm_fp32_ms=1e3*t_gemm, contin_rank1_ms=1e3*t_r1,
                          rank1_per_eigh64=t_eigh64/t_r1)))
