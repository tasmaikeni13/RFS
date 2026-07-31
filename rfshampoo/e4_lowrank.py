"""E4: CONT's native regime — low per-step rank with FACTORED gradients.
In real training each layer's gradient arrives factored (G = delta^T x from
backprop). Feeding the exact rank-r factors makes every part of the step
O(N^2 r): the continuation update AND the preconditioner application.
    G = A B^T (N x r factors)  =>  G G^T = V V^T with V = A chol(B^T B),
    L^{-1/4} G R^{-1/4} = (SL.apply A) (SR.apply B)^T.
Layer 1024x1024, r = 4, 150 steps, fp64, vs fp64-eigh-every-step gold.
"""
import torch, time, sys
sys.path.insert(0, '/home/navin/shampoo/research')
from rfshampoo.engine import SpectralState

torch.manual_seed(3)
dev = 'cuda:1' if torch.cuda.is_available() else 'cpu'
N, r, STEPS, ETA, EPS = 1024, 4, 150, 0.05, 1e-6

def batch(step):
    g = torch.Generator(device='cpu').manual_seed(999 + step)
    X = torch.randn(N, r, generator=g).to(dev).double()
    T = torch.randn(N, r, generator=g).to(dev).double()
    return X, T

def factor_updates(W, X, T):
    A = (W @ X - T) / r          # N x r
    B = X                        # N x r ; G = A B^T
    def half(M):  # rank-safe (r x r): M^{1/2}
        if not torch.isfinite(M).all():
            raise FloatingPointError('diverged')
        dd, Q = torch.linalg.eigh(M)
        return Q @ (dd.clamp_min(0).sqrt().unsqueeze(1) * Q.t())
    VL = A @ half(B.t() @ B)
    VR = B @ half(A.t() @ A)
    return A, B, VL, VR

def run_cont():
    W = (0.1 * torch.eye(N, device=dev)).double()
    SL = SpectralState(N, EPS, dev, torch.float64)
    SR = SpectralState(N, EPS, dev, torch.float64)
    times, snaps = [], {}
    for t in range(STEPS):
        X, T = batch(t)
        torch.cuda.synchronize(); t0 = time.time()
        A, B, VL, VR = factor_updates(W, X, T)
        SL.add_gram(VL); SR.add_gram(VR)
        PG = SL.apply_invroot(A) @ SR.apply_invroot(B).t()
        W -= ETA * PG
        torch.cuda.synchronize(); times.append(time.time() - t0)
        if t % 30 == 0 or t == STEPS - 1:
            snaps[t] = W.clone()
    return snaps, times, SL

def run_eigh(every=1):
    W = (0.1 * torch.eye(N, device=dev)).double()
    L = EPS * torch.eye(N, device=dev, dtype=torch.float64)
    R = EPS * torch.eye(N, device=dev, dtype=torch.float64)
    PL = EPS ** -0.25 * torch.eye(N, device=dev, dtype=torch.float64)
    PR = PL.clone()
    times, snaps = [], {}
    for t in range(STEPS):
        X, T = batch(t)
        torch.cuda.synchronize(); t0 = time.time()
        A, B, _, _ = factor_updates(W, X, T)
        G = A @ B.t()
        L += G @ G.t(); R += G.t() @ G
        if t == 0 or (t + 1) % every == 0:
            dL, QL = torch.linalg.eigh(L); PL = QL @ (dL.clamp_min(0).pow(-0.25).unsqueeze(1) * QL.t())
            dR, QR = torch.linalg.eigh(R); PR = QR @ (dR.clamp_min(0).pow(-0.25).unsqueeze(1) * QR.t())
        W -= ETA * (PL @ G @ PR)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        if t % 30 == 0 or t == STEPS - 1:
            snaps[t] = W.clone()
    return snaps, times

gold, tg = run_eigh(1)
cont, tc, SL = run_cont()
try:
    stale, ts = run_eigh(100)
    stale_ok = True
except FloatingPointError:
    stale_ok = False
med = lambda v: sorted(v[5:])[len(v[5:]) // 2] * 1e3
ks = sorted(gold)
ec = [float((cont[k] - gold[k]).norm() / gold[k].norm()) for k in ks]
es = [float((stale[k] - gold[k]).norm() / gold[k].norm()) for k in ks] if stale_ok else None
print(f"eigh1-gold med_step {med(tg):7.1f} ms")
print(f"cont      med_step {med(tc):7.1f} ms  speedup {med(tg)/med(tc):4.1f}x  "
      f"traj-vs-gold {ec[0]:.1e} -> {ec[-1]:.1e}  orth {SL.diagnostics()['orth']:.1e}")
if stale_ok:
    print(f"eighK=100 med_step {med(ts):7.1f} ms  traj-vs-gold {es[0]:.1e} -> {es[-1]:.1e}")
else:
    print("eighK=100 DIVERGED (stale eps-scale preconditioner at this lr)")
print("E4-DONE")
