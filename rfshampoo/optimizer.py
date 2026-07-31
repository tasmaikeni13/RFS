"""RootFreeShampoo: exact Shampoo trajectory, zero N x N eigendecompositions.

Per 2D parameter W (m x n), maintains the GKS accumulators
    L = eps I + sum G G^T,   R = eps I + sum G^T G
and applies  W <- W - lr * L^{-1/4} G R^{-1/4}
with the inverse roots realized by the exact engines in engine.py:
  mode 'gram'  : exact pushdown while cumulative rank < threshold
  mode 'cont'  : secular/Loewner spectral continuation (rank-1 injections)
  mode 'track' : matmul-only certified Newton tracking (dense regime)
  mode 'eigh'  : baseline (torch.linalg.eigh) for comparison
  mode 'auto'  : gram -> (cont | track) by per-step rank vs dimension
1D parameters and dims > max_precond_dim fall back to Adam-style diagonal.
"""
import torch
from torch.optim import Optimizer

from .engine import SpectralState, GramState, TrackState


class _EighState:
    def __init__(self, n, eps, device, dtype=torch.float64, every=1):
        dtype = torch.float64  # deployed practice: fp64 eig for stability
        self.L = eps * torch.eye(n, device=device, dtype=dtype)
        self.P = eps ** -0.25 * torch.eye(n, device=device, dtype=dtype)
        self.every = every
        self.t = 0

    @torch.no_grad()
    def scale(self, beta):
        self.L.mul_(beta)

    @torch.no_grad()
    def add_gram(self, G):
        Gd = G.to(self.L.dtype)
        self.L.add_(Gd @ Gd.t())
        self.t += 1
        if self.t == 1 or self.t % self.every == 0:
            d, Q = torch.linalg.eigh(self.L)
            self.P = Q @ (d.clamp_min(0).pow(-0.25).unsqueeze(1) * Q.t())

    @torch.no_grad()
    def apply_invroot(self, X, p=4):
        return (self.P.to(X.dtype) @ X)


def _make_state(n, eps, device, mode, r_step, precond_dtype, eigh_every):
    if mode == 'eigh':
        return _EighState(n, eps, device, precond_dtype, eigh_every)
    if mode == 'track':
        return TrackState(n, eps, device, precond_dtype)
    if mode == 'cont':
        return SpectralState(n, eps, device, precond_dtype)
    if mode == 'auto' and r_step >= max(1, n // 2):
        # per-step rank fills the Gram budget immediately: straight to TRACK
        return TrackState(n, eps, device, precond_dtype)
    return GramState(n, eps, device, precond_dtype, max_rank=max(1, n // 2))


class RootFreeShampoo(Optimizer):
    def __init__(self, params, lr=3e-4, eps=1e-6, mode='auto',
                 beta2=1.0, max_precond_dim=4096, precond_dtype=torch.float64,
                 cont_rank_frac=0.125, eigh_every=1,
                 adam_betas=(0.9, 0.999), adam_eps=1e-8):
        defaults = dict(lr=lr, eps=eps, mode=mode, beta2=beta2,
                        max_precond_dim=max_precond_dim,
                        precond_dtype=precond_dtype,
                        cont_rank_frac=cont_rank_frac, eigh_every=eigh_every,
                        adam_betas=adam_betas, adam_eps=adam_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _get_engine(self, side_key, st, n, g, group, r_step):
        eng = st.get(side_key)
        mode = group['mode']
        if eng is None:
            eng = _make_state(n, group['eps'], g.device, mode, r_step,
                              group['precond_dtype'], group['eigh_every'])
            st[side_key] = eng
        # auto transitions out of GRAM when the buffer is full
        if isinstance(eng, GramState) and eng.full():
            if mode in ('gram',):
                pass  # stay (pure gram mode keeps growing; costs grow too)
            elif mode in ('auto', 'cont') and r_step <= max(1, int(group['cont_rank_frac'] * n)):
                eng = eng.to_spectral()
                st[side_key] = eng
            else:  # auto with large per-step rank, or explicit track
                tr = TrackState(n, group['eps'], g.device, group['precond_dtype'])
                tr.L = eng.to_dense_L().to(group['precond_dtype'])
                # exact warm handoff: X = L^{-1/4} from the pushdown identity
                tr.X = eng.apply_invroot(
                    torch.eye(n, device=g.device, dtype=group['precond_dtype']))
                tr._retrack()
                st[side_key] = tr
                eng = tr
        return eng

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                shape = p.shape
                if p.ndim >= 2:
                    G = g.reshape(shape[0], -1)
                    m, n = G.shape
                else:
                    G = None
                if G is None or max(G.shape) > group['max_precond_dim'] \
                        or min(G.shape) < 2:
                    self._adam_step(p, g, st, group)
                    continue
                r_step = min(m, n)
                if group['beta2'] != 1.0:
                    pass  # engines scale below
                engL = self._get_engine('L', st, m, g, group, r_step)
                engR = self._get_engine('R', st, n, g, group, r_step)
                if group['beta2'] != 1.0:
                    engL.scale(group['beta2'])
                    engR.scale(group['beta2'])
                engL.add_gram(G)
                engR.add_gram(G.t())
                PG = engL.apply_invroot(G)
                PGP = engR.apply_invroot(PG.t()).t()
                p.add_(PGP.reshape(shape), alpha=-lr)

    @torch.no_grad()
    def _adam_step(self, p, g, st, group):
        if 'exp_avg' not in st:
            st['exp_avg'] = torch.zeros_like(p)
            st['exp_avg_sq'] = torch.zeros_like(p)
            st['t'] = 0
        st['t'] += 1
        b1, b2 = group['adam_betas']
        st['exp_avg'].mul_(b1).add_(g, alpha=1 - b1)
        st['exp_avg_sq'].mul_(b2).addcmul_(g, g, value=1 - b2)
        bc1 = 1 - b1 ** st['t']
        bc2 = 1 - b2 ** st['t']
        denom = (st['exp_avg_sq'] / bc2).sqrt_().add_(group['adam_eps'])
        p.addcdiv_(st['exp_avg'] / bc1, denom, value=-group['lr'])

    def diagnostics(self):
        out = []
        for group in self.param_groups:
            for p in group['params']:
                st = self.state[p]
                for k in ('L', 'R'):
                    eng = st.get(k)
                    if isinstance(eng, SpectralState):
                        out.append((k, 'cont', eng.diagnostics()))
                    elif isinstance(eng, TrackState):
                        out.append((k, 'track', dict(res=eng.last_residual,
                                                     iters=eng.last_iters)))
                    elif isinstance(eng, GramState):
                        out.append((k, 'gram', dict(rank=eng.rank)))
        return out
