"""E1: torch engines vs high-precision ideal — all four modes must produce the
same trajectory on the same quadratic problem (fp64), matching the numpy/mpmath
witness results."""
import torch, sys
sys.path.insert(0, '/home/navin/shampoo/research')
from rfshampoo.optimizer import RootFreeShampoo

torch.manual_seed(0)
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
m, n, pa, pb, STEPS = 48, 36, 20, 24, 120

A = torch.randn(pa, m, dtype=torch.float64, device=dev) / m ** 0.5
B = torch.randn(n, pb, dtype=torch.float64, device=dev) / n ** 0.5
C = torch.randn(pa, pb, dtype=torch.float64, device=dev)
W0 = torch.randn(m, n, dtype=torch.float64, device=dev) / m ** 0.5

def run(mode, steps=STEPS, eigh_every=1):
    W = torch.nn.Parameter(W0.clone())
    opt = RootFreeShampoo([W], lr=0.05, eps=1e-3, mode=mode,
                          precond_dtype=torch.float64, eigh_every=eigh_every)
    traj = []
    for t in range(steps):
        loss = 0.5 * ((A @ W @ B - C) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        traj.append(W.detach().clone())
    return traj, float(loss)

ref, l0 = run('eigh')
for mode, steps in [('cont', STEPS), ('track', STEPS), ('auto', STEPS),
                    ('gram', 30)]:
    tr, l = run(mode, steps=steps)
    errs = [float((a - b).abs().max() / b.abs().max()) for a, b in zip(tr, ref)]
    print(f"{mode:5s} loss={l:.6e}  max|dW|/|W|: t=1 {errs[0]:.2e}  "
          f"t=mid {errs[len(errs)//2]:.2e}  t=end {errs[-1]:.2e}", flush=True)
print("eigh loss:", l0)
