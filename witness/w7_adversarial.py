"""w7: adversarial stress for the continuation engine — the counterexample hunt.
(a) repeated near-parallel directions (deflation/cluster stress),
(b) huge dynamic range injections (kappa up to ~1e12),
(c) exact duplicate directions,
(d) alternating tiny/huge magnitudes.
Metrics vs fresh eigh at fp64 after every 100 injections."""
import numpy as np, sys
sys.path.insert(0, '/home/navin/shampoo/research/witness')
from secular import rank1_update, apply_invroot

rng = np.random.default_rng(11)
n, STEPS, EPS0 = 128, 1200, 1e-9

d = np.full(n, EPS0); U = np.eye(n); L = EPS0 * np.eye(n)
base = rng.standard_normal((n, 4))
worst = dict(orth=0.0, eig=0.0, app=0.0)
for t in range(1, STEPS + 1):
    kind = t % 4
    if kind == 0:      # near-parallel to a fixed direction
        v = base[:, 0] + 1e-9 * rng.standard_normal(n)
    elif kind == 1:    # exact repeat
        v = base[:, 1].copy()
    elif kind == 2:    # huge
        v = 1e4 * rng.standard_normal(n)
    else:              # tiny
        v = 1e-4 * rng.standard_normal(n)
    d, U = rank1_update(d, U, v, 1.0)
    L += np.outer(v, v)
    if t % 100 == 0:
        orth = np.abs(U.T @ U - np.eye(n)).max()
        dref = np.linalg.eigvalsh(L)
        eigerr = np.abs(np.sort(dref) - np.sort(d)).max() / dref.max()
        X = rng.standard_normal((n, 4))
        dr, Ur = np.linalg.eigh(L)
        dr = np.clip(dr, EPS0, None)
        ref = Ur @ ((dr[:, None] ** -0.25) * (Ur.T @ X))
        got = apply_invroot(d, U, X)
        app = np.abs(ref - got).max() / np.abs(ref).max()
        worst = {k: max(worst[k], v2) for k, v2 in
                 dict(orth=orth, eig=eigerr, app=app).items()}
        print(f"t={t:5d} kappa={dref.max()/max(dref.min(),1e-300):.1e} "
              f"orth={orth:.2e} eig={eigerr:.2e} app={app:.2e}", flush=True)
print("WORST:", worst)
