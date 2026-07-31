"""H1 kill test: orthogonality/accuracy drift of the continuation engine over
thousands of rank-1 injections, vs recompute-from-scratch eigh at fp64."""
import numpy as np, time, json
from secular import rank1_update, apply_invroot

rng = np.random.default_rng(1)

def run(n, steps, eps=1e-6, probe_every=50):
    d = np.full(n, eps); U = np.eye(n)
    L = eps * np.eye(n)
    out = []
    t0 = time.time()
    for t in range(1, steps + 1):
        v = rng.standard_normal(n) / np.sqrt(n)
        d, U = rank1_update(d, U, v, 1.0)
        L += np.outer(v, v)
        if t % probe_every == 0 or t == steps:
            orth = np.abs(U.T @ U - np.eye(n)).max()
            recon = np.abs(U @ (d[:, None] * U.T) - L).max() / np.abs(L).max()
            dref = np.linalg.eigvalsh(L)
            eigerr = np.abs(dref - d).max() / dref.max()
            X = rng.standard_normal((n, 8))
            dr, Ur = np.linalg.eigh(L)
            ref = Ur @ ((dr[:, None] ** -0.25) * (Ur.T @ X))
            got = apply_invroot(d, U, X)
            app = np.abs(ref - got).max() / np.abs(ref).max()
            out.append(dict(t=t, orth=float(orth), recon=float(recon),
                            eig=float(eigerr), app=float(app)))
    dt = time.time() - t0
    print(f"n={n} steps={steps} wall={dt:.1f}s  final:", json.dumps(out[-1]))
    for row in out[:: max(1, len(out)//10)]:
        print("   ", json.dumps(row))
    return out

run(64, 4000)
run(128, 2000)
