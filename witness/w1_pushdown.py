"""H2 witness: Gram pushdown identity
   f(eps I_m + Y Y^T) = f(eps) I_m + Y g(Y^T Y) Y^T,  g(s) = (f(eps+s)-f(eps))/s
   with f(x) = x^{-1/4}; checked at fp64 and at 50-digit precision via mpmath."""
import numpy as np
import mpmath as mp

rng = np.random.default_rng(7)
m, k, eps = 60, 13, 1e-3
Y = rng.standard_normal((m, k))

# fp64
def invroot_eig(M, p=4):
    d, Q = np.linalg.eigh(M)
    return Q @ ((d[:, None] ** (-1.0 / p)) * Q.T)

lhs = invroot_eig(eps * np.eye(m) + Y @ Y.T)
s, V = np.linalg.eigh(Y.T @ Y)                      # k x k only
g = ((eps + s) ** -0.25 - eps ** -0.25) / s
rhs = eps ** -0.25 * np.eye(m) + Y @ (V @ (g[:, None] * V.T)) @ Y.T
print("fp64  max|lhs-rhs| / |lhs| =", np.abs(lhs - rhs).max() / np.abs(lhs).max())

# 50-digit
mp.mp.dps = 50
Ym = mp.matrix(Y.tolist())
Mm = mp.eye(m) * mp.mpf(eps) + Ym * Ym.T
E, Q = mp.eigsy(Mm)
L = mp.zeros(m, m)
for i in range(m):
    L[i, i] = E[i] ** mp.mpf(-0.25)
lhs_m = Q * L * Q.T
Gm = Ym.T * Ym
E2, Q2 = mp.eigsy(Gm)
D2 = mp.zeros(k, k)
for i in range(k):
    D2[i, i] = ((mp.mpf(eps) + E2[i]) ** mp.mpf(-0.25) - mp.mpf(eps) ** mp.mpf(-0.25)) / E2[i]
rhs_m = mp.eye(m) * mp.mpf(eps) ** mp.mpf(-0.25) + Ym * (Q2 * D2 * Q2.T) * Ym.T
err = max(abs(lhs_m[i, j] - rhs_m[i, j]) for i in range(m) for j in range(m))
print("50dps max|lhs-rhs| =", mp.nstr(err, 5))
