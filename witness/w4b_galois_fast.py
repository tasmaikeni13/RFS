"""Fast exact construction of Q(x) = prod_{eps in {±1}^5} (x - sum_i eps_i sqrt(lambda_i)).

Identity: sum over the 32 sign patterns of exp((sum eps_i sqrt(lambda_i)) t)
        = 32 * prod_i cosh(sqrt(lambda_i) t),
so the power sums of Q's roots are p_k(Q) = 32 * k! * [t^k] prod_i cosh(sqrt(lambda_i) t),
and  log prod_i cosh(sqrt(lambda_i) t) = sum_m a_m q_m t^{2m}
with a_m the rational Taylor coefficients of log cosh(u) in u^2 = (lambda t^2),
q_m = sum_i lambda_i^m in Q (Newton's identities on charpoly(L)).
All exact rational arithmetic; then Newton -> coefficients of Q; check
irreducibility over Q. Certificates: (C1) charpoly irreducible, Galois order
120; (C2) Q irreducible of degree 32.
"""
from fractions import Fraction as Fr
import sympy as sp
import random, json, time

x = sp.symbols('x')
DEG = 32
M = DEG // 2 + 1   # series in u = t^2 up to u^{16}

def logcosh_series(M):
    """log cosh(t) = sum_{m>=1} a_m t^{2m}: exact via series composition."""
    t = sp.symbols('t')
    s = sp.log(sp.cosh(t)).series(t, 0, 2 * M + 1).removeO()
    p = sp.Poly(s, t)
    return [Fr(int(sp.numer(p.coeff_monomial(t ** (2 * m)))),
               int(sp.denom(p.coeff_monomial(t ** (2 * m)))))
            if p.coeff_monomial(t ** (2 * m)) != 0 else Fr(0)
            for m in range(M)]

A = logcosh_series(M)   # A[m] = a_m (A[0] = 0)

def series_exp(c, M):
    """exp of a power series c (c[0]=0), truncated to M terms: e' = e * c'."""
    e = [Fr(0)] * M
    e[0] = Fr(1)
    for n in range(1, M):
        acc = Fr(0)
        for k in range(1, n + 1):
            acc += k * c[k] * e[n - k] if k < M else 0
        e[n] = acc / n
    return e

def build_Q_fast(charpoly_coeffs):
    """charpoly = x^5 + c4 x^4 + ... + c0; returns integer coeff list of Q."""
    c4, c3, c2, c1, c0 = [Fr(v) for v in charpoly_coeffs]
    # Newton's identities: q_m = sum lambda_i^m, with e1..e5 from charpoly signs
    e = [Fr(0)] * 6
    e[1], e[2], e[3], e[4], e[5] = -c4, c3, -c2, c1, -c0
    q = [Fr(5)] + [Fr(0)] * (M)
    for m in range(1, M + 1):
        # Newton: q_m = e1 q_{m-1} - e2 q_{m-2} + ... + (-1)^{m-1} m e_m
        if m <= 5:
            q[m] = sum((-1) ** (j - 1) * e[j] * q[m - j] for j in range(1, m)) \
                   + (-1) ** (m - 1) * m * e[m]
        else:
            q[m] = sum((-1) ** (j - 1) * e[j] * q[m - j] for j in range(1, 6))
    # log prod cosh = sum_m A[m] * q[m] * u^m   (u = t^2)
    lc = [Fr(0)] * M
    for m in range(1, M):
        lc[m] = A[m] * q[m]
    pc = series_exp(lc, M)          # prod_i cosh, coefficients of u^m = t^{2m}
    # power sums of Q's roots: p_k = 32 * k! * [t^k] prod cosh  (odd k: 0)
    fact = [1] * (DEG + 1)
    for i in range(1, DEG + 1):
        fact[i] = fact[i - 1] * i
    p = [Fr(0)] * (DEG + 1)
    p[0] = Fr(DEG)
    for k in range(2, DEG + 1, 2):
        p[k] = Fr(32) * fact[k] * pc[k // 2]
    # Newton -> elementary symmetric E of Q's roots: k E_k = sum_{i=1..k} (-1)^{i-1} E_{k-i} p_i
    E = [Fr(0)] * (DEG + 1)
    E[0] = Fr(1)
    for k in range(1, DEG + 1):
        acc = Fr(0)
        for i in range(1, k + 1):
            acc += (-1) ** (i - 1) * E[k - i] * p[i]
        E[k] = acc / k
    # Q(x) = sum_{k} (-1)^k E_k x^{DEG-k}
    coeffs = [(-1) ** k * E[k] for k in range(DEG + 1)]
    assert all(c.denominator == 1 for c in coeffs), "non-integer coeff — bug"
    return [int(c) for c in coeffs]

def main():
    for seed in range(60):
        rng = random.Random(seed)
        B = sp.Matrix(5, 5, lambda i, j: rng.randint(-3, 3))
        L = B * B.T + sp.eye(5)
        p = sp.Poly(L.charpoly(x).as_expr(), x, domain='QQ')
        if not p.is_irreducible:
            continue
        try:
            G, _ = p.galois_group()
        except Exception:
            continue
        if G.order() != 120:
            continue
        cs = [int(v) for v in p.all_coeffs()[1:]]
        t0 = time.time()
        Qc = build_Q_fast(cs)
        Q = sp.Poly(Qc, x, domain='QQ')
        irr = Q.is_irreducible
        dt = time.time() - t0
        print(json.dumps(dict(seed=seed, charpoly=str(p.as_expr()),
                              galois_order=120, Q_degree=Q.degree(),
                              Q_irreducible=bool(irr), seconds=round(dt, 2))),
              flush=True)
        if irr:
            with open('/home/navin/shampoo/research/witness/w4_certificate.json', 'w') as f:
                json.dump(dict(seed=seed,
                               L=[[int(v) for v in row] for row in L.tolist()],
                               charpoly=str(p.as_expr()),
                               charpoly_coeffs=[str(v) for v in p.all_coeffs()],
                               galois_group_order=120,
                               Q_coeffs=[str(c) for c in Qc],
                               Q_degree=32, Q_irreducible=True,
                               method='cosh-generating-function, exact rational'),
                          f, indent=1)
            print("CERTIFICATE WRITTEN")
            return
    print("no witness in 60 seeds")

main()
