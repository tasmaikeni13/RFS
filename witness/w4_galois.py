"""P4 witness + certificates (all exact computations):
  (C1) L: rational symmetric PD 5x5; p := charpoly(L) irreducible /Q with
       Galois group of order 120 (=> S5, unsolvable).
  (C2) Q(x) := prod over all 32 sign patterns of (x - sum_i eps_i sqrt(lambda_i)),
       an integer polynomial computed exactly by pairing recursion + symmetric
       reduction; certificate: Q irreducible over Q.
Consequences (paper Thm): deg minpoly(tr L^{1/2}) = 32 => all sign-conjugates
realized => sqrt(lambda_i) in splitting field E => F subset E => Gal(E/Q)
surjects onto S5 => unsolvable => tr(L^{1/2}) not in solvableByRad Q R.
Combined with the Lean bridge (Impossible.lean), this proves: no finite
radical-arithmetic algorithm outputs L^{-1/4} for this (hence for generic) L.
"""
import sympy as sp
import random, json, time

x, u = sp.symbols('x u')
lam = sp.symbols('l1 l2 l3 l4 l5')

def build_Q(pcoeffs):
    """Exact Q for the quintic with elementary symmetric values pcoeffs=(e1..e5)."""
    F = sp.Poly(x, x, *lam)  # F_0 = x
    Fexpr = x
    for k in range(5):
        # F_k(x) -> F_k(x-u)*F_k(x+u) with u^2 -> lam_k
        A = Fexpr.subs(x, x - u)
        B = Fexpr.subs(x, x + u)
        prod = sp.expand(A * B)
        # substitute u^2 = lam_k: collect powers of u (only even survive)
        prod = sp.Poly(prod, u)
        terms = 0
        for (deg,), c in prod.terms():
            assert deg % 2 == 0, "odd power of u survived — bug"
            terms += c * lam[k] ** (deg // 2)
        Fexpr = sp.expand(terms)
    # Fexpr: degree-32 poly in x, coefficients symmetric in lam
    e1, e2, e3, e4, e5 = pcoeffs
    P = sp.Poly(Fexpr, x)
    from sympy.polys.polyfuncs import symmetrize
    svars = sp.symbols('s1:6')
    Qcoeffs = []
    for c in P.all_coeffs():
        cexpr = sp.expand(sp.sympify(c))
        if cexpr.free_symbols & set(lam):
            sym, rem = symmetrize(cexpr, *lam, formal=True)
            assert sp.simplify(rem) == 0, "coefficient not symmetric — bug"
            val = sym.subs(dict(zip(svars, (e1, e2, e3, e4, e5))))
            Qcoeffs.append(sp.Rational(sp.nsimplify(val)))
        else:
            Qcoeffs.append(sp.Rational(cexpr))
    return sp.Poly(Qcoeffs, x, domain='QQ')

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
        # elementary symmetric values: p = x^5 - e1 x^4 + e2 x^3 - e3 x^2 + e4 x - e5
        c = p.all_coeffs()  # [1, c4, c3, c2, c1, c0]
        e = (-c[1], c[2], -c[3], c[4], -c[5])
        t0 = time.time()
        Q = build_Q(e)
        tQ = time.time() - t0
        irr = Q.is_irreducible
        print(json.dumps(dict(seed=seed, L=[[int(v) for v in row] for row in L.tolist()],
                              charpoly=str(p.as_expr()), galois_order=120,
                              Q_degree=Q.degree(), Q_irreducible=bool(irr),
                              build_seconds=round(tQ, 1))), flush=True)
        if irr:
            with open('w4_certificate.json', 'w') as f:
                json.dump(dict(seed=seed, L=[[int(v) for v in row] for row in L.tolist()],
                               charpoly=str(p.as_expr()),
                               charpoly_coeffs=[str(v) for v in c],
                               galois_group_order=120,
                               Q=str(Q.as_expr()), Q_degree=int(Q.degree()),
                               Q_irreducible=True), f, indent=1)
            print("CERTIFICATE WRITTEN")
            return
    print("no witness found")

main()
