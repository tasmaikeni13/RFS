# RootFree-Shampoo — theory in short

Everything needed to re-implement from scratch. No derivations; those are in
`NOTES.md`/`REPORT.md`, machine-checked statements in `leanproofs/`.

## 0. Object of study

Shampoo (GKS'18), per weight matrix `W ∈ R^{m×n}`, gradient `G_t`:

    L_t = L_{t-1} + G_t G_tᵀ ,  L_0 = εI_m          (m×m)
    R_t = R_{t-1} + G_tᵀ G_t ,  R_0 = εI_n          (n×n)
    W_{t+1} = W_t − η · L_t^{-1/4} · G_t · R_t^{-1/4}

`L^{-1/4}` means the principal (symmetric PD) inverse fourth root. The whole
problem: compute/apply it without an N×N eigendecomposition, exactly.

## 1. The exactness barrier (what "exact" can mean)

- **Rational:** no finite algorithm over (+,−,×,÷) of the entries outputs
  `L^{-1/4}` — already for 1×1: `X·P(a)⁴ = Q(a)⁴` forces `1+4degP = 4degQ`,
  impossible mod 4. Holds on any infinite input set ⇒ covers branching programs.
- **Radical:** with k-th roots allowed, still impossible. Witness `L = BBᵀ+I`,
  charpoly `x⁵−98x⁴+2887x³−32586x²+134052x−114660` (irreducible, Galois S₅).
  If all entries of `X = L^{-1/4}` were radical, then `tr(L·X²) = Σ√λᵢ` is
  radical ⇒ its minpoly has solvable Galois group. But its minpoly is the
  degree-32 polynomial `Q(x) = Π_{ε∈{±1}⁵}(x − Σεᵢ√λᵢ)` (irreducible over ℚ ⇒
  all sign-conjugates realized ⇒ splitting field ⊇ every √λᵢ ⇒ surjects onto
  S₅ ⇒ unsolvable). Contradiction.
  - Practical construction of Q (exact, milliseconds): power sums of Q's roots
    `p_k(Q) = 32·k!·[t^k] Πᵢ cosh(√λᵢ t)`, and
    `log Πᵢ cosh(√λᵢ t) = Σ_m a_m q_m t^{2m}` with `a_m` = Taylor coefficients
    of `log cosh`, `q_m = Σλᵢ^m ∈ ℚ` by Newton's identities on the charpoly.
    Then Newton p→e gives Q's integer coefficients. Check irreducibility.

**Consequence.** Every method (including QR-based `eigh`) is an iteration
truncated at machine ε. The only coherent target: *the same exact-arithmetic
dynamical system, no structural approximation (no diagonal / low-rank /
extra-Kronecker), realized at certified machine precision.*

## 2. The equivalence principle (why certificates suffice)

**Spectral certificate** of `L`: pair `(U, d)` with `U·Uᵀ = I`,
`L = U·diag(d)·Uᵀ`.

**Trajectory-equivalence theorem** (Lean: `trajectory_equivalence`): if at
every step the preconditioner is applied as

    P = U · diag(d^{-1/4}) · Uᵀ

through *any* exact certificate, the whole optimizer state sequence (W and
accumulators) equals ideal Shampoo's — for every spectral function f, not just
x^{-1/4}. So the entire problem reduces to: **maintain exact certificates
cheaply**. Three engines, one per rank regime.

## 3. Engine GRAM — exact cold phase (cumulative rank < N)

Intertwining theorem: `A·Y = Y·B ⇒ f(A)·Y = Y·f(B)` for all f. Applied to
`A = εI + YYᵀ`, `B = εI + YᵀY` (Y = all gradient columns collected so far,
N×k, k = cumulative rank):

    (εI + YYᵀ)^{-1/4} = ε^{-1/4}·I + Y · g(YᵀY) · Yᵀ
    g(s) = ((ε+s)^{-1/4} − ε^{-1/4}) / s        (spectral function of the k×k Gram;
                                                 value at s=0 irrelevant — kernel dirs)

Implementation: eigendecompose the k×k Gram `YᵀY = V diag(s) Vᵀ` (cheap, k≪N;
do it in fp64), then

    P·X = ε^{-1/4}·X + Y · V · diag(g(s)) · Vᵀ · (Yᵀ X).

Cost per apply: `O(k³ + N·k·cols(X))`. Strictly O(N²)-or-better while
`k ≲ N^{2/3}`. No N×N spectral work exists anywhere in this phase — and since
`L_0 = εI` is diagonal, the run needs **no initial eigendecomposition ever**.

## 4. Engine CONT — secular continuation (the core)

Maintain `(U, d)` (d ascending) across steps. Never recompute; only continue.

**Per rank-1 injection** `L ← L + ρ·v·vᵀ` (ρ>0):

1. `z = Uᵀ v`  — O(N²).
2. **Relative clustering:** group adjacent d's with gap `d_{i+1}−d_i ≤ 4·eps·d_{i+1}`.
   Within each cluster, apply a Householder reflection (to z and to the same
   columns of U) concentrating the cluster's z-mass into one component; the
   rest become exact zeros.
3. **Relative deflation:** drop index i if `ρ·z_i² ≤ ⅛·eps·d_i`.
   (NOT the LAPACK absolute rule — the consumed operator λ^{-1/4} conditions
   *relatively*; absolute deflation costs ~4 digits at small eigenvalues.)
4. **Secular solve** on the kept k indices: new eigenvalues are the roots of

       f(μ) = 1 + ρ·Σᵢ zᵢ²/(dᵢ − μ) = 0,

   one root per interval: `μᵢ ∈ (dᵢ, d_{i+1})`, last in `(d_k, d_k + ρ‖z‖²)`.
   Solve in the shifted variable `tᵢ = μᵢ − dᵢ` (never form μ−d by subtraction):
   f is monotone ↑ on each interval; safeguarded Newton + bisection bracket;
   fp64 always (even if the GEMMs are fp32). All k roots are independent —
   batch/Triton-parallelize.
5. **Gu–Eisenstat reconstruction** (self-correction — this is what keeps 10⁴
   continuations orthogonal): recompute the weights from the computed roots,

       ẑᵢ² = ( Πⱼ (μⱼ − dᵢ) ) / ( ρ · Π_{j≠i} (dⱼ − dᵢ) ),   sign(ẑᵢ) = sign(zᵢ),

   using `μⱼ − dᵢ = (dⱼ − dᵢ) + tⱼ`.
6. **Löwner eigenvectors** (exact, given ẑ and roots):

       q̂ⱼ ∝ ( ẑᵢ / (dᵢ − μⱼ) )ᵢ ,   normalized;   use dᵢ − μⱼ = (dᵢ − dⱼ) − tⱼ.

7. Update: `d[kept] ← d + t`; `U[:,kept] ← U[:,kept] @ Q̂`; re-sort ascending.

**Rank-r step:** feed r rank-1s. CRITICAL: feed the *factored* gradient, not
its dense columns. Backprop gives `G = A·Bᵀ` (A = δ, B = x, both N×r); then

    G·Gᵀ = V·Vᵀ  with  V = A · (BᵀB)^{1/2}   (r×r half — rank-safe via eigh)

so the injection is exactly rank r regardless of N. Application is also
factored: `L^{-1/4}·G·R^{-1/4} = (S_L.apply A) · (S_R.apply B)ᵀ`.

**EMA / β₂:** `L ← βL` is exact and O(N): `d ← β·d`.

**Costs per step (rank r):** O(N·r) scalar secular work, `O(N²·r)` in GEMMs
(`Uᵀv` and `U·Q̂`); Q̂ is Cauchy-structured, so `U·Q̂` is `O(N²·log)`-able by 1D
FMM on the kernel 1/(d−μ) (machine-precision; interlacing makes geometry benign).
Same complexity class as Shampoo's own `G·Gᵀ` accumulation — the O(N³) eig and
its staleness simply disappear.

## 5. Engine TRACK — dense regime (per-step rank ≈ N)

**Do not warm-start any one-sided iteration on X.** Linearized at the fixed
point, the map `X ← X(5I − X⁴L)/4` multiplies error mode (i,j) by

    φ(r) = 1 − (r⁻¹ + r⁻² + r⁻³ + r⁻⁴)/4 ,   r = (λᵢ/λⱼ)^{1/4},

which is ≈ −κ/4 for λᵢ≪λⱼ: non-commuting rounding error amplifies ~κ per
anchored step (empirically 1e-15 → NaN). Stable warm carry-over *requires*
carrying the eigenbasis — i.e., CONT. TRACK is therefore a **cold, stable,
matmul-only refresh** per update:

    s = ‖L‖_F ;  A = L/s                       (spec ⊂ (0,1])
    (Y,Z) NS pair:  T = (3I − ZY)/2 ;  Y ← YT ;  Z ← TZ ;  symmetrize both
        → Y → A^{1/2},  Z → A^{-1/2}
    run NS twice:  B = A^{1/2} → W = (B/‖B‖_F)^{-1/2}
    X = L^{-1/4} = ‖B/…‖-unscale · s^{-1/4} · W

Stop rules: `‖T−I‖_F/√N < tol` (tol ≈ 50·eps(dtype)); plus a floor detector —
once past the lift plateau (residual < 0.1), stop after 4 iterations without a
5% new best (finite-precision floor ≈ eps·√κ; certificates record it).
**Certificate** (always log): `‖X⁴L − I‖_F/√N`. Iterations ≈ log₂.₂₅(κ)+O(1)
per stage; norm-scaling acceleration is known headroom (~3×).

## 6. The algorithm (auto mode, per Kronecker factor of dimension N)

    if 1D param or N > max_precond_dim:  diagonal/Adam fallback
    phase GRAM   while cumulative rank < min(N/2, cost crossover ~ N^{2/3}·r^{1/3})
    then CONT    if per-step rank r ≤ ρ₀·N      (ρ₀ ≈ 1/8; needs factored grads)
    else TRACK   (refresh per update, certified)
    handoffs are exact:  GRAM→CONT: continue (U,d) from εI through the buffer;
                         GRAM→TRACK: X = pushdown-apply(I).

All phases realize the *same* operator (Theorem §2) — they differ only in cost.

## 7. Complexity summary

| phase | maintenance + apply per step | O(N²)-or-better when |
|---|---|---|
| GRAM  | O(r_cum³ + N·r_cum·r_step) | r_cum ≲ N^{2/3} |
| CONT  | O(N²·r_step) GEMM · O(N²·r_step·log N) FMM | always (given factored grads) |
| TRACK | GEMM-only, ~4·(#NS iters) GEMMs | — (dense regime is GEMM-bound for any exact method) |

Shampoo's own accumulation is already Θ(N²·r_step): CONT adds **no complexity
class**; it removes the O(N³)-bad-constants eig and the staleness.

## 8. Numerical rules that matter (hard-won)

1. Relative (not absolute) deflation and clustering — §4 steps 2–3.
2. Always reconstruct ẑ from computed roots (Gu–Eisenstat) before forming
   eigenvectors; this replaces re-orthogonalization entirely.
3. Secular solves in fp64 even when everything else is fp32; shifted variable
   `t = μ − dᵢ`; bisection-safeguarded Newton.
4. Never re-anchor a warm Newton iterate on X (φ(r) blow-up, §5). Coupled NS
   only, symmetrized every iteration, with floor detection.
5. Feed factored gradients (`G = ABᵀ`) or you silently pay rank = min(m,n).
6. Certificates: CONT — `‖UᵀU − I‖_max`; TRACK — `‖X⁴L−I‖_F/√N`. Log them; they
   are the "no approximation" guarantee at runtime.
7. Comparisons vs `eigh` are ill-posed when κ > 1/eps (both emit noise); with
   ε-regularization κ ≤ λ_max/ε keeps you in range. `eigh` itself has *absolute*
   backward error — at small eigenvalues under λ^{-1/4}, continuation is the
   more accurate one (verified vs 50-digit reference).

## 9. Measured ground truth (for regression-testing a new implementation)

- Pushdown identity: exact to 4.9e-46 at 50-digit arithmetic.
- 4000 rank-1 continuations (incl. degenerate εI start): orth/eig errors flat ~1e-14.
- Full Shampoo, all 4 modes vs eigh-per-step, fp64 GPU, 120 steps: ≤ 1.8e-12.
- GPT-85M, 250 steps: RootFree ≡ fp64-eig-gold to 4 decimals in loss at every
  checkpoint; ‖W−gold‖/‖gold‖ = 2.3e-7 (vs 4.1e-3 for eig-every-100 staleness).
- Factored low-rank regime (N=1024, r=4): trajectory 5e-12→4.2e-11 vs gold,
  orth 7.5e-15; stale variant diverges at the same lr.
- Continuation refresh vs fp64 eigh (GPU): 8.5× (N=2048), 12.9× (N=4096);
  Triton secular kernel ~1e-15 agreement, 8× over torch-vectorized (N=1024).

## 10. Pointers

Lean names: `cfc_intertwine`, `cfc_gram_pushdown(_explicit)`,
`secular_charpoly_eval`, `loewner_eigenvector`, `gu_eisenstat_identity`,
`SpectralPair.cfc_eq`, `trajectory_equivalence`, `invQuart_defining`,
`no_rational_inverse_quartic_root`, `radical_entries_give_radical_trace`,
`no_radical_root_of_gal`. Reference code: `rfshampoo/engine.py` (three engines),
`rfshampoo/secular_triton.py` (kernel), `witness/secular.py` (clean numpy CONT),
`witness/w4b_galois_fast.py` (certificate), `witness/w6_fmm.py` (Cauchy FMM).
