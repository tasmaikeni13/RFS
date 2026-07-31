# RootFree-Shampoo: Shampoo's exact trajectory without explicit inverse fourth roots

**Research report — 2026-07-16.** Full artifacts under `/home/navin/shampoo/research/`:
Lean proofs (`leanproofs/`, Mathlib v4.31.0, `lake build` clean, **0 sorries, 0 new axioms**),
witnesses (`witness/`), optimizer + experiments (`rfshampoo/`), notes (`NOTES.md`),
research state (`STATE.md`).

---

## 1. What was asked, and what is mathematically possible

Goal: replace the explicit computation of `L^{-1/4}`, `R^{-1/4}` in Shampoo with an
alternative operator at `O(N²)`-or-better per step, **exact same trajectory, no
approximations**.

The first result is a pair of impossibility theorems that pin down what "exact"
can even mean — the FLT-style indirect move: prove a barrier, and let the barrier
define the achievable target.

**Theorem I (rational impossibility; Lean-complete:
`Shampoo.no_rational_inverse_quartic_root`).** No algorithm using only field
operations (+, −, ×, ÷) on the entries computes `L ↦ L^{-1/4}` — already
impossible for 1×1 matrices, by a degree-mod-4 argument fully machine-checked.

**Theorem II (radical impossibility; Lean skeleton + exact certificate).** Even
allowing arbitrary k-th roots, no finite computation outputs `L^{-1/4}` for the
explicit witness

    L = B·Bᵀ + I,  charpoly = x⁵ − 98x⁴ + 2887x³ − 32586x² + 134052x − 114660.

Chain: entries radical ⇒ `tr(L·X²) = Σ√λᵢ` radical (Lean:
`radical_entries_give_radical_trace`) ⇒ its minpoly has solvable Galois group
(Mathlib: `isSolvable_gal_minpoly`). But the *exact* certificate computation
(`witness/w4b_galois_fast.py`, all rational arithmetic via a new cosh
generating-function construction, 0.03 s) shows: charpoly is irreducible with
Galois group S₅, and the degree-32 sign-product polynomial
`Q(x) = Π_{ε∈{±}⁵}(x − Σεᵢ√λᵢ)` is **irreducible over ℚ** ⇒ minpoly(Σ√λᵢ) = Q
⇒ all 32 sign-conjugates are realized ⇒ the splitting field contains every √λᵢ,
hence the S₅ splitting field of charpoly(L) ⇒ its Galois group surjects onto S₅
(Lean: `not_solvable_of_surjective_perm5`) — unsolvable. Contradiction.
(Cross-check: `|Q(Σ√λᵢ)| = 7.9e-46` at 60-digit precision.)

**Consequence.** *Every* algorithm — including eig-based Shampoo itself, whose
QR eigensolver is an infinite iteration truncated at machine ε — fails literal
exactness. The only coherent target is: **the same exact-arithmetic dynamical
system, with no structural approximation, realized at (certified) machine
precision.** That target is achieved below, with a realization that is at
least as exact as the eigendecomposition baseline it replaces — strictly more
accurate in the small-eigenvalue regime (w3) — and fresh at every step, which
deployed Shampoo is not. (The Lean statement covers branching algorithms too:
the identity is refuted on any infinite input set, and a finite branching
algorithm sends an infinite set of inputs through some single branch.)

## 2. The replacement operator: exact algebraic continuation

Everything rests on a machine-checked representation-independence theorem:

**Theorem III (trajectory equivalence; Lean: `Shampoo.trajectory_equivalence`).**
For every spectral function f, every gradient oracle, and *any* refresh policy
that returns exact spectral certificates `(U, d)` with `L = U·diag(d)·Uᵀ`,
`U·Uᵀ = 1`, the certificate optimizer's weights and accumulators equal ideal
`cfc f`-Shampoo's at every step. (Non-vacuity is also machine-checked, by
instantiating the oracle with Mathlib's spectral theorem.)

So it suffices to *maintain* exact spectral certificates cheaply. Three exact,
eigendecomposition-free engines do this, one per rank regime (auto-switched):

**GRAM (cold phase; Lean: `cfc_intertwine`, `cfc_gram_pushdown_explicit`).**
The new intertwining theorem — `A·Y = Y·B ⇒ f(A)·Y = Y·f(B)` for all f — gives

    (εI + Y·Yᵀ)^{-1/4} = ε^{-1/4}·I + Y · g(εI + YᵀY) · Yᵀ,   g(x) = (x^{-1/4} − ε^{-1/4})/(x − ε),

so while the accumulated gradient rank r is below the dimension, the exact
inverse root acts through an r×r spectral problem: `O(r³ + N·r·r_step)` per
step, no N×N spectral computation at all. Verified to 4.9e-46 (50-digit run).

**CONT (secular continuation; Lean: `secular_charpoly_eval`,
`loewner_eigenvector`, `gu_eisenstat_identity`).** Rank-1 injections update the
certificate *exactly*: new eigenvalues are the secular roots (an algebraic
characterization, machine-checked), eigenvectors are Löwner vectors
(machine-checked), and z is *reconstructed* from the computed roots via the
Gu–Eisenstat product identity (machine-checked), which self-corrects rounding:
orthogonality stays ~1e-14, flat, across 4000 consecutive injections. Cost
`O(N²·r_step)` per step (GEMM), `O(N²·r_step·log N)` via the Cauchy-FMM path
(demonstrated: 8e-15 accuracy, linear scaling, 22× over dense at N = 32k).
A custom Triton kernel solves all N secular roots in parallel at machine
precision (1e-15 vs eigh), 8× over the vectorized torch solver at N = 1024.
Two design deltas vs. classical divide-and-conquer eigensolvers:
- **relative deflation** (`ρzᵢ² ≤ ⅛·EPS·dᵢ`, relative cluster gaps): required
  because Shampoo consumes `λ^{-1/4}`, whose conditioning is relative; LAPACK's
  absolute rules lose ~4 digits at small eigenvalues,
- **lifetime continuation**: the certificate is never recomputed, only
  continued — from the exactly-diagonal `L₀ = εI`, so no eigendecomposition is
  ever needed, at any step, including t = 0.

**TRACK (dense regime).** When per-step rank ≈ N, updates are GEMM-class for
*any* exact method. A stable coupled Newton–Schulz refresh (matmul-only,
`L^{-1/4} = (L^{1/2})^{-1/2}`, symmetrized, online residual certificate
`‖X⁴L − I‖_F/√N`) realizes the operator at machine precision (1.3e-15 steady).
**Negative result with mechanism:** warm-starting a one-sided Newton iteration
on X alone is *provably unstable* — the linearized map has modes
`φ = 1 − (r⁻¹+r⁻²+r⁻³+r⁻⁴)/4`, `r = (λᵢ/λⱼ)^{1/4}`, i.e. error amplification
~κ/4 per anchored step (observed: 1e-15 → NaN). Stable warm carry-over requires
auxiliary spectral state — precisely what CONT maintains. This elevates the
spectral certificate from "an option" to the unique stable warm-carryable
representation among polynomial-update candidates.

## 3. Verification summary

**Lean 4 + Mathlib (30 lemmas/theorems, 0 sorries):**
| Pillar | Machine-checked statements |
|---|---|
| Intertwine | `pow/aeval_intertwine`, `cfc_eq_aeval_interpolate`, `cfc_intertwine`, `cfc_gram_pushdown(_explicit)` |
| Secular | `det_one_sub_smul_vecMulVec`, `secular_charpoly_eval`, `loewner_eigenvector`, `gu_eisenstat_identity` |
| Equivalence | `SpectralPair.cfc_eq` (representation independence), `spectrum_subset`, `trajectory_equivalence`, `mathlibRefresh_isFor` |
| Impossible | `no_rational_inverse_quartic_root`, `radical_entries_give_radical_trace`, `no_radical_root_of_gal`, `not_solvable_of_surjective_perm5` |

**Numerical witnesses (fp64 + 50–60 digit references):**
- w1: pushdown identity exact to 4.9e-46 (50 dps).
- w2: 4000 rank-1 continuations — orthogonality/eigenvalue/application errors
  flat at ~1e-14 (no drift growth), including the degenerate εI start.
- w3: full two-sided Shampoo, 150 steps vs 50-digit mpmath reference: the
  continuation trajectory is *closer to truth than eigh-per-step* at early steps
  (error ratios 0.38–0.84) and at parity thereafter (≤1.16).
- E1 (GPU, fp64): all four modes vs eigh-per-step over 120 steps:
  gram 3.1e-13, track 7.0e-13, auto 7.4e-13, cont 1.8e-12 — machine-level.
- w4/w4b: the Galois certificate (S₅ + irreducible Q₃₂), exact, cross-checked
  at 60 digits.
- w6: Cauchy-FMM apply — 2.2e-15…8.5e-15 relative error, linear scaling.
- w7 (adversarial): 1200 engineered injections (duplicates, near-parallel
  floods, 1e-4/1e+4 magnitude alternation, κ→1e17): certificate orthogonality
  ≤ 2.8e-14 and eigenvalue error ≤ 1.7e-13 throughout; the applied-operator
  comparison is ill-posed only while κ > 1/eps (both methods emit noise there)
  and recovers to 2e-13 immediately — no corruption carried.
- w5/Triton: continuation refresh 8.5–12.9× faster than fp64 eigh at
  N = 2048–4096; Triton secular kernel 8× over torch-vectorized at N = 1024.

**Training (GPT-85M-class, d=768, 8 layers, 250 steps, identical seeds/data,
2×Quadro RTX 6000, both cards saturated):**

| run | median opt ms | final loss | mean loss (last 50) | ‖W − gold‖/‖gold‖ @250 |
|---|---:|---:|---:|---:|
| eigh1 — fp64 eig every step (gold) | 15 710 | 7.8803 | 7.9213 | — |
| **rf — RootFree, zero eig, certified** | **14 085** | **7.8803** | **7.9213** | **2.3e-7** |
| eighK=100 — deployed staleness (t=1 + every 100) | 908 | 6.6122 | 6.6117 | 4.1e-3 |
| adamw (reference, different algorithm) | 3 | 5.3191 | 5.3340 | 2.5e-2 |

RootFree reproduces the gold trajectory to 2.5e-9…2.3e-7 across all 250 steps
(loss identical to four decimals at every checkpoint) — **four orders of
magnitude closer to ideal Shampoo than the deployed-staleness variant**, at
10% less per-step cost than the fp64-eig gold, with zero eigendecompositions.
The stale variant is not "cheaper ideal Shampoo": it is a *different algorithm*
(trajectory 4.1e-3 away; different final loss — lower here, catastrophically
higher, 13.81, in a variant without the t=1 refresh). Shampoo hyperparameters
were untuned (pure GKS sum, flat lr): the experiment demonstrates trajectory
equivalence and cost, not leaderboard placement — AdamW's better loss on this
synthetic task reflects tuning, not a verdict on Shampoo.

**E4 (CONT's native regime — 1024×1024 layer, factored gradients of rank 4,
150 steps, fp64):** continuation tracks the fresh-eigh gold trajectory at
5.0e-12 → 4.2e-11 with certificate orthogonality 7.5e-15 — machine-exact,
fresh every step — while the eigh-K=100 stale variant *diverges outright* at
the same learning rate (99 steps under an ε-scale preconditioner). Honest
wall-clock: at this small size the torch-level engine is 0.5× cusolver-eigh
(166 vs 83 ms/step) — per-rank-1 launch overhead and the fixed-90-iteration
secular solver dominate; the asymptotics win from N≈2048 (w5: 8.5×/12.9× vs
fp64 eigh at N = 2048/4096; the Triton kernel is a further 8× on the solver).
Design note: the O(N²r) regime consumes gradients in factored form G = ABᵀ —
which is exactly how backprop produces them per layer (δᵀx) — so the exact
rank-r path is available in real training without any rank detection.

## 4. Honest complexity claims

For factor dimension N, per-step gradient rank r_step, cumulative rank r_cum:
- GRAM: `O(r_cum³ + N·r_cum·r_step)` — strictly `O(N²)`-or-better while
  `r_cum ≲ N^{2/3}`; analytic handoff at `r_cum ≈ N^{2/3}·r_step^{1/3}`.
- CONT: `O(N²·r_step)` eager, `O(N²·r_step·log N)` FMM — the same complexity
  class as Shampoo's own unavoidable accumulation GEMMs (`Θ(N²·r_step)`); the
  `O(N³)`-with-bad-constants eigendecomposition and its staleness are gone.
- TRACK: GEMM-only certified refresh (no eig; fresh every step) — dense-regime
  cost is GEMM-class for any exact method.
- The impossibility theorems bound the semantics for *all* methods; whether
  `O(N²·polylog)` total is achievable in the dense full-rank regime is stated
  as open (with the warm-instability theorem as evidence for "no" via
  X-only iterations).

## 5. What is genuinely new here

1. The **intertwining + Gram pushdown** packaging of Shampoo's cold phase as an
   exact identity (Lean-checked), making the entire early trajectory free of
   N×N spectral computation from `L₀ = εI`.
2. **Lifetime secular/Löwner continuation** of the preconditioner's spectral
   certificate across optimizer steps — divide-and-conquer eigensolver algebra
   repurposed as a *temporal* continuation — with the **relative-deflation**
   rule required by the inverse-root use-case, machine-checked component
   identities, and the machine-checked **trajectory-equivalence theorem**.
3. The **warm-tracking instability mechanism** `φ(r) = 1 − (Σ r⁻ⁱ)/4` — a
   negative result explaining why the spectral certificate is the unique stable
   warm representation.
4. The **impossibility bookend** (rational: Lean-complete; radical: Lean bridge
   + exact S₅/Q₃₂ certificate via a new cosh generating-function construction
   computing `Π_ε(x − Σεᵢ√λᵢ)` in exact arithmetic in milliseconds).
5. A working optimizer (`rfshampoo`) with certified diagnostics, a Triton
   secular kernel, and machine-precision trajectory parity demonstrated on GPU.

Prior-art positioning: matmul-only Newton root solvers for Shampoo exist
(Distributed Shampoo; DASH '26 — cold, uncertified, no equivalence theorems);
SOAP tracks eigenbases *approximately* (QR/power iteration, algorithm changed).
No prior work: exact continuation, machine-checked equivalence, relative
deflation, impossibility framing.

## 6. Loose ends / future work

- In-Lean discharge of the Galois certificate (needs Dedekind's theorem in
  Mathlib) — currently: Lean bridge + exact external certificate.
- FMM path fused into the GPU engine (CPU demo done); scaled-NS acceleration
  for TRACK (~3× fewer iterations); Kronecker-structured multi-GPU sharding.
- Lower bound for the dense regime (is Ω(N^ω) forced for exact fresh-every-step
  maintenance when r_step = Θ(N)?).
