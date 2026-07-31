/-
Copyright (c) 2026 . All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: RootFree-Shampoo development (Pillar L4)
-/
/-
Impossibility theorems: no finite exact-arithmetic algorithm computes the
Shampoo inverse quartic root.

* `no_rational_inverse_quartic_root` — UNCONDITIONAL: no rational function
  equals a^{-1/4} on (0,∞). Any entrywise-rational algorithm for L ↦ L^{-1/4}
  restricts on the ray {a•1 : a > 0} to such a function: impossible already
  for 1×1 matrices.

* `radical_entries_give_radical_trace` + `no_radical_root_of_gal` — the radical
  case: if every entry of X (= L^{-1/4}) were expressible by radicals, then
  t := tr(L·X²) = tr(L^{1/2}) = Σᵢ √λᵢ would lie in `solvableByRad ℚ ℝ`, so by
  Mathlib's Abel–Ruffini bridge its minimal polynomial would have solvable
  Galois group. The exact computational certificate (w4_certificate.json:
  deg minpoly(t) = 32 via irreducibility of the sign-product polynomial Q, and
  Gal(charpoly L) ≅ S₅) forces that Galois group to surject onto S₅ —
  unsolvable. Hence some entry of L^{-1/4} is not radical-expressible.

* `not_solvable_of_surjective_perm5` — the group-theoretic step: any group
  surjecting onto S₅ is unsolvable.
-/
import Mathlib

open Matrix Polynomial

namespace Shampoo

/-- **Rational impossibility (scalar core).** No rational function P/Q can
satisfy (P(a)/Q(a))⁴ · a = 1 even on just an infinite set of positive inputs.
Consequently no algorithm using only field operations (+,−,×,÷) on the input
entries — including algorithms that branch, since some branch's input set is
infinite — can output L^{-1/4}: restrict to L = a•1 and read one entry. -/
theorem no_rational_inverse_quartic_root
    (P Q : Polynomial ℝ) (S : Set ℝ) (hS : S.Infinite) (hpos : ∀ a ∈ S, 0 < a)
    (h : ∀ a ∈ S, Q.eval a ≠ 0 ∧ (P.eval a) ^ 4 * a = (Q.eval a) ^ 4) :
    False := by
  obtain ⟨a₀, ha₀⟩ := hS.nonempty
  have hPne : P ≠ 0 := by
    rintro rfl
    have h1 := (h a₀ ha₀)
    simp at h1
    exact h1.1 (pow_eq_zero_iff (n := 4) (by norm_num) |>.mp h1.2.symm)
  have hQne : Q ≠ 0 := by
    rintro rfl
    have h1 := h a₀ ha₀
    simp at h1
  -- polynomial identity X * P⁴ = Q⁴ from agreement on the infinite set S
  have hpoly : Polynomial.X * P ^ 4 = Q ^ 4 := by
    apply Polynomial.eq_of_infinite_eval_eq
    apply Set.Infinite.mono (s := S)
    · intro a ha
      have := (h a ha).2
      simp only [Set.mem_setOf_eq, eval_mul, eval_X, eval_pow]
      linarith [this]
    · exact hS
  -- degree contradiction: 1 + 4·deg P = 4·deg Q
  have hdeg := congrArg natDegree hpoly
  rw [natDegree_mul (X_ne_zero) (pow_ne_zero 4 hPne), natDegree_X,
    natDegree_pow, natDegree_pow] at hdeg
  omega

/-- Any group surjecting onto S₅ is unsolvable. -/
theorem not_solvable_of_surjective_perm5 {G : Type*} [Group G]
    (f : G →* Equiv.Perm (Fin 5)) (hf : Function.Surjective f) :
    ¬ IsSolvable G := fun h =>
  Equiv.Perm.fin_5_not_solvable (@solvable_of_surjective _ _ _ _ f hf h)

/-- If L has rational entries and every entry of X lies in the radical closure
`solvableByRad ℚ ℝ`, then tr(L·X²) lies there too (it is a polynomial in the
entries). With X = L^{-1/4} this element is tr(L^{1/2}) = Σ √λᵢ. -/
theorem radical_entries_give_radical_trace
    {n : Type*} [Fintype n] [DecidableEq n]
    (L X : Matrix n n ℝ)
    (hL : ∀ i j, ∃ q : ℚ, L i j = (q : ℝ))
    (hX : ∀ i j, X i j ∈ solvableByRad ℚ ℝ) :
    (L * X ^ 2).trace ∈ solvableByRad ℚ ℝ := by
  have hXmem : ∀ i j, (X ^ 2) i j ∈ solvableByRad ℚ ℝ := by
    intro i j
    rw [pow_two, Matrix.mul_apply]
    exact sum_mem fun k _ => mul_mem (hX i k) (hX k j)
  have hLmem : ∀ i j, L i j ∈ solvableByRad ℚ ℝ := by
    intro i j
    obtain ⟨q, hq⟩ := hL i j
    rw [hq]
    exact SubfieldClass.ratCast_mem _ q
  rw [Matrix.trace]
  refine sum_mem fun i _ => ?_
  rw [Matrix.diag_apply, Matrix.mul_apply]
  exact sum_mem fun k _ => mul_mem (hLmem i k) (hXmem k i)

/-- **Radical impossibility, bridged form.** If the Galois group of the minimal
polynomial of tr(L·X²) is not solvable (discharged for the concrete witness by
the exact certificate: Q irreducible of degree 32 + Gal(charpoly L) = S₅ forces
a surjection onto S₅), then not all entries of X are radical-expressible. -/
theorem no_radical_root_of_gal
    {n : Type*} [Fintype n] [DecidableEq n]
    (L X : Matrix n n ℝ)
    (hL : ∀ i j, ∃ q : ℚ, L i j = (q : ℝ))
    (hGal : ¬ IsSolvable ((minpoly ℚ ((L * X ^ 2).trace)).Gal)) :
    ¬ (∀ i j, X i j ∈ solvableByRad ℚ ℝ) := by
  intro hX
  exact hGal (isSolvable_gal_minpoly
    (radical_entries_give_radical_trace L X hL hX))

end Shampoo
