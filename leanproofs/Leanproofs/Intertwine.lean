/-
Copyright (c) 2026 . All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: RootFree-Shampoo development (Pillar L1)
-/
/-
The intertwining theorem for the matrix functional calculus, and the Gram
pushdown identity giving Shampoo's exact eigendecomposition-free cold phase.

Main results:
* `Matrix.aeval_intertwine`  : A*Y = Y*B → p(A)*Y = Y*p(B) for polynomials p.
* `Matrix.cfc_intertwine`    : A*Y = Y*B (A,B Hermitian) → f(A)*Y = Y*f(B) for ALL f.
* `Matrix.cfc_gram_pushdown` : f(ε•1 + Y*Yᴴ)*Y = Y*f(ε•1 + Yᴴ*Y).
* `Matrix.cfc_gram_pushdown_explicit` :
      f(ε•1 + Y*Yᴴ) = f(ε)•1 + Y * g(ε•1 + Yᴴ*Y) * Yᴴ,  g x = (f x − f ε)/(x − ε),
  i.e. any spectral function (in particular the inverse 4th root) of the m×m
  accumulator is EXACTLY computable from the k×k Gram spectrum.
-/
import Mathlib

open Matrix Polynomial

variable {m k : Type*} [Fintype m] [DecidableEq m] [Fintype k] [DecidableEq k]

namespace Shampoo

/-- The spectrum of a real Hermitian matrix is finite. -/
lemma spectrum_finite (A : Matrix m m ℝ) (hA : A.IsHermitian) :
    (spectrum ℝ A).Finite := by
  rw [hA.spectrum_real_eq_range_eigenvalues]
  exact Set.finite_range _

/-- Powers intertwine: from A*Y = Y*B conclude Aⁿ*Y = Y*Bⁿ. -/
lemma pow_intertwine {A : Matrix m m ℝ} {B : Matrix k k ℝ} {Y : Matrix m k ℝ}
    (h : A * Y = Y * B) : ∀ n : ℕ, A ^ n * Y = Y * B ^ n := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
    rw [pow_succ, pow_succ, Matrix.mul_assoc, h, ← Matrix.mul_assoc, ih,
      Matrix.mul_assoc]

/-- Polynomials intertwine. -/
lemma aeval_intertwine {A : Matrix m m ℝ} {B : Matrix k k ℝ} {Y : Matrix m k ℝ}
    (h : A * Y = Y * B) (p : ℝ[X]) : aeval A p * Y = Y * aeval B p := by
  induction p using Polynomial.induction_on' with
  | add p q hp hq => rw [map_add, map_add, Matrix.add_mul, Matrix.mul_add, hp, hq]
  | monomial n a =>
    rw [aeval_monomial, aeval_monomial, Algebra.algebraMap_eq_smul_one,
      Algebra.algebraMap_eq_smul_one, smul_mul_assoc, smul_mul_assoc, one_mul,
      one_mul, Matrix.smul_mul, Matrix.mul_smul, pow_intertwine h]

/-- The functional-calculus interpolation bridge: `cfc f` equals the evaluation
of the Lagrange interpolant of f on any finite superset of the spectrum. -/
lemma cfc_eq_aeval_interpolate (A : Matrix m m ℝ) (hA : A.IsHermitian) (f : ℝ → ℝ)
    (s : Finset ℝ) (hs : spectrum ℝ A ⊆ ↑s) :
    cfc f A = aeval A (Lagrange.interpolate s id f) := by
  rw [← cfc_polynomial (R := ℝ) (Lagrange.interpolate s id f) A hA.isSelfAdjoint]
  exact cfc_congr fun x hx =>
    (Lagrange.eval_interpolate_at_node f (Set.injOn_id _) (hs hx)).symm

/-- **The intertwining theorem.** If Hermitian A, B satisfy A*Y = Y*B, then
f(A)*Y = Y*f(B) for every function f : ℝ → ℝ. -/
theorem cfc_intertwine {A : Matrix m m ℝ} {B : Matrix k k ℝ} {Y : Matrix m k ℝ}
    (hA : A.IsHermitian) (hB : B.IsHermitian) (h : A * Y = Y * B) (f : ℝ → ℝ) :
    cfc f A * Y = Y * cfc f B := by
  classical
  obtain ⟨sA, hsA⟩ := (spectrum_finite A hA).exists_finset_coe
  obtain ⟨sB, hsB⟩ := (spectrum_finite B hB).exists_finset_coe
  rw [cfc_eq_aeval_interpolate A hA f (sA ∪ sB)
      (by rw [Finset.coe_union, ← hsA]; exact Set.subset_union_left),
    cfc_eq_aeval_interpolate B hB f (sA ∪ sB)
      (by rw [Finset.coe_union, ← hsB]; exact Set.subset_union_right)]
  exact aeval_intertwine h _

/-- The algebraic seed of the pushdown: (ε•1 + Y*Yᴴ)*Y = Y*(ε•1 + Yᴴ*Y). -/
lemma gram_seed (Y : Matrix m k ℝ) (ε : ℝ) :
    (ε • 1 + Y * Yᴴ) * Y = Y * (ε • 1 + Yᴴ * Y) := by
  rw [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.one_mul,
    Matrix.mul_smul, Matrix.mul_one, Matrix.mul_assoc]

lemma isHermitian_gram_left (Y : Matrix m k ℝ) (ε : ℝ) :
    (ε • 1 + Y * Yᴴ).IsHermitian :=
  ((Matrix.isHermitian_one).smul (star_trivial ε)).add
    (isHermitian_mul_conjTranspose_self Y)

lemma isHermitian_gram_right (Y : Matrix m k ℝ) (ε : ℝ) :
    (ε • 1 + Yᴴ * Y).IsHermitian :=
  ((Matrix.isHermitian_one).smul (star_trivial ε)).add
    (isHermitian_conjTranspose_mul_self Y)

/-- **Gram pushdown, intertwining form.** -/
theorem cfc_gram_pushdown (Y : Matrix m k ℝ) (ε : ℝ) (f : ℝ → ℝ) :
    cfc f (ε • 1 + Y * Yᴴ) * Y = Y * cfc f (ε • 1 + Yᴴ * Y) :=
  cfc_intertwine (isHermitian_gram_left Y ε) (isHermitian_gram_right Y ε)
    (gram_seed Y ε) f

/-- **Gram pushdown, explicit form**: with g x := (f x − f ε)/(x − ε) (the value
at x = ε is junk and irrelevant),
  f(ε•1 + Y*Yᴴ) = f(ε)•1 + Y * g(ε•1 + Yᴴ*Y) * Yᴴ. -/
theorem cfc_gram_pushdown_explicit (Y : Matrix m k ℝ) (ε : ℝ) (f : ℝ → ℝ) :
    cfc f (ε • 1 + Y * Yᴴ) =
      f ε • 1 + Y * cfc (fun x => if x = ε then 0 else (f x - f ε) / (x - ε))
        (ε • 1 + Yᴴ * Y) * Yᴴ := by
  classical
  set A := ε • 1 + Y * Yᴴ with hAdef
  set g : ℝ → ℝ := fun x => if x = ε then 0 else (f x - f ε) / (x - ε) with hgdef
  have hA : A.IsHermitian := isHermitian_gram_left Y ε
  have hcont : ∀ (h : ℝ → ℝ), ContinuousOn h (spectrum ℝ A) :=
    fun h => (spectrum_finite A hA).continuousOn h
  have key : ∀ x : ℝ, f x = f ε + (x - ε) * g x := by
    intro x
    by_cases hx : x = ε
    · simp [hgdef, hx]
    · have h0 : x - ε ≠ 0 := sub_ne_zero.mpr hx
      rw [hgdef]
      simp only [if_neg hx]
      field_simp
      ring
  have h1 : cfc f A = cfc (fun x => f ε + (x - ε) * g x) A :=
    cfc_congr fun x _ => key x
  have h2 : cfc (fun x => f ε + (x - ε) * g x) A
      = cfc (fun _ => f ε) A + cfc (fun x => (x - ε) * g x) A :=
    cfc_add (a := A) (f := fun _ => f ε) (g := fun x => (x - ε) * g x)
      (hf := hcont _) (hg := hcont _)
  have h3 : cfc (fun x => (x - ε) * g x) A
      = cfc (fun x => x - ε) A * cfc g A :=
    cfc_mul (a := A) (f := fun x => x - ε) (g := g) (hf := hcont _) (hg := hcont _)
  have h4 : cfc (fun x : ℝ => x - ε) A = Y * Yᴴ := by
    have hsplit : cfc (fun x : ℝ => x - ε) A = cfc id A - cfc (fun _ : ℝ => ε) A :=
      cfc_sub (f := id) (g := fun _ : ℝ => ε) (a := A) (hf := hcont _) (hg := hcont _)
    rw [hsplit, cfc_id ℝ A hA.isSelfAdjoint, cfc_const ε A hA.isSelfAdjoint,
      Algebra.algebraMap_eq_smul_one, hAdef, add_sub_cancel_left]
  have h5 : cfc (fun _ : ℝ => f ε) A = f ε • 1 := by
    rw [cfc_const (f ε) A hA.isSelfAdjoint, Algebra.algebraMap_eq_smul_one]
  have h6 : Y * Yᴴ * cfc g A = Y * cfc g (ε • 1 + Yᴴ * Y) * Yᴴ := by
    have push := cfc_gram_pushdown Yᴴ ε g
    rw [conjTranspose_conjTranspose] at push
    -- push : cfc g (ε•1 + Yᴴ*Y) * Yᴴ = Yᴴ * cfc g (ε•1 + Y*Yᴴ)
    rw [Matrix.mul_assoc, hAdef, ← push, ← Matrix.mul_assoc]
  rw [h1, h2, h3, h4, h5, h6]

end Shampoo
