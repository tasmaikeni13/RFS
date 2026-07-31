/-
Copyright (c) 2026 RootFree-Shampoo development. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: RootFree-Shampoo development (Pillar L2)
-/
/-
The exact algebra of the spectral continuation engine.

* `secular_charpoly_eval` — χ_{D+ρzzᵀ}(μ) = ∏ᵢ(μ−dᵢ) − ρ·Σᵢ zᵢ²·∏_{j≠i}(μ−dⱼ):
  the new eigenvalues are EXACTLY the secular roots (algebraic characterization,
  not an approximation).
* `loewner_eigenvector` — q := (zᵢ/(dᵢ − μ))ᵢ is an exact eigenvector.
* `gu_eisenstat_identity` — ρ·zᵢ²·∏_{j≠i}(dᵢ − dⱼ) = −∏ⱼ(dᵢ − μⱼ): the product
  identity behind the self-correcting reconstruction of z from computed roots.
-/
import Mathlib

open Matrix Polynomial Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

namespace Shampoo

/-- Rank-one determinant: det(1 − ρ • vecMulVec w z) = 1 − ρ·(z ⬝ᵥ w). -/
lemma det_one_sub_smul_vecMulVec (ρ : ℝ) (w z : n → ℝ) :
    (1 - ρ • vecMulVec w z).det = 1 - ρ * (z ⬝ᵥ w) := by
  have h : (1 : Matrix n n ℝ) - ρ • vecMulVec w z
      = 1 + replicateCol (Fin 1) (-ρ • w) * replicateRow (Fin 1) z := by
    ext i j
    simp [Matrix.mul_apply, vecMulVec_apply, sub_eq_add_neg]
    ring
  rw [h, Matrix.det_one_add_mul_comm, Matrix.det_fin_one, Matrix.add_apply,
    Matrix.one_apply_eq]
  have hmul : (replicateRow (Fin 1) z * replicateCol (Fin 1) (-ρ • w)) 0 0
      = ∑ j, z j * ((-ρ • w) j) := by
    rw [Matrix.mul_apply]
    exact Finset.sum_congr rfl fun j _ => rfl
  rw [hmul]
  have hstep : ∑ j, z j * ((-ρ • w) j) + ρ * (z ⬝ᵥ w) = 0 := by
    rw [dotProduct, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_eq_zero fun j _ => ?_
    rw [Pi.smul_apply, smul_eq_mul]
    ring
  linarith [hstep]

/-- **The secular characterization.** For every μ,
  χ_{D+ρzzᵀ}(μ) = ∏ᵢ(μ−dᵢ) − ρ Σᵢ zᵢ² ∏_{j≠i}(μ−dⱼ). -/
theorem secular_charpoly_eval (d z : n → ℝ) (ρ : ℝ) (μ : ℝ) :
    ((diagonal d + ρ • vecMulVec z z).charpoly).eval μ
      = ∏ i, (μ - d i) - ρ * ∑ i, z i ^ 2 * ∏ j ∈ univ.erase i, (μ - d j) := by
  classical
  set M := diagonal d + ρ • vecMulVec z z with hM
  set RHSp : Polynomial ℝ :=
    ∏ i, (X - C (d i)) - C ρ * ∑ i, C (z i) ^ 2 * ∏ j ∈ univ.erase i, (X - C (d j))
    with hRHSp
  have hRHSeval : ∀ x : ℝ, RHSp.eval x
      = ∏ i, (x - d i) - ρ * ∑ i, z i ^ 2 * ∏ j ∈ univ.erase i, (x - d j) := by
    intro x
    simp [hRHSp, eval_prod, eval_finsetSum]
  suffices hpoly : M.charpoly = RHSp by rw [hpoly, hRHSeval]
  apply Polynomial.eq_of_infinite_eval_eq
  have hsub : {x : ℝ | ∀ i, x ≠ d i} ⊆ {x | M.charpoly.eval x = RHSp.eval x} := by
    intro μ hμ
    have hne : ∀ i, μ - d i ≠ 0 := fun i => sub_ne_zero.mpr (hμ i)
    have hfact : Matrix.scalar n μ - M
        = diagonal (fun i => μ - d i)
            * (1 - ρ • vecMulVec (fun i => z i / (μ - d i)) z) := by
      ext i j
      rw [Matrix.diagonal_mul]
      rcases eq_or_ne i j with rfl | hij
      · simp only [hM, Matrix.scalar_apply, Matrix.sub_apply, Matrix.add_apply,
          Matrix.smul_apply, vecMulVec_apply, diagonal_apply_eq,
          Matrix.one_apply_eq, smul_eq_mul]
        have hnei := hne i
        field_simp
        try ring
      · simp only [hM, Matrix.scalar_apply, Matrix.sub_apply, Matrix.add_apply,
          Matrix.smul_apply, vecMulVec_apply, diagonal_apply_ne _ hij,
          Matrix.one_apply_ne hij, smul_eq_mul]
        have hnei := hne i
        field_simp
        try ring
    rw [Set.mem_setOf_eq, Matrix.eval_charpoly, hfact, det_mul, det_diagonal,
      det_one_sub_smul_vecMulVec, hRHSeval]
    have hdot : (z ⬝ᵥ fun i => z i / (μ - d i)) = ∑ i, z i ^ 2 / (μ - d i) := by
      simp only [dotProduct]
      exact Finset.sum_congr rfl fun i _ => by rw [pow_two]; ring
    have per : ∀ i, (∏ j, (μ - d j)) * (z i ^ 2 / (μ - d i))
        = z i ^ 2 * ∏ j ∈ univ.erase i, (μ - d j) := by
      intro i
      rw [← Finset.mul_prod_erase univ (fun j => μ - d j) (Finset.mem_univ i)]
      have hnei := hne i
      field_simp
    rw [hdot, mul_sub, mul_one, mul_left_comm, Finset.mul_sum]
    congr 1
    exact congrArg (fun t => ρ * t) (Finset.sum_congr rfl fun i _ => per i)
  refine Set.Infinite.mono hsub ?_
  have hset : {x : ℝ | ∀ i, x ≠ d i} = (Set.range d)ᶜ := by
    ext x
    simp [not_exists, eq_comm]
  rw [hset]
  exact (Set.finite_range d).infinite_compl

/-- **Löwner eigenvector formula.** If μ avoids the poles and satisfies the
secular equation 1 + ρ·Σᵢ zᵢ²/(dᵢ − μ) = 0, then q := (zᵢ/(dᵢ − μ))ᵢ satisfies
(D + ρzzᵀ)q = μ·q — an exact eigenpair. -/
theorem loewner_eigenvector (d z : n → ℝ) (ρ μ : ℝ)
    (hpole : ∀ i, d i ≠ μ)
    (hsec : 1 + ρ * ∑ i, z i ^ 2 / (d i - μ) = 0) :
    (diagonal d + ρ • vecMulVec z z).mulVec (fun i => z i / (d i - μ))
      = μ • (fun i => z i / (d i - μ)) := by
  classical
  have hne : ∀ i, d i - μ ≠ 0 := fun i => sub_ne_zero.mpr (hpole i)
  have hsum : ρ * ∑ i, z i ^ 2 / (d i - μ) = -1 := by linarith
  funext i
  have hdvec : (diagonal d).mulVec (fun j => z j / (d j - μ)) i
      = d i * (z i / (d i - μ)) := by
    rw [Matrix.mulVec_diagonal]
  have hrank1 : ((ρ • vecMulVec z z).mulVec fun j => z j / (d j - μ)) i
      = z i * (ρ * ∑ j, z j ^ 2 / (d j - μ)) := by
    calc ((ρ • vecMulVec z z).mulVec fun j => z j / (d j - μ)) i
        = ∑ x, ρ * (z i * z x) * (z x / (d x - μ)) := by
          simp [Matrix.mulVec, dotProduct, vecMulVec_apply]
      _ = ∑ x, z i * (ρ * (z x ^ 2 / (d x - μ))) :=
          Finset.sum_congr rfl fun x _ => by rw [pow_two]; ring
      _ = z i * (ρ * ∑ j, z j ^ 2 / (d j - μ)) := by
          rw [Finset.mul_sum, Finset.mul_sum]
  rw [Matrix.add_mulVec, Pi.add_apply, hdvec, hrank1, hsum, Pi.smul_apply,
    smul_eq_mul]
  have hnei := hne i
  field_simp
  ring

/-- **Gu–Eisenstat reconstruction identity.** If χ_{D+ρzzᵀ} = ∏ⱼ(X − μⱼ), then
  ρ·zᵢ²·∏_{j≠i}(dᵢ − dⱼ) = −∏ⱼ(dᵢ − μⱼ) for every i. -/
theorem gu_eisenstat_identity (d z : n → ℝ) (ρ : ℝ) (μ : n → ℝ)
    (hsplit : (diagonal d + ρ • vecMulVec z z).charpoly = ∏ j, (X - C (μ j)))
    (i : n) :
    ρ * z i ^ 2 * ∏ j ∈ univ.erase i, (d i - d j) = -∏ j, (d i - μ j) := by
  have h := secular_charpoly_eval d z ρ (d i)
  rw [hsplit] at h
  simp only [eval_prod, eval_sub, eval_X, eval_C] at h
  have hzero : ∏ j, (d i - d j) = 0 :=
    Finset.prod_eq_zero (Finset.mem_univ i) (by simp)
  have hsum : ∑ k, z k ^ 2 * ∏ j ∈ univ.erase k, (d i - d j)
      = z i ^ 2 * ∏ j ∈ univ.erase i, (d i - d j) := by
    rw [Finset.sum_eq_single i]
    · intro k _ hki
      have hz : ∏ j ∈ univ.erase k, (d i - d j) = 0 :=
        Finset.prod_eq_zero (Finset.mem_erase.mpr ⟨Ne.symm hki, Finset.mem_univ i⟩)
          (by simp)
      rw [hz, mul_zero]
    · intro hmem
      exact absurd (Finset.mem_univ i) hmem
  rw [hzero, hsum] at h
  linear_combination h

end Shampoo
