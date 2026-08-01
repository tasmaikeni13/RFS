/-
Copyright (c) 2026 RootFree-Shampoo development. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: RootFree-Shampoo development (capstone)
-/
/-
The capstone lemma: the spectral function used throughout the development,
f(x) = x^(-1/4) (Real.rpow), applied through the functional calculus to a
positive definite accumulator, IS the inverse quartic root in the defining
sense: (cfc f L)⁴ · L = 1. Together with `trajectory_equivalence` this closes
the loop: the certificate optimizer applies exactly the operator Shampoo means.
-/
import Mathlib
import Leanproofs.Equivalence

open Matrix Polynomial

namespace Shampoo

noncomputable def invQuart : ℝ → ℝ := fun x => x ^ (-(1 : ℝ) / 4)

/-- On a positive definite matrix, `cfc invQuart` satisfies the defining
equation of the inverse quartic root. -/
theorem invQuart_defining {n : Type*} [Fintype n] [DecidableEq n]
    {L : Matrix n n ℝ} (hL : L.PosDef) :
    (cfc invQuart L) ^ 4 * L = 1 := by
  have hherm : L.IsHermitian := hL.isHermitian
  have hspec : ∀ x ∈ spectrum ℝ L, 0 < x := by
    intro x hx
    rw [hherm.spectrum_real_eq_range_eigenvalues] at hx
    obtain ⟨i, rfl⟩ := hx
    exact hL.eigenvalues_pos i
  have hcont : ∀ (h : ℝ → ℝ), ContinuousOn h (spectrum ℝ L) :=
    fun h => (spectrum_finite L hherm).continuousOn h
  have hpow : (cfc invQuart L) ^ 4 = cfc (fun x => invQuart x ^ 4) L :=
    (cfc_pow invQuart 4 L (hcont _) hherm.isSelfAdjoint).symm
  have hmul : cfc (fun x => invQuart x ^ 4) L * L
      = cfc (fun x => invQuart x ^ 4 * id x) L := by
    have h := cfc_mul (fun x => invQuart x ^ 4) id L (hcont _) (hcont _)
    rw [cfc_id ℝ L hherm.isSelfAdjoint] at h
    exact h.symm
  have hone : cfc (fun x => invQuart x ^ 4 * id x) L = 1 := by
    have hcg : cfc (fun x => invQuart x ^ 4 * id x) L
        = cfc (fun _ : ℝ => (1 : ℝ)) L :=
      cfc_congr fun x hx => by
        have hxpos := hspec x hx
        unfold invQuart
        simp only [id_eq]
        rw [← Real.rpow_natCast (x ^ (-(1 : ℝ) / 4)) 4,
          ← Real.rpow_mul hxpos.le]
        norm_num
        have hxne : x ≠ 0 := ne_of_gt hxpos
        rw [Real.rpow_neg_one]
        exact inv_mul_cancel₀ hxne
    rw [hcg, cfc_const 1 L hherm.isSelfAdjoint, map_one]
  rw [hpow, hmul, hone]

end Shampoo
