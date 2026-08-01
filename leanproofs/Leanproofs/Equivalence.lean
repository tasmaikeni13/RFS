/-
Copyright (c) 2026 RootFree-Shampoo development. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: RootFree-Shampoo development (Pillar L3)
-/
/-
The trajectory-equivalence theorem.

`trajectory_equivalence`: running Shampoo with preconditioners applied through
ANY exact spectral certificate (orthogonal U, eigenvalues d, L = U·diag(d)·Uᴴ —
e.g. those maintained by the secular/Löwner continuation of Pillar L2) produces
EXACTLY the same weight iterates, at every step, as the ideal algorithm
applying the matrix function `cfc f` (f = x^{-1/4} for Shampoo) — for every
spectral function f. Representation independence is total: the trajectory
cannot see how the inverse root was computed.

`mathlibRefresh_isFor` instantiates the refresh oracle with Mathlib's spectral
theorem, so the main theorem's hypotheses are satisfiable (non-vacuity).
-/
import Mathlib
import Leanproofs.Intertwine

open Matrix Polynomial

variable {m k : Type*} [Fintype m] [DecidableEq m] [Fintype k] [DecidableEq k]

namespace Shampoo

/-- A spectral certificate: an orthogonal matrix and an eigenvalue vector. -/
structure SpectralPair (n : Type*) [Fintype n] [DecidableEq n] where
  U : Matrix n n ℝ
  d : n → ℝ

/-- The certificate is exact for A. -/
def SpectralPair.IsFor {n : Type*} [Fintype n] [DecidableEq n]
    (P : SpectralPair n) (A : Matrix n n ℝ) : Prop :=
  P.U * P.Uᴴ = 1 ∧ A = P.U * diagonal P.d * P.Uᴴ

/-- x•1 as a diagonal matrix. -/
lemma smul_one_eq_diagonal' {n : Type*} [Fintype n] [DecidableEq n] (x : ℝ) :
    (x • (1 : Matrix n n ℝ)) = diagonal (fun _ => x) := by
  ext i j
  rcases eq_or_ne i j with rfl | hij
  · simp
  · simp [Matrix.one_apply_ne hij, diagonal_apply_ne _ hij]

namespace SpectralPair

variable {n : Type*} [Fintype n] [DecidableEq n] {P : SpectralPair n}
variable {A : Matrix n n ℝ}

lemma left_inv (h : P.IsFor A) : P.Uᴴ * P.U = 1 :=
  Matrix.mul_eq_one_comm.mp h.1

/-- The certified matrix is Hermitian. -/
lemma isHermitian (h : P.IsFor A) : A.IsHermitian := by
  rw [h.2]
  unfold Matrix.IsHermitian
  rw [conjTranspose_mul, conjTranspose_mul, conjTranspose_conjTranspose,
    diagonal_conjTranspose]
  simp [Matrix.mul_assoc, Pi.star_def]

lemma conj_pow (h : P.IsFor A) (D : Matrix n n ℝ) (j : ℕ) :
    (P.U * D * P.Uᴴ) ^ j = P.U * D ^ j * P.Uᴴ := by
  induction j with
  | zero => simpa using h.1.symm
  | succ j ih =>
    rw [pow_succ, ih, pow_succ]
    calc P.U * D ^ j * P.Uᴴ * (P.U * D * P.Uᴴ)
        = P.U * D ^ j * ((P.Uᴴ * P.U) * (D * P.Uᴴ)) := by
          simp only [Matrix.mul_assoc]
      _ = P.U * (D ^ j * D) * P.Uᴴ := by
          rw [left_inv h, Matrix.one_mul]
          simp only [Matrix.mul_assoc]

lemma conj_aeval (h : P.IsFor A) (D : Matrix n n ℝ) (p : ℝ[X]) :
    aeval (P.U * D * P.Uᴴ) p = P.U * aeval D p * P.Uᴴ := by
  induction p using Polynomial.induction_on' with
  | add p q hp hq =>
    rw [map_add, map_add, hp, hq, Matrix.mul_add, Matrix.add_mul]
  | monomial j a =>
    rw [aeval_monomial, aeval_monomial, Algebra.algebraMap_eq_smul_one,
      smul_mul_assoc, smul_mul_assoc, one_mul, one_mul, conj_pow h,
      Matrix.mul_smul, Matrix.smul_mul]

lemma aeval_diagonal' (d : n → ℝ) (p : ℝ[X]) :
    aeval (diagonal d) p = diagonal (fun i => p.eval (d i)) := by
  induction p using Polynomial.induction_on' with
  | add p q hp hq =>
    rw [map_add, hp, hq, diagonal_add]
    congr 1
    funext i
    rw [eval_add]
  | monomial j a =>
    rw [aeval_monomial, Algebra.algebraMap_eq_smul_one, smul_mul_assoc, one_mul,
      diagonal_pow, ← diagonal_smul]
    congr 1
    funext i
    rw [eval_monomial]
    simp [mul_comm]

/-- The spectrum is contained in the certified eigenvalues. -/
lemma spectrum_subset (h : P.IsFor A) : spectrum ℝ A ⊆ Set.range P.d := by
  intro x hx
  by_contra hxr
  rw [Set.mem_range, not_exists] at hxr
  refine spectrum.notMem_iff.mpr ?_ hx
  have key : algebraMap ℝ (Matrix n n ℝ) x - A
      = P.U * diagonal (fun i => x - P.d i) * P.Uᴴ := by
    rw [h.2, Algebra.algebraMap_eq_smul_one]
    have hx1 : (x • 1 : Matrix n n ℝ) = P.U * (x • (1 : Matrix n n ℝ)) * P.Uᴴ := by
      rw [Matrix.mul_smul, Matrix.mul_one, Matrix.smul_mul, h.1]
    rw [hx1, ← Matrix.sub_mul, ← Matrix.mul_sub, smul_one_eq_diagonal',
      diagonal_sub]
  rw [key]
  have hdetU : IsUnit P.U.det := by
    have h1 : P.U.det * P.Uᴴ.det = 1 := by
      rw [← det_mul, h.1, det_one]
    exact IsUnit.of_mul_eq_one _ h1
  have hdetD : IsUnit (diagonal (fun i => x - P.d i)).det := by
    rw [det_diagonal]
    rw [isUnit_iff_ne_zero]
    refine Finset.prod_ne_zero_iff.mpr fun i _ => sub_ne_zero.mpr ?_
    exact fun hxd => hxr i hxd.symm
  have hdetUH : IsUnit P.Uᴴ.det := by
    have h1 : P.Uᴴ.det * P.U.det = 1 := by
      rw [← det_mul, left_inv h, det_one]
    exact IsUnit.of_mul_eq_one _ h1
  refine (Matrix.isUnit_iff_isUnit_det _).mpr ?_
  rw [det_mul, det_mul]
  exact (hdetU.mul hdetD).mul hdetUH

/-- **Representation independence of the functional calculus.** Any exact
spectral certificate evaluates any spectral function of A exactly. -/
theorem cfc_eq (h : P.IsFor A) (f : ℝ → ℝ) :
    cfc f A = P.U * diagonal (fun i => f (P.d i)) * P.Uᴴ := by
  classical
  obtain ⟨sd, hsd⟩ := (Set.finite_range P.d).exists_finset_coe
  rw [cfc_eq_aeval_interpolate A (isHermitian h) f sd
      (by rw [hsd]; exact spectrum_subset h)]
  conv_lhs => rw [h.2]
  rw [conj_aeval h, aeval_diagonal']
  have hfun : (fun i => (Lagrange.interpolate sd id f).eval (P.d i))
      = fun i => f (P.d i) := by
    funext i
    have hmem : P.d i ∈ sd := by
      have hr : P.d i ∈ Set.range P.d := Set.mem_range_self i
      rwa [← hsd, Finset.mem_coe] at hr
    exact Lagrange.eval_interpolate_at_node f (Set.injOn_id _) hmem
  rw [hfun]

end SpectralPair

/-! ## The two optimizers and the equivalence theorem -/

/-- Optimizer state: weights and the two accumulators. -/
structure OptState (m k : Type*) [Fintype m] [DecidableEq m] [Fintype k]
    [DecidableEq k] where
  W : Matrix m k ℝ
  L : Matrix m m ℝ
  R : Matrix k k ℝ

variable (η : ℝ) (f : ℝ → ℝ) (grad : Matrix m k ℝ → Matrix m k ℝ)

/-- One step of ideal Shampoo: accumulate, then precondition through the exact
matrix function `cfc f` (Shampoo: f = x^{-1/4}). -/
noncomputable def idealStep (s : OptState m k) : OptState m k :=
  ⟨s.W - η • (cfc f (s.L + grad s.W * (grad s.W)ᴴ) * grad s.W
      * cfc f (s.R + (grad s.W)ᴴ * grad s.W)),
    s.L + grad s.W * (grad s.W)ᴴ, s.R + (grad s.W)ᴴ * grad s.W⟩

/-- One step of spectral-certificate Shampoo: accumulate, refresh the
certificates by ANY exact policy, precondition through the certificates. -/
noncomputable def spectralStep (refL : Matrix m m ℝ → SpectralPair m)
    (refR : Matrix k k ℝ → SpectralPair k) (s : OptState m k) : OptState m k :=
  ⟨s.W - η • ((refL (s.L + grad s.W * (grad s.W)ᴴ)).U
      * diagonal (fun i => f ((refL (s.L + grad s.W * (grad s.W)ᴴ)).d i))
      * (refL (s.L + grad s.W * (grad s.W)ᴴ)).Uᴴ * grad s.W
      * ((refR (s.R + (grad s.W)ᴴ * grad s.W)).U
        * diagonal (fun i => f ((refR (s.R + (grad s.W)ᴴ * grad s.W)).d i))
        * (refR (s.R + (grad s.W)ᴴ * grad s.W)).Uᴴ)),
    s.L + grad s.W * (grad s.W)ᴴ, s.R + (grad s.W)ᴴ * grad s.W⟩

/-- Accumulators stay Hermitian along the ideal trajectory. -/
lemma idealStep_hermitian (s : OptState m k) (hL : s.L.IsHermitian)
    (hR : s.R.IsHermitian) :
    ((idealStep η f grad s).L).IsHermitian ∧
      ((idealStep η f grad s).R).IsHermitian :=
  ⟨hL.add (isHermitian_mul_conjTranspose_self _),
    hR.add (isHermitian_conjTranspose_mul_self _)⟩

/-- **Trajectory equivalence.** With exact refresh oracles, the spectral
optimizer's trajectory coincides with ideal Shampoo's — every weight matrix,
every accumulator, every step, for every spectral function f. -/
theorem trajectory_equivalence
    (refL : Matrix m m ℝ → SpectralPair m) (refR : Matrix k k ℝ → SpectralPair k)
    (hrefL : ∀ M : Matrix m m ℝ, M.IsHermitian → (refL M).IsFor M)
    (hrefR : ∀ M : Matrix k k ℝ, M.IsHermitian → (refR M).IsFor M)
    (s0 : OptState m k) (hL0 : s0.L.IsHermitian) (hR0 : s0.R.IsHermitian) :
    ∀ t : ℕ, (spectralStep η f grad refL refR)^[t] s0 = (idealStep η f grad)^[t] s0 := by
  -- strengthen: carry the Hermitian invariant through the induction
  suffices H : ∀ t : ℕ, (spectralStep η f grad refL refR)^[t] s0 = (idealStep η f grad)^[t] s0
      ∧ ((idealStep η f grad)^[t] s0).L.IsHermitian
      ∧ ((idealStep η f grad)^[t] s0).R.IsHermitian from fun t => (H t).1
  intro t
  induction t with
  | zero => exact ⟨rfl, hL0, hR0⟩
  | succ t ih =>
    obtain ⟨heq, hLh, hRh⟩ := ih
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', heq]
    set s := (idealStep η f grad)^[t] s0 with hs
    have hstep : spectralStep η f grad refL refR s = idealStep η f grad s := by
      unfold spectralStep idealStep
      have hLh' : (s.L + grad s.W * (grad s.W)ᴴ).IsHermitian :=
        hLh.add (isHermitian_mul_conjTranspose_self _)
      have hRh' : (s.R + (grad s.W)ᴴ * grad s.W).IsHermitian :=
        hRh.add (isHermitian_conjTranspose_mul_self _)
      rw [SpectralPair.cfc_eq (hrefL _ hLh') f, SpectralPair.cfc_eq (hrefR _ hRh') f]
    rw [hstep]
    exact ⟨rfl, (idealStep_hermitian η f grad s hLh hRh).1,
      (idealStep_hermitian η f grad s hLh hRh).2⟩

/-! ## Non-vacuity: Mathlib's spectral theorem satisfies the oracle spec -/

/-- The refresh oracle given by Mathlib's spectral theorem. -/
noncomputable def mathlibRefresh (n : Type*) [Fintype n] [DecidableEq n]
    (M : Matrix n n ℝ) : SpectralPair n :=
  if h : M.IsHermitian then ⟨h.eigenvectorUnitary, h.eigenvalues⟩
  else ⟨1, fun _ => 0⟩

lemma mathlibRefresh_isFor {n : Type*} [Fintype n] [DecidableEq n]
    (M : Matrix n n ℝ) (h : M.IsHermitian) :
    (mathlibRefresh n M).IsFor M := by
  unfold mathlibRefresh
  rw [dif_pos h]
  constructor
  · have hu := Matrix.mem_unitaryGroup_iff.mp h.eigenvectorUnitary.2
    rwa [Matrix.star_eq_conjTranspose] at hu
  · have hsp := h.spectral_theorem
    rw [Unitary.conjStarAlgAut_apply, Matrix.star_eq_conjTranspose,
      RCLike.ofReal_real_eq_id, Function.id_comp] at hsp
    exact hsp

end Shampoo
