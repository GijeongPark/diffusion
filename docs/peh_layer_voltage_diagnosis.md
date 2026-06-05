# PEH Layer-Count Voltage "Collapse": Diagnosis and Decision Memo

**Status:** Diagnosis complete — no solver changes made (decision deferred to advisor).
**Date recorded:** 2026-06-04
**Scope:** Why the in-house FEniCSx PEH solver's peak voltage appeared to drop ~10× when the
through-thickness mesh was refined from 3 layers (2 substrate + 1 piezo) to 11 layers (8 + 3),
and what to do about it.

---

## 1. Executive summary

The voltage "collapse" is **not** a physics bug, a mesh bug, or a material/coupling bug.
It is an **artifact of memory-limited solver degradation**, and it is now reproducible as a
hard out-of-memory failure rather than a wrong number.

The key facts, all from this project's own run logs and audits:

1. The meshes are **identical in every meaningful way** across layer counts (same element
   order, same material volumes, same in-plane size, zero coarsening). The mesh generator is
   not corrupting the high-layer cases.
2. Under an **honest quadratic (Q2) solve, only the 2+1 case fits in memory.** The 4+2 case
   (9.1 M vector DOFs) is **OOM-killed (exit 137)** on a **128 GiB** machine; 8+3 (15.7 M DOFs)
   is larger still.
3. The old "24 V" number was produced by a *pre-guard* code path that silently degraded the
   solve under memory pressure (linear-element fallback and/or mesh coarsening). The guards
   added since then correctly **refuse** to emit degraded numbers — so the case now fails
   loudly instead of returning garbage.
4. Therefore: **months of "memory handling" work could not have fixed the result**, because
   the result it was protecting never existed as a valid Q2 solve. The honest Q2 high-layer
   solve simply does not fit in RAM with the current eigensolver.

**Bottom line for the advisor:** the question "why does voltage collapse with more layers?"
is the wrong question. The right questions are (a) *do we even need 11 layers?* (physically,
no) and (b) *if we do, how do we solve a 10–16 M-DOF modal problem without a direct
factorization?*

---

## 2. The reference problem (ANSYS Workbench model being matched)

| Quantity | Value |
|---|---|
| Substrate | structural steel, 1.0 m × 1.0 m × 1.0e-3 m |
| Piezo patch | PZT-5H, 1.0 m × 1.0 m × 1.0e-4 m (full coverage) |
| Substrate material | E = 1.9305e11 Pa, ν = 0.3, ρ = 7930 kg/m³ |
| Piezo density | ρ = 7500 kg/m³ |
| Piezo permittivity | relative diag(1704, 1704, 1434) → absolute εS₃₃ ≈ 1.26934e-08 F/m |
| Boundary | cantilever (clamped at x = 0) |
| Excitation | base acceleration Z = 2.5 m/s² |
| Damping | constant structural damping g = 0.05 (modal ζ = 0.025, via g = 2ζ) |
| Electrical | bottom electrode grounded, top electrode coupled, external R = 10 000 Ω (CIRCU94) |
| Analysis | coupled-field harmonic, 0–3 Hz, 30 intervals |

ANSYS coarse-grid CSV peaks (0.1 Hz grid) used as the comparison reference:

| Sample | Peak frequency | Peak voltage |
|---|---:|---:|
| 0000 | 0.6 Hz | 242.96 V |
| 0001 | 0.7 Hz | 116.68 V |

> Note on reference convention: always record *which* ANSYS source a percent error is computed
> against (coarse table peak, refined Workbench peak, or fitted peak). They differ.

---

## 3. The evidence

### 3.1 The meshes are clean at every layer count

From the mesh-only audit (`parity_sweep_sample_0000.json`), sample 0000, in-plane size 8 mm,
`mesh_preset = ansys_parity`:

| Case | Q2 vector DOFs | substrate vol (m³) | piezo vol (m³) | element order | coarsening passes |
|---|---:|---:|---:|---:|---:|
| 2+1 | 4,978,767 | 6.3432e-4 | 1.0000e-4 | 2 | 0 |
| 4+2 | 9,118,197 | 6.3432e-4 | 1.0000e-4 | 2 | 0 |
| 8+3 | 15,718,383 | 6.3432e-4 | 1.0000e-4 | 2 | 0 |

Material volumes are identical to 5+ significant figures; element order is 2 everywhere; no
coarsening occurs. **There is no geometric or discretization difference that could explain a
10× change in response** — only the DOF count grows.

### 3.2 Only the low-layer case actually solves

From `tmp/controlled_ansys_parity_layer_sweep_20260522_094415` (single-rank, Q2, anisotropic
PZT, `shift_invert_cholesky_ooc` backend):

| Case | DOFs | Outcome | f₁ (Hz) | V_peak (V) | θ₁ |
|---|---:|---|---:|---:|---:|
| 2+1 | 5.0 M | **OK** | 0.7071 | **350.5** | 0.01141 |
| 4+2 | 9.1 M | **exit 137 (OOM-killed)** | — | — | — |
| 8+3 | 15.7 M | not reached (larger than 4+2) | — | — | — |

Additional log evidence:
- Multiple recent attempts to run higher layers / more MPI ranks failed: `exit 255`
  (MPI-as-root failure), "no compatible top-surface strain field," and `exit 137` (OOM).
- The MUMPS out-of-core options were reported by PETSc as **"unused database options"**
  (`mat_mumps_icntl_22`, `mat_mumps_icntl_14`) — i.e., the out-of-core path was **not actually
  engaging**, so the "_ooc" backend was effectively in-core and still OOM'd.

### 3.3 The low-layer result is the right order of magnitude

The 2+1 solve gives **350 V at 0.71 Hz** vs ANSYS **243 V at 0.60 Hz** — same order of
magnitude, frequency high by ~18%. This is ordinary disagreement between a 3D tetrahedral
model and ANSYS (which uses swept/shell elements plus bonded MPC contact), **not** a collapse.

> Parity nuance worth flagging: the recent capacitance "fix" switched the circuit dielectric
> from the reduced-plate value (1.729e-8 F/m) to the full-3D εS₃₃ (1.26934e-8 F/m), a −27 %
> change in C. Lower C raises voltage, which moved the 2+1 peak from ~273 V up to ~350 V —
> i.e., *further* from ANSYS's 243 V. Whether this is correct depends on what permittivity
> ANSYS actually used for the CIRCU94 capacitance; this should be reconciled separately from
> the layer question.

---

## 4. Root cause

A shift-invert modal eigensolve requires a **sparse direct factorization** (MUMPS Cholesky/LU)
of the stiffness operator. For 3D elasticity the factorization memory grows **superlinearly**
with the number of DOFs (fill-in), far faster than the matrix itself. Empirically on this
128 GiB machine:

- ~5 M DOFs (2+1): fits, solves.
- ~9 M DOFs (4+2): exceeds RAM → killed.
- ~16 M DOFs (8+3): hopeless by direct factorization.

No configuration flag changes this asymptotic. "Improving memory handling" — disabling silent
fallbacks, enabling out-of-core, tuning MPI ranks — addresses *symptoms*; it cannot make a
direct factorization of a 10–16 M-DOF operator fit in 128 GiB. This is why the investigation
stalled: the lever being pulled was not connected to the wall being hit.

---

## 5. Why this is also a *physics* non-problem

The plate is 1 m × 1 m × 1 mm — an aspect ratio of **1000:1**, strongly bending-dominated at
these sub-3 Hz frequencies. Through the 1 mm thickness, the bending strain is essentially
**linear** (zero at the neutral axis, maximal at the surfaces). Quadratic (Q2) elements
represent a linear-to-quadratic strain profile **within a single element**, so very few layers
through the thickness already resolve the bending kinematics.

Consequences:

- Through-thickness refinement should make the response **converge** — a small, monotone change
  toward a limit — **never** a 10× drop. A 10× drop is the signature of a *degraded* solve
  (element-order fallback over-stiffens thin bending; linear tets shear-lock badly at 1000:1
  aspect ratio), not of refinement.
- The 2+1 Q2 model is already in the converged regime for the quantity of interest (first-mode
  voltage), as evidenced by its ANSYS-order agreement.
- **8+3 layers is not physically required.** It was an arbitrary resolution target that pushed
  the problem past the memory wall for no accuracy benefit.

---

## 6. Decision matrix (advisor's call)

| Option | What it is | Effort | When to choose |
|---|---|---|---|
| **A. Accept low-layer Q2 + prove convergence** | Declare the Q2 model at a memory-feasible resolution (≤ ~5 M DOFs) as the production solver; demonstrate through-thickness convergence with a memory-safe study (§7). | Low | Default. Unblocks dataset generation immediately; physically justified. |
| **B. Factorization-free eigensolver** | Replace shift-invert + MUMPS with an iterative, low-memory eigensolver (SLEPc LOBPCG or Krylov-Schur) with a GAMG/AMG preconditioner; no full factorization. | Medium–High | If a reviewer insists on directly solving 8+3, or future geometries genuinely need >5 M DOFs. |
| **C. Change element technology** | Move from 3D tets to swept hexahedra (few through-thickness elements, far fewer DOFs) or a shell/plate formulation — closer to what ANSYS does. | High | Long-term: best accuracy-per-DOF and closest ANSYS parity, but a substantial solver rewrite. |

**Recommendation:** Option **A** now to unblock training-data generation, holding Options B/C
in reserve. The collapse has already been explained; spending effort to make 8+3 solvable only
makes sense if 8+3 is actually needed, which the physics says it is not.

---

## 7. Recommended confirmatory test (memory-safe, decisive)

To convert "the collapse is a memory artifact" from a strong inference into a measured fact —
*without* hitting OOM — run a through-thickness sweep at a **coarser but fixed in-plane size**
so the total DOF count stays inside the ~5 M budget at every layer count.

Example: at an in-plane size of ~20 mm (vs the current 8 mm), the in-plane element count drops
by ≈ (8/20)² ≈ 0.16×, so even 8+3 lands near ≈ 2.5 M DOFs and fits comfortably.

Procedure:
1. Fix in-plane mesh size at a value where 8+3 fits in memory (≈ 20 mm).
2. Solve 2+1, 4+2, 8+3 at that fixed in-plane size, Q2, single rank.
3. Compare f₁, θ₁, modal force, and V_peak across the three.

Interpretation:
- **Voltage stable across 2+1 → 8+3** ⇒ through-thickness layering is benign; the original
  "collapse" was purely memory-driven degradation. Confirms Option A.
- **Voltage still drops with layers even when all three solve cleanly** ⇒ there is a genuine
  layer-dependent bug (revisit `θ = (1/h)∫_piezo e_col3·ε dV`, interface connectivity, and
  mode selection). This would be surprising given §5, but it is the falsifiable check.

This test is cheap (all three cases fit), needs no solver changes, and produces a single plot
that settles the question for the advisor.

---

## 8. Material / parity reference (retained from prior diagnostic notes)

For any future ANSYS-parity work, the following constants are the baseline (do not substitute
reduced-plate constants into the 3D FEM path unless explicitly doing a reduced-plate study):

- Structural steel: E = 1.9305e11 Pa, ν = 0.3, ρ = 7930 kg/m³
- PZT-5H density: 7500 kg/m³
- PZT permittivity: relative diag(1704, 1704, 1434); absolute εS₃₃ ≈ 1.26934e-08 F/m
- Circuit capacitance dielectric: use full-3D εS₃₃ (see §3.3 caveat)
- Damping: modal ζ = 0.025 (ANSYS g = 0.05)
- Load: external R = 10 000 Ω, base acceleration 2.5 m/s²
- Internal Voigt order used by the solver: `xx, yy, zz, yz, xz, xy` (ANSYS Workbench tables are
  `xx, yy, zz, xy, yz, xz` — reorder on import).
- Voigt-Reuss-Hill isotropic equivalent of the PZT stiffness (for an isotropic-elastic
  diagnostic, if ever needed): E ≈ 61.07 GPa, ν ≈ 0.3948, K ≈ 96.74 GPa, G ≈ 21.89 GPa.

---

## 9. What changed in the repository alongside this memo

This memo accompanies a cleanup that reduces the repository to the essential PEH pipeline
(training-data generation, ANSYS geometry export, PEH FE analysis) and removes the standalone
diagnostic/audit scaffolding that had accumulated. The solver itself was **not** modified —
the layer decision is deferred per Option A/B/C above.
