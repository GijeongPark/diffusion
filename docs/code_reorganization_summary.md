# Code Reorganization — Review Summary

*Prepared for review (2026-06-04). Audience: non-programmer. Scope: code organization only.*

**In one sentence:** the project was reorganized so that every setting and material
property is edited directly in the two notebooks, and the code is grouped into clearly
named folders — and it produces the **identical result (350 V for sample 0000)**, so this
changed only the *organization* of the code, not the *computation*.

## Why this was done
- Material properties were hidden in a separate configuration file; the notebook could change almost nothing.
- The code was 13 files dumped in one folder — hard to find anything.
- Goal: make the notebooks the single place to control everything, and group the files by role.

## Before → After

### 1. Where you set material properties & parameters

| | Before | After |
|---|---|---|
| Materials (Young's modulus, piezo matrices, damping, resistance, …) | only in `configs/peh_inverse_design_spec.yaml`; the notebook could change just 2 densities | all in **notebook 2 → cell ② "Materials & physics"**, as plain labeled variables |
| Example: change piezo density | open and edit the YAML file | set `PIEZO_DENSITY_KG_PER_M3 = …` in the notebook |
| Frequency window (0.9–1.1 × peak) | hard-coded; editing it did nothing | a notebook setting that now actually takes effect |

### 2. Code folder organization (the `peh_inverse_design` package)

| Before | After |
|---|---|
| 13 `.py` files in one flat folder | grouped into 7 role-named folders: `core/`, `geometry/`, `meshing/`, `solver/`, `datasets/`, `pipeline/`, `viz/` |

### 3. Notebook structure

| Before | After |
|---|---|
| one large, mixed settings cell | three labeled sections: **① Run settings**, **② Materials & physics**, **③ Build & run** |
| notebook 1 settings scattered across cells | one **Settings** cell at the top |

## Key takeaways for the meeting
1. **Results are unchanged** — sample 0000 = **350 V before and after**. This verifies the reorganization did not affect the science.
2. **One place to change everything** — the notebooks are now the single source of truth; the old YAML file remains only as a backup default.
3. **Easier to read, run, and maintain** — files are grouped by what they do.
4. **The voltage discrepancy vs. ANSYS is unchanged and still open** — a separate physics/modeling question, not part of this reorganization.
5. During testing we also fixed two *environment* issues (a Docker MPI-launcher flag; choosing a memory-feasible mesh preset) — unrelated to the code's correctness.

## What did NOT change
- The physics, the FEM equations, the solver math, the mesh-generation algorithm.
- The numerical output (350 V for sample 0000).
- The open voltage-discrepancy investigation.
