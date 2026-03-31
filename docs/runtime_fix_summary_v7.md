# Runtime fix summary (v7)

## What the latest ANSYS feedback changed

The split-body workaround in v6 solved the CAD-topology conflict, but it violated a hard workflow constraint: **the Workbench handoff must stay one STEP file**.

That means the export has to satisfy three practical requirements at the same time:

1. the ANSYS handoff must stay a **single combined STEP file**,
2. the file must stay on the **meshable / conformal-interface** side of the old trade-off,
3. the user must not be forced to manually click hundreds of piezo-bottom face fragments.

## What was wrong with the previous options

### 1. The split-body export fixed topology, but broke the required workflow

v6 recommended:

- `plate3d_XXXX_substrate.step`
- `plate3d_XXXX_piezo.step`

That avoided the one-file CAD conflict, but it no longer matched the actual Workbench import requirement.

### 2. The single-face combined STEP still was not a safe default

The legacy `plate3d_XXXX_single_face_probe.step` can keep the piezo bottom continuous, but the latest Workbench error log still points to the same class of failure there:

- surface mesh generation succeeds,
- tetrahedral fill fails,
- the boundary mesh contains an edge intersection / near-self-intersection.

So that file remains useful only as a probe, not as the default Workbench meshing geometry.

### 3. The meshable combined STEP still leaves a manual-selection burden

The conformal combined STEP is still the safer meshing option, but the piezo bottom can be partitioned into many regions whenever the tiled metaplate topology is complicated.

That is not a contact failure; it is a **selection/scoping problem** for the bottom electrode or interface definition.

## What I changed in v7

### 1. Restored the default ANSYS handoff to a single combined STEP file

The `layered_tet` export now again writes:

- `plate3d_XXXX.step`

as the **recommended** Workbench geometry handoff.

This file uses the **partitioned / conformal interface** path, not the single-face probe path, because that is the meshable side of the trade-off for one-file export.

### 2. Added a Workbench face-group manifest to remove the manual clicking problem

Each export now also writes:

- `plate3d_XXXX_ansys_face_groups.json`

This file records simple geometric recipes for grouping faces in Workbench, especially:

- `piezo_bottom_electrode`
- `piezo_top_electrode`
- `clamped_edge`

The intended use is:

- keep the one-file, conformal STEP for meshing,
- build the bottom-electrode/interface group from the piezo body and the interface `z` plane,
- avoid manual clicking of every fragmented piezo-bottom region.

So the bottom-face issue is handled as a **selection recipe problem**, not by forcing the default geometry back to the non-meshable single-face variant.

### 3. Added conservative CAD simplification before STEP export

The cleaned substrate planform now goes through one more CAD-oriented simplification stage before the STEP export:

- simplify the boundary at the CAD reference scale,
- prune only holes below that reference-scale area threshold,
- preserve the single connected substrate requirement and the volume checks.

This is meant to:

- reduce pointless micro-segmentation inherited from the raw GRF contour,
- shrink the exported STEP complexity,
- reduce Workbench face fragmentation and general meshing stress without switching to a different physical handoff mode.

### 4. Annotated the exported combined STEP with body/face names and colors where the CAD kernel supports it

Before STEP export, the combined model is now annotated with labels such as:

- `substrate`
- `piezo`
- `piezo_top_electrode`
- `piezo_bottom_electrode`
- `clamped_edge`

Workbench import behavior for those annotations can still depend on the downstream CAD kernel, so the new **face-group manifest** is the reliable path and the annotations are treated as a best-effort bonus.

### 5. Updated the handoff JSON back to the one-file workflow

`plate3d_XXXX_ansys_workbench.json` now again prefers:

- `files.step_path`

and also records:

- `files.face_selection_manifest_path`
- `files.inspection_single_face_step_path`

The handoff notes now explain that:

- the combined STEP is the recommended meshing geometry,
- the selection manifest is the recommended fix for bottom-electrode scoping,
- the single-face probe remains inspection-only.

## Resulting export behavior

### Recommended Workbench file set

- `plate3d_XXXX.step`
- `plate3d_XXXX_ansys_face_groups.json`
- `plate3d_XXXX_ansys_workbench.json`

### Optional legacy debug file

- `plate3d_XXXX_single_face_probe.step`

## What this does and does not claim

This patch does **not** claim that one exact combined STEP can simultaneously have:

- a fully conformal shared interface,
- a truly single piezo-bottom CAD face,
- and identical meshing robustness.

The latest feedback still points the other way.

So v7 makes the one-file workflow practical by doing two things together:

- keep the **meshable combined STEP** as the default,
- move the piezo-bottom convenience problem into an explicit **face-group manifest** instead of trying to force the CAD itself into the unstable single-face state.

## Files changed in this round

- `peh_inverse_design/volume_mesh.py`
- `peh_inverse_design/build_volume_meshes.py`
- `peh_inverse_design/problem_spec.py`
- `README.md`
- `docs/inverse_design_pipeline.md`
- `docs/runtime_fix_summary_v7.md`
- `docs/runtime_fix_summary.md`
- `tests/test_problem_spec_handoff.py`
- `tests/test_volume_mesh_cad_export.py`
