# Runtime fix summary (v4)

## What turned out to be wrong after v3

Two separate issues were being conflated.

### 1. The data-model report was visually misleading even when the solver mesh itself was reasonable

The report panel was plotting the **piezo top surface**, not the exposed metaplate surface.

With the current problem definition:

- the piezo patch fully covers the plate
- the top piezo surface is therefore almost a full solid rectangle in plan view
- the substrate unit-cell pattern is hidden underneath that patch

So the panel could easily look like “an almost solid mesh with irregular element sizes” even when the underlying tiled substrate geometry was still patterned correctly.

A second bug made that confusion worse:

- `visualize_run_outputs.py` loaded dataset rows with `data[key][sample_id]`
- that only works if `sample_id == row_index`
- after filtered runs, subset runs, or `target_ok` screening, that assumption can break
- in that case the report can show the wrong binary geometry against the correct mesh/response files

That means the old report could mislead you in **two** ways at once:

1. it showed the piezo top surface instead of the metaplate footprint
2. it could attach the wrong dataset geometry row to a valid response sample

### 2. The v3 ANSYS STEP handoff optimized for “single piezo-bottom face” at the expense of meshability

In v3, the default layered-tet ANSYS STEP export changed to an **unpartitioned** piezo/substrate interface so that the piezo bottom stayed one continuous CAD face.

That helped face selection, but it also created a partial-overlap CAD situation:

- the piezo bottom remained one large face
- the substrate top still had the detailed metaplate topology underneath
- their shared region was no longer imprinted into matching face boundaries on the default STEP handoff

That is a risky geometry state for downstream solid meshing in Workbench, especially for a very thin fully covering piezo layer. In other words, v3 fixed the selection problem by making the default STEP handoff less robust for meshing.

## What I changed in v4

### 1. Fixed the report/data lookup path

`peh_inverse_design/visualize_run_outputs.py` now:

- resolves dataset rows by matching the stored `sample_id` array instead of assuming `sample_id == array index`
- falls back to positional indexing only when no `sample_id` field exists
- reads `tile_counts` and `cell_size_m` from the dataset row when present

This removes the geometry/response mismatch bug for filtered and subset runs.

### 2. Made the report figure explicit about what surface it is plotting

The surface panel now explicitly describes the field as the **piezo top-surface** strain/mesh view.

I also overlaid the tiled substrate footprint onto that panel and its zoomed mesh inset. This keeps the current physics interpretation intact while making the hidden metaplate pattern visible on top of the piezo-surface plot.

Practically, this means:

- the panel can still look like a nearly solid plate, which is physically expected for a fully covering patch
- but the underlying unit-cell pattern is now drawn on top, so you can tell whether the geometry path is actually patterned or not

### 3. Split the ANSYS STEP export into a meshable default and an inspection-only single-face probe

For the layered-tet path, the export is now intentionally dual-track:

- `plate3d_XXXX.step`
  - **recommended Workbench handoff**
  - exports the substrate-piezo interface in a **partitioned / mesh-compatible** way
- `plate3d_XXXX_single_face_probe.step`
  - optional inspection-only variant
  - keeps the piezo bottom as one continuous face so you can still inspect electrode continuity

This restores the safer default for Workbench meshing without throwing away the “single continuous bottom face” check entirely.

### 4. Updated the CAD and Workbench metadata

The CAD JSON and ANSYS handoff JSON now record that:

- the recommended STEP for Workbench meshing is the partitioned `plate3d_XXXX.step`
- the single-face probe STEP is separate and should not be preferred for default Workbench meshing

## Files changed in this round

- `peh_inverse_design/visualize_run_outputs.py`
- `peh_inverse_design/volume_mesh.py`
- `peh_inverse_design/build_volume_meshes.py`
- `peh_inverse_design/problem_spec.py`
- `README.md`
- `docs/inverse_design_pipeline.md`
- `docs/runtime_fix_summary_v4.md`

## Expected effect

### Data-model / result-interpretation path

- summary figures should no longer silently mismatch sample geometry rows after filtered runs
- the strain/mesh panel should stop looking like a “mysterious almost-solid metaplate” because it now clearly shows the piezo top surface with the substrate footprint overlaid
- the strain plots are easier to judge physically because the hidden unit-cell support pattern is visible again

### ANSYS Workbench path

- the default `plate3d_XXXX.step` is now chosen for meshability, not for single-face inspection
- the Workbench handoff should therefore be more robust than the v3 single-face default
- the optional `plate3d_XXXX_single_face_probe.step` remains available when you want to inspect electrode continuity separately

## Important caveat

I could patch the export logic and the reporting path here, but I could not re-run ANSYS Workbench itself inside this environment. So the Workbench fix is a geometry-export correction based on the CAD topology conflict in v3, not a direct Workbench benchmark executed from this container.
