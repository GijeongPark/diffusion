# Runtime fix summary (v6)

## What the latest ANSYS feedback confirmed

The previous “two combined STEP variants” approach was still trapped in a real CAD-topology loop.

### 1. The partitioned combined STEP was selectable only as hundreds of regions

The previous default `plate3d_XXXX.step` kept the substrate-piezo interface conformal by partitioning the piezo bottom with the metaplate topology.

That made the combined CAD more mesh-compatible, but it also meant the piezo bottom was no longer a single selectable face. In practice, Workbench exposed it as many separate regions.

### 2. The single-face combined STEP still tended to fail meshing

The previous `plate3d_XXXX_single_face_probe.step` preserved one continuous piezo bottom face, but it did so by removing the conformal interface partition.

That brought back the old problem: one combined file could keep the piezo bottom continuous **or** keep the interface conformal for meshing, but not both reliably.

### 3. So the real issue was not just “which combined STEP is better”

The important conclusion is that, for this geometry,

- a **single combined STEP** with exact shared topology naturally fragments the piezo bottom
- a **single combined STEP** with one continuous piezo bottom naturally loses the shared-topology condition that Workbench likes for meshing

So the previous v3/v4/v5 fixes were still choosing between two bad single-file compromises.

## What I changed in v6

### 1. Switched the default ANSYS handoff from one combined STEP to two standalone body STEP files

For the `layered_tet` path, the recommended Workbench export is now:

- `plate3d_XXXX_substrate.step`
- `plate3d_XXXX_piezo.step`

instead of relying on one combined `plate3d_XXXX.step`.

This removes the topology conflict entirely:

- the substrate STEP contains only the metaplate body
- the piezo STEP contains only the piezo body
- the piezo bottom remains one continuous face because it is now its own standalone solid
- Workbench can mesh the two solids independently after importing both into the same Geometry cell

### 2. Kept the legacy combined single-face probe only as an optional debug export

`plate3d_XXXX_single_face_probe.step` is still available when explicitly requested, but it is now treated as a **legacy inspection probe**, not the recommended Workbench handoff.

### 3. Updated the CAD report to describe the new split-body strategy

`plate3d_XXXX_cad.json` now records:

- `ansys_step_strategy = split_body_steps`
- the standalone substrate STEP path
- the standalone piezo STEP path
- separate pre-export and roundtrip validation blocks for each body

### 4. Updated the ANSYS handoff JSON accordingly

`plate3d_XXXX_ansys_workbench.json` now tells Workbench to prefer the split-body handoff:

- `files.substrate_step_path`
- `files.piezo_step_path`

The handoff notes now explicitly state that the split-body import is the recommended way to avoid the single-file topology conflict.

## Why this should behave better in Workbench

This new handoff no longer asks one STEP file to satisfy two contradictory goals at once.

Instead:

- the piezo body is imported as a simple standalone solid, so its bottom face is one face
- the metaplate body is imported as its own standalone solid, so its topology is preserved without imprinting that topology onto the piezo bottom face inside the same CAD body
- meshing can proceed body-by-body in Workbench, with the bonded/contact relationship created there if needed

## Important caveat

I could patch the export logic and the handoff/reporting logic here, but this environment still does **not** have gmsh or ANSYS Workbench available, so I could only verify the patch at the Python syntax level in this container.

That means the new fix is a geometry-export strategy correction based on the topology conflict revealed by your Workbench feedback, not a direct Workbench meshing benchmark executed from this environment.

## Files changed in this round

- `peh_inverse_design/volume_mesh.py`
- `peh_inverse_design/problem_spec.py`
- `peh_inverse_design/build_volume_meshes.py`
- `README.md`
- `docs/inverse_design_pipeline.md`
- `docs/runtime_fix_summary_v6.md`
- `docs/runtime_fix_summary.md`
