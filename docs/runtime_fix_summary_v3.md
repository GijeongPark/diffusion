# Runtime fix summary (v3)

## What was wrong across the previous iterations

The earlier versions had three separate issues that were interacting in confusing ways.

### 1. The in-house 3D solver mesh was still too expensive in practice

In the older path, the effective solver mesh size could become thickness-limited instead of staying at the intended in-plane scale. For a thin `1.0 m x 1.0 m` plate, that drove the tetrahedral mesh count far too high and made even small screening runs take much longer than they should.

### 2. The ANSYS geometry export and the Python solver mesh path were still too tightly coupled

ANSYS only needs:

- a solid STEP assembly
- exactly two solid bodies
- Workbench to generate its own mesh

The Python path, by contrast, needs a Gmsh-derived mesh for the FEniCSx solve. Keeping these two paths too closely tied made preprocessing heavier than necessary and made it easier for ANSYS-oriented geometry decisions to leak into the Python solver path.

### 3. The piezo/substrate interface in the exported STEP geometry was fragmented

This turned out to be a separate CAD-quality issue for the ANSYS handoff.

The previous `layered_tet` STEP export built the substrate solid and piezo solid, then fragmented them against each other at the CAD level. That preserved the substrate topology on the piezo bottom face, but it also split the piezo bottom into hundreds of separate faces.

In practice, that caused two ANSYS Workbench problems:

- the `PZT_BOTTOM` region could not be selected as one continuous face
- the default Workbench mesher tried to honor the fragmented interface topology, which led to visibly poor and unstable surface meshing behavior compared with a normal Parasolid/SolidWorks import

For example, the previous exported CAD reports recorded piezo-bottom face counts such as:

- sample `0000`: `422`
- sample `0001`: `332`
- sample `0002`: `321`

That matched the observed Workbench behavior exactly.

### 4. The top-surface strain visualization path was incomplete

The report figure was supposed to show the equivalent strain on the piezo top surface, but the modal files were being saved without the displacement data needed to reconstruct that strain field. As a result:

- `Top Surface FEM Mesh` fell back to a plain mesh panel
- `max_top_surface_strain` stayed `nan`
- the report images did not show the requested piezo top-surface strain distribution


## What I changed in v3

### 1. Kept the faster runtime strategy from the earlier fixes

The practical runtime fixes remain in place:

- CAD reference size is decoupled from the solver mesh size
- the default Python mesh backend stays `layered_tet`
- the solver keeps explicit through-thickness layers
- batched Docker execution stays enabled
- skip-existing solver outputs stays enabled
- the default solid displacement interpolation stays quadratic (`element_order = 2`)

This preserves the main runtime improvement while keeping the same physical piezoelectric plate problem definition.

### 2. Continued to separate the ANSYS geometry path from the Python solver mesh path

The intended split is still:

- ANSYS: import the STEP assembly only
- Python/FEniCSx: use the dedicated Gmsh-derived solver mesh

The `layered_tet` backend still generates the Python solver mesh from a partitioned 2D surface extrusion, but the ANSYS handoff is treated as its own solid-CAD export problem rather than as an extension of the solver-mesh topology.

### 3. Fixed top-surface strain saving for the report pipeline

The modal solver now supports storing the information needed to reconstruct top-surface strain even for the quadratic displacement field used in the current runtime profile.

Concretely:

- the notebook now enables `SOLVER_STORE_MODE_SHAPES = True`
- the modal solver evaluates mode shapes directly at the raw top-surface mesh vertices
- the modal files now store non-empty `top_surface_strain_eqv`
- the integrated dataset now gets finite `max_top_surface_strain` values
- the report figure now shows the piezo top-surface equivalent strain instead of falling back to a black mesh panel

This was verified on the regenerated `0329TEST` outputs before the later geometry refresh.

### 4. Fixed the ANSYS STEP export so the piezo bottom stays one continuous face

For the `layered_tet` ANSYS STEP handoff, the CAD export no longer partitions the piezo/substrate interface in the STEP model.

The important detail is:

- the Python solver mesh path still uses a partitioned surface representation where it needs one
- the ANSYS STEP export now keeps the piezo bottom as one continuous CAD face

The `layered_tet` STEP round-trip validation now explicitly requires:

- exactly two solid bodies
- exactly one continuous piezo bottom face in the exported STEP handoff

After this change, regenerating a sample produced:

- `pre_export piezo_bottom_face_count = 1`
- `step_roundtrip piezo_bottom_face_count = 1`

This is the expected CAD condition for clean Workbench face selection and much more normal default meshing behavior.

### 5. Regenerated the current `0329TEST` ANSYS geometry folder for the active notebook settings

The active notebook configuration currently uses:

- `RUN_NAME = "0329TEST"`
- `PIEZO_THICKNESS_M = 1e-4`
- `MESH_SIZE_SCALE = 0.08`
- `CAD_REFERENCE_SIZE_SCALE = 0.01`
- `SOLVER_MESH_BACKEND = "layered_tet"`
- `SUBSTRATE_LAYERS = 2`
- `PIEZO_LAYERS = 1`
- `SOLVER_ELEMENT_ORDER = 2`
- `SOLVER_STORE_MODE_SHAPES = True`

Because interrupted runs had left mixed geometry artifacts in `runs/0329TEST/meshes/volumes`, I refreshed that folder cleanly for the current notebook settings.

To avoid losing the previous artifacts, the old folder was moved to:

- `runs/0329TEST/meshes/volumes_backup_20260329_1945`

The clean regenerated folder is:

- `runs/0329TEST/meshes/volumes`

The regenerated ANSYS handoff JSON files now record:

- `piezo_thickness_m = 0.0001`
- `total_thickness_m = 0.0011`

And the regenerated CAD reports now record:

- sample `0000`: `piezo_bottom_face_count = 1`
- sample `0001`: `piezo_bottom_face_count = 1`
- sample `0002`: `piezo_bottom_face_count = 1`


## Files changed in this round

- `peh_inverse_design/volume_mesh.py`
- `peh_inverse_design/fenicsx_modal_solver.py`
- `2 integrated_peh_pipeline.ipynb`

Related behavior from the earlier runtime split remains in:

- `peh_inverse_design/build_volume_meshes.py`
- `peh_inverse_design/problem_spec.py`
- `peh_inverse_design/pipeline_runner.py`
- `README.md`
- `docs/inverse_design_pipeline.md`


## Expected effect

### ANSYS path

- the STEP handoff remains a two-body solid assembly
- the piezo bottom should import as one continuous face
- Workbench face selection for the bottom electrode should be much cleaner
- default Workbench meshing should behave much more like a normal solid CAD import, instead of inheriting hundreds of interface-face fragments

### Python path

- preprocessing remains much faster than the old full-3D tetrahedralization route
- the layered-tet solver mesh stays practical for small screening runs
- the report pipeline now shows top-surface equivalent strain instead of only the fallback mesh panel
- integrated datasets now contain finite top-surface strain summaries when the modal outputs are regenerated with stored mode shapes


## Important caveats

### 1. Downstream physics outputs may still need regeneration after a geometry refresh

Refreshing `runs/0329TEST/meshes/volumes` updates:

- STEP geometry
- CAD report
- ANSYS handoff JSON
- Python solver mesh NPZ

But it does **not** automatically refresh:

- `runs/0329TEST/data/fem_responses`
- `runs/0329TEST/data/modal_data`
- `runs/0329TEST/data/integrated_dataset.npz`
- `runs/0329TEST/reports`

Those downstream outputs should be regenerated if you want the full run to be internally consistent with the newly exported geometry folder.

### 2. The solver strain fix and the ANSYS face fix address different problems

They are both important, but they solve different issues:

- the strain fix is for the Python/FEniCSx reporting path
- the continuous-face fix is for the ANSYS STEP geometry handoff

The Python solver still uses its own dedicated mesh path, and the ANSYS path still uses the STEP assembly directly.


## Recommended verification workflow

1. Import one regenerated STEP file from `runs/0329TEST/meshes/volumes` into ANSYS Workbench.
2. Confirm the piezo body bottom can be selected as one face instead of many patterned sub-faces.
3. Let Workbench generate a default mesh and compare its behavior against the previous fragmented export.
4. If you want the Python outputs to stay synchronized with the refreshed geometry folder, rerun the modal solve, integrated dataset build, and report generation for the same run.
