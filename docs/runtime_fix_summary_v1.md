# Runtime and Accuracy Fix Summary (v6)

## What was wrong in v5

The main runtime problem was in the 3D solid mesh sizing logic.

`volume_mesh.py` used a single global mesh size for the in-house FEniCSx mesh and defined it as:

- in-plane target = `cell_size * mesh_size_scale`
- thickness target = `total_thickness / 2.5`
- actual mesh size = `min(in-plane target, thickness target)`

For the default problem setup:

- unit-cell size = `0.10 m`
- `mesh_size_scale = 0.06` → intended in-plane size = `0.006 m`
- total thickness = `0.0012667 m`
- thickness target = `0.0012667 / 2.5 = 0.00050668 m`

So the actual mesh size became about **0.507 mm everywhere**, not 6 mm in-plane.

That is catastrophic for a `1.0 m x 1.0 m` thin plate. It drives the in-plane mesh count into the millions of elements before the eigensolve even starts, which is why a 3-sample run can stall for many hours.

There were two secondary issues:

1. The pipeline launched **one Docker container per sample**, which added avoidable startup overhead.
2. The default in-house solve used **linear solid interpolation**, which is a poor accuracy/runtime tradeoff for thin bending-dominated plates once you coarsen the mesh to something practical.

## What I changed

### 1. Decoupled CAD precision from solver mesh size

I split the old single mesh-size concept into two roles:

- **CAD reference size**: used for exact-CAD feature checks and OCC tolerances
- **solver mesh size**: used for the FEniCSx 3D mesh

New default behavior:

- `cad_reference_size_scale = 0.01`
- `mesh_size_scale = 0.06`
- `limit_solver_mesh_by_thickness = False`

This keeps CAD validation strict enough for exact solid export, while allowing the solver mesh to stay at the intended **in-plane** resolution.

If you ever want the old behavior back for debugging, it is still available with:

- `--limit-solver-mesh-by-thickness`

### 2. Kept the ANSYS path fully solid

The ANSYS handoff is still based on the validated solid STEP geometry with exactly two solid bodies:

- substrate solid
- piezo solid

So the ANSYS path remains a **true 3D solid-geometry workflow**.

### 3. Made the in-house solver faster without changing the problem definition

The in-house solver still solves the same problem:

- finite tiled cantilever plate
- base excitation in `z`
- modal reduction
- resistor-coupled voltage FRF
- same substrate and piezo material data

What changed is the numerical strategy:

- all meshes are now solved in **one Docker batch run**
- existing outputs are skipped by default
- the default retained mode count is reduced to a more practical screening value (`8`)
- the default coarse peak-search grid is reduced to `401` points, followed by local refinement
- the default solid displacement field is now **quadratic** (`element_order = 2`)

That last change is important: quadratic solid interpolation is a much better fit for thin, bending-driven structures than first-order solids when the mesh is intentionally kept coarse enough to be practical.

## Why this should still stay close to ANSYS

The improved path is still based on:

- the same full 3D solid geometry
- the same mechanical boundary condition
- the same support excitation interpretation
- the same piezoelectric coupling data
- the same reduced modal voltage-FRF formulation

So this is **not** a switch to a different physical problem. It is a fix to the discretization and execution strategy so that the in-house route is usable for dataset generation and screening.

## Expected effect

Compared with v5, the updated code should:

- cut mesh sizes from thickness-driven over-refinement to practical thin-plate meshes
- avoid repeated container startup
- resume partial runs much more efficiently
- keep better bending accuracy at coarse mesh sizes through quadratic solid interpolation

In practice, the expected behavior is:

- screening runs of a few samples should finish in **minutes to tens of minutes**, not many hours
- the in-house FRF should stay much closer to a solid ANSYS reference than the old coarse-first-order alternative

## Files changed

- `peh_inverse_design/volume_mesh.py`
- `peh_inverse_design/build_volume_meshes.py`
- `peh_inverse_design/fenicsx_modal_solver.py`
- `peh_inverse_design/pipeline_runner.py`
- `2 integrated_peh_pipeline.ipynb`
- `README.md`
- `docs/inverse_design_pipeline.md`

## New practical defaults

- exact solid CAD for ANSYS remains the default
- solver mesh uses in-plane scale by default
- quadratic solid interpolation is the default for the in-house solver
- batched solver execution is the default
- skip-existing outputs is the default

## Recommended verification workflow

1. Run 1 sample with the updated defaults.
2. Compare `f_peak` and the normalized voltage FRF against your ANSYS reference for that same geometry.
3. If needed, tighten only `mesh_size_scale` a little.
4. Keep `limit_solver_mesh_by_thickness = False` unless you are doing a one-off mesh-convergence study.

This gives you a practical path for fast screening while keeping the ANSYS path as the high-fidelity solid reference.
