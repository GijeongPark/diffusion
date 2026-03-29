# Runtime fix summary (v7)

## What was wrong in the previous version

The previous pipeline still tied two unrelated tasks together:

1. **ANSYS geometry export**
   - The ANSYS path only needs a **solid STEP assembly** with two bodies:
     - substrate
     - piezo layer
   - ANSYS Workbench should import that STEP assembly and create its own mesh.

2. **Python solver meshing**
   - The in-house pipeline needs a Gmsh-based mesh for the Python/FEniCSx solve.

In `v6`, those two paths were still too tightly coupled. The code exported the STEP model **and** built a full 3D gmsh volume mesh from the solid geometry in the same stage. That made the preprocessing much heavier than necessary.

The second problem was the **Python solver mesh construction itself**. The old route tetrahedralized the full 3D solid body, which is expensive for a thin, meter-scale plate with many geometric details. Even after fixing the accidental thickness-driven over-refinement, this remained too slow for routine 3-sample screening runs.

## What I changed in v7

### 1. I separated the ANSYS geometry path from the Python solver mesh path

The ANSYS handoff is now explicitly treated as:

- **STEP only** for geometry
- **two solid bodies** only
- **no exported gmsh volume mesh required by ANSYS**

The handoff JSON now records the STEP file as the ANSYS geometry source and records the Python solver mesh separately.

### 2. I changed the default Python mesh backend to a much faster one

The new default backend is:

- `solver_mesh_backend = "layered_tet"`

Instead of tetrahedralizing the full 3D STEP body, the pipeline now:

1. builds and validates the exact STEP solids for ANSYS,
2. creates a **partitioned 2D gmsh surface mesh** of the plate planform,
3. extrudes that surface into a **layered tetrahedral mesh** for the Python solver.

This preserves the same bending-type piezoelectric problem while avoiding the cost of a full free 3D tetrahedralization of the solid CAD body.

### 3. I added explicit through-thickness layer controls

The fast backend now uses:

- `substrate_layers = 2`
- `piezo_layers = 1`

This gives the Python solver a thickness-resolved mesh without forcing gmsh to generate an expensive fully unstructured 3D mesh.

### 4. I changed the notebook and pipeline defaults to a faster profile

The integrated notebook now defaults to:

- `MESH_SIZE_SCALE = 0.08`
- `SOLVER_MESH_BACKEND = "layered_tet"`
- `SUBSTRATE_LAYERS = 2`
- `PIEZO_LAYERS = 1`
- `SOLVER_SEARCH_POINTS = 301`
- `SOLVER_ELEMENT_ORDER = 2`

This is intended to keep the Python path in a practical screening regime while still targeting a result that is close to the ANSYS trend for the same geometry and loading definition.

### 5. I kept a legacy fallback path

If you ever need the old behavior for debugging, the code still supports:

- `solver_mesh_backend = "gmsh_volume"`

That backend can optionally keep:

- native `.msh`
- optional `.xdmf`

But it is no longer the default.

## Files changed

- `peh_inverse_design/volume_mesh.py`
- `peh_inverse_design/build_volume_meshes.py`
- `peh_inverse_design/problem_spec.py`
- `peh_inverse_design/pipeline_runner.py`
- `peh_inverse_design/fenicsx_modal_solver.py`
- `2 integrated_peh_pipeline.ipynb`
- `README.md`

## Expected effect

### ANSYS path

- faster preprocessing
- STEP assembly stays solid
- no unnecessary gmsh volume mesh generation for ANSYS

### Python path

- much faster mesh generation
- substantially lower 3D meshing overhead
- more controlled thin-plate discretization
- better runtime for small screening runs

## Important caveat

I verified the modified Python files compile successfully, but I did **not** run a full end-to-end gmsh + DOLFINx solve in this environment because the required gmsh/FEniCSx runtime stack is not available here. So this is a code-level fix aimed at the right bottleneck, not a benchmarked runtime claim from this container.
