# Diffusion-Based Inverse Design for Piezoelectric Energy Harvesting

This project aims to perform inverse design in the piezoelectric energy harvesting (PEH) domain, focusing on a plate metamaterial substrate with a fully covering piezoelectric patch configuration.

## Overview

The foundational framework draws from the work of Qibang Liu et al., *"Towards SDF-based Metamaterial Design: Neural Operator Transformer for Forward Prediction and Diffusion Model for Inverse Design."* However, unlike the original study which addresses static problems, this research extends the approach to **dynamic problems** — for example, inverse designing a geometry unit cell that yields a target voltage frequency response function (FRF).

Beyond the difference in problem domain, this work also seeks to incorporate more sophisticated methodologies or improvements to the overall framework where applicable.

## Current Geometry Pipeline

The repository currently contains a notebook, [periodic_grf_sdf.ipynb](/home/gijeong/Inverse%20Design/periodic_grf_sdf.ipynb), that:

- generates periodic Gaussian-random-field unit-cell geometries
- converts them to binary maps and signed distance fields
- builds `gmsh` meshes for downstream simulation

## Fixed Inverse-Design Setup

The current inverse-design problem is now fixed as:

- one unit-cell geometry repeated `10 x 10` times to form a finite cantilevered plate
- fully covered piezoelectric patch
- target input given as the magnitude of the voltage FRF around the fundamental resonance
- normalized frequency axis `f / f_peak in [0.9, 1.1]`
- `f_peak` stored separately as a conditioning scalar

The detailed physics setup, dataset schema, and recommended diffusion pipeline are documented in [docs/inverse_design_pipeline.md](/home/gijeong/Inverse%20Design/docs/inverse_design_pipeline.md).

The machine-readable problem specification is stored in [configs/peh_inverse_design_spec.yaml](/home/gijeong/Inverse%20Design/configs/peh_inverse_design_spec.yaml).

## Package Layout

The reusable Python package `peh_inverse_design/` is organized by role so you can find
the relevant code without reading every file:

| Subpackage | What lives there |
| --- | --- |
| `core/` | shared primitives: physical-group tags (`mesh_tags`), the problem specification loader (`problem_spec`), and `paths` (repository-root helper) |
| `geometry/` | unit-cell geometry construction (`geometry_pipeline`) and modal surface-field extraction (`modal_surface_fields`) |
| `meshing/` | volume meshing + CAD/STEP export (`volume_mesh`) and its CLI driver (`build_volume_meshes`) |
| `solver/` | the FEniCSx modal FEM solver that runs inside the dolfinx Docker image (`fenicsx_modal_solver`) and the numpy-only reduced-order FRF evaluator shared with host tools (`modal_frf`) |
| `datasets/` | dataset assembly and I/O (`response_dataset`, `build_geometry_dataset`, `build_response_dataset`, `build_integrated_dataset`, `subset_unit_cell_dataset`) |
| `pipeline/` | the end-to-end orchestrator (`pipeline_runner`) used by the notebook and `run_all.sh` |
| `viz/` | run-output figures and reports (`visualize_run_outputs`) |
| `validation/` | ANSYS cross-validation: physical-units FRF CSV export (`export_physical_frf`) and solver-mesh vs STEP geometry parity (`geometry_parity`) |

The top-level package API is unchanged: `from peh_inverse_design import PipelineConfig, run_pipeline`
still works. Command-line modules are now addressed by their subpackage, e.g.
`python -m peh_inverse_design.pipeline.pipeline_runner`.

## Dataset Utilities

The package handles the next pipeline step after the unit-cell notebook:

- building `data/geometry_dataset.npz` from the unit-cell notebook output
- generating `10 x 10` tiled full-plate meshes for FEM
- exporting ANSYS Workbench Geometry-compatible `plate3d_*.step` CAD files
- aggregating per-sample FEM outputs into `data/response_dataset.npz`
- collecting everything into one `integrated_dataset.npz`

`dataset_100.npz` is only a legacy filename from an early test. You can use any `.npz` filename for the unit-cell dataset, for example `data/unit_cell_dataset.npz` or `data/test_runs/test3/unit_cell_dataset.npz`.

Build the geometry dataset and full-plate meshes:

```bash
./.venv/bin/python -m peh_inverse_design.datasets.build_geometry_dataset \
  --unit-cell-npz data/dataset_100.npz \
  --geometry-output data/geometry_dataset.npz \
  --manifest data/samples.csv \
  --mesh-dir meshes/plates
```

Aggregate per-sample FEM responses:

```bash
./.venv/bin/python -m peh_inverse_design.datasets.build_response_dataset \
  --response-dir data/fem_responses \
  --output data/response_dataset.npz \
  --manifest data/samples.csv
```

Build one integrated dataset after FEM is done:

```bash
./.venv/bin/python -m peh_inverse_design.datasets.build_integrated_dataset \
  --unit-cell-npz data/unit_cell_dataset.npz \
  --response-dir runs/test3/data/fem_responses \
  --modal-dir runs/test3/data/modal_data \
  --mesh-dir runs/test3/meshes/volumes \
  --output runs/test3/data/integrated_dataset.npz
```

This aligned dataset contains, per sample:

- unit-cell geometry fields such as `grf`, `binary`, `sdf`, `threshold`
- FRF outputs such as `f_peak_hz`, `freq_hz`, `freq_ratio`, `voltage_mag`
- modal diagnostics such as `eigenfreq_hz`, `field_frequency_hz`, `top_surface_strain_eqv`
- path indices to the corresponding mesh, response, and modal files

The in-house response files and aggregated datasets now store `voltage_mag` and `peak_voltage` as peak amplitudes only. Legacy RMS-tagged response files are rejected so the stored voltage convention stays explicit and uniform.

Export solid STEP geometry for manual ANSYS Workbench handoff and build the fast Python solver meshes from the same planform:

```bash
./.venv/bin/python -m peh_inverse_design.meshing.build_volume_meshes \
  --unit-cell-npz data/dataset_100.npz \
  --mesh-dir meshes/volumes
```

This writes one set of files per sample under `meshes/volumes/`, including:

- `plate3d_XXXX.step` as the **recommended** combined STEP for ANSYS Workbench
- `plate3d_XXXX_single_face_probe.step` as an optional legacy inspection-only combined STEP when explicitly requested
- `plate3d_XXXX_ansys_face_groups.json` as the face-selection recipe for Workbench electrode/interface grouping
- `plate3d_XXXX_fenicsx.npz` for the in-house solver
- `plate3d_XXXX_cad.json` for the CAD validation report
- `plate3d_XXXX_ansys_workbench.json` for the ANSYS Workbench handoff bundle, including the shared problem specification and expected solid-body layout
- `mesh_build_summary.json` for per-run mesh/CAD success and rejection reasons

By default the ANSYS path stays STEP-only and solid, while the Python path uses the faster `layered_tet` solver mesh backend. That backend meshes a partitioned 2D plate surface with gmsh and extrudes it into a layered tetrahedral mesh for FEniCSx, which is much faster than tetrahedralizing the full 3D STEP body for every sample.

Important ANSYS note:

- use `plate3d_XXXX.step` as the default Workbench geometry; it keeps the substrate/piezo interface conformal so the combined solid is meshable in one file
- the piezo bottom can therefore be split into multiple CAD regions when the substrate pattern is complex; use `plate3d_XXXX_ansys_face_groups.json` instead of manually clicking every fragment for the bottom electrode/interface selection
- `plate3d_XXXX_single_face_probe.step` is off by default because it is inspection-only and can still trigger piezo meshing failures in Workbench
- the CAD report and the Workbench handoff JSON now record the combined-file handoff plus the face-selection recipe explicitly

If you explicitly want the old full 3D gmsh volume-mesh route for the Python solver, switch to the legacy backend:

```bash
./.venv/bin/python -m peh_inverse_design.meshing.build_volume_meshes \
  --unit-cell-npz data/dataset_100.npz \
  --mesh-dir meshes/volumes \
  --solver-mesh-backend gmsh_volume \
  --write-native-msh \
  --write-xdmf
```

The CAD export defaults to `exact` mode:

- preserve the tiled substrate topology exactly
- reject disconnected or under-resolved planforms instead of silently healing them
- export and validate exactly two solid bodies: substrate and piezo
- keep the ANSYS path on solid STEP bodies rather than 2D or reduced-order geometry

Runtime note for the in-house solver:

- the STEP/ANSYS path stays fully solid, but the in-house FEniCSx mesh no longer ties the global in-plane element size to the ~1.27 mm total thickness by default
- CAD validation now uses its own small reference size, while the solver mesh uses the requested in-plane scale
- the default FEniCSx solve now batches all samples in one Docker run, skips already-finished outputs, and uses quadratic solid displacement interpolation (`element_order = 2`) for thin-plate bending accuracy on a much coarser mesh

If you intentionally want repaired CAD for disconnected samples, opt in with:

```bash
./.venv/bin/python -m peh_inverse_design.meshing.build_volume_meshes \
  --unit-cell-npz data/dataset_100.npz \
  --mesh-dir meshes/volumes \
  --repair-cad \
  --repair-bridge-width-m 0.0008
```

That mode adds explicit bridge geometry between disconnected substrate components and records the repair in the CAD report.

Run the FEniCSx modal-reduction solver in the official Docker image:

```bash
./scripts/run_fenicsx_solver.sh \
  --mesh /workspace/meshes/volumes/plate3d_0000_fenicsx.npz \
  --response-dir /workspace/data/fem_responses \
  --modes-dir /workspace/data/modal_data \
  --house-voltage-amplitude-convention peak
```

Manual Workbench handoff stays outside the automated pipeline. The mesh/CAD step still writes `plate3d_XXXX.step`, `plate3d_XXXX_ansys_face_groups.json`, and `plate3d_XXXX_ansys_workbench.json` so you can import the geometry into ANSYS Workbench and scope electrodes/interfaces manually when needed.

Create human-readable summary figures after a run:

```bash
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python peh_inverse_design/viz/visualize_run_outputs.py \
  --dataset data/dataset_100.npz \
  --mesh-dir meshes/volumes \
  --response-dir data/fem_responses \
  --modal-dir data/modal_data \
  --output-dir reports/run_outputs
```

This generates per-sample summary PNGs, a gallery image, and `summary.csv`. The CSV records the modal frequency, FRF peak frequency, and peak voltage directly in peak-amplitude units.

The surface-strain panel in those figures is now explicitly the **piezo top-surface** strain field. Because the patch fully covers the metaplate, that panel is expected to look almost solid in plan view; the visualizer now overlays the tiled substrate footprint so the underlying unit-cell pattern remains visible.

## ANSYS Cross-Validation

Two host-side tools support cross-checking a finished run against ANSYS Workbench. They consume
the per-sample artifacts a pipeline run already produces (modal NPZs, solver-mesh NPZs, STEP
files) and need no Docker. The notebook enables both automatically via
`FRF_PHYSICAL_EXPORT_RANGE_HZ` and `VERIFY_GEOMETRY_PARITY`.

Export each sample's voltage FRF in physical units (Hz, peak volts) over an absolute sweep, so it
lives on the same axis as an ANSYS harmonic sweep instead of the per-sample normalized window:

```bash
./.venv/bin/python -m peh_inverse_design.validation.export_physical_frf \
  --modal-dir runs/0605/data/modal_data \
  --output-dir runs/0605/data/frf_physical \
  --freq-min-hz 0 --freq-max-hz 2 --points 201
```

Verify per sample that the solver mesh and the ANSYS STEP share the same geometry: planform XOR
(symmetric-difference) area, planform area, substrate/piezo volumes, and bounding box, with a
cross-check against `plate3d_XXXX_cad.json`:

```bash
./.venv/bin/python -m peh_inverse_design.validation.geometry_parity \
  --mesh-dir runs/0605/meshes/volumes \
  --output-dir runs/0605/reports/geometry_parity \
  --summary-csv runs/0605/reports/geometry_parity.csv
```

## Quick Test Run

If you want a clean 3-sample test run without mixing outputs with older runs:

```bash
./.venv/bin/python -m peh_inverse_design.datasets.subset_unit_cell_dataset \
  --input data/unit_cell_dataset.npz \
  --output data/test_runs/test3/unit_cell_dataset.npz \
  --limit 3
```

Then run the full pipeline into a dedicated output folder:

```bash
bash scripts/run_all.sh \
  --unit-cell-npz data/test_runs/test3/unit_cell_dataset.npz \
  --limit 3 \
  --run-name test3
```

`run_all.sh` now delegates to the same Python pipeline as the notebook. When `--limit 3` is used without explicit `--sample-ids`, the pipeline requests **3 successful solid exports** and keeps scanning/rejecting candidate unit cells until it has 3 samples that survive the full 3D solid-build stage, or until the candidate dataset is exhausted.

This writes everything under `runs/test3/`:

- `runs/test3/meshes/volumes/`
- `runs/test3/data/fem_responses/`
- `runs/test3/data/modal_data/`
- `runs/test3/data/response_dataset.npz`
- `runs/test3/data/integrated_dataset.npz`
- `runs/test3/reports/`

If you already have a small unit-cell dataset, you can skip the subset step and point `--unit-cell-npz` directly at it.

## Beginner Notebook

If you prefer a notebook workflow over terminal commands, use [integrated_peh_pipeline.ipynb](/home/gijeong/Inverse%20Design/integrated_peh_pipeline.ipynb).

Recommended order:

1. run [periodic_grf_sdf.ipynb](/home/gijeong/Inverse%20Design/periodic_grf_sdf.ipynb)
2. save or reuse the generated unit-cell dataset NPZ
3. open [integrated_peh_pipeline.ipynb](/home/gijeong/Inverse%20Design/integrated_peh_pipeline.ipynb)
4. in cell **① Run & pipeline settings**, update `SOURCE_UNIT_CELL_NPZ`, `RUN_NAME`, and `LIMIT`
5. in cell **② Materials & physics**, edit any material or physical value you need — every property (the full anisotropic piezo matrices, densities, damping, base excitation, load resistance, plate geometry, and the frequency window) is a plain variable here, assembled into `PROBLEM_SPEC`
6. click `Run All`

The notebook calls the same Python pipeline underneath and creates STEP geometry, Python solver meshes, FEM results, the integrated dataset, and report images in one run.

The notebook pipeline now also exposes CAD mode:

- `EXACT_CAD = True`, `REPAIR_CAD = False` rejects disconnected tiled substrates
- `EXACT_CAD = False`, `REPAIR_CAD = True` adds explicit bridge geometry for repair CAD

For closer ANSYS parity in the notebook:

- set `MESH_PRESET = "ansys_parity"`
- leave `SUBSTRATE_LAYERS = None` and `PIEZO_LAYERS = None` so the preset supplies `8` substrate layers and `3` piezo layers together with the uncapped parity mesh profile
- if you explicitly type `SUBSTRATE_LAYERS = 8` and `PIEZO_LAYERS = 3` while still leaving `mesh_preset="default"`, the pipeline now auto-aligns that combination to `ansys_parity` unless you also explicitly set `solver_max_q2_vector_dofs`
- use `EIGENSOLVER_BACKEND = "shift_invert_cholesky"` first; reserve `shift_invert_lu`, `iterative_lobpcg_gamg`, `iterative_gd_gamg`, and `shift_invert_cholesky_ooc` for explicit follow-up runs
- set `SOLVER_MPI_RANKS = 4` for the first distributed Docker solve attempt, then adjust to 2 or 8 if needed
- use `--eigensolver-fallback-backends` only when you intentionally want the pipeline to restart failed isolated solves in fresh Docker containers with named fallback backends

If Step 1 fails, inspect `runs/<RUN_NAME>/meshes/volumes/mesh_build_summary.json` for the exact CAD rejection reasons.

All material and physical properties are now edited directly in the notebook's **② Materials & physics** cell and assembled into a `PROBLEM_SPEC` dictionary that becomes the single source of truth and overrides `configs/peh_inverse_design_spec.yaml`. This includes substrate and piezo stiffness, densities (e.g. `SUBSTRATE_DENSITY_KG_PER_M3`, `PIEZO_DENSITY_KG_PER_M3`), the full 3D anisotropic piezo matrices, modal damping, base excitation, the external load resistance, the plate geometry, and the normalized frequency window (`FREQ_RATIO_MIN`, `FREQ_RATIO_MAX`, `FRF_POINTS`). The YAML file remains as a documented default used by the command-line tools.

Both the notebook and `run_all.sh` use the same `peh_inverse_design.pipeline.pipeline_runner` implementation under the hood.

## ANSYS-Parity Layer Diagnosis

The earlier "voltage collapse" when refining from 2+1 to 8+3 through-thickness layers has been diagnosed as a **memory-limited solver artifact**, not a physics or mesh bug. Under an honest quadratic (Q2) solve only the low-layer case fits in RAM: the 4+2 (9.1 M DOFs) and 8+3 (15.7 M DOFs) cases exceed a 128 GiB machine and are OOM-killed, while the old "24 V" came from a pre-guard run that silently degraded the solve. Full evidence, root cause, and the decision options are recorded in [docs/peh_layer_voltage_diagnosis.md](docs/peh_layer_voltage_diagnosis.md).

Key conclusions (superseding the earlier element-order / coarsening hypothesis):

- The meshes are identical across layer counts (same Q2 order, same material volumes, same in-plane size, zero coarsening); only the DOF count grows, so the collapse is not a mesh defect.
- A shift-invert modal eigensolve needs a sparse direct factorization whose memory grows superlinearly with DOFs; ~5 M DOFs already nears the 128 GiB ceiling, so no "memory handling" flag can make 9-16 M DOFs fit.
- For a 1000:1 thin plate the low-layer Q2 model is already converged for first-mode voltage (~350 V vs ANSYS ~243 V); 8+3 layers is not physically required.
- For ANSYS parity, the modal circuit capacitance uses the full 3D PZT `eps33`, not the reduced-plate value (see the memo for the parity caveat this introduces).
