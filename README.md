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

## Dataset Utilities

The repository now includes a reusable Python package, [peh_inverse_design](/home/gijeong/Inverse%20Design/peh_inverse_design), for the next pipeline step:

- building `data/geometry_dataset.npz` from the unit-cell notebook output
- generating `10 x 10` tiled full-plate meshes for FEM
- exporting ANSYS Workbench Geometry-compatible `plate3d_*.step` CAD files
- aggregating per-sample FEM outputs into `data/response_dataset.npz`
- collecting everything into one `integrated_dataset.npz`

`dataset_100.npz` is only a legacy filename from an early test. You can use any `.npz` filename for the unit-cell dataset, for example `data/unit_cell_dataset.npz` or `data/test_runs/test3/unit_cell_dataset.npz`.

Build the geometry dataset and full-plate meshes:

```bash
./.venv/bin/python -m peh_inverse_design.build_geometry_dataset \
  --unit-cell-npz data/dataset_100.npz \
  --geometry-output data/geometry_dataset.npz \
  --manifest data/samples.csv \
  --mesh-dir meshes/plates
```

Aggregate per-sample FEM responses:

```bash
./.venv/bin/python -m peh_inverse_design.build_response_dataset \
  --response-dir data/fem_responses \
  --output data/response_dataset.npz \
  --manifest data/samples.csv
```

Build one integrated dataset after FEM is done:

```bash
./.venv/bin/python -m peh_inverse_design.build_integrated_dataset \
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

Export solid STEP geometry for ANSYS and build the fast Python solver meshes from the same planform:

```bash
./.venv/bin/python -m peh_inverse_design.build_volume_meshes \
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
./.venv/bin/python -m peh_inverse_design.build_volume_meshes \
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
./.venv/bin/python -m peh_inverse_design.build_volume_meshes \
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
  --modes-dir /workspace/data/modal_data
```

Create human-readable summary figures after a run:

```bash
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python peh_inverse_design/visualize_run_outputs.py \
  --dataset data/dataset_100.npz \
  --mesh-dir meshes/volumes \
  --response-dir data/fem_responses \
  --modal-dir data/modal_data \
  --output-dir reports/run_outputs
```

This generates per-sample summary PNGs, a gallery image, and `summary.csv`.

The surface-strain panel in those figures is now explicitly the **piezo top-surface** strain field. Because the patch fully covers the metaplate, that panel is expected to look almost solid in plan view; the visualizer now overlays the tiled substrate footprint so the underlying unit-cell pattern remains visible.

## Quick Test Run

If you want a clean 3-sample test run without mixing outputs with older runs:

```bash
./.venv/bin/python -m peh_inverse_design.subset_unit_cell_dataset \
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
4. update `SOURCE_UNIT_CELL_NPZ`, `RUN_NAME`, and `LIMIT`
5. click `Run All`

The notebook calls the same Python pipeline underneath and creates STEP geometry, Python solver meshes, FEM results, the integrated dataset, and report images in one run.

The notebook pipeline now also exposes CAD mode:

- `EXACT_CAD = True`, `REPAIR_CAD = False` rejects disconnected tiled substrates
- `EXACT_CAD = False`, `REPAIR_CAD = True` adds explicit bridge geometry for repair CAD

If Step 1 fails, inspect `runs/<RUN_NAME>/meshes/volumes/mesh_build_summary.json` for the exact CAD rejection reasons.

Density values used by the solver can now be set explicitly:

- `SUBSTRATE_RHO = 7930.0`
- `PIEZO_RHO = 7500.0` or your desired patch density

Both the notebook and `run_all.sh` use the same `peh_inverse_design.pipeline_runner` implementation under the hood.
