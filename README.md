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
- exporting ANSYS Mechanical-compatible `plate3d_*_ansys.inp` files for External Model import
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

Build 3D layered substrate+piezo meshes for the FEniCSx solver:

```bash
./.venv/bin/python -m peh_inverse_design.build_volume_meshes \
  --unit-cell-npz data/dataset_100.npz \
  --mesh-dir meshes/volumes
```

This writes one set of files per sample under `meshes/volumes/`, including:

- `plate3d_XXXX.msh` for the native gmsh mesh
- `plate3d_XXXX.xdmf` and `plate3d_XXXX_facets.xdmf` for FEniCSx-style inspection
- `plate3d_XXXX_fenicsx.npz` for the in-house solver
- `plate3d_XXXX_ansys.inp` for ANSYS Mechanical External Model import

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

The notebook calls the same Python pipeline underneath and creates meshes, FEM results, the integrated dataset, and report images in one run.

Density values used by the solver can now be set explicitly:

- `SUBSTRATE_RHO = 7930.0`
- `PIEZO_RHO = 7500.0` or your desired patch density
