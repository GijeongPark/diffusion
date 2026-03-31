# Runtime fix summary (v5)

## Why Step 1/5 suddenly felt much slower after v4

The v4 Workbench fix unintentionally introduced a performance regression in the mesh-export stage.

For every sample, the layered-tet path was doing **two full CAD export cycles** before it even started the solver-mesh build:

- one export for the new default meshable STEP (`plate3d_XXXX.step`)
- one more export for the optional single-face inspection STEP (`plate3d_XXXX_single_face_probe.step`)

Each export cycle included:

- OCC model build
- STEP write
- STEP re-import
- roundtrip validation

So Step 1/5 was paying roughly an extra per-sample CAD roundtrip cost even when the inspection-only STEP was not actually needed.

A second issue made this feel worse in the notebook: the subprocess output from `build_volume_meshes` could be buffered, so the notebook often showed only the Step 1/5 banner for a long time instead of streaming the per-sample progress lines.

## What I changed

### 1. Made the inspection-only STEP opt-in instead of always-on

The single-face probe STEP is now **disabled by default**.

That means the standard notebook/pipeline path now builds only:

- the meshable Workbench STEP
- the Python solver mesh

The inspection-only file is still supported, but only when explicitly requested via `export_inspection_single_face_step=True` in `PipelineConfig` or `--export-inspection-single-face-step` in `build_volume_meshes`.

### 2. Enabled unbuffered subprocess logging for pipeline Python steps

`pipeline_runner.py` now launches Python subprocesses with `PYTHONUNBUFFERED=1`, so the notebook should show progress messages from Step 1/5 as they are produced instead of appearing frozen until the subprocess exits.

### 3. Forced the mesh-build progress prints to flush

`build_volume_meshes.py` now flushes the most important progress lines immediately:

- resolved plate dimensions
- per-sample success/failure counters
- final mesh-build summary

## Expected effect

- Step 1/5 should be noticeably faster again for the normal workflow because it no longer generates the inspection-only STEP unless you explicitly ask for it.
- The notebook should now show live progress instead of sitting at the Step 1/5 banner with no visible activity.
- The Workbench robustness fix from v4 is preserved because the default exported STEP is still the partitioned, meshable one.

## Files changed in this round

- `peh_inverse_design/volume_mesh.py`
- `peh_inverse_design/build_volume_meshes.py`
- `peh_inverse_design/pipeline_runner.py`
- `2 integrated_peh_pipeline.ipynb`
- `README.md`
- `docs/inverse_design_pipeline.md`
- `docs/runtime_fix_summary_v5.md`
- `docs/runtime_fix_summary.md`
