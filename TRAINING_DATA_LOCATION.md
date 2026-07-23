# 📦 Where is the 50,000-sample training data?

**One file. This is it:**

```
runs/final50k/integrated_dataset.npz        ← 52,570 training pairs, 7.8 GB
```

(full WSL path: `/home/user/GJ/diffusion/runs/final50k/integrated_dataset.npz`,
or from Windows Explorer:
`\\wsl.localhost\Ubuntu-22.04\home\user\GJ\diffusion\runs\final50k\`)

There are two more copies/variants:

| file | what it is |
|---|---|
| `C:\Users\admin\GJ_transfer\integrated_dataset_final50k.npz` | identical copy on the Windows drive, made for transferring to the main machine. Verified: its sha256 (in the `.sha256` file next to it) matches the original. |
| `runs/final3000_3x3/integrated_dataset.npz` | smaller 3,329-pair subset (474 MB) — handy for quick experiments before committing to the full file. |

## How to load it

```python
import numpy as np
d = np.load("runs/final50k/integrated_dataset.npz", allow_pickle=True)
d["sdf"]          # (52570, 120, 120)  model input: signed distance fields
d["f_peak_hz"]    # (52570,)           conditioning target: resonance frequency
d["voltage_mag"]  # (52570, 256)       target: voltage FRF magnitude
```

Full schema + inspection plots: **notebook `2 integrated_peh_pipeline.ipynb`, §4**.

## Map of this folder (what everything is)

| item | role |
|---|---|
| `1 periodic_grf_sdf.ipynb` | **entry point 1** — generates unit-cell geometry pools |
| `2 integrated_peh_pipeline.ipynb` | **entry point 2** — runs the FEM pipeline, merges & inspects datasets |
| `runs/` | **finished training datasets** (see above) |
| `data/` | geometry pools (~9.5 GB of `.npz`) — the *inputs* the campaign consumed; kept because pools are reusable |
| `peh_inverse_design/`, `scripts/`, `configs/` | the pipeline library + launcher + physics spec that the notebooks call |
| `README.md`, `SETUP.md`, `docs/`, `tests/`, `meshes/` | the upstream repo's own docs/tests |
| `../archive_ops_scripts_20260718/` | ~40 archived one-off scripts from the finished campaign — not needed, safe to delete |
| `../archive_10x10_20260709.tar.gz` | results of the abandoned 10×10-plate campaign (481 pairs) |

The dataset generation campaign finished and passed QA on **2026-07-16**
(52,570 pairs; 0 NaN; unique ids; f_peak 3.56–9.55 Hz; peak voltage
18.5–38.1 V). To generate *more* data, follow the recipe in notebook 2 §2 —
both notebooks are self-sufficient; no extra scripts are required.
