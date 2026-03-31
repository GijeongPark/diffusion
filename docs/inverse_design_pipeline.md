# Conditional Inverse-Design Pipeline for a Cantilevered PEH Plate

This document fixes the current problem setup and defines a practical data and model pipeline for inverse design from a target voltage FRF to a repeated unit-cell geometry.

## 1. Problem Definition

The design variable is **one periodic unit-cell geometry**. That geometry is tiled `10 x 10` times to build a **finite cantilevered plate**, and the global voltage response of the full plate is used as the target signal.

This is **not** an infinite periodic-cell problem. The final forward model should therefore evaluate the **full repeated plate with cantilever boundary conditions**, not a single cell with Bloch-periodic dynamics.

## 2. Fixed Physical Setup

### Geometry

| Quantity | Value |
| --- | --- |
| Unit-cell size | `0.10 m x 0.10 m` |
| Number of repeated cells | `10 x 10` |
| Total plate size | `1.00 m x 1.00 m` |
| Substrate thickness `h_s` | `1.0 mm` |
| Piezo patch thickness `h_pz` | `0.2667 mm` |
| Piezo coverage | Fully covered patch |

### Substrate material

| Quantity | Value |
| --- | --- |
| Young's modulus `E` | `193.05 GPa` |
| Poisson ratio `nu` | `0.30` |
| Density `rho` | `7930 kg/m^3` |

### Piezoelectric data currently fixed

| Quantity | Value |
| --- | --- |
| Reduced piezoelectric constant `e_31` | `-23.38 C/m^2` |
| Dielectric permittivity at constant strain `eps_33^S` | `17.29e-09 F/m` |
| Polarization direction | Thickness direction (`z`, axis 3) |

For the 3D coupled FEM, use the full ANSYS material data below in a single consistent constitutive model.

Stress-charge form:

`sigma = c^E * epsilon - e^T * E`

`D = e * epsilon + eps^S * E`

Full piezoelectric matrix `e` (stress/electric-field form, poling in `z`):

```text
[[ 0.0,          0.0,         -6.622811852 ],
 [ 0.0,          0.0,         -6.622811852 ],
 [ 0.0,          0.0,         23.24031303  ],
 [ 0.0,          0.0,          0.0         ],
 [ 0.0,         17.03448276,   0.0         ],
 [17.03448276,   0.0,          0.0         ]]
```

Full stiffness matrix at constant electric field `c^E`:

```text
[[1.27205e11, 8.0212213391e10, 8.4670187076e10, 0.0,         0.0,         0.0        ],
 [8.0212213391e10, 1.27205e11, 8.4670187076e10, 0.0,         0.0,         0.0        ],
 [8.4670187076e10, 8.4670187076e10, 1.17436e11, 0.0,         0.0,         0.0        ],
 [0.0,         0.0,         0.0,         2.3474178404e10, 0.0,         0.0        ],
 [0.0,         0.0,         0.0,         0.0,         2.30e10,     0.0        ],
 [0.0,         0.0,         0.0,         0.0,         0.0,         2.30e10    ]]
```

Permittivity at constant strain `eps^S`:

```text
[[1.50911e-08, 0.0,         0.0        ],
 [0.0,         1.50911e-08, 0.0        ],
 [0.0,         0.0,         1.26934e-08]]
```

Important note: keep the reduced plate constants only for analytical reduced-order derivations. For the 3D FEM dataset used to train the diffusion pipeline, reuse the full ANSYS constitutive data consistently for every sample.

### Mechanical and electrical conditions

| Quantity | Value |
| --- | --- |
| Structural boundary condition | Clamped-free-free-free cantilevered plate |
| Base excitation | Harmonic support acceleration applied at the clamped base in `z` |
| Base acceleration magnitude | `2.5 m/s^2` |
| Damping model | Modal superposition |
| Modal damping ratio | `0.025` |
| Measured response | Total electrode voltage magnitude |
| External load resistance | `10 kOhm` |
| Peak of interest | Fundamental-mode resonance peak |

Interpretation of base acceleration for this problem:

- the clamped edge is attached to a moving base
- that base undergoes harmonic motion in the `z` direction
- the imposed support acceleration is

`a_base(t) = 2.5 * sin(omega * t) [m/s^2]`

- the plate response is measured relative to that support excitation

In an FEM implementation this can be handled in either of two equivalent ways:

1. impose the support/base motion at the clamped boundary
2. transform to relative coordinates and apply the equivalent inertia loading associated with the support acceleration

Since your analytical model already uses a clamped base with harmonic support excitation, the FEM should follow the same interpretation.

## 3. Modeling Interpretation

### 3.1 Global model, not cell-only dynamics

The geometry generator may still work at the unit-cell level, but the target FRF is the response of the **entire tiled cantilever plate**:

1. Generate one periodic unit-cell geometry.
2. Tile it `10 x 10` to construct the full plate topology.
3. Apply clamped boundary conditions on one outer plate edge.
4. Apply support/base acceleration in the `z` direction.
5. Compute the total electrode voltage over the fully covered piezo patch.

### 3.2 Recommended geometry representation

For learning, use the **SDF** as the primary design representation rather than the raw mesh:

`target FRF -> conditional diffusion -> unit-cell SDF -> threshold/postprocess -> tile 10x10 -> gmsh -> FEM verification`

This keeps the generative model smooth and mesh-agnostic while preserving compatibility with the existing GRF-to-SDF and `gmsh` workflow.

### 3.3 Frequency-axis decision

Use the user-selected choice:

- normalized frequency axis: `f / f_peak in [0.9, 1.1]`
- save `f_peak` separately as a scalar conditioning variable

Recommended addition:

- also store the original absolute frequency samples used by the FEM solve for traceability and later ablation studies

## 4. Training Data Specification

## 4.1 Geometry dataset

The current notebook already produces most of this information. Keep the existing fields and add a model-ready SDF tensor.

Recommended file: `data/geometry_dataset.npz`

| Key | Shape | Description |
| --- | --- | --- |
| `sample_id` | `(N,)` | Integer sample identifier |
| `grf` | `(N, H_grf, W_grf)` | Raw Gaussian random field |
| `binary` | `(N, H_grf, W_grf)` | Solid/void unit-cell map |
| `sdf` | `(N, H_sdf, W_sdf)` | Native SDF from the current notebook |
| `sdf_ml` | `(N, 128, 128)` | Resampled SDF for ML |
| `volume_fraction` | `(N,)` | Solid fraction |
| `threshold` | `(N,)` | GRF threshold used to define geometry |
| `cell_size_m` | `(N, 2)` | Always `[0.1, 0.1]` for now |
| `tile_counts` | `(N, 2)` | Always `[10, 10]` for now |

Recommendation: keep the native meshing SDF resolution if it is convenient, but store an ML-friendly `128 x 128` tensor because diffusion backbones work much more naturally on powers of two than on the current `120 x 120`.

## 4.2 FEM response dataset

Recommended file: `data/response_dataset.npz`

| Key | Shape | Description |
| --- | --- | --- |
| `sample_id` | `(N,)` | Matches the geometry dataset |
| `f_peak_hz` | `(N,)` | Fundamental resonance frequency of the full tiled plate |
| `freq_ratio` | `(N, N_f)` | Normalized grid in `[0.9, 1.1]` |
| `freq_hz` | `(N, N_f)` | Absolute frequencies used in the solve |
| `voltage_mag` | `(N, N_f)` | Magnitude of the voltage response |
| `peak_voltage` | `(N,)` | Maximum voltage in the stored band |
| `quality_flag` | `(N,)` | Mesh/FEM success flag |

Recommended defaults:

- `N_f = 256`
- denser sampling near the peak if modal superposition makes this cheap
- always save the same number of samples per curve

## 4.3 Metadata manifest

Recommended file: `data/samples.csv`

Columns:

- `sample_id`
- `mesh_path`
- `xdmf_path`
- `geometry_npz_key`
- `response_npz_key`
- `mesh_ok`
- `fem_ok`
- `split`

This makes it much easier to handle re-runs, failed FEM jobs, and train/validation/test partitioning without rewriting arrays.

## 5. Forward Data-Generation Pipeline

The recommended end-to-end pipeline is:

1. Generate periodic unit-cell fields with the current GRF-to-SDF notebook.
2. Convert the unit-cell SDF to a clean binary geometry.
3. Tile the unit cell `10 x 10` to build the finite plate geometry.
4. Mesh the tiled plate with `gmsh`.
5. Run the in-house FEM with:
   - clamped outer edge
   - `z`-direction base excitation
   - modal superposition
   - damping ratio `0.025`
   - fully covered piezo layer
   - external load `10 kOhm`
6. Extract the fundamental peak `f_peak`.
7. Save `|V(f)|` on `f/f_peak in [0.9, 1.1]`.
8. Save `f_peak` as a separate scalar label.

The implemented mesh generator currently writes these physical groups for FEM:

- `solid` for all 2D material regions
- `clamped` for the outer `x = 0` boundary segments
- `free_x_max` for the outer `x = Lx` boundary segments
- `free_y_min` for the outer `y = 0` boundary segments
- `free_y_max` for the outer `y = Ly` boundary segments

The current FEniCSx implementation in this repository uses a 3D layered mesh and computes the voltage FRF through:

- structural eigenmodes of the full substrate+piezo plate
- modal superposition with damping ratio `0.025`
- one electrical DOF for the fully covered electrode pair
- resistor coupling with `R = 10 kOhm`

Current implementation note:

- the solver currently assumes `piezo density = 7800 kg/m^3` unless overridden at runtime
- the mesh builder now emits a **combined one-file** ANSYS handoff by default:
  - `plate3d_XXXX.step`: combined metaplate + piezo STEP with a conformal, meshable interface
  - `plate3d_XXXX_single_face_probe.step`: optional legacy combined-file probe when explicitly requested
  - `plate3d_XXXX_ansys_face_groups.json`: face-selection recipe for bottom electrode/interface grouping in Workbench
- the CAD STEP export now defaults to `exact` mode: preserve the tiled substrate topology, reject disconnected or under-resolved planforms, and require exactly two watertight solids after STEP round-trip validation
- for Workbench meshing, prefer `plate3d_XXXX.step`; if the piezo bottom imports as many CAD regions, use the face-group recipe instead of manually selecting every fragment
- if disconnected samples must still be exported for manufacturing-style repair, use the explicit `repair` mode to add real connector geometry rather than a global morphological closing
- each CAD export writes `plate3d_XXXX_cad.json`, `plate3d_XXXX_ansys_workbench.json`, and `plate3d_XXXX_ansys_face_groups.json`, and each run writes `mesh_build_summary.json`, so rejected samples and Workbench scoping rules are traceable instead of failing later in the solver stage
- the in-house 3D solver mesh is now intentionally decoupled from the total thickness: CAD validation uses a small reference size, while the FEniCSx mesh uses the requested in-plane scale unless you explicitly restore thickness-limited meshing
- the default FEniCSx batch solve now runs all samples in one container invocation, skips already-finished outputs, and uses quadratic solid interpolation for better thin-plate bending accuracy at practical mesh sizes

Result-visualization note:

- the report strain panel is the **piezo top-surface** strain field, not the exposed metaplate top surface
- because the piezo patch fully covers the plate, the raw top view can look almost solid even when the substrate contains a strong unit-cell pattern underneath
- the visualizer therefore overlays the tiled substrate footprint on top of the piezo-surface strain/mesh plot so the geometry and the response panel stay interpretable together

## 6. Recommended ML Pipeline

## 6.1 Why diffusion is appropriate here

This inverse problem is naturally **one-to-many**: multiple geometries can produce similar FRFs. A conditional diffusion model is a good fit because it can return multiple candidate unit cells for the same target curve.

## 6.2 Recommended model structure

Use a **latent conditional diffusion model** instead of direct pixel-space diffusion.

### Geometry side

1. Train an autoencoder on `sdf_ml`.
2. Compress each unit-cell SDF to a latent tensor.
3. Decode the latent back to an SDF.

### Condition side

Encode the target response with:

- `voltage_mag(f/f_peak)` as a 1D sequence
- `f_peak_hz` as an extra scalar

A practical conditioning encoder is:

- 1D CNN or small Transformer for `voltage_mag`
- MLP for `f_peak_hz`
- concatenate the two embeddings into one conditioning vector

### Diffusion side

Train a conditional latent diffusion model to predict the latent SDF from:

- noisy latent geometry
- FRF embedding
- `f_peak_hz` embedding

This is the most robust path if the dataset size is moderate.

## 6.3 Why not generate meshes directly

Direct mesh generation is possible, but it adds unnecessary complexity:

- variable topology
- element-quality issues
- hard geometric validity constraints

SDF generation is smoother, easier to condition, and already matches your current geometry pipeline.

## 7. Inference and Re-Ranking

At inference time:

1. Input the target `voltage_mag` and `f_peak_hz`.
2. Sample multiple candidate unit-cell SDFs from the conditional diffusion model.
3. Threshold and postprocess each SDF.
4. Tile the candidate cell `10 x 10`.
5. Re-mesh and run FEM on each candidate.
6. Re-rank by FRF mismatch and geometry validity.

Recommended re-ranking score:

`score = w1 * relative_L2_error(voltage_mag) + w2 * peak_frequency_error + w3 * geometry_penalty`

The geometry penalty can include:

- disconnected solids
- minimum feature-size violations
- FEM/meshing failure

## 8. Data Split Strategy

Split by **sample identity**, not by repeated FRF windows derived from the same geometry. Avoid leakage between train/validation/test sets.

Recommended split:

- `70%` train
- `15%` validation
- `15%` test

If later you introduce multiple operating conditions for the same geometry, keep all condition variants of the same geometry in the same split.

## 9. Practical Recommendations for the Next Implementation Step

The most efficient next coding step is:

1. keep the current unit-cell GRF and SDF generation
2. add a tiled-plate geometry builder
3. export both:
   - unit-cell SDF for ML
   - full-plate mesh for FEM
4. standardize the FEM output into `response_dataset.npz`

Only after that should the diffusion model be trained.

## 10. Open Technical Items to Fix Once

These items should be fixed once before large-scale data generation:

1. Decide the exact constitutive model used by the in-house FEM:
   - reduced plate model
   - or full 3D coupled piezoelectric model
2. Fix the exact support-acceleration implementation:
   - imposed support acceleration
   - or equivalent inertial loading in modal coordinates
3. Fix the FEM frequency sampling rule around the fundamental peak.

The rest of the pipeline can remain unchanged after those are set.
