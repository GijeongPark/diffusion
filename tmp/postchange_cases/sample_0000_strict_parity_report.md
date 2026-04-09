# Sample 0000 Strict Parity Report

Date: 2026-04-09

## Historical baseline

Source: `tmp/parity_exec/default_sample0000/default_sub2_pz1_mesh0p08/audit/sample_0000_audit_summary.json`

| case | mesh preset | mesh backend | eigensolver backend | fallback | mode1 (Hz) | dominant drive-coupling (Hz) | f_peak (Hz) | peak voltage (V) | ANSYS freq error | ANSYS peak-voltage error |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| historical default | `default` | `layered_tet` | `unknown` | unknown | 0.707029186063 | n/a | 0.723582418503 | 342.213627256 | +24.7556% vs 0.58 Hz | +40.8518% vs 242.96 V |

## Post-change strict rerun

Source mesh: `tmp/postchange_cases/ansys_parity_sub8_pz3_mesh0p08/meshes/volumes/plate3d_0000_fenicsx.npz`

| case | mesh preset | mesh backend | eigensolver backend | fallback | mode1 (Hz) | dominant drive-coupling (Hz) | f_peak (Hz) | peak voltage (V) | ANSYS freq error | ANSYS peak-voltage error |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| post-change strict run | `ansys_parity` (`8+3`) | `layered_tet` | `shift_invert_lu` | no fallback permitted; none used | n/a | n/a | n/a | n/a | n/a | n/a |

Strict-parity status: invalid.

Diagnostic: `tmp/postchange_cases/ansys_parity_sub8_pz3_mesh0p08/data/fem_responses/sample_0000_solver_diagnostic.json`

Reason: `shift_invert_lu` failed during MUMPS numerical factorization, and strict parity correctly refused to fall back to `iterative_gd`.

## Ranking / seed outcome

- Combined parity ranking now excludes or heavily penalizes invalid runs, so the failed strict `ansys_parity` candidate is not allowed to win by frequency alone.
- Best run changed after switching ranking and seed: no new valid strict-parity winner was produced in this session because the strict `ansys_parity` case failed hard before modal/FRF outputs were generated.
- Voltage-convention diagnosis: no longer applicable as the default explanation. The code now defaults to ANSYS peak voltage and only surfaces convention-mismatch checks when `unknown` mode is explicitly requested.
