from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from peh_inverse_design.audit_ansys_alignment import audit_run_sample


class AuditAnsysAlignmentTests(unittest.TestCase):
    def test_audit_run_sample_exports_summary_and_surface_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "demo"
            mesh_dir = run_dir / "meshes" / "volumes"
            response_dir = run_dir / "data" / "fem_responses"
            modal_dir = run_dir / "data" / "modal_data"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            response_dir.mkdir(parents=True, exist_ok=True)
            modal_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "data" / "problem_spec_used.yaml").write_text(
                "\n".join(
                    [
                        "geometry:",
                        "  substrate_thickness_m: 0.001",
                        "  piezo_patch_thickness_m: 0.0001",
                        "materials:",
                        "  substrate:",
                        "    density_kg_per_m3: 7930.0",
                        "  piezoelectric:",
                        "    density_kg_per_m3: 7500.0",
                        "mechanics:",
                        "  damping:",
                        "    modal_damping_ratio: 0.025",
                        "  base_excitation:",
                        "    amplitude_m_per_s2: 2.5",
                        "electrical:",
                        "  external_load_resistance_ohm: 10000.0",
                    ]
                ),
                encoding="utf-8",
            )

            np.savez_compressed(
                mesh_dir / "plate3d_0000_fenicsx.npz",
                points=np.asarray(
                    [
                        [0.0, 0.0, 0.0011],
                        [1.0, 0.0, 0.0011],
                        [1.0, 1.0, 0.0011],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
                tetra_cells=np.asarray([[3, 0, 1, 2]], dtype=np.int64),
                tetra_tags=np.asarray([12], dtype=np.int32),
                triangle_cells=np.asarray([[0, 1, 2]], dtype=np.int32),
                triangle_tags=np.asarray([105], dtype=np.int32),
            )
            np.savez_compressed(
                response_dir / "sample_0000_response.npz",
                sample_id=np.asarray(0, dtype=np.int32),
                f_peak_hz=np.asarray(0.5, dtype=np.float64),
                freq_hz=np.asarray([0.45, 0.5, 0.55], dtype=np.float64),
                voltage_mag=np.asarray([1.0, 2.0, 1.5], dtype=np.float64),
                quality_flag=np.asarray(1, dtype=np.int32),
            )
            np.savez_compressed(
                modal_dir / "sample_0000_modal.npz",
                sample_id=np.asarray(0, dtype=np.int32),
                eigenfreq_hz=np.asarray([0.48], dtype=np.float64),
                field_frequency_hz=np.asarray(0.5, dtype=np.float64),
                top_surface_strain_eqv=np.asarray([123.0], dtype=np.float64),
            )
            (mesh_dir / "plate3d_0000_ansys_workbench.json").write_text(
                json.dumps(
                    {
                        "geometry": {
                            "plate_size_m": [1.0, 1.0],
                            "substrate_thickness_m": 0.001,
                            "piezo_thickness_m": 0.0001,
                            "total_thickness_m": 0.0011,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (mesh_dir / "plate3d_0000_ansys_face_groups.json").write_text(
                json.dumps(
                    {
                        "selection_strategy": "worksheet_by_body_and_z_plane",
                        "named_selection_recipes": {
                            "piezo_bottom_electrode": {
                                "expected_region_count": 1,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            output_dir = run_dir / "audit"
            summary = audit_run_sample(run_dir=run_dir, sample_id=0, output_dir=output_dir)

            self.assertEqual(summary["frequency_comparison"]["ansys_modal_target_hz"], 0.48)
            self.assertEqual(summary["frequency_comparison"]["f_peak_hz"], 0.5)
            self.assertEqual(summary["voltage_comparison"]["peak_voltage_v"], 2.0)
            self.assertEqual(summary["voltage_comparison"]["peak_voltage_peak_v"], 2.0)
            self.assertAlmostEqual(summary["voltage_comparison"]["peak_voltage_rms_v"], 2.0 / np.sqrt(2.0))
            self.assertEqual(summary["top_surface_strain"]["max_strain_eqv"], 123.0)
            self.assertEqual(summary["top_surface_strain"]["field_name"], "top_surface_strain_eqv")
            self.assertTrue((output_dir / "sample_0000_top_surface_strain.csv").exists())
            self.assertTrue((output_dir / "sample_0000_top_surface_strain_mesh.npz").exists())
            self.assertTrue((output_dir / "sample_0000_audit_summary.json").exists())

    def test_audit_prefers_mode1_surface_strain_when_explicit_field_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "demo"
            mesh_dir = run_dir / "meshes" / "volumes"
            response_dir = run_dir / "data" / "fem_responses"
            modal_dir = run_dir / "data" / "modal_data"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            response_dir.mkdir(parents=True, exist_ok=True)
            modal_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "data" / "problem_spec_used.yaml").write_text(
                "\n".join(
                    [
                        "geometry:",
                        "  substrate_thickness_m: 0.001",
                        "  piezo_patch_thickness_m: 0.0001",
                        "materials:",
                        "  substrate:",
                        "    density_kg_per_m3: 7930.0",
                        "  piezoelectric:",
                        "    density_kg_per_m3: 7500.0",
                        "mechanics:",
                        "  damping:",
                        "    modal_damping_ratio: 0.025",
                        "  base_excitation:",
                        "    amplitude_m_per_s2: 2.5",
                        "electrical:",
                        "  external_load_resistance_ohm: 10000.0",
                    ]
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                mesh_dir / "plate3d_0000_fenicsx.npz",
                points=np.asarray(
                    [
                        [0.0, 0.0, 0.0011],
                        [0.1, 0.0, 0.0011],
                        [0.1, 1.0, 0.0011],
                        [0.9, 0.0, 0.0011],
                        [1.0, 0.0, 0.0011],
                        [1.0, 1.0, 0.0011],
                        [0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0],
                        [0.1, 1.0, 0.0],
                        [0.9, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
                tetra_cells=np.asarray(
                    [
                        [6, 7, 8, 2],
                        [9, 10, 11, 5],
                    ],
                    dtype=np.int64,
                ),
                tetra_tags=np.asarray([12, 12], dtype=np.int32),
                triangle_cells=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
                triangle_tags=np.asarray([105, 105], dtype=np.int32),
            )
            np.savez_compressed(
                response_dir / "sample_0000_response.npz",
                sample_id=np.asarray(0, dtype=np.int32),
                f_peak_hz=np.asarray(0.5, dtype=np.float64),
                freq_hz=np.asarray([0.45, 0.5, 0.55], dtype=np.float64),
                voltage_mag=np.asarray([1.0, 2.0, 1.5], dtype=np.float64),
                quality_flag=np.asarray(1, dtype=np.int32),
            )
            np.savez_compressed(
                modal_dir / "sample_0000_modal.npz",
                sample_id=np.asarray(0, dtype=np.int32),
                eigenfreq_hz=np.asarray([0.48], dtype=np.float64),
                mode1_frequency_hz=np.asarray(0.48, dtype=np.float64),
                mode1_top_surface_strain_eqv=np.asarray([10.0, 1.0], dtype=np.float64),
                harmonic_field_frequency_hz=np.asarray(0.5, dtype=np.float64),
                harmonic_top_surface_strain_eqv=np.asarray([2.0, 20.0], dtype=np.float64),
                top_surface_strain_eqv=np.asarray([2.0, 20.0], dtype=np.float64),
                modal_theta=np.asarray([0.01], dtype=np.float64),
                modal_force=np.asarray([0.02], dtype=np.float64),
                modal_mass=np.asarray([1.0], dtype=np.float64),
                capacitance_f=np.asarray([3.0e-7], dtype=np.float64),
            )
            (mesh_dir / "plate3d_0000_ansys_workbench.json").write_text(
                json.dumps(
                    {
                        "geometry": {
                            "plate_size_m": [1.0, 1.0],
                            "substrate_thickness_m": 0.001,
                            "piezo_thickness_m": 0.0001,
                            "total_thickness_m": 0.0011,
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = audit_run_sample(run_dir=run_dir, sample_id=0)

            self.assertEqual(summary["top_surface_strain"]["field_name"], "mode1_top_surface_strain_eqv")
            self.assertLess(summary["top_surface_strain"]["top_1pct_x_mean_m"], 0.5)
            self.assertIn("harmonic_top_surface_strain", summary)
            self.assertGreater(summary["harmonic_top_surface_strain"]["tip_mean_strain_eqv"], summary["harmonic_top_surface_strain"]["root_mean_strain_eqv"])

    def test_audit_detects_rms_vs_peak_voltage_mismatch_and_single_frequency_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "demo"
            mesh_dir = run_dir / "meshes" / "volumes"
            response_dir = run_dir / "data" / "fem_responses"
            modal_dir = run_dir / "data" / "modal_data"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            response_dir.mkdir(parents=True, exist_ok=True)
            modal_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "data" / "problem_spec_used.yaml").write_text(
                "\n".join(
                    [
                        "geometry:",
                        "  substrate_thickness_m: 0.001",
                        "  piezo_patch_thickness_m: 0.0001",
                        "materials:",
                        "  substrate:",
                        "    density_kg_per_m3: 7930.0",
                        "  piezoelectric:",
                        "    density_kg_per_m3: 7500.0",
                        "mechanics:",
                        "  damping:",
                        "    modal_damping_ratio: 0.025",
                        "  base_excitation:",
                        "    amplitude_m_per_s2: 2.5",
                        "electrical:",
                        "  external_load_resistance_ohm: 10000.0",
                    ]
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                mesh_dir / "plate3d_0000_fenicsx.npz",
                points=np.asarray(
                    [
                        [0.0, 0.0, 0.0011],
                        [1.0, 0.0, 0.0011],
                        [1.0, 1.0, 0.0011],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
                tetra_cells=np.asarray([[3, 0, 1, 2]], dtype=np.int64),
                tetra_tags=np.asarray([12], dtype=np.int32),
                triangle_cells=np.asarray([[0, 1, 2]], dtype=np.int32),
                triangle_tags=np.asarray([105], dtype=np.int32),
            )
            np.savez_compressed(
                response_dir / "sample_0000_response.npz",
                sample_id=np.asarray(0, dtype=np.int32),
                f_peak_hz=np.asarray(0.5, dtype=np.float64),
                freq_hz=np.asarray([0.45, 0.5, 0.55], dtype=np.float64),
                voltage_mag=np.asarray([1.0, 2.0, 1.5], dtype=np.float64),
                quality_flag=np.asarray(1, dtype=np.int32),
            )
            np.savez_compressed(
                modal_dir / "sample_0000_modal.npz",
                sample_id=np.asarray(0, dtype=np.int32),
                eigenfreq_hz=np.asarray([0.48], dtype=np.float64),
                mode1_frequency_hz=np.asarray(0.48, dtype=np.float64),
                harmonic_field_frequency_hz=np.asarray(0.5, dtype=np.float64),
                mode1_top_surface_strain_eqv=np.asarray([10.0], dtype=np.float64),
                harmonic_top_surface_strain_eqv=np.asarray([12.0], dtype=np.float64),
                top_surface_strain_eqv=np.asarray([12.0], dtype=np.float64),
            )
            (mesh_dir / "plate3d_0000_ansys_workbench.json").write_text(
                json.dumps(
                    {
                        "geometry": {
                            "plate_size_m": [1.0, 1.0],
                            "substrate_thickness_m": 0.001,
                            "piezo_thickness_m": 0.0001,
                            "total_thickness_m": 0.0011,
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = audit_run_sample(
                run_dir=run_dir,
                sample_id=0,
                ansys_modal_hz=0.58,
                ansys_voltage_v=2.0 / np.sqrt(2.0),
                ansys_voltage_form="unknown",
            )

            self.assertTrue(summary["frequency_comparison"]["frequency_reference_ambiguous"])
            self.assertAlmostEqual(summary["frequency_comparison"]["ambiguous_frequency_reference_hz"], 0.58)
            self.assertTrue(summary["voltage_comparison"]["voltage_convention_mismatch_likely"])
            self.assertAlmostEqual(summary["voltage_comparison"]["error_percent_assuming_ansys_rms"], 0.0, places=9)
            self.assertGreater(abs(summary["voltage_comparison"]["error_percent_assuming_ansys_peak"]), 40.0)

    def test_audit_defaults_to_peak_voltage_form(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "demo"
            mesh_dir = run_dir / "meshes" / "volumes"
            response_dir = run_dir / "data" / "fem_responses"
            modal_dir = run_dir / "data" / "modal_data"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            response_dir.mkdir(parents=True, exist_ok=True)
            modal_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "data" / "problem_spec_used.yaml").write_text(
                "\n".join(
                    [
                        "geometry:",
                        "  substrate_thickness_m: 0.001",
                        "  piezo_patch_thickness_m: 0.0001",
                        "mechanics:",
                        "  damping:",
                        "    modal_damping_ratio: 0.025",
                        "  base_excitation:",
                        "    amplitude_m_per_s2: 2.5",
                        "electrical:",
                        "  external_load_resistance_ohm: 10000.0",
                    ]
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                mesh_dir / "plate3d_0000_fenicsx.npz",
                points=np.asarray(
                    [
                        [0.0, 0.0, 0.0011],
                        [1.0, 0.0, 0.0011],
                        [1.0, 1.0, 0.0011],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
                tetra_cells=np.asarray([[3, 0, 1, 2]], dtype=np.int64),
                tetra_tags=np.asarray([12], dtype=np.int32),
                triangle_cells=np.asarray([[0, 1, 2]], dtype=np.int32),
                triangle_tags=np.asarray([105], dtype=np.int32),
            )
            np.savez_compressed(
                response_dir / "sample_0000_response.npz",
                sample_id=np.asarray(0, dtype=np.int32),
                f_peak_hz=np.asarray(0.5, dtype=np.float64),
                freq_hz=np.asarray([0.45, 0.5, 0.55], dtype=np.float64),
                voltage_mag=np.asarray([1.0, 2.0, 1.5], dtype=np.float64),
                quality_flag=np.asarray(1, dtype=np.int32),
            )
            np.savez_compressed(
                modal_dir / "sample_0000_modal.npz",
                sample_id=np.asarray(0, dtype=np.int32),
                eigenfreq_hz=np.asarray([0.48], dtype=np.float64),
                mode1_frequency_hz=np.asarray(0.48, dtype=np.float64),
                harmonic_field_frequency_hz=np.asarray(0.5, dtype=np.float64),
                top_surface_strain_eqv=np.asarray([12.0], dtype=np.float64),
            )
            (mesh_dir / "plate3d_0000_ansys_workbench.json").write_text(
                json.dumps(
                    {
                        "geometry": {
                            "plate_size_m": [1.0, 1.0],
                            "substrate_thickness_m": 0.001,
                            "piezo_thickness_m": 0.0001,
                            "total_thickness_m": 0.0011,
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = audit_run_sample(
                run_dir=run_dir,
                sample_id=0,
                ansys_voltage_v=2.0,
            )

            self.assertEqual(summary["voltage_comparison"]["ansys_voltage_form"], "peak")
            self.assertAlmostEqual(summary["voltage_comparison"]["selected_voltage_error_percent"], 0.0, places=9)
            self.assertFalse(summary["voltage_comparison"]["voltage_convention_mismatch_likely"])


if __name__ == "__main__":
    unittest.main()
