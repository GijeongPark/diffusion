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
                    ],
                    dtype=np.float64,
                ),
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
            self.assertEqual(summary["top_surface_strain"]["max_strain_eqv"], 123.0)
            self.assertTrue((output_dir / "sample_0000_top_surface_strain.csv").exists())
            self.assertTrue((output_dir / "sample_0000_top_surface_strain_mesh.npz").exists())
            self.assertTrue((output_dir / "sample_0000_audit_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
