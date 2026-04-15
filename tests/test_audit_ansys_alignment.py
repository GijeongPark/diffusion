from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from peh_inverse_design.audit_ansys_alignment import audit_run_sample
from peh_inverse_design.response_dataset import save_fem_response


class AuditAnsysAlignmentTests(unittest.TestCase):
    def _build_minimal_run(self, root: Path) -> Path:
        run_dir = root / "runs" / "demo"
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
                    "electrical:",
                    "  external_load_resistance_ohm: 10000.0",
                    "  house_voltage_amplitude_convention: peak",
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
                    [0.0, 1.0, 0.0011],
                    [0.0, 0.0, 0.0010],
                    [0.0, 0.0, 0.0000],
                ],
                dtype=np.float64,
            ),
            tetra_cells=np.asarray(
                [
                    [3, 0, 1, 2],
                    [4, 3, 1, 2],
                ],
                dtype=np.int64,
            ),
            tetra_tags=np.asarray([12, 11], dtype=np.int32),
            triangle_cells=np.asarray([[0, 1, 2]], dtype=np.int32),
            triangle_tags=np.asarray([105], dtype=np.int32),
        )
        save_fem_response(
            sample_id=0,
            f_peak_hz=0.60,
            freq_hz=np.asarray([0.55, 0.60, 0.65], dtype=np.float64),
            voltage_mag=np.asarray([1.0, 2.0, 1.5], dtype=np.float64),
            output_dir=response_dir,
        )
        np.savez_compressed(
            modal_dir / "sample_0000_modal.npz",
            sample_id=np.asarray(0, dtype=np.int32),
            eigenfreq_hz=np.asarray([0.58], dtype=np.float64),
            mode1_frequency_hz=np.asarray(0.58, dtype=np.float64),
            harmonic_field_frequency_hz=np.asarray(0.60, dtype=np.float64),
            mode1_top_surface_strain_eqv=np.asarray([120.0], dtype=np.float64),
            harmonic_top_surface_strain_eqv=np.asarray([123.0], dtype=np.float64),
            top_surface_strain_eqv=np.asarray([123.0], dtype=np.float64),
            modal_theta=np.asarray([0.2], dtype=np.float64),
            modal_force=np.asarray([0.1], dtype=np.float64),
            modal_mass=np.asarray([1.0], dtype=np.float64),
            capacitance_f=np.asarray([5.0e-7], dtype=np.float64),
            capacitance_eps33s_f_per_m=np.asarray([1.729e-8], dtype=np.float64),
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
        return run_dir

    def test_audit_run_sample_exports_summary_and_compares_peak_voltage_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._build_minimal_run(Path(tmpdir))
            output_dir = Path(tmpdir) / "exports"

            summary = audit_run_sample(
                run_dir=run_dir,
                sample_id=0,
                output_dir=output_dir,
                ansys_modal_hz=0.58,
                ansys_frf_peak_hz=0.60,
                ansys_voltage_v=2.0,
                ansys_voltage_form="peak",
            )

            self.assertAlmostEqual(summary["frequency_comparison"]["mode1_frequency_hz"], 0.58)
            self.assertAlmostEqual(summary["frequency_comparison"]["f_peak_hz"], 0.60)
            self.assertAlmostEqual(summary["voltage_comparison"]["peak_voltage_peak_v"], 2.0)
            self.assertAlmostEqual(summary["voltage_comparison"]["selected_voltage_error_percent"], 0.0)
            self.assertNotIn("peak_voltage_rms_v", summary["voltage_comparison"])
            self.assertNotIn("error_percent_assuming_ansys_rms", summary["voltage_comparison"])
            self.assertAlmostEqual(summary["electromechanical"]["capacitance_f"], 5.0e-7)
            self.assertAlmostEqual(summary["electromechanical"]["capacitance_eps33s_f_per_m"], 1.729e-8)
            self.assertEqual(summary["ansys_face_groups"]["piezo_bottom_expected_region_count"], 1)

            csv_path = Path(summary["exports"]["top_surface_csv_path"])
            self.assertTrue(csv_path.exists())
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["strain_eqv"], "120")

            summary_path = Path(summary["exports"]["audit_summary_path"])
            self.assertTrue(summary_path.exists())

    def test_audit_flags_single_frequency_reference_as_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._build_minimal_run(Path(tmpdir))

            summary = audit_run_sample(
                run_dir=run_dir,
                sample_id=0,
                ansys_modal_hz=None,
                ansys_frf_peak_hz=0.60,
                ansys_voltage_v=None,
                ansys_voltage_form="unknown",
            )

            self.assertTrue(summary["frequency_comparison"]["frequency_reference_ambiguous"])
            self.assertAlmostEqual(summary["frequency_comparison"]["ambiguous_frequency_reference_hz"], 0.60)
            self.assertTrue(any("Only one ANSYS frequency reference was provided" in w for w in summary["warnings"]))


if __name__ == "__main__":
    unittest.main()
