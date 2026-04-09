from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from peh_inverse_design.build_integrated_dataset import build_integrated_dataset
from peh_inverse_design.response_dataset import save_fem_response
from peh_inverse_design.solver_diagnostics import build_solver_provenance_arrays, compute_drive_coupling_diagnostics


class BuildIntegratedDatasetTests(unittest.TestCase):
    def test_integrated_dataset_persists_voltage_and_mesh_provenance_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unit_cell_npz = root / "unit_cell_dataset.npz"
            response_dir = root / "responses"
            modal_dir = root / "modal"
            mesh_dir = root / "meshes"
            output_path = root / "integrated_dataset.npz"
            index_csv_path = root / "integrated_dataset.csv"
            modal_dir.mkdir(parents=True, exist_ok=True)
            mesh_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                unit_cell_npz,
                grf=np.zeros((1, 2, 2), dtype=np.float64),
                threshold=np.asarray([0.0], dtype=np.float64),
                sample_id=np.asarray([0], dtype=np.int32),
                source_sample_id=np.asarray([0], dtype=np.int32),
            )
            save_fem_response(
                sample_id=0,
                f_peak_hz=0.72,
                freq_hz=np.asarray([0.6, 0.72, 0.84], dtype=np.float64),
                voltage_mag=np.asarray([200.0, 342.0, 210.0], dtype=np.float64),
                output_dir=response_dir,
                solver_provenance=build_solver_provenance_arrays(
                    eigensolver_backend="shift_invert_lu",
                    solver_element_order=2,
                    requested_solver_element_order=2,
                    requested_eigensolver_backend="shift_invert_lu",
                    used_eigensolver_fallback=False,
                    used_element_order_fallback=False,
                    solver_parity_valid=True,
                    parity_invalid_reason="",
                    strict_parity_requested=True,
                    diagnostic_only=False,
                ),
                modal_diagnostics=compute_drive_coupling_diagnostics(
                    eigenfreq_hz=np.asarray([0.7, 1.1], dtype=np.float64),
                    modal_force=np.asarray([1.0, 0.2], dtype=np.float64),
                    modal_theta=np.asarray([2.0, 5.0], dtype=np.float64),
                )
                | {
                    "frf_search_seed_source": np.asarray(["dominant_coupling"]),
                    "frf_search_seed_frequency_hz": np.asarray([1.1], dtype=np.float64),
                },
            )
            np.savez_compressed(
                modal_dir / "sample_0000_modal.npz",
                sample_id=np.asarray(0, dtype=np.int32),
                eigenfreq_hz=np.asarray([0.7], dtype=np.float64),
                mode1_frequency_hz=np.asarray(0.7, dtype=np.float64),
                harmonic_field_frequency_hz=np.asarray(0.72, dtype=np.float64),
                modal_force=np.asarray([1.0], dtype=np.float64),
                modal_theta=np.asarray([2.0], dtype=np.float64),
                modal_mass=np.asarray([1.0], dtype=np.float64),
                capacitance_f=np.asarray([1.0e-4], dtype=np.float64),
            )
            np.savez_compressed(
                mesh_dir / "plate3d_0000_fenicsx.npz",
                points=np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                tetra_cells=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
                tetra_tags=np.asarray([1], dtype=np.int32),
                triangle_cells=np.asarray([[0, 1, 2]], dtype=np.int32),
                triangle_tags=np.asarray([1], dtype=np.int32),
                solver_mesh_size_m=np.asarray([0.008], dtype=np.float64),
                mesh_preset=np.asarray(["ansys_parity"]),
                max_solver_vector_dofs=np.asarray([-1], dtype=np.int64),
                solver_mesh_coarsening_allowed=np.asarray([0], dtype=np.int32),
            )
            (mesh_dir / "plate3d_0000_cad.json").write_text(
                json.dumps(
                    {
                        "sample_id": 0,
                        "mesh_preset": "ansys_parity",
                        "substrate_layers": 8,
                        "piezo_layers": 3,
                        "allow_solver_mesh_coarsening": False,
                        "max_solver_vector_dofs": None,
                    }
                ),
                encoding="utf-8",
            )

            build_integrated_dataset(
                unit_cell_npz=unit_cell_npz,
                response_dir=response_dir,
                modal_dir=modal_dir,
                mesh_dir=mesh_dir,
                output_path=output_path,
                index_csv_path=index_csv_path,
            )

            with np.load(output_path, allow_pickle=True) as integrated:
                self.assertAlmostEqual(float(integrated["peak_voltage"][0]), 342.0)
                self.assertAlmostEqual(float(integrated["peak_voltage_peak_v"][0]), 342.0)
                self.assertAlmostEqual(float(integrated["peak_voltage_rms_v"][0]), 342.0 / np.sqrt(2.0))
                self.assertEqual(str(integrated["peak_voltage_form"][0]), "peak")
                self.assertEqual(str(integrated["mesh_preset"][0]), "ansys_parity")
                self.assertEqual(int(integrated["substrate_layers"][0]), 8)
                self.assertEqual(int(integrated["piezo_layers"][0]), 3)
                self.assertAlmostEqual(float(integrated["solver_mesh_size_m"][0]), 0.008)
                self.assertEqual(int(integrated["mesh_point_count"][0]), 4)
                self.assertEqual(int(integrated["mesh_tetra_count"][0]), 1)
                self.assertEqual(int(integrated["solver_max_q2_vector_dofs"][0]), -1)
                self.assertEqual(int(integrated["solver_max_q2_vector_dofs_unlimited"][0]), 1)
                self.assertEqual(str(integrated["eigensolver_backend"][0]), "shift_invert_lu")
                self.assertEqual(int(integrated["solver_element_order"][0]), 2)
                self.assertEqual(int(integrated["dominant_drive_coupling_mode_index"][0]), 0)
                self.assertTrue(bool(integrated["solver_parity_valid"][0]))
                self.assertEqual(str(integrated["frf_search_seed_source"][0]), "dominant_coupling")
                self.assertAlmostEqual(float(integrated["frf_search_seed_frequency_hz"][0]), 1.1)

            with index_csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["peak_voltage_peak_v"], "342")
            self.assertEqual(rows[0]["peak_voltage_form"], "peak")
            self.assertEqual(rows[0]["mesh_preset"], "ansys_parity")
            self.assertEqual(rows[0]["substrate_layers"], "8")
            self.assertEqual(rows[0]["piezo_layers"], "3")
            self.assertEqual(rows[0]["solver_max_q2_vector_dofs"], "")
            self.assertEqual(rows[0]["solver_max_q2_vector_dofs_unlimited"], "True")
            self.assertEqual(rows[0]["eigensolver_backend"], "shift_invert_lu")
            self.assertEqual(rows[0]["solver_parity_valid"], "True")
            self.assertEqual(rows[0]["frf_search_seed_source"], "dominant_coupling")


if __name__ == "__main__":
    unittest.main()
