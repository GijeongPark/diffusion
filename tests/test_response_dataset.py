from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from peh_inverse_design.response_dataset import aggregate_response_directory, save_fem_response
from peh_inverse_design.solver_diagnostics import build_solver_provenance_arrays, compute_drive_coupling_diagnostics


class ResponseDatasetTests(unittest.TestCase):
    def test_save_and_aggregate_persist_peak_and_rms_voltage_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            response_dir = root / "responses"
            output_path = root / "response_dataset.npz"
            save_fem_response(
                sample_id=7,
                f_peak_hz=1.5,
                freq_hz=np.asarray([1.0, 1.5, 2.0], dtype=np.float64),
                voltage_mag=np.asarray([2.0, 4.0, 3.0], dtype=np.float64),
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
                    eigenfreq_hz=np.asarray([1.4, 2.8], dtype=np.float64),
                    modal_force=np.asarray([1.0, 3.0], dtype=np.float64),
                    modal_theta=np.asarray([2.0, 1.0], dtype=np.float64),
                )
                | {
                    "frf_search_seed_source": np.asarray(["dominant_coupling"]),
                    "frf_search_seed_frequency_hz": np.asarray([2.8], dtype=np.float64),
                },
            )

            with np.load(response_dir / "sample_0007_response.npz", allow_pickle=True) as response:
                self.assertAlmostEqual(float(response["peak_voltage_peak_v"]), 4.0)
                self.assertAlmostEqual(float(response["peak_voltage_rms_v"]), 4.0 / np.sqrt(2.0))
                self.assertEqual(str(np.asarray(response["peak_voltage_form"]).reshape(-1)[0]), "peak")
                self.assertAlmostEqual(float(response["peak_voltage"]), 4.0)
                self.assertEqual(str(np.asarray(response["eigensolver_backend"]).reshape(-1)[0]), "shift_invert_lu")
                self.assertTrue(bool(np.asarray(response["solver_parity_valid"]).reshape(-1)[0]))
                self.assertEqual(int(np.asarray(response["dominant_drive_coupling_mode_index"]).reshape(-1)[0]), 1)
                self.assertEqual(str(np.asarray(response["frf_search_seed_source"]).reshape(-1)[0]), "dominant_coupling")
                self.assertAlmostEqual(float(np.asarray(response["frf_search_seed_frequency_hz"]).reshape(-1)[0]), 2.8)

            aggregated = aggregate_response_directory(response_dir=response_dir, output_path=output_path)

            self.assertIn("peak_voltage_peak_v", aggregated)
            self.assertIn("peak_voltage_rms_v", aggregated)
            self.assertIn("peak_voltage_form", aggregated)
            self.assertAlmostEqual(float(aggregated["peak_voltage_peak_v"][0]), 4.0)
            self.assertAlmostEqual(float(aggregated["peak_voltage_rms_v"][0]), 4.0 / np.sqrt(2.0))
            self.assertEqual(str(aggregated["peak_voltage_form"][0]), "peak")
            self.assertAlmostEqual(float(aggregated["peak_voltage"][0]), 4.0)
            self.assertEqual(str(aggregated["eigensolver_backend"][0]), "shift_invert_lu")
            self.assertEqual(int(aggregated["solver_element_order"][0]), 2)
            self.assertEqual(int(aggregated["dominant_drive_coupling_mode_index"][0]), 1)
            self.assertTrue(bool(aggregated["solver_parity_valid"][0]))
            self.assertEqual(str(aggregated["frf_search_seed_source"][0]), "dominant_coupling")
            self.assertAlmostEqual(float(aggregated["frf_search_seed_frequency_hz"][0]), 2.8)


if __name__ == "__main__":
    unittest.main()
