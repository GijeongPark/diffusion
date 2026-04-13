from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from peh_inverse_design.response_dataset import aggregate_response_directory, save_fem_response


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
            )

            with np.load(response_dir / "sample_0007_response.npz", allow_pickle=True) as response:
                self.assertAlmostEqual(float(response["peak_voltage_peak_v"]), 4.0)
                self.assertAlmostEqual(float(response["peak_voltage_rms_v"]), 4.0 / np.sqrt(2.0))
                self.assertEqual(str(np.asarray(response["peak_voltage_form"]).reshape(-1)[0]), "peak")
                self.assertAlmostEqual(float(response["peak_voltage"]), 4.0)

            aggregated = aggregate_response_directory(response_dir=response_dir, output_path=output_path)

            self.assertIn("peak_voltage_peak_v", aggregated)
            self.assertIn("peak_voltage_rms_v", aggregated)
            self.assertIn("peak_voltage_form", aggregated)
            self.assertAlmostEqual(float(aggregated["peak_voltage_peak_v"][0]), 4.0)
            self.assertAlmostEqual(float(aggregated["peak_voltage_rms_v"][0]), 4.0 / np.sqrt(2.0))
            self.assertEqual(str(aggregated["peak_voltage_form"][0]), "peak")
            self.assertAlmostEqual(float(aggregated["peak_voltage"][0]), 4.0)

    def test_save_fem_response_can_store_rms_canonical_voltage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            response_dir = root / "responses"
            output_path = root / "response_dataset.npz"
            save_fem_response(
                sample_id=8,
                f_peak_hz=1.5,
                freq_hz=np.asarray([1.0, 1.5, 2.0], dtype=np.float64),
                voltage_mag=np.asarray([2.0, 4.0, 3.0], dtype=np.float64),
                output_dir=response_dir,
                voltage_amplitude_convention="rms",
            )

            with np.load(response_dir / "sample_0008_response.npz", allow_pickle=True) as response:
                np.testing.assert_allclose(
                    np.asarray(response["voltage_mag"], dtype=np.float64),
                    np.asarray([2.0, 4.0, 3.0], dtype=np.float64) / np.sqrt(2.0),
                )
                self.assertAlmostEqual(float(response["peak_voltage_peak_v"]), 4.0)
                self.assertAlmostEqual(float(response["peak_voltage_rms_v"]), 4.0 / np.sqrt(2.0))
                self.assertEqual(str(np.asarray(response["peak_voltage_form"]).reshape(-1)[0]), "rms")
                self.assertAlmostEqual(float(response["peak_voltage"]), 4.0 / np.sqrt(2.0))

            aggregated = aggregate_response_directory(response_dir=response_dir, output_path=output_path)

            np.testing.assert_allclose(
                np.asarray(aggregated["voltage_mag"][0], dtype=np.float64),
                np.asarray([2.0, 4.0, 3.0], dtype=np.float64) / np.sqrt(2.0),
            )
            self.assertAlmostEqual(float(aggregated["peak_voltage"][0]), 4.0 / np.sqrt(2.0))
            self.assertAlmostEqual(float(aggregated["peak_voltage_peak_v"][0]), 4.0)
            self.assertAlmostEqual(float(aggregated["peak_voltage_rms_v"][0]), 4.0 / np.sqrt(2.0))
            self.assertEqual(str(aggregated["peak_voltage_form"][0]), "rms")


if __name__ == "__main__":
    unittest.main()
