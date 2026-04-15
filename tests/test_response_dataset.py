from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from peh_inverse_design.response_dataset import aggregate_response_directory, save_fem_response


class ResponseDatasetTests(unittest.TestCase):
    def test_save_and_aggregate_persist_peak_voltage_fields_only(self) -> None:
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
                self.assertAlmostEqual(float(response["peak_voltage"]), 4.0)
                self.assertNotIn("peak_voltage_rms_v", response.files)
                self.assertNotIn("peak_voltage_form", response.files)

            aggregated = aggregate_response_directory(response_dir=response_dir, output_path=output_path)

            self.assertIn("peak_voltage_peak_v", aggregated)
            self.assertAlmostEqual(float(aggregated["peak_voltage_peak_v"][0]), 4.0)
            self.assertAlmostEqual(float(aggregated["peak_voltage"][0]), 4.0)
            self.assertNotIn("peak_voltage_rms_v", aggregated)
            self.assertNotIn("peak_voltage_form", aggregated)

    def test_save_fem_response_rejects_rms_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            response_dir = root / "responses"
            with self.assertRaisesRegex(ValueError, "RMS handling was removed"):
                save_fem_response(
                    sample_id=8,
                    f_peak_hz=1.5,
                    freq_hz=np.asarray([1.0, 1.5, 2.0], dtype=np.float64),
                    voltage_mag=np.asarray([2.0, 4.0, 3.0], dtype=np.float64),
                    output_dir=response_dir,
                    voltage_amplitude_convention="rms",
                )

    def test_aggregate_response_directory_rejects_tagged_rms_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            response_dir = root / "responses"
            response_dir.mkdir(parents=True, exist_ok=True)
            output_path = root / "response_dataset.npz"
            np.savez_compressed(
                response_dir / "sample_0001_response.npz",
                sample_id=np.asarray(1, dtype=np.int32),
                f_peak_hz=np.asarray(1.0, dtype=np.float64),
                freq_hz=np.asarray([0.9, 1.0, 1.1], dtype=np.float64),
                voltage_mag=np.asarray([1.0, 2.0, 1.5], dtype=np.float64),
                peak_voltage=np.asarray(2.0, dtype=np.float64),
                peak_voltage_peak_v=np.asarray(2.0, dtype=np.float64),
                peak_voltage_form=np.asarray("rms"),
                quality_flag=np.asarray(1, dtype=np.int32),
            )

            with self.assertRaisesRegex(ValueError, "RMS handling was removed"):
                aggregate_response_directory(response_dir=response_dir, output_path=output_path)


if __name__ == "__main__":
    unittest.main()
