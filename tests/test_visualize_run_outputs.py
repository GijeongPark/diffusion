from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from peh_inverse_design.response_dataset import save_fem_response
from peh_inverse_design.visualize_run_outputs import _dataset_plate_size_m, _load_dataset_row, _load_response
from peh_inverse_design.modal_surface_fields import preferred_surface_strain_field


class VisualizeRunOutputsTests(unittest.TestCase):
    def test_load_dataset_row_matches_sample_id_instead_of_positional_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.npz"
            np.savez_compressed(
                dataset_path,
                sample_id=np.asarray([11, 42], dtype=np.int32),
                binary=np.asarray(
                    [
                        [[0, 1], [1, 0]],
                        [[1, 1], [0, 0]],
                    ],
                    dtype=bool,
                ),
                sdf=np.asarray(
                    [
                        [[0.0, 1.0], [1.0, 0.0]],
                        [[2.0, 3.0], [3.0, 2.0]],
                    ],
                    dtype=np.float64,
                ),
                threshold=np.asarray([0.1, 0.9], dtype=np.float64),
                volume_fraction=np.asarray([0.25, 0.75], dtype=np.float64),
                tile_counts=np.asarray([[3, 2], [5, 4]], dtype=np.int32),
                cell_size_m=np.asarray([[0.2, 0.1], [0.3, 0.05]], dtype=np.float64),
            )

            row = _load_dataset_row(dataset_path, sample_id=42)

            self.assertEqual(int(row["dataset_index"]), 1)
            np.testing.assert_array_equal(row["binary"], np.asarray([[True, True], [False, False]], dtype=bool))
            np.testing.assert_allclose(row["threshold"], np.asarray(0.9))
            np.testing.assert_allclose(row["volume_fraction"], np.asarray(0.75))
            np.testing.assert_array_equal(row["tile_counts"], np.asarray([5, 4], dtype=np.int32))
            np.testing.assert_allclose(row["cell_size_m"], np.asarray([0.3, 0.05], dtype=np.float64))

    def test_dataset_plate_size_uses_dataset_metadata(self) -> None:
        row = {
            "tile_counts": np.asarray([5, 4], dtype=np.int32),
            "cell_size_m": np.asarray([0.3, 0.05], dtype=np.float64),
        }
        self.assertEqual(_dataset_plate_size_m(row), (1.5, 0.2))

    def test_visualizer_prefers_true_mode1_surface_field_when_present(self) -> None:
        modal = {
            "mode1_frequency_hz": np.asarray(1.0, dtype=np.float64),
            "mode1_top_surface_strain_eqv": np.asarray([1.0, 2.0], dtype=np.float64),
            "harmonic_field_frequency_hz": np.asarray(0.98, dtype=np.float64),
            "harmonic_top_surface_strain_eqv": np.asarray([5.0, 6.0], dtype=np.float64),
            "top_surface_strain_eqv": np.asarray([5.0, 6.0], dtype=np.float64),
        }

        field = preferred_surface_strain_field(modal, triangle_count=2)

        self.assertIsNotNone(field)
        self.assertEqual(field.kind, "modal")
        np.testing.assert_array_equal(field.strain, np.asarray([1.0, 2.0], dtype=np.float64))

    def test_load_response_uses_stored_voltage_convention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            response_dir = Path(tmpdir) / "responses"
            save_fem_response(
                sample_id=3,
                f_peak_hz=0.8,
                freq_hz=np.asarray([0.7, 0.8, 0.9], dtype=np.float64),
                voltage_mag=np.asarray([2.0, 4.0, 3.0], dtype=np.float64),
                output_dir=response_dir,
                voltage_amplitude_convention="rms",
            )

            response = _load_response(response_dir / "sample_0003_response.npz")

            np.testing.assert_allclose(
                np.asarray(response["voltage_mag"], dtype=np.float64),
                np.asarray([2.0, 4.0, 3.0], dtype=np.float64) / np.sqrt(2.0),
            )
            self.assertAlmostEqual(float(response["peak_voltage"]), 4.0 / np.sqrt(2.0))
            self.assertEqual(str(np.asarray(response["peak_voltage_form"]).reshape(-1)[0]), "rms")


if __name__ == "__main__":
    unittest.main()
