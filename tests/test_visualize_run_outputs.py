from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from peh_inverse_design.modal_surface_fields import preferred_surface_strain_field
from peh_inverse_design.response_dataset import save_fem_response
from peh_inverse_design.visualize_run_outputs import (
    _dataset_plate_size_m,
    _load_dataset_row,
    _load_response,
    _plot_stats,
    _write_summary_csv,
)


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

    def test_load_response_reads_peak_only_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            response_dir = Path(tmpdir) / "responses"
            save_fem_response(
                sample_id=3,
                f_peak_hz=0.8,
                freq_hz=np.asarray([0.7, 0.8, 0.9], dtype=np.float64),
                voltage_mag=np.asarray([2.0, 4.0, 3.0], dtype=np.float64),
                output_dir=response_dir,
            )

            response = _load_response(response_dir / "sample_0003_response.npz")

            np.testing.assert_allclose(
                np.asarray(response["voltage_mag"], dtype=np.float64),
                np.asarray([2.0, 4.0, 3.0], dtype=np.float64),
            )
            self.assertAlmostEqual(float(response["peak_voltage"]), 4.0)
            self.assertNotIn("peak_voltage_form", response)
            self.assertNotIn("peak_voltage_rms_v", response)

    def test_write_summary_csv_tracks_mode1_frequency_and_peak_voltage_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.csv"
            _write_summary_csv(
                [
                    {
                        "sample_id": 3.0,
                        "volume_fraction": 0.25,
                        "mode1_frequency_hz": 0.78,
                        "f_peak_hz": 0.8,
                        "peak_voltage": 4.0,
                        "n_nodes": 10.0,
                        "n_tetra": 20.0,
                        "n_modes": 4.0,
                        "max_top_surface_strain": 1.2e-3,
                    }
                ],
                output_path,
            )

            with output_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    [
                        "sample_id",
                        "volume_fraction",
                        "mode1_frequency_hz",
                        "f_peak_hz",
                        "peak_voltage",
                        "n_nodes",
                        "n_tetra",
                        "n_modes",
                        "max_top_surface_strain",
                    ],
                )
                rows = list(reader)

            self.assertEqual(rows[0]["mode1_frequency_hz"], "0.78")
            self.assertEqual(rows[0]["peak_voltage"], "4.0")
            self.assertNotIn("peak_voltage_form", rows[0])

    def test_plot_stats_labels_peak_voltage_without_response_form_field(self) -> None:
        fig, ax = plt.subplots()
        try:
            stats = _plot_stats(
                ax=ax,
                sample_id=7,
                dataset_row={
                    "volume_fraction": np.asarray(0.25, dtype=np.float64),
                    "binary": np.asarray([[True, False], [False, True]], dtype=bool),
                },
                mesh={
                    "points": np.asarray(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float64,
                    ),
                    "tetra_cells": np.asarray([[0, 1, 2, 3]], dtype=np.int64),
                    "triangle_cells": np.asarray([[0, 1, 2]], dtype=np.int32),
                    "triangle_tags": np.asarray([105], dtype=np.int32),
                },
                response={
                    "f_peak_hz": np.asarray(123.0, dtype=np.float64),
                    "peak_voltage": np.asarray(4.5, dtype=np.float64),
                },
                modal={
                    "eigenfreq_hz": np.asarray([120.0, 240.0], dtype=np.float64),
                },
                strain_max=1.2e-3,
            )
        finally:
            plt.close(fig)

        self.assertEqual(stats["peak_voltage"], 4.5)
        self.assertEqual(stats["mode1_frequency_hz"], 120.0)
        self.assertIn("peak voltage (peak): 4.5000", ax.texts[0].get_text())


if __name__ == "__main__":
    unittest.main()
