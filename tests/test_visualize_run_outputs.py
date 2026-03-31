from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from peh_inverse_design.visualize_run_outputs import _dataset_plate_size_m, _load_dataset_row


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


if __name__ == "__main__":
    unittest.main()
