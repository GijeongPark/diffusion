from __future__ import annotations

import unittest

import numpy as np

from peh_inverse_design.fenicsx_modal_solver import (
    MechanicalConfig,
    PiezoConfig,
    _build_modal_save_payload,
    _compute_top_surface_cellwise_strain,
    _remap_raw_cell_tags_to_created_mesh,
    _resolve_peak_search_seed,
)


class FenicsxModalSolverTests(unittest.TestCase):
    def test_remap_raw_cell_tags_matches_created_mesh_after_point_reordering(self) -> None:
        raw_tetra_cells = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        raw_tetra_tags = np.asarray([12], dtype=np.int32)
        input_global_indices = np.asarray([3, 1, 0, 2], dtype=np.int64)
        created_cell_geometry_dofs = np.asarray([[2, 1, 3, 0]], dtype=np.int64)

        remapped = _remap_raw_cell_tags_to_created_mesh(
            raw_tetra_cells=raw_tetra_cells,
            raw_tetra_tags=raw_tetra_tags,
            created_cell_geometry_dofs=created_cell_geometry_dofs,
            input_global_indices=input_global_indices,
        )

        np.testing.assert_array_equal(remapped, np.asarray([12], dtype=np.int32))

    def test_remap_raw_cell_tags_matches_created_mesh_after_cell_reordering(self) -> None:
        raw_tetra_cells = np.asarray(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
            ],
            dtype=np.int64,
        )
        raw_tetra_tags = np.asarray([11, 12], dtype=np.int32)
        input_global_indices = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
        created_cell_geometry_dofs = np.asarray(
            [
                [1, 2, 3, 4],
                [0, 1, 2, 3],
            ],
            dtype=np.int64,
        )

        remapped = _remap_raw_cell_tags_to_created_mesh(
            raw_tetra_cells=raw_tetra_cells,
            raw_tetra_tags=raw_tetra_tags,
            created_cell_geometry_dofs=created_cell_geometry_dofs,
            input_global_indices=input_global_indices,
        )

        np.testing.assert_array_equal(remapped, np.asarray([12, 11], dtype=np.int32))

    def test_remap_raw_cell_tags_matches_created_mesh_after_cell_and_point_reordering(self) -> None:
        raw_tetra_cells = np.asarray(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
            ],
            dtype=np.int64,
        )
        raw_tetra_tags = np.asarray([11, 12], dtype=np.int32)

        # The created mesh reorders geometry points relative to the raw NPZ input.
        input_global_indices = np.asarray([2, 0, 3, 1, 4], dtype=np.int64)
        # These created cells use geometry-dof indices, not the original raw-point ids.
        created_cell_geometry_dofs = np.asarray(
            [
                [1, 3, 0, 2],  # maps back to raw vertices [0, 1, 2, 3]
                [3, 0, 2, 4],  # maps back to raw vertices [1, 2, 3, 4]
            ],
            dtype=np.int64,
        )

        remapped = _remap_raw_cell_tags_to_created_mesh(
            raw_tetra_cells=raw_tetra_cells,
            raw_tetra_tags=raw_tetra_tags,
            created_cell_geometry_dofs=created_cell_geometry_dofs,
            input_global_indices=input_global_indices,
        )

        np.testing.assert_array_equal(remapped, np.asarray([11, 12], dtype=np.int32))

    def test_build_modal_save_payload_writes_explicit_modal_and_harmonic_fields(self) -> None:
        modal_model = {
            "eigenfreq_hz": np.asarray([1.25, 3.5], dtype=np.float64),
            "modal_force": np.asarray([0.1, 0.2], dtype=np.float64),
            "modal_theta": np.asarray([0.01, 0.02], dtype=np.float64),
            "modal_mass": np.asarray([1.0, 1.0], dtype=np.float64),
            "capacitance_f": np.asarray([5.0e-7], dtype=np.float64),
            "substrate_volume_m3": np.asarray([1.0e-4], dtype=np.float64),
            "piezo_volume_m3": np.asarray([2.0e-5], dtype=np.float64),
            "substrate_cell_count": np.asarray([12], dtype=np.int32),
            "piezo_cell_count": np.asarray([8], dtype=np.int32),
        }

        payload = _build_modal_save_payload(
            sample_id=7,
            element_order=2,
            mechanical=MechanicalConfig(),
            piezo=PiezoConfig(),
            modal_model=modal_model,
            mode1_top_surface_strain_eqv=np.asarray([10.0, 20.0], dtype=np.float64),
            harmonic_top_surface_strain_eqv=np.asarray([30.0, 40.0], dtype=np.float64),
            harmonic_field_frequency_hz=1.2,
            frf_search_seed_source="dominant_coupling",
            frf_search_seed_frequency_hz=3.5,
        )

        self.assertIn("mode1_frequency_hz", payload)
        self.assertIn("mode1_top_surface_strain_eqv", payload)
        self.assertIn("harmonic_field_frequency_hz", payload)
        self.assertIn("harmonic_top_surface_strain_eqv", payload)
        self.assertIn("frf_search_seed_source", payload)
        self.assertIn("frf_search_seed_frequency_hz", payload)
        np.testing.assert_array_equal(payload["top_surface_strain_eqv"], np.asarray([30.0, 40.0], dtype=np.float64))
        self.assertEqual(float(payload["mode1_frequency_hz"]), 1.25)
        self.assertEqual(float(payload["harmonic_field_frequency_hz"]), 1.2)
        self.assertEqual(str(np.asarray(payload["frf_search_seed_source"]).reshape(-1)[0]), "dominant_coupling")
        self.assertEqual(float(np.asarray(payload["frf_search_seed_frequency_hz"]).reshape(-1)[0]), 3.5)

    def test_resolve_peak_search_seed_prefers_dominant_coupling_for_auto(self) -> None:
        modal_model = {
            "eigenfreq_hz": np.asarray([0.7, 1.5, 3.7], dtype=np.float64),
            "dominant_drive_coupling_mode_frequency_hz": np.asarray([3.7], dtype=np.float64),
            "suspect_mode_ordering": np.asarray([True], dtype=np.bool_),
        }

        source, frequency_hz = _resolve_peak_search_seed(
            modal_model=modal_model,
            peak_search_seed="auto",
        )

        self.assertEqual(source, "dominant_coupling")
        self.assertEqual(frequency_hz, 3.7)

    def test_resolve_peak_search_seed_uses_f1_when_requested(self) -> None:
        modal_model = {
            "eigenfreq_hz": np.asarray([0.7, 1.5, 3.7], dtype=np.float64),
            "dominant_drive_coupling_mode_frequency_hz": np.asarray([3.7], dtype=np.float64),
            "suspect_mode_ordering": np.asarray([True], dtype=np.bool_),
        }

        source, frequency_hz = _resolve_peak_search_seed(
            modal_model=modal_model,
            peak_search_seed="f1",
        )

        self.assertEqual(source, "f1")
        self.assertEqual(frequency_hz, 0.7)

    def test_top_surface_cellwise_strain_is_root_dominant_for_cantilever_like_mode(self) -> None:
        x_nodes = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        y_nodes = np.asarray([0.0, 1.0], dtype=np.float64)
        z_nodes = np.asarray([0.0, 1.0], dtype=np.float64)

        def node_id(ix: int, iy: int, iz: int) -> int:
            return iz * (len(x_nodes) * len(y_nodes)) + iy * len(x_nodes) + ix

        points = []
        for z in z_nodes:
            for y in y_nodes:
                for x in x_nodes:
                    points.append([x, y, z])
        points = np.asarray(points, dtype=np.float64)

        tetra_cells: list[list[int]] = []
        triangle_cells: list[list[int]] = []
        for ix in range(len(x_nodes) - 1):
            v000 = node_id(ix, 0, 0)
            v100 = node_id(ix + 1, 0, 0)
            v010 = node_id(ix, 1, 0)
            v110 = node_id(ix + 1, 1, 0)
            v001 = node_id(ix, 0, 1)
            v101 = node_id(ix + 1, 0, 1)
            v011 = node_id(ix, 1, 1)
            v111 = node_id(ix + 1, 1, 1)
            tetra_cells.extend(
                [
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                ]
            )
            triangle_cells.extend(
                [
                    [v001, v101, v111],
                    [v001, v111, v011],
                ]
            )

        tetra_cells_array = np.asarray(tetra_cells, dtype=np.int64)
        triangle_cells_array = np.asarray(triangle_cells, dtype=np.int32)
        triangle_tags = np.full(triangle_cells_array.shape[0], 105, dtype=np.int32)

        def w(x: np.ndarray) -> np.ndarray:
            return x ** 2 * (3.0 - x)

        def dw_dx(x: np.ndarray) -> np.ndarray:
            return 6.0 * x - 3.0 * x ** 2

        nodal_displacement = np.zeros((points.shape[0], 3), dtype=np.float64)
        x = points[:, 0]
        z_centered = points[:, 2] - 0.5
        nodal_displacement[:, 0] = -z_centered * dw_dx(x)
        nodal_displacement[:, 2] = w(x)

        strain = _compute_top_surface_cellwise_strain(
            points=points,
            tetra_cells=tetra_cells_array,
            triangle_cells=triangle_cells_array,
            triangle_tags=triangle_tags,
            nodal_displacement=nodal_displacement,
        )
        centroids = np.mean(points[triangle_cells_array], axis=1)
        root_mean = float(np.mean(strain[centroids[:, 0] <= 0.25]))
        tip_mean = float(np.mean(strain[centroids[:, 0] >= 0.75]))
        top_decile = strain >= np.percentile(strain, 90.0)
        top_decile_x = float(np.mean(centroids[top_decile, 0]))

        self.assertGreater(root_mean, tip_mean)
        self.assertLess(top_decile_x, 0.35)


if __name__ == "__main__":
    unittest.main()
