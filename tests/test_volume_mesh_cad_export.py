from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gmsh
import numpy as np
from shapely.geometry import box

from peh_inverse_design.geometry_pipeline import GeometryBuildConfig
from peh_inverse_design.volume_mesh import (
    VolumeMeshConfig,
    _build_layered_tet_solver_mesh,
    _build_substrate_planform,
    _export_occ_geometry_to_step,
    _mesh_polygons_volume_sample_layered_tet,
    _normalize_step_length_units_to_metre,
    _prepare_cad_occ_model,
    _reload_occ_geometry_from_step,
    _resolve_cad_reference_size_m,
    _validate_current_occ_model,
    _write_ansys_face_selection_manifest,
    _write_cad_report,
)


class VolumeMeshCadExportTests(unittest.TestCase):
    def _geometry_config(
        self,
        tile_counts: tuple[int, int] = (4, 3),
        cell_size_m: tuple[float, float] = (5.0e-3, 5.0e-3),
    ) -> GeometryBuildConfig:
        return GeometryBuildConfig(
            cell_size_m=cell_size_m,
            tile_counts=tile_counts,
        )

    def _exact_volume_config(self) -> VolumeMeshConfig:
        return VolumeMeshConfig(
            substrate_thickness_m=2.0e-3,
            piezo_thickness_m=1.0e-3,
            mesh_size_relative_to_cell=0.2,
            require_connected_substrate=True,
        )

    def _repair_volume_config(self) -> VolumeMeshConfig:
        return VolumeMeshConfig(
            substrate_thickness_m=2.0e-3,
            piezo_thickness_m=1.0e-3,
            mesh_size_relative_to_cell=0.2,
            require_connected_substrate=True,
            exact_cad=False,
            repair_cad=True,
            repair_bridge_width_m=1.5e-3,
        )

    def _full_plate(self, geometry_config: GeometryBuildConfig):
        return box(0.0, 0.0, geometry_config.plate_size_m[0], geometry_config.plate_size_m[1])

    def _hole_planform(self, geometry_config: GeometryBuildConfig):
        outer = self._full_plate(geometry_config)
        hole = box(7.0e-3, 4.0e-3, 1.3e-2, 1.0e-2)
        return outer.difference(hole)

    def _disconnected_parts(self):
        return [
            box(0.0, 0.0, 8.0e-3, 1.5e-2),
            box(1.2e-2, 0.0, 2.0e-2, 1.5e-2),
        ]

    def _roundtrip_cad(
        self,
        polygons,
        geometry_config: GeometryBuildConfig,
        volume_config: VolumeMeshConfig,
        *,
        partition_interface: bool = True,
    ):
        cad_reference_size_m = _resolve_cad_reference_size_m(geometry_config, volume_config)
        planform = _build_substrate_planform(
            polygons=polygons,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=cad_reference_size_m,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            step_path = Path(tmpdir) / "probe.step"
            cad_report_path = Path(tmpdir) / "probe_cad.json"
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.model.add("probe")
            try:
                pre_export_report = _prepare_cad_occ_model(
                    planform=planform,
                    geometry_config=geometry_config,
                    volume_config=volume_config,
                    mesh_size_m=cad_reference_size_m,
                    partition_interface=partition_interface,
                )
                _export_occ_geometry_to_step(step_path)
                _reload_occ_geometry_from_step(step_path)
                roundtrip_report = _validate_current_occ_model(
                    stage="step_roundtrip",
                    planform=planform,
                    geometry_config=geometry_config,
                    volume_config=volume_config,
                    mesh_size_m=cad_reference_size_m,
                )
            finally:
                gmsh.finalize()

            _write_cad_report(
                report_path=cad_report_path,
                sample_id=1,
                geometry_config=geometry_config,
                volume_config=volume_config,
                planform=planform,
                pre_export_report=pre_export_report,
                roundtrip_report=roundtrip_report,
            )
            payload = json.loads(cad_report_path.read_text(encoding="utf-8"))

        return planform, pre_export_report, roundtrip_report, payload

    def test_roundtrip_hole_planform_keeps_two_solids_and_hole_walls(self) -> None:
        geometry_config = self._geometry_config()
        volume_config = self._exact_volume_config()
        _, _, report, _ = self._roundtrip_cad([self._hole_planform(geometry_config)], geometry_config, volume_config)

        self.assertEqual(report.solid_body_count, 2)
        self.assertEqual(report.stray_surface_tags, ())
        self.assertEqual(report.stray_curve_tags, ())
        self.assertGreater(len(report.internal_vertical_substrate_face_tags), 0)
        self.assertGreater(report.piezo_bottom_face_count, 1)

        full_plate_volume = (
            geometry_config.plate_size_m[0]
            * geometry_config.plate_size_m[1]
            * volume_config.substrate_thickness_m
        )
        self.assertLess(report.substrate_volume_m3, full_plate_volume)

        self.assertAlmostEqual(
            report.substrate_volume_m3,
            report.expected_substrate_volume_m3,
            delta=report.expected_substrate_volume_m3 * volume_config.volume_relative_tolerance + 1.0e-15,
        )
        self.assertAlmostEqual(
            report.piezo_volume_m3,
            report.expected_piezo_volume_m3,
            delta=report.expected_piezo_volume_m3 * volume_config.volume_relative_tolerance + 1.0e-15,
        )

    def test_roundtrip_full_plate_has_no_internal_vertical_faces(self) -> None:
        geometry_config = self._geometry_config(tile_counts=(3, 2))
        volume_config = self._exact_volume_config()
        _, _, report, _ = self._roundtrip_cad([self._full_plate(geometry_config)], geometry_config, volume_config)

        self.assertEqual(report.solid_body_count, 2)
        self.assertEqual(report.stray_surface_tags, ())
        self.assertEqual(report.stray_curve_tags, ())
        self.assertEqual(report.internal_vertical_substrate_face_tags, ())
        self.assertEqual(report.piezo_bottom_face_count, 1)
        self.assertAlmostEqual(
            report.substrate_volume_m3,
            report.expected_substrate_volume_m3,
            delta=report.expected_substrate_volume_m3 * volume_config.volume_relative_tolerance + 1.0e-15,
        )

    def test_step_export_labels_length_units_in_metres(self) -> None:
        geometry_config = self._geometry_config(tile_counts=(3, 2))
        volume_config = self._exact_volume_config()
        cad_reference_size_m = _resolve_cad_reference_size_m(geometry_config, volume_config)
        planform = _build_substrate_planform(
            polygons=[self._full_plate(geometry_config)],
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=cad_reference_size_m,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            step_path = Path(tmpdir) / "probe.step"
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.model.add("probe_units")
            try:
                _prepare_cad_occ_model(
                    planform=planform,
                    geometry_config=geometry_config,
                    volume_config=volume_config,
                    mesh_size_m=cad_reference_size_m,
                    partition_interface=False,
                )
                _export_occ_geometry_to_step(step_path)
            finally:
                gmsh.finalize()

            _normalize_step_length_units_to_metre(step_path)
            step_payload = step_path.read_text(encoding="utf-8")

        self.assertIn("SI_UNIT($,.METRE.)", step_payload)
        self.assertNotIn("SI_UNIT(.MILLI.,.METRE.)", step_payload)

    def test_unpartitioned_roundtrip_keeps_single_piezo_bottom_face(self) -> None:
        geometry_config = self._geometry_config()
        volume_config = self._exact_volume_config()
        _, _, report, _ = self._roundtrip_cad(
            [self._hole_planform(geometry_config)],
            geometry_config,
            volume_config,
            partition_interface=False,
        )

        self.assertEqual(report.solid_body_count, 2)
        self.assertEqual(report.piezo_bottom_face_count, 1)
        self.assertGreater(len(report.internal_vertical_substrate_face_tags), 0)

    def test_exact_cad_rejects_disconnected_planform(self) -> None:
        geometry_config = self._geometry_config(tile_counts=(4, 3))
        volume_config = self._exact_volume_config()

        with self.assertRaisesRegex(RuntimeError, "disconnected substrate planform"):
            self._roundtrip_cad(self._disconnected_parts(), geometry_config, volume_config)

    def test_repair_cad_adds_explicit_bridges_and_marks_report(self) -> None:
        geometry_config = self._geometry_config(tile_counts=(4, 3))
        volume_config = self._repair_volume_config()
        planform, _, report, payload = self._roundtrip_cad(self._disconnected_parts(), geometry_config, volume_config)

        self.assertTrue(planform.was_repaired)
        self.assertEqual(planform.initial_component_count, 2)
        self.assertEqual(report.solid_body_count, 2)
        self.assertEqual(report.stray_surface_tags, ())
        self.assertEqual(report.stray_curve_tags, ())
        self.assertGreater(report.piezo_bottom_face_count, 1)

        self.assertEqual(payload["cad_mode"], "repair")
        self.assertTrue(payload["repair_applied"])
        self.assertEqual(payload["initial_component_count"], 2)
        self.assertEqual(payload["step_roundtrip"]["solid_body_count"], 2)
        self.assertGreater(payload["step_roundtrip"]["piezo_bottom_face_count"], 1)

    def test_layered_tet_step_falls_back_to_raw_planform_when_cad_cleanup_changes_geometry_too_much(self) -> None:
        geometry_config = self._geometry_config(tile_counts=(4, 3))
        volume_config = VolumeMeshConfig(
            substrate_thickness_m=2.0e-3,
            piezo_thickness_m=1.0e-3,
            mesh_size_relative_to_cell=0.1,
            cad_reference_size_relative_to_cell=0.2,
            require_connected_substrate=True,
            solver_mesh_backend="layered_tet",
            cad_planform_simplify_relative_to_reference=0.0,
            cad_min_hole_area_relative_to_reference_squared=4.0,
        )
        small_hole_planform = self._full_plate(geometry_config).difference(box(9.0e-3, 6.75e-3, 1.05e-2, 8.25e-3))
        captured: dict[str, float] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            def fake_build_step_export_variant(*, planform, **_kwargs):
                captured["cad_planform_area_m2"] = float(planform.area_m2)
                return object(), object()

            def fake_build_layered_tet_solver_mesh(*, planform, **_kwargs):
                captured["solver_planform_area_m2"] = float(planform.area_m2)
                solver_path = output_dir / "plate3d_0007_fenicsx.npz"
                solver_path.write_bytes(b"stub")
                return solver_path

            with mock.patch(
                "peh_inverse_design.volume_mesh._build_step_export_variant",
                side_effect=fake_build_step_export_variant,
            ), mock.patch(
                "peh_inverse_design.volume_mesh._build_layered_tet_solver_mesh",
                side_effect=fake_build_layered_tet_solver_mesh,
            ), mock.patch(
                "peh_inverse_design.volume_mesh._write_cad_report",
            ), mock.patch(
                "peh_inverse_design.volume_mesh._write_ansys_face_selection_manifest",
            ):
                artifacts = _mesh_polygons_volume_sample_layered_tet(
                    polygons=[small_hole_planform],
                    sample_id=7,
                    output_dir=output_dir,
                    geometry_config=geometry_config,
                    volume_config=volume_config,
                )

        self.assertIsNotNone(artifacts)
        self.assertAlmostEqual(captured["cad_planform_area_m2"], captured["solver_planform_area_m2"])
        self.assertAlmostEqual(artifacts.planform.area_m2, captured["solver_planform_area_m2"])


    def test_single_face_manifest_reports_one_expected_bottom_region(self) -> None:
        geometry_config = self._geometry_config()
        volume_config = VolumeMeshConfig(
            substrate_thickness_m=2.0e-3,
            piezo_thickness_m=1.0e-3,
            mesh_size_relative_to_cell=0.2,
            require_connected_substrate=True,
            ansys_step_strategy="single_face_assembly",
        )
        _, _, roundtrip_report, _ = self._roundtrip_cad(
            [self._hole_planform(geometry_config)],
            geometry_config,
            volume_config,
            partition_interface=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "plate3d_0001_ansys_face_groups.json"
            _write_ansys_face_selection_manifest(
                manifest_path=manifest_path,
                sample_id=1,
                geometry_config=geometry_config,
                volume_config=volume_config,
                roundtrip_report=roundtrip_report,
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        notes = " ".join(payload["notes"])
        self.assertIn("continuous", notes)
        self.assertEqual(payload["named_selection_recipes"]["piezo_bottom_electrode"]["expected_region_count"], 1)

    def test_layered_tet_primary_ansys_step_can_switch_to_single_face_strategy(self) -> None:
        geometry_config = self._geometry_config(tile_counts=(4, 3))
        volume_config = VolumeMeshConfig(
            substrate_thickness_m=2.0e-3,
            piezo_thickness_m=1.0e-3,
            mesh_size_relative_to_cell=0.1,
            cad_reference_size_relative_to_cell=0.2,
            require_connected_substrate=True,
            solver_mesh_backend="layered_tet",
            ansys_step_strategy="single_face_assembly",
        )
        small_hole_planform = self._full_plate(geometry_config).difference(box(9.0e-3, 6.75e-3, 1.05e-2, 8.25e-3))
        captured: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            def fake_build_step_export_variant(*, partition_interface, require_single_piezo_bottom_face=False, **_kwargs):
                captured["partition_interface"] = bool(partition_interface)
                captured["require_single_piezo_bottom_face"] = bool(require_single_piezo_bottom_face)
                return object(), object()

            def fake_build_layered_tet_solver_mesh(**_kwargs):
                solver_path = output_dir / "plate3d_0008_fenicsx.npz"
                solver_path.write_bytes(b"stub")
                return solver_path

            with mock.patch(
                "peh_inverse_design.volume_mesh._build_step_export_variant",
                side_effect=fake_build_step_export_variant,
            ), mock.patch(
                "peh_inverse_design.volume_mesh._build_layered_tet_solver_mesh",
                side_effect=fake_build_layered_tet_solver_mesh,
            ), mock.patch(
                "peh_inverse_design.volume_mesh._write_cad_report",
            ), mock.patch(
                "peh_inverse_design.volume_mesh._write_ansys_face_selection_manifest",
            ):
                artifacts = _mesh_polygons_volume_sample_layered_tet(
                    polygons=[small_hole_planform],
                    sample_id=8,
                    output_dir=output_dir,
                    geometry_config=geometry_config,
                    volume_config=volume_config,
                )

        self.assertIsNotNone(artifacts)
        self.assertFalse(captured["partition_interface"])
        self.assertTrue(captured["require_single_piezo_bottom_face"])

    def test_layered_tet_solver_mesh_coarsens_until_q2_dofs_fit_limit(self) -> None:
        geometry_config = self._geometry_config(tile_counts=(1, 1))
        volume_config = VolumeMeshConfig(
            substrate_thickness_m=2.0e-3,
            piezo_thickness_m=1.0e-3,
            mesh_size_relative_to_cell=0.2,
            require_connected_substrate=True,
            substrate_layers=1,
            piezo_layers=1,
            max_solver_vector_dofs=4_000_000,
            solver_mesh_growth_factor=1.25,
            solver_mesh_limit_retries=3,
        )
        initial_mesh_size_m = geometry_config.cell_size_m[0] * volume_config.mesh_size_relative_to_cell
        planform = _build_substrate_planform(
            polygons=[self._full_plate(geometry_config)],
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=initial_mesh_size_m,
        )
        xy = np.asarray(
            [
                [0.0, 0.0],
                [geometry_config.plate_size_m[0], 0.0],
                [geometry_config.plate_size_m[0], geometry_config.plate_size_m[1]],
                [0.0, geometry_config.plate_size_m[1]],
            ],
            dtype=np.float64,
        )
        triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        substrate_mask = np.asarray([True, True], dtype=bool)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with mock.patch(
                "peh_inverse_design.volume_mesh._mesh_partitioned_full_plate_triangles",
                return_value=(xy, triangles, substrate_mask),
            ), mock.patch(
                "peh_inverse_design.volume_mesh._estimate_quadratic_vector_dofs",
                side_effect=[4_500_000, 3_900_000],
            ):
                solver_path = _build_layered_tet_solver_mesh(
                    planform=planform,
                    sample_id=11,
                    output_dir=output_dir,
                    geometry_config=geometry_config,
                    volume_config=volume_config,
                )

            payload = np.load(solver_path)
            self.assertEqual(int(payload["solver_mesh_coarsening_passes"][0]), 1)
            self.assertAlmostEqual(
                float(payload["solver_mesh_size_m"][0]),
                initial_mesh_size_m * volume_config.solver_mesh_growth_factor,
            )
            self.assertEqual(int(payload["estimated_q2_vector_dofs"][0]), 3_900_000)


if __name__ == "__main__":
    unittest.main()
