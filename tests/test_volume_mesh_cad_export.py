from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import gmsh
from shapely.geometry import box

from peh_inverse_design.geometry_pipeline import GeometryBuildConfig
from peh_inverse_design.volume_mesh import (
    VolumeMeshConfig,
    _build_substrate_planform,
    _export_occ_geometry_to_step,
    _prepare_cad_occ_model,
    _reload_occ_geometry_from_step,
    _resolve_cad_reference_size_m,
    _validate_current_occ_model,
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

    def _roundtrip_cad(self, polygons, geometry_config: GeometryBuildConfig, volume_config: VolumeMeshConfig):
        cad_reference_size_m = _resolve_cad_reference_size_m(geometry_config, volume_config)
        planform = _build_substrate_planform(
            polygons=polygons,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=cad_reference_size_m,
        )
        partition_interface = str(volume_config.solver_mesh_backend).strip().lower() == "gmsh_volume"

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
        self.assertEqual(report.piezo_bottom_face_count, 1)

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
        self.assertEqual(report.piezo_bottom_face_count, 1)

        self.assertEqual(payload["cad_mode"], "repair")
        self.assertTrue(payload["repair_applied"])
        self.assertEqual(payload["initial_component_count"], 2)
        self.assertEqual(payload["step_roundtrip"]["solid_body_count"], 2)
        self.assertEqual(payload["step_roundtrip"]["piezo_bottom_face_count"], 1)


if __name__ == "__main__":
    unittest.main()
