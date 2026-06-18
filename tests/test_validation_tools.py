from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from peh_inverse_design.geometry.geometry_pipeline import GeometryBuildConfig
from peh_inverse_design.meshing.volume_mesh import (
    VolumeMeshConfig,
    _normalize_step_length_units_to_metre,
    gmsh,
    mesh_plain_plate_volume_sample,
)
from peh_inverse_design.pipeline.pipeline_runner import PipelineConfig
from peh_inverse_design.solver.modal_frf import evaluate_voltage_frf, load_modal_model
from peh_inverse_design.validation.export_physical_frf import (
    build_physical_frequency_grid,
    compute_physical_voltage_frf,
    export_physical_voltage_frf_csv,
)
from peh_inverse_design.validation.geometry_parity import (
    solver_mesh_geometry_summary,
    step_geometry_summary,
    verify_sample_geometry_parity,
)


SINGLE_MODE_MODEL = {
    "eigenfreq_hz": np.asarray([1.2], dtype=np.float64),
    "modal_theta": np.asarray([0.4], dtype=np.float64),
    "modal_force": np.asarray([2.5], dtype=np.float64),
    "capacitance_f": np.asarray([1.5e-8], dtype=np.float64),
}


def _single_mode_voltage_closed_form(
    f_hz: float,
    damping_ratio: float,
    resistance_ohm: float,
) -> complex:
    omega = 2.0 * math.pi * f_hz
    omega_n = 2.0 * math.pi * float(SINGLE_MODE_MODEL["eigenfreq_hz"][0])
    theta = float(SINGLE_MODE_MODEL["modal_theta"][0])
    force = float(SINGLE_MODE_MODEL["modal_force"][0])
    capacitance = float(SINGLE_MODE_MODEL["capacitance_f"][0])
    mechanical = omega_n ** 2 - omega ** 2 + 2j * damping_ratio * omega_n * omega
    electrical = (1.0 / resistance_ohm) + 1j * omega * capacitance
    return -1j * omega * theta * force / (electrical * mechanical + 1j * omega * theta ** 2)


def _write_modal_npz(path: Path, sample_id: int = 7) -> Path:
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_id, dtype=np.int32),
        eigenfreq_hz=SINGLE_MODE_MODEL["eigenfreq_hz"],
        modal_theta=SINGLE_MODE_MODEL["modal_theta"],
        modal_force=SINGLE_MODE_MODEL["modal_force"],
        modal_mass=np.asarray([1.0], dtype=np.float64),
        capacitance_f=SINGLE_MODE_MODEL["capacitance_f"],
        damping_ratio=np.asarray(0.02, dtype=np.float64),
        resistance_ohm=np.asarray(1.0e4, dtype=np.float64),
        base_acceleration_m_per_s2=np.asarray(2.5, dtype=np.float64),
    )
    return path


class ModalFrfTests(unittest.TestCase):
    def test_single_mode_voltage_matches_closed_form(self) -> None:
        frequencies = np.asarray([0.4, 1.0, 1.2, 1.7], dtype=np.float64)
        voltage = evaluate_voltage_frf(
            frequencies_hz=frequencies,
            modal_model=SINGLE_MODE_MODEL,
            damping_ratio=0.02,
            resistance_ohm=1.0e4,
        )
        expected = np.asarray(
            [_single_mode_voltage_closed_form(f, 0.02, 1.0e4) for f in frequencies],
            dtype=np.complex128,
        )
        np.testing.assert_allclose(voltage, expected, rtol=1.0e-12)

    def test_zero_frequency_gives_zero_voltage(self) -> None:
        voltage = evaluate_voltage_frf(
            frequencies_hz=np.asarray([0.0]),
            modal_model=SINGLE_MODE_MODEL,
            damping_ratio=0.02,
            resistance_ohm=1.0e4,
        )
        self.assertEqual(abs(voltage[0]), 0.0)

    def test_load_modal_model_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            modal_path = _write_modal_npz(Path(tmp) / "sample_0007_modal.npz")
            model = load_modal_model(modal_path)
        self.assertEqual(int(model["sample_id"]), 7)
        self.assertEqual(float(model["damping_ratio"]), 0.02)
        self.assertEqual(float(model["resistance_ohm"]), 1.0e4)
        np.testing.assert_allclose(model["eigenfreq_hz"], SINGLE_MODE_MODEL["eigenfreq_hz"])

    def test_load_modal_model_rejects_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_0001_modal.npz"
            np.savez_compressed(path, eigenfreq_hz=np.asarray([1.0]))
            with self.assertRaises(KeyError):
                load_modal_model(path)


class ExportPhysicalFrfTests(unittest.TestCase):
    def test_grid_validation(self) -> None:
        with self.assertRaises(ValueError):
            build_physical_frequency_grid(-0.1, 2.0, 10)
        with self.assertRaises(ValueError):
            build_physical_frequency_grid(2.0, 2.0, 10)
        with self.assertRaises(ValueError):
            build_physical_frequency_grid(0.0, 2.0, 1)

    def test_export_csv_roundtrip_matches_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            modal_path = _write_modal_npz(Path(tmp) / "sample_0007_modal.npz")
            csv_path = export_physical_voltage_frf_csv(
                modal_npz_path=modal_path,
                output_csv_path=Path(tmp) / "sample_0007_voltage_frf_hz.csv",
                freq_min_hz=0.0,
                freq_max_hz=2.0,
                num_points=21,
            )
            freq, voltage = np.loadtxt(csv_path, delimiter=",", skiprows=7, unpack=True)
        expected = np.abs(
            np.asarray(
                [_single_mode_voltage_closed_form(f, 0.02, 1.0e4) for f in freq],
                dtype=np.complex128,
            )
        )
        np.testing.assert_allclose(freq, np.linspace(0.0, 2.0, 21), rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(voltage, expected, rtol=1.0e-9)

    def test_sweep_beyond_retained_modes_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            modal_path = _write_modal_npz(Path(tmp) / "sample_0007_modal.npz")
            with self.assertWarns(UserWarning):
                compute_physical_voltage_frf(
                    modal_npz_path=modal_path,
                    frequencies_hz=np.asarray([0.5, 5.0]),
                )


class PipelineConfigValidationTests(unittest.TestCase):
    _BASE_KWARGS = {"source_unit_cell_npz": "data/unit_cell_dataset.npz"}

    def test_frf_export_bounds_must_come_together(self) -> None:
        with self.assertRaises(ValueError):
            PipelineConfig(**self._BASE_KWARGS, frf_physical_export_freq_min_hz=0.0)

    def test_frf_export_bounds_must_be_ordered(self) -> None:
        with self.assertRaises(ValueError):
            PipelineConfig(
                **self._BASE_KWARGS,
                frf_physical_export_freq_min_hz=2.0,
                frf_physical_export_freq_max_hz=1.0,
            )

    def test_frf_export_accepts_valid_sweep(self) -> None:
        config = PipelineConfig(
            **self._BASE_KWARGS,
            frf_physical_export_freq_min_hz=0.0,
            frf_physical_export_freq_max_hz=2.0,
            frf_physical_export_points=201,
            verify_geometry_parity=True,
        )
        self.assertEqual(float(config.frf_physical_export_freq_max_hz), 2.0)
        self.assertTrue(config.verify_geometry_parity)


def _write_two_body_step(
    step_path: Path,
    plate_lx: float,
    plate_ly: float,
    substrate_thickness_m: float,
    piezo_thickness_m: float,
    substrate_lx: float | None = None,
) -> Path:
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("parity_test_step")
    try:
        gmsh.model.occ.addBox(
            0.0, 0.0, 0.0, substrate_lx if substrate_lx is not None else plate_lx, plate_ly, substrate_thickness_m
        )
        gmsh.model.occ.addBox(0.0, 0.0, substrate_thickness_m, plate_lx, plate_ly, piezo_thickness_m)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_path))
    finally:
        gmsh.finalize()
    _normalize_step_length_units_to_metre(step_path)
    return step_path


class GeometryParityTests(unittest.TestCase):
    _PLATE_LX = 0.1
    _PLATE_LY = 0.1
    _SUBSTRATE_T = 1.0e-3
    _PIEZO_T = 1.0e-4

    def _build_plain_plate_mesh(self, output_dir: Path) -> Path:
        geometry_config = GeometryBuildConfig(cell_size_m=(0.05, 0.05), tile_counts=(2, 2))
        volume_config = VolumeMeshConfig(
            substrate_thickness_m=self._SUBSTRATE_T,
            piezo_thickness_m=self._PIEZO_T,
            mesh_size_relative_to_cell=0.5,
            substrate_layers=1,
            piezo_layers=1,
            max_solver_vector_dofs=None,
        )
        mesh_path = mesh_plain_plate_volume_sample(
            sample_id=0,
            output_dir=output_dir,
            geometry_config=geometry_config,
            volume_config=volume_config,
        )
        self.assertIsNotNone(mesh_path)
        return Path(mesh_path)

    def test_solver_mesh_summary_measures_plain_plate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = self._build_plain_plate_mesh(Path(tmp))
            summary = solver_mesh_geometry_summary(mesh_path)
        self.assertAlmostEqual(
            float(summary["planform_area_m2"]), self._PLATE_LX * self._PLATE_LY, places=12
        )
        self.assertAlmostEqual(
            float(summary["substrate_volume_m3"]),
            self._PLATE_LX * self._PLATE_LY * self._SUBSTRATE_T,
            places=15,
        )
        self.assertAlmostEqual(
            float(summary["piezo_volume_m3"]),
            self._PLATE_LX * self._PLATE_LY * self._PIEZO_T,
            places=15,
        )

    def test_matching_step_passes_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mesh_path = self._build_plain_plate_mesh(tmp_path)
            step_path = _write_two_body_step(
                tmp_path / "plate3d_0000.step",
                self._PLATE_LX,
                self._PLATE_LY,
                self._SUBSTRATE_T,
                self._PIEZO_T,
            )
            report = verify_sample_geometry_parity(mesh_npz_path=mesh_path, step_path=step_path)
        self.assertTrue(report["parity_ok"], msg=str(report["checks"]))
        self.assertLess(float(report["metrics"]["planform_symdiff_rel"]), 1.0e-9)

    def test_mismatched_step_fails_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mesh_path = self._build_plain_plate_mesh(tmp_path)
            step_path = _write_two_body_step(
                tmp_path / "plate3d_0000.step",
                self._PLATE_LX,
                self._PLATE_LY,
                self._SUBSTRATE_T,
                self._PIEZO_T,
                substrate_lx=0.95 * self._PLATE_LX,
            )
            report = verify_sample_geometry_parity(mesh_npz_path=mesh_path, step_path=step_path)
        self.assertFalse(report["parity_ok"])
        self.assertAlmostEqual(float(report["metrics"]["planform_symdiff_rel"]), 0.05, places=6)
        self.assertAlmostEqual(float(report["metrics"]["substrate_volume_rel_diff"]), 0.05, places=6)
        failed_names = {str(check["name"]) for check in report["checks"] if not check["ok"]}
        self.assertIn("planform_symdiff_rel", failed_names)
        self.assertIn("substrate_volume_rel_diff", failed_names)

    def test_step_summary_reads_two_body_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            step_path = _write_two_body_step(
                Path(tmp) / "plate3d_0000.step",
                self._PLATE_LX,
                self._PLATE_LY,
                self._SUBSTRATE_T,
                self._PIEZO_T,
            )
            summary = step_geometry_summary(step_path)
        self.assertAlmostEqual(
            float(summary["planform_area_m2"]), self._PLATE_LX * self._PLATE_LY, places=10
        )
        self.assertAlmostEqual(
            float(summary["substrate_volume_m3"]),
            self._PLATE_LX * self._PLATE_LY * self._SUBSTRATE_T,
            places=14,
        )
        # OCC bounding boxes are padded by their internal tolerance (~1e-7 m),
        # which is why geometry parity uses an absolute 1e-6 m bbox tolerance.
        self.assertAlmostEqual(
            float(summary["piezo_z_range_m"][1]),
            self._SUBSTRATE_T + self._PIEZO_T,
            delta=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
