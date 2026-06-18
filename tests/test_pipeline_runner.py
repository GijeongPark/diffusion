from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from peh_inverse_design.pipeline.pipeline_runner import (
    PipelineConfig,
    _build_frf_physical_export_command,
    _build_geometry_parity_command,
    _build_mesh_command,
    _build_solver_docker_command,
    _effective_mesh_builder_settings,
    _build_solver_inner_args,
    _cli_parser,
    _run_solver_with_isolated_retry,
    _solver_outputs_exist,
)
from peh_inverse_design.datasets.response_dataset import save_fem_response


def _mesh_provenance(mesh_path: Path) -> dict[str, np.ndarray]:
    return {"mesh_file_sha256": np.asarray([hashlib.sha256(mesh_path.read_bytes()).hexdigest()])}


class PipelineRunnerTests(unittest.TestCase):
    def test_build_mesh_command_uses_preset_without_explicit_overrides(self) -> None:
        config = PipelineConfig(
            source_unit_cell_npz="dummy.npz",
            exact_cad=True,
            repair_cad=False,
            mesh_preset="ansys_parity",
        )

        cmd = _build_mesh_command(
            project_python=Path("/tmp/python"),
            candidate_unit_cell_npz=Path("/tmp/unit_cell.npz"),
            mesh_dir=Path("/tmp/meshes"),
            config=config,
            runtime_problem_spec_path=None,
        )

        self.assertIn("--mesh-preset", cmd)
        self.assertIn("ansys_parity", cmd)
        self.assertNotIn("--substrate-layers", cmd)
        self.assertNotIn("--piezo-layers", cmd)
        self.assertNotIn("--solver-max-q2-vector-dofs", cmd)

    def test_build_frf_physical_export_command_forwards_sweep(self) -> None:
        config = PipelineConfig(
            source_unit_cell_npz="dummy.npz",
            frf_physical_export_freq_min_hz=0.0,
            frf_physical_export_freq_max_hz=2.0,
            frf_physical_export_points=201,
        )

        cmd = _build_frf_physical_export_command(
            project_python=Path("/tmp/python"),
            modal_dir=Path("/tmp/modal"),
            frf_physical_dir=Path("/tmp/frf_physical"),
            config=config,
        )

        rendered = [str(value) for value in cmd]
        self.assertIn("peh_inverse_design.validation.export_physical_frf", rendered)
        self.assertEqual(rendered[rendered.index("--freq-min-hz") + 1], "0.0")
        self.assertEqual(rendered[rendered.index("--freq-max-hz") + 1], "2.0")
        self.assertEqual(rendered[rendered.index("--points") + 1], "201")

    def test_build_geometry_parity_command_forwards_paths(self) -> None:
        cmd = _build_geometry_parity_command(
            project_python=Path("/tmp/python"),
            mesh_dir=Path("/tmp/meshes"),
            geometry_parity_dir=Path("/tmp/reports/geometry_parity"),
            summary_csv_path=Path("/tmp/reports/geometry_parity.csv"),
        )

        rendered = [str(value) for value in cmd]
        self.assertIn("peh_inverse_design.validation.geometry_parity", rendered)
        self.assertEqual(rendered[rendered.index("--mesh-dir") + 1], str(Path("/tmp/meshes")))
        self.assertEqual(
            rendered[rendered.index("--summary-csv") + 1],
            str(Path("/tmp/reports/geometry_parity.csv")),
        )

    def test_cli_parser_accepts_frf_export_and_parity_flags(self) -> None:
        parser = _cli_parser()
        args = parser.parse_args(
            [
                "--unit-cell-npz",
                "dummy.npz",
                "--frf-export-freq-min-hz",
                "0.0",
                "--frf-export-freq-max-hz",
                "2.0",
                "--frf-export-points",
                "101",
                "--verify-geometry-parity",
            ]
        )
        self.assertEqual(float(args.frf_export_freq_min_hz), 0.0)
        self.assertEqual(float(args.frf_export_freq_max_hz), 2.0)
        self.assertEqual(int(args.frf_export_points), 101)
        self.assertTrue(bool(args.verify_geometry_parity))

    def test_build_mesh_command_keeps_explicit_overrides(self) -> None:
        config = PipelineConfig(
            source_unit_cell_npz="dummy.npz",
            exact_cad=True,
            repair_cad=False,
            mesh_preset="ansys_parity",
            substrate_layers=6,
            piezo_layers=2,
            solver_max_q2_vector_dofs=4_000_000,
        )

        cmd = _build_mesh_command(
            project_python=Path("/tmp/python"),
            candidate_unit_cell_npz=Path("/tmp/unit_cell.npz"),
            mesh_dir=Path("/tmp/meshes"),
            config=config,
            runtime_problem_spec_path=None,
        )

        self.assertEqual(cmd[cmd.index("--substrate-layers") + 1], "6")
        self.assertEqual(cmd[cmd.index("--piezo-layers") + 1], "2")
        self.assertEqual(cmd[cmd.index("--solver-max-q2-vector-dofs") + 1], "4000000")

    def test_explicit_parity_layers_auto_align_mesh_preset(self) -> None:
        config = PipelineConfig(
            source_unit_cell_npz="dummy.npz",
            exact_cad=True,
            repair_cad=False,
            mesh_preset="default",
            substrate_layers=8,
            piezo_layers=3,
        )

        effective = _effective_mesh_builder_settings(config)
        self.assertEqual(effective["mesh_preset"], "ansys_parity")
        self.assertEqual(effective["substrate_layers"], 8)
        self.assertEqual(effective["piezo_layers"], 3)
        self.assertIsNone(effective["solver_max_q2_vector_dofs"])

        cmd = _build_mesh_command(
            project_python=Path("/tmp/python"),
            candidate_unit_cell_npz=Path("/tmp/unit_cell.npz"),
            mesh_dir=Path("/tmp/meshes"),
            config=config,
            runtime_problem_spec_path=None,
        )

        self.assertEqual(cmd[cmd.index("--mesh-preset") + 1], "ansys_parity")
        self.assertEqual(cmd[cmd.index("--substrate-layers") + 1], "8")
        self.assertEqual(cmd[cmd.index("--piezo-layers") + 1], "3")
        self.assertNotIn("--solver-max-q2-vector-dofs", cmd)

    def test_pipeline_config_disables_oom_fallback_by_default(self) -> None:
        config = PipelineConfig(
            source_unit_cell_npz="dummy.npz",
            exact_cad=True,
            repair_cad=False,
        )

        self.assertIsNone(config.solver_oom_fallback_element_order)

    def test_pipeline_config_treats_nonpositive_oom_fallback_as_disabled(self) -> None:
        config = PipelineConfig(
            source_unit_cell_npz="dummy.npz",
            exact_cad=True,
            repair_cad=False,
            solver_oom_fallback_element_order=0,
        )

        self.assertIsNone(config.solver_oom_fallback_element_order)

    def test_pipeline_config_validates_eigensolver_backend_and_mpi_ranks(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported eigensolver backend"):
            PipelineConfig(
                source_unit_cell_npz="dummy.npz",
                exact_cad=True,
                repair_cad=False,
                solver_eigensolver_backend="unknown",
            )
        with self.assertRaisesRegex(ValueError, "solver_mpi_ranks"):
            PipelineConfig(
                source_unit_cell_npz="dummy.npz",
                exact_cad=True,
                repair_cad=False,
                solver_mpi_ranks=0,
            )
        config = PipelineConfig(
            source_unit_cell_npz="dummy.npz",
            exact_cad=True,
            repair_cad=False,
            solver_eigensolver_backend="shift_invert_cholesky",
            solver_eigensolver_fallback_backends="shift_invert_cholesky,iterative_gd",
        )
        self.assertEqual(config.solver_eigensolver_fallback_backends, ("iterative_gd_gamg",))

    def test_isolated_retry_falls_back_to_lower_order_after_oom(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_dir = root / "meshes"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = mesh_dir / "plate3d_0002_fenicsx.npz"
            mesh_path.write_bytes(b"stub")
            response_dir = root / "responses"
            modal_dir = root / "modal"
            config = PipelineConfig(
                source_unit_cell_npz=root / "dummy.npz",
                exact_cad=True,
                repair_cad=False,
                solver_element_order=2,
                solver_oom_fallback_element_order=1,
            )
            requested_orders: list[int | None] = []

            def fake_build_solver_docker_command(*, element_order=None, **_kwargs):
                requested_orders.append(None if element_order is None else int(element_order))
                return ["docker", "run"]

            with mock.patch(
                "peh_inverse_design.pipeline.pipeline_runner._build_solver_docker_command",
                side_effect=fake_build_solver_docker_command,
            ), mock.patch(
                "peh_inverse_design.pipeline.pipeline_runner._run_command",
                side_effect=[
                    subprocess.CalledProcessError(137, ["docker", "run"]),
                    subprocess.CalledProcessError(137, ["docker", "run"]),
                    None,
                ],
            ), mock.patch(
                "peh_inverse_design.pipeline.pipeline_runner._solver_outputs_exist",
                return_value=False,
            ):
                _run_solver_with_isolated_retry(
                    mesh_files=[mesh_path],
                    project_root=root,
                    response_dir=response_dir,
                    modal_dir=modal_dir,
                    config=config,
                    runtime_problem_spec_path=None,
                )

        self.assertEqual(requested_orders, [None, None, 1])

    def test_isolated_retry_does_not_fallback_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_dir = root / "meshes"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = mesh_dir / "plate3d_0002_fenicsx.npz"
            mesh_path.write_bytes(b"stub")
            response_dir = root / "responses"
            modal_dir = root / "modal"
            config = PipelineConfig(
                source_unit_cell_npz=root / "dummy.npz",
                exact_cad=True,
                repair_cad=False,
                solver_element_order=2,
            )
            requested_orders: list[int | None] = []

            def fake_build_solver_docker_command(*, element_order=None, **_kwargs):
                requested_orders.append(None if element_order is None else int(element_order))
                return ["docker", "run"]

            with mock.patch(
                "peh_inverse_design.pipeline.pipeline_runner._build_solver_docker_command",
                side_effect=fake_build_solver_docker_command,
            ), mock.patch(
                "peh_inverse_design.pipeline.pipeline_runner._run_command",
                side_effect=[
                    subprocess.CalledProcessError(137, ["docker", "run"]),
                    subprocess.CalledProcessError(137, ["docker", "run"]),
                ],
            ), mock.patch(
                "peh_inverse_design.pipeline.pipeline_runner._solver_outputs_exist",
                return_value=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "exit status 137"):
                    _run_solver_with_isolated_retry(
                        mesh_files=[mesh_path],
                        project_root=root,
                        response_dir=response_dir,
                        modal_dir=modal_dir,
                        config=config,
                        runtime_problem_spec_path=None,
                    )

        self.assertEqual(requested_orders, [None, None])

    def test_isolated_retry_can_restart_with_backend_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_dir = root / "meshes"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = mesh_dir / "plate3d_0002_fenicsx.npz"
            mesh_path.write_bytes(b"stub")
            response_dir = root / "responses"
            modal_dir = root / "modal"
            config = PipelineConfig(
                source_unit_cell_npz=root / "dummy.npz",
                exact_cad=True,
                repair_cad=False,
                solver_eigensolver_fallback_backends=("iterative_lobpcg_gamg",),
            )
            requested_backends: list[str | None] = []

            def fake_build_solver_docker_command(*, eigensolver_backend=None, **_kwargs):
                requested_backends.append(None if eigensolver_backend is None else str(eigensolver_backend))
                return ["docker", "run"]

            with mock.patch(
                "peh_inverse_design.pipeline.pipeline_runner._build_solver_docker_command",
                side_effect=fake_build_solver_docker_command,
            ), mock.patch(
                "peh_inverse_design.pipeline.pipeline_runner._run_command",
                side_effect=[
                    subprocess.CalledProcessError(137, ["docker", "run"]),
                    subprocess.CalledProcessError(137, ["docker", "run"]),
                    None,
                ],
            ), mock.patch(
                "peh_inverse_design.pipeline.pipeline_runner._solver_outputs_exist",
                return_value=False,
            ):
                _run_solver_with_isolated_retry(
                    mesh_files=[mesh_path],
                    project_root=root,
                    response_dir=response_dir,
                    modal_dir=modal_dir,
                    config=config,
                    runtime_problem_spec_path=None,
                )

        self.assertEqual(requested_backends, [None, None, "iterative_lobpcg_gamg"])

    def test_build_solver_inner_args_forwards_peak_voltage_convention(self) -> None:
        config = PipelineConfig(
            source_unit_cell_npz="dummy.npz",
            exact_cad=True,
            repair_cad=False,
            substrate_rho=7930.0,
            piezo_rho=7500.0,
            house_voltage_amplitude_convention="peak",
        )

        cmd = _build_solver_inner_args(
            project_root=Path("/tmp/project"),
            response_dir=Path("/tmp/project/data/fem_responses"),
            modal_dir=Path("/tmp/project/data/modal_data"),
            config=config,
            runtime_problem_spec_path=None,
            mesh_path=Path("/tmp/project/meshes/plate3d_0000_fenicsx.npz"),
        )

        self.assertEqual(
            cmd[cmd.index("--house-voltage-amplitude-convention") + 1],
            "peak",
        )
        self.assertEqual(
            cmd[cmd.index("--eigensolver-backend") + 1],
            "shift_invert_cholesky",
        )

    def test_build_solver_inner_args_uses_mpiexec_when_requested(self) -> None:
        config = PipelineConfig(
            source_unit_cell_npz="dummy.npz",
            exact_cad=True,
            repair_cad=False,
            substrate_rho=7930.0,
            piezo_rho=7500.0,
            house_voltage_amplitude_convention="peak",
            solver_mpi_ranks=4,
        )

        cmd = _build_solver_inner_args(
            project_root=Path("/tmp/project"),
            response_dir=Path("/tmp/project/data/fem_responses"),
            modal_dir=Path("/tmp/project/data/modal_data"),
            config=config,
            runtime_problem_spec_path=None,
            mesh_path=Path("/tmp/project/meshes/plate3d_0000_fenicsx.npz"),
        )

        self.assertEqual(cmd[:3], ["mpiexec", "-n", "4"])
        self.assertIn("python3", cmd)

    def test_build_solver_docker_command_sets_mpi_environment(self) -> None:
        config = PipelineConfig(
            source_unit_cell_npz="dummy.npz",
            exact_cad=True,
            repair_cad=False,
            substrate_rho=7930.0,
            piezo_rho=7500.0,
            house_voltage_amplitude_convention="peak",
            solver_mpi_ranks=4,
        )

        cmd = _build_solver_docker_command(
            project_root=Path("/tmp/project"),
            response_dir=Path("/tmp/project/data/fem_responses"),
            modal_dir=Path("/tmp/project/data/modal_data"),
            config=config,
            runtime_problem_spec_path=None,
            mesh_path=Path("/tmp/project/meshes/plate3d_0000_fenicsx.npz"),
        )

        shell_command = cmd[-1]
        self.assertIn("OMPI_ALLOW_RUN_AS_ROOT=1", shell_command)
        self.assertIn("OMP_NUM_THREADS=1 mpiexec -n 4", shell_command)

    def test_pipeline_config_rejects_rms_voltage_convention(self) -> None:
        with self.assertRaisesRegex(ValueError, "RMS handling was removed"):
            PipelineConfig(
                source_unit_cell_npz="dummy.npz",
                exact_cad=True,
                repair_cad=False,
                house_voltage_amplitude_convention="rms",
            )

    def test_cli_parser_no_longer_exposes_ansys_audit_flags_and_only_accepts_peak_voltage_convention(self) -> None:
        parser = _cli_parser()
        option_strings = {option for action in parser._actions for option in action.option_strings}

        self.assertNotIn("--audit-ansys-modal-hz", option_strings)
        self.assertNotIn("--audit-ansys-frf-peak-hz", option_strings)
        self.assertNotIn("--audit-ansys-voltage-v", option_strings)
        self.assertNotIn("--audit-ansys-voltage-form", option_strings)
        self.assertNotIn("--audit-sample-id", option_strings)

        action = next(action for action in parser._actions if "--house-voltage-amplitude-convention" in action.option_strings)
        self.assertEqual(action.choices, ["peak"])
        fallback_action = next(action for action in parser._actions if "--solver-oom-fallback-element-order" in action.option_strings)
        self.assertEqual(fallback_action.default, 0)

    def test_solver_outputs_exist_accepts_peak_and_rejects_rms_tagged_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_dir = root / "meshes"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = mesh_dir / "plate3d_0002_fenicsx.npz"
            mesh_path.write_bytes(b"stub")
            response_dir = root / "responses"
            modal_dir = root / "modal"
            modal_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(modal_dir / "sample_0002_modal.npz", **_mesh_provenance(mesh_path))
            response_path = save_fem_response(
                sample_id=2,
                f_peak_hz=1.0,
                freq_hz=[0.9, 1.0, 1.1],
                voltage_mag=[1.0, 2.0, 1.5],
                output_dir=response_dir,
                metadata=_mesh_provenance(mesh_path),
            )

            self.assertTrue(_solver_outputs_exist(mesh_path, response_dir, modal_dir, "peak"))

            with np.load(response_path, allow_pickle=True) as response:
                tagged_rms_payload = {key: np.asarray(response[key]) for key in response.files}
            tagged_rms_payload["peak_voltage_form"] = np.asarray("rms")
            np.savez_compressed(response_path, **tagged_rms_payload)

            self.assertFalse(_solver_outputs_exist(mesh_path, response_dir, modal_dir, "peak"))

    def test_solver_outputs_exist_rejects_wrong_modal_element_order_when_expected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_dir = root / "meshes"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = mesh_dir / "plate3d_0002_fenicsx.npz"
            mesh_path.write_bytes(b"stub")
            response_dir = root / "responses"
            modal_dir = root / "modal"
            modal_dir.mkdir(parents=True, exist_ok=True)
            save_fem_response(
                sample_id=2,
                f_peak_hz=1.0,
                freq_hz=[0.9, 1.0, 1.1],
                voltage_mag=[1.0, 2.0, 1.5],
                output_dir=response_dir,
                metadata=_mesh_provenance(mesh_path),
            )
            np.savez_compressed(
                modal_dir / "sample_0002_modal.npz",
                element_order=np.asarray([1], dtype=np.int32),
                **_mesh_provenance(mesh_path),
            )

            self.assertFalse(
                _solver_outputs_exist(
                    mesh_path,
                    response_dir,
                    modal_dir,
                    "peak",
                    expected_element_order=2,
                )
            )
            self.assertTrue(
                _solver_outputs_exist(
                    mesh_path,
                    response_dir,
                    modal_dir,
                    "peak",
                    expected_element_order=1,
                )
            )


if __name__ == "__main__":
    unittest.main()
