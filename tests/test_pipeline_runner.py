from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from peh_inverse_design.pipeline_runner import (
    PipelineConfig,
    _apply_strict_ansys_parity_overrides,
    _build_mesh_command,
    _cli_parser,
    _run_solver_with_isolated_retry,
)


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

    def test_pipeline_config_defaults_to_peak_audit_voltage_form(self) -> None:
        config = PipelineConfig(
            source_unit_cell_npz="dummy.npz",
            exact_cad=True,
            repair_cad=False,
        )

        self.assertEqual(config.audit_ansys_voltage_form, "peak")

    def test_strict_ansys_parity_overrides_pipeline_args(self) -> None:
        parser = _cli_parser()
        args = parser.parse_args(
            [
                "--unit-cell-npz",
                "dummy.npz",
                "--strict-ansys-parity",
                "--mesh-preset",
                "default",
                "--solver-eigensolver-backend",
                "auto",
                "--allow-eigensolver-fallback",
            ]
        )

        updated = _apply_strict_ansys_parity_overrides(
            args,
            explicit_audit_voltage_form=False,
        )

        self.assertEqual(updated.mesh_preset, "ansys_parity")
        self.assertTrue(updated.strict_parity)
        self.assertEqual(updated.solver_eigensolver_backend, "shift_invert_lu")
        self.assertFalse(updated.allow_eigensolver_fallback)
        self.assertEqual(updated.audit_ansys_voltage_form, "peak")

    def test_strict_ansys_parity_keeps_explicit_audit_voltage_override(self) -> None:
        parser = _cli_parser()
        args = parser.parse_args(
            [
                "--unit-cell-npz",
                "dummy.npz",
                "--strict-ansys-parity",
                "--audit-ansys-voltage-form",
                "rms",
            ]
        )

        updated = _apply_strict_ansys_parity_overrides(
            args,
            explicit_audit_voltage_form=True,
        )

        self.assertEqual(updated.audit_ansys_voltage_form, "rms")

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
                "peh_inverse_design.pipeline_runner._build_solver_docker_command",
                side_effect=fake_build_solver_docker_command,
            ), mock.patch(
                "peh_inverse_design.pipeline_runner._run_command",
                side_effect=[
                    subprocess.CalledProcessError(137, ["docker", "run"]),
                    subprocess.CalledProcessError(137, ["docker", "run"]),
                    None,
                ],
            ), mock.patch(
                "peh_inverse_design.pipeline_runner._solver_outputs_exist",
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

    def test_isolated_retry_disables_lower_order_fallback_for_strict_parity_runs(self) -> None:
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
                mesh_preset="ansys_parity",
                strict_parity=True,
                solver_element_order=2,
                solver_oom_fallback_element_order=1,
            )
            requested_orders: list[int | None] = []

            def fake_build_solver_docker_command(*, element_order=None, **_kwargs):
                requested_orders.append(None if element_order is None else int(element_order))
                return ["docker", "run"]

            with mock.patch(
                "peh_inverse_design.pipeline_runner._build_solver_docker_command",
                side_effect=fake_build_solver_docker_command,
            ), mock.patch(
                "peh_inverse_design.pipeline_runner._run_command",
                side_effect=[
                    subprocess.CalledProcessError(137, ["docker", "run"]),
                    subprocess.CalledProcessError(137, ["docker", "run"]),
                ],
            ), mock.patch(
                "peh_inverse_design.pipeline_runner._solver_outputs_exist",
                return_value=False,
            ):
                with self.assertRaises(RuntimeError) as raised:
                    _run_solver_with_isolated_retry(
                        mesh_files=[mesh_path],
                        project_root=root,
                        response_dir=response_dir,
                        modal_dir=modal_dir,
                        config=config,
                        runtime_problem_spec_path=None,
                    )
                self.assertIn("parity-invalid", str(raised.exception))
                self.assertEqual(requested_orders, [None, None])
                self.assertTrue((response_dir / "sample_0002_solver_diagnostic.json").exists())


if __name__ == "__main__":
    unittest.main()
