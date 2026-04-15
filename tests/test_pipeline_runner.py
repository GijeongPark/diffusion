from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from peh_inverse_design.pipeline_runner import (
    PipelineConfig,
    _build_mesh_command,
    _build_solver_inner_args,
    _cli_parser,
    _run_solver_with_isolated_retry,
    _solver_outputs_exist,
)
from peh_inverse_design.response_dataset import save_fem_response


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
            (modal_dir / "sample_0002_modal.npz").write_bytes(b"stub")
            response_path = save_fem_response(
                sample_id=2,
                f_peak_hz=1.0,
                freq_hz=[0.9, 1.0, 1.1],
                voltage_mag=[1.0, 2.0, 1.5],
                output_dir=response_dir,
            )

            self.assertTrue(_solver_outputs_exist(mesh_path, response_dir, modal_dir, "peak"))

            with np.load(response_path, allow_pickle=True) as response:
                tagged_rms_payload = {key: np.asarray(response[key]) for key in response.files}
            tagged_rms_payload["peak_voltage_form"] = np.asarray("rms")
            np.savez_compressed(response_path, **tagged_rms_payload)

            self.assertFalse(_solver_outputs_exist(mesh_path, response_dir, modal_dir, "peak"))


if __name__ == "__main__":
    unittest.main()
