from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from peh_inverse_design.pipeline_runner import PipelineConfig, _run_solver_with_isolated_retry


class PipelineRunnerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
