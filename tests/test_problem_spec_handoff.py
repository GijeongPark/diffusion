from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from peh_inverse_design.problem_spec import write_ansys_workbench_handoff


@dataclass(frozen=True)
class _GeometryConfigStub:
    cell_size_m: tuple[float, float]
    tile_counts: tuple[int, int]

    @property
    def plate_size_m(self) -> tuple[float, float]:
        return (
            float(self.cell_size_m[0]) * int(self.tile_counts[0]),
            float(self.cell_size_m[1]) * int(self.tile_counts[1]),
        )


@dataclass(frozen=True)
class _VolumeConfigStub:
    substrate_thickness_m: float
    piezo_thickness_m: float
    ansys_step_strategy: str = "single_face_assembly"


class ProblemSpecHandoffTests(unittest.TestCase):
    def test_combined_workbench_handoff_includes_face_selection_manifest(self) -> None:
        geometry_config = _GeometryConfigStub(cell_size_m=(5.0e-3, 5.0e-3), tile_counts=(4, 3))
        volume_config = _VolumeConfigStub(
            substrate_thickness_m=1.0e-3,
            piezo_thickness_m=1.0e-4,
        )
        problem_spec = {
            "project": {"name": "unit-test"},
            "geometry": {},
            "materials": {},
            "mechanics": {},
            "electrical": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            handoff_path = Path(tmpdir) / "handoff.json"
            write_ansys_workbench_handoff(
                sample_id=7,
                output_path=handoff_path,
                step_path=Path(tmpdir) / "plate3d_0007.step",
                msh_path=None,
                cad_report_path=Path(tmpdir) / "plate3d_0007_cad.json",
                solver_mesh_path=Path(tmpdir) / "plate3d_0007_fenicsx.npz",
                geometry_config=geometry_config,
                volume_config=volume_config,
                problem_spec=problem_spec,
                inspection_single_face_step_path=Path(tmpdir) / "plate3d_0007_single_face_probe.step",
                face_selection_manifest_path=Path(tmpdir) / "plate3d_0007_ansys_face_groups.json",
            )
            payload = json.loads(handoff_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["ansys_workbench_geometry_import"]["recommended_step_variant"], "step_path")
        self.assertEqual(payload["files"]["step_path"].split("/")[-1], "plate3d_0007.step")
        self.assertEqual(
            payload["files"]["face_selection_manifest_path"].split("/")[-1],
            "plate3d_0007_ansys_face_groups.json",
        )
        notes = " ".join(payload["notes"])
        self.assertIn("one-file", notes)
        self.assertIn("face_selection_manifest_path", notes)
        self.assertIn("continuous", notes)
        import_sequence = " ".join(payload["ansys_workbench_geometry_import"]["recommended_import_sequence"])
        self.assertIn("step_path", import_sequence)
        self.assertIn("face_selection_manifest_path", import_sequence)


if __name__ == "__main__":
    unittest.main()
