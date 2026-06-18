"""Per-sample geometry parity between the solver mesh and the ANSYS STEP export.

The surrogate FEM path analyzes the layered tetra mesh built from the *source*
substrate planform, while the STEP file handed to ANSYS is built from the
*CAD-simplified* planform (the simplification is capped by
``max_cad_planform_symdiff_relative_to_source``, default 1e-3). This tool
measures the difference that actually shipped, per sample:

* planform shape: XOR (symmetric-difference) area between the solver-mesh
  substrate footprint and the STEP substrate footprint, relative to the mesh
  footprint area;
* planform area, substrate volume, piezo volume: relative differences;
* plate bounding box and layer thicknesses: absolute differences.

The solver-mesh footprint is reconstructed from the bottom-plate facet
triangles stored in ``plate3d_XXXX_fenicsx.npz``; the STEP footprint is
reconstructed by importing the STEP with OpenCASCADE, surface-meshing the
substrate bottom faces, and unioning their triangles (all planform edges are
straight lines, so the triangulated footprint is exact). Results are
cross-checked against the areas recorded in ``plate3d_XXXX_cad.json``.

Default tolerances are derived from the export-time guarantees: the CAD
planform may differ from the source planform by up to 1e-3 relative symdiff,
and each geometry stage validates volumes to 5e-4 relative, so mesh-vs-STEP
volumes can legitimately differ by up to ~2e-3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from peh_inverse_design.core.mesh_tags import (
        FACET_BOTTOM_PLATE_TAG,
        VOLUME_PIEZO_TAG,
        VOLUME_SUBSTRATE_TAG,
    )
    from peh_inverse_design.meshing.volume_mesh import (
        _silence_native_output,
        _tetra_signed_volumes,
        gmsh,
    )
else:
    from ..core.mesh_tags import FACET_BOTTOM_PLATE_TAG, VOLUME_PIEZO_TAG, VOLUME_SUBSTRATE_TAG
    from ..meshing.volume_mesh import _silence_native_output, _tetra_signed_volumes, gmsh


@dataclass(frozen=True)
class GeometryParityTolerances:
    """Acceptance thresholds for solver-mesh vs ANSYS-STEP geometry parity.

    The planform tolerances bound the CAD-simplification symdiff cap (1e-3
    relative by default) plus floating-point slack; the volume tolerance adds
    the two 5e-4 relative volume validations performed at export time.
    """

    rel_volume_tol: float = 2.5e-3
    rel_planform_area_tol: float = 1.5e-3
    rel_planform_symdiff_tol: float = 1.5e-3
    bbox_abs_tol_m: float = 1.0e-6
    # The STEP footprint must reproduce the cad.json record (same polygon),
    # and the mesh footprint must reproduce the recorded source planform.
    rel_recorded_area_tol: float = 1.0e-6


def _extract_sample_id(path: Path) -> int:
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        raise ValueError(f"Could not infer sample id from {path}.")
    return int(matches[-1])


def _relative_difference(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1.0e-30)


def _triangles_to_polygon(xy: np.ndarray, triangles: np.ndarray) -> BaseGeometry:
    """Union conforming triangles into the exact planform polygon they tile."""
    xy = np.asarray(xy, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.size == 0:
        raise ValueError("No triangles available to reconstruct a planform polygon.")
    coords = xy[triangles]
    triangle_polygons = shapely.polygons(coords)
    expected_area = float(np.sum(shapely.area(triangle_polygons)))

    union: BaseGeometry | None = None
    try:
        union = shapely.coverage_union_all(triangle_polygons)
    except Exception:
        union = None
    if union is None or not union.is_valid or not math.isclose(
        float(union.area), expected_area, rel_tol=1.0e-9, abs_tol=1.0e-18
    ):
        union = unary_union([geom for geom in triangle_polygons])
    if union.is_empty:
        raise ValueError("Triangle union produced an empty planform polygon.")
    return union


def solver_mesh_geometry_summary(mesh_npz_path: str | Path) -> dict[str, object]:
    """Measure the geometry actually analyzed by the surrogate solver mesh."""
    mesh_npz_path = Path(mesh_npz_path)
    with np.load(mesh_npz_path, allow_pickle=True) as data:
        points = np.asarray(data["points"], dtype=np.float64)
        tetra_cells = np.asarray(data["tetra_cells"], dtype=np.int64)
        tetra_tags = np.asarray(data["tetra_tags"], dtype=np.int32).reshape(-1)
        triangle_cells = np.asarray(data["triangle_cells"], dtype=np.int64)
        triangle_tags = np.asarray(data["triangle_tags"], dtype=np.int32).reshape(-1)

    cell_volumes = np.abs(_tetra_signed_volumes(points, tetra_cells))
    substrate_mask = tetra_tags == VOLUME_SUBSTRATE_TAG
    piezo_mask = tetra_tags == VOLUME_PIEZO_TAG
    if not np.any(substrate_mask) or not np.any(piezo_mask):
        raise ValueError(f"{mesh_npz_path} does not contain both substrate and piezo tetra tags.")

    bottom_triangles = triangle_cells[triangle_tags == FACET_BOTTOM_PLATE_TAG]
    if bottom_triangles.shape[0] == 0:
        raise ValueError(
            f"{mesh_npz_path} has no FACET_BOTTOM_PLATE_TAG ({FACET_BOTTOM_PLATE_TAG}) triangles; "
            "cannot reconstruct the substrate planform footprint."
        )
    planform = _triangles_to_polygon(points[:, :2], bottom_triangles)

    substrate_points = points[np.unique(tetra_cells[substrate_mask].reshape(-1))]
    piezo_points = points[np.unique(tetra_cells[piezo_mask].reshape(-1))]
    return {
        "planform_polygon": planform,
        "planform_area_m2": float(planform.area),
        "substrate_volume_m3": float(np.sum(cell_volumes[substrate_mask])),
        "piezo_volume_m3": float(np.sum(cell_volumes[piezo_mask])),
        "bbox_min_m": [float(value) for value in np.min(points, axis=0)],
        "bbox_max_m": [float(value) for value in np.max(points, axis=0)],
        "substrate_z_range_m": [float(np.min(substrate_points[:, 2])), float(np.max(substrate_points[:, 2]))],
        "piezo_z_range_m": [float(np.min(piezo_points[:, 2])), float(np.max(piezo_points[:, 2]))],
        "n_points": int(points.shape[0]),
        "n_tetra": int(tetra_cells.shape[0]),
    }


def _import_step_in_metres(step_path: Path) -> None:
    """Import a STEP file with its coordinates converted into metres.

    OpenCASCADE always converts STEP coordinates from the file's declared
    length unit into the process-global session unit (mm unless overridden),
    so the target unit is pinned to metres for the import. This mirrors how
    ANSYS interprets the file: a mislabeled unit header shows up here as the
    same orders-of-magnitude scale error ANSYS would see.
    """
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    with _silence_native_output():
        gmsh.model.occ.importShapes(str(step_path))
    gmsh.model.occ.synchronize()


def _restore_default_occ_length_unit(step_path: Path) -> None:
    """Re-arm OpenCASCADE's default mm session unit after a metre-target import.

    The OCCTargetUnit option flips a process-global OpenCASCADE static that
    also rescales any STEP file *written* later in the same process. gmsh only
    applies the option during an import, so a throwaway re-import is used to
    push the default back.
    """
    try:
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.add("geometry_parity_unit_restore")
        with _silence_native_output():
            gmsh.model.occ.importShapes(str(step_path))
    except Exception:
        pass


def _collect_triangles_for_faces(face_tags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    if len(node_tags) == 0:
        raise RuntimeError("gmsh did not return any mesh nodes for the STEP planform reconstruction.")
    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(node_tags)
    node_tags = node_tags[order]
    node_coords = node_coords[order]

    triangle_blocks: list[np.ndarray] = []
    for tag in face_tags:
        elem_types, _, elem_nodes = gmsh.model.mesh.getElements(2, int(tag))
        for elem_type, node_block in zip(elem_types, elem_nodes):
            props = gmsh.model.mesh.getElementProperties(int(elem_type))
            name = str(props[0]).lower()
            dim = int(props[1])
            num_nodes = int(props[3])
            if dim != 2 or not name.startswith("triangle"):
                continue
            block = np.asarray(node_block, dtype=np.int64).reshape(-1, num_nodes)[:, :3]
            local = np.searchsorted(node_tags, block)
            if np.any(local >= node_tags.shape[0]) or not np.array_equal(node_tags[local], block):
                raise RuntimeError("Could not map gmsh node tags for the STEP bottom-face mesh.")
            triangle_blocks.append(np.asarray(local, dtype=np.int64))
    if not triangle_blocks:
        raise RuntimeError("The STEP substrate bottom faces produced no surface triangles.")
    return node_coords[:, :2], np.vstack(triangle_blocks)


def step_geometry_summary(
    step_path: str | Path,
    planform_mesh_size_m: float | None = None,
) -> dict[str, object]:
    """Measure the geometry ANSYS imports: reload the STEP with OpenCASCADE and survey it.

    The substrate footprint is recovered by 2D-meshing the substrate bottom
    faces (z = substrate bottom plane) and unioning the triangles; planform
    edges are straight, so the reconstruction is exact regardless of mesh size.
    """
    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("geometry_parity_step")
    try:
        _import_step_in_metres(step_path)

        solids = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
        if len(solids) != 2:
            raise RuntimeError(
                f"{step_path} contains {len(solids)} solid bodies; expected the 2-body "
                "substrate+piezo assembly."
            )
        solid_bboxes = {tag: gmsh.model.getBoundingBox(3, tag) for tag in solids}
        substrate_tag = min(solids, key=lambda tag: (solid_bboxes[tag][2], solid_bboxes[tag][5]))
        piezo_tag = next(tag for tag in solids if tag != substrate_tag)
        substrate_bbox = solid_bboxes[substrate_tag]
        piezo_bbox = solid_bboxes[piezo_tag]

        substrate_volume_m3 = float(gmsh.model.occ.getMass(3, substrate_tag))
        piezo_volume_m3 = float(gmsh.model.occ.getMass(3, piezo_tag))

        bbox_min = [min(substrate_bbox[axis], piezo_bbox[axis]) for axis in range(3)]
        bbox_max = [max(substrate_bbox[axis + 3], piezo_bbox[axis + 3]) for axis in range(3)]
        total_thickness_m = float(bbox_max[2] - bbox_min[2])
        z_tol = max(1.0e-9, 1.0e-3 * total_thickness_m)

        bottom_face_tags = []
        for dim, tag in gmsh.model.getBoundary([(3, int(substrate_tag))], combined=False, oriented=False, recursive=False):
            if dim != 2:
                continue
            face_bbox = gmsh.model.getBoundingBox(2, tag)
            if abs(face_bbox[2] - substrate_bbox[2]) <= z_tol and abs(face_bbox[5] - substrate_bbox[2]) <= z_tol:
                bottom_face_tags.append(int(tag))
        if not bottom_face_tags:
            raise RuntimeError(f"Could not find substrate bottom faces at z={substrate_bbox[2]:.6g} in {step_path}.")

        if planform_mesh_size_m is None:
            planform_mesh_size_m = max(
                1.0e-6,
                min(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]) / 16.0,
            )
        point_dimtags = gmsh.model.getEntities(0)
        if point_dimtags:
            gmsh.model.mesh.setSize(point_dimtags, float(planform_mesh_size_m))
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        with _silence_native_output():
            gmsh.model.mesh.generate(2)
        xy, triangles = _collect_triangles_for_faces(bottom_face_tags)
        planform = _triangles_to_polygon(xy, triangles)
    finally:
        _restore_default_occ_length_unit(step_path)
        gmsh.finalize()

    return {
        "planform_polygon": planform,
        "planform_area_m2": float(planform.area),
        "substrate_volume_m3": substrate_volume_m3,
        "piezo_volume_m3": piezo_volume_m3,
        "bbox_min_m": [float(value) for value in bbox_min],
        "bbox_max_m": [float(value) for value in bbox_max],
        "substrate_z_range_m": [float(substrate_bbox[2]), float(substrate_bbox[5])],
        "piezo_z_range_m": [float(piezo_bbox[2]), float(piezo_bbox[5])],
        "bottom_face_count": int(len(bottom_face_tags)),
    }


def _strip_polygon(summary: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in summary.items() if key != "planform_polygon"}


def verify_sample_geometry_parity(
    mesh_npz_path: str | Path,
    step_path: str | Path,
    cad_report_path: str | Path | None = None,
    tolerances: GeometryParityTolerances | None = None,
    planform_mesh_size_m: float | None = None,
) -> dict[str, object]:
    """Compare one sample's solver-mesh geometry against its ANSYS STEP export."""
    tolerances = tolerances or GeometryParityTolerances()
    mesh_npz_path = Path(mesh_npz_path)
    step_path = Path(step_path)

    mesh_summary = solver_mesh_geometry_summary(mesh_npz_path)
    step_summary = step_geometry_summary(step_path, planform_mesh_size_m=planform_mesh_size_m)

    mesh_planform = mesh_summary["planform_polygon"]
    step_planform = step_summary["planform_polygon"]
    mesh_planform_area = float(mesh_summary["planform_area_m2"])
    symdiff_area_m2 = float(mesh_planform.symmetric_difference(step_planform).area)

    plate_extent_diffs_m = [
        abs(
            (float(mesh_summary["bbox_max_m"][axis]) - float(mesh_summary["bbox_min_m"][axis]))
            - (float(step_summary["bbox_max_m"][axis]) - float(step_summary["bbox_min_m"][axis]))
        )
        for axis in range(3)
    ]

    metrics: dict[str, float] = {
        "planform_symdiff_area_m2": symdiff_area_m2,
        "planform_symdiff_rel": symdiff_area_m2 / max(mesh_planform_area, 1.0e-30),
        "planform_area_rel_diff": _relative_difference(
            step_summary["planform_area_m2"], mesh_planform_area
        ),
        "substrate_volume_rel_diff": _relative_difference(
            step_summary["substrate_volume_m3"], mesh_summary["substrate_volume_m3"]
        ),
        "piezo_volume_rel_diff": _relative_difference(
            step_summary["piezo_volume_m3"], mesh_summary["piezo_volume_m3"]
        ),
        "bbox_max_abs_diff_m": float(max(plate_extent_diffs_m)),
    }

    checks: list[dict[str, object]] = [
        {
            "name": "planform_symdiff_rel",
            "value": metrics["planform_symdiff_rel"],
            "tolerance": float(tolerances.rel_planform_symdiff_tol),
            "ok": metrics["planform_symdiff_rel"] <= float(tolerances.rel_planform_symdiff_tol),
        },
        {
            "name": "planform_area_rel_diff",
            "value": metrics["planform_area_rel_diff"],
            "tolerance": float(tolerances.rel_planform_area_tol),
            "ok": metrics["planform_area_rel_diff"] <= float(tolerances.rel_planform_area_tol),
        },
        {
            "name": "substrate_volume_rel_diff",
            "value": metrics["substrate_volume_rel_diff"],
            "tolerance": float(tolerances.rel_volume_tol),
            "ok": metrics["substrate_volume_rel_diff"] <= float(tolerances.rel_volume_tol),
        },
        {
            "name": "piezo_volume_rel_diff",
            "value": metrics["piezo_volume_rel_diff"],
            "tolerance": float(tolerances.rel_volume_tol),
            "ok": metrics["piezo_volume_rel_diff"] <= float(tolerances.rel_volume_tol),
        },
        {
            "name": "bbox_max_abs_diff_m",
            "value": metrics["bbox_max_abs_diff_m"],
            "tolerance": float(tolerances.bbox_abs_tol_m),
            "ok": metrics["bbox_max_abs_diff_m"] <= float(tolerances.bbox_abs_tol_m),
        },
    ]

    cad_report: dict[str, object] = {}
    if cad_report_path is not None and Path(cad_report_path).exists():
        cad_payload = json.loads(Path(cad_report_path).read_text(encoding="utf-8"))
        cad_report = {
            "cad_report_path": str(cad_report_path),
            "recorded_cad_planform_area_m2": cad_payload.get("planform_area_m2"),
            "recorded_source_planform_area_m2": cad_payload.get("source_planform_area_m2"),
            "recorded_source_to_cad_planform_symdiff_relative": cad_payload.get(
                "source_to_cad_planform_symdiff_relative"
            ),
            "recorded_max_cad_planform_symdiff_relative_to_source": cad_payload.get(
                "max_cad_planform_symdiff_relative_to_source"
            ),
        }
        recorded_cad_area = cad_payload.get("planform_area_m2")
        if recorded_cad_area is not None:
            value = _relative_difference(step_summary["planform_area_m2"], float(recorded_cad_area))
            checks.append(
                {
                    "name": "step_matches_recorded_cad_planform_area",
                    "value": value,
                    "tolerance": float(tolerances.rel_recorded_area_tol),
                    "ok": value <= float(tolerances.rel_recorded_area_tol),
                }
            )
        recorded_source_area = cad_payload.get("source_planform_area_m2")
        if recorded_source_area is not None:
            value = _relative_difference(mesh_planform_area, float(recorded_source_area))
            checks.append(
                {
                    "name": "mesh_matches_recorded_source_planform_area",
                    "value": value,
                    "tolerance": float(tolerances.rel_recorded_area_tol),
                    "ok": value <= float(tolerances.rel_recorded_area_tol),
                }
            )

    return {
        "sample_id": _extract_sample_id(mesh_npz_path),
        "mesh_npz_path": str(mesh_npz_path),
        "step_path": str(step_path),
        "parity_ok": bool(all(check["ok"] for check in checks)),
        "metrics": metrics,
        "checks": checks,
        "tolerances": asdict(tolerances),
        "mesh": _strip_polygon(mesh_summary),
        "step": _strip_polygon(step_summary),
        "cad_report": cad_report,
    }


_SUMMARY_CSV_FIELDS = [
    "sample_id",
    "parity_ok",
    "planform_symdiff_rel",
    "planform_symdiff_area_m2",
    "planform_area_rel_diff",
    "mesh_planform_area_m2",
    "step_planform_area_m2",
    "substrate_volume_rel_diff",
    "mesh_substrate_volume_m3",
    "step_substrate_volume_m3",
    "piezo_volume_rel_diff",
    "mesh_piezo_volume_m3",
    "step_piezo_volume_m3",
    "bbox_max_abs_diff_m",
    "failed_checks",
    "mesh_npz_path",
    "step_path",
]


def _summary_row(report: dict[str, object]) -> dict[str, str]:
    metrics = report["metrics"]
    failed = [str(check["name"]) for check in report["checks"] if not check["ok"]]
    return {
        "sample_id": str(int(report["sample_id"])),
        "parity_ok": str(bool(report["parity_ok"])),
        "planform_symdiff_rel": f"{float(metrics['planform_symdiff_rel']):.12g}",
        "planform_symdiff_area_m2": f"{float(metrics['planform_symdiff_area_m2']):.12g}",
        "planform_area_rel_diff": f"{float(metrics['planform_area_rel_diff']):.12g}",
        "mesh_planform_area_m2": f"{float(report['mesh']['planform_area_m2']):.12g}",
        "step_planform_area_m2": f"{float(report['step']['planform_area_m2']):.12g}",
        "substrate_volume_rel_diff": f"{float(metrics['substrate_volume_rel_diff']):.12g}",
        "mesh_substrate_volume_m3": f"{float(report['mesh']['substrate_volume_m3']):.12g}",
        "step_substrate_volume_m3": f"{float(report['step']['substrate_volume_m3']):.12g}",
        "piezo_volume_rel_diff": f"{float(metrics['piezo_volume_rel_diff']):.12g}",
        "mesh_piezo_volume_m3": f"{float(report['mesh']['piezo_volume_m3']):.12g}",
        "step_piezo_volume_m3": f"{float(report['step']['piezo_volume_m3']):.12g}",
        "bbox_max_abs_diff_m": f"{float(metrics['bbox_max_abs_diff_m']):.12g}",
        "failed_checks": ";".join(failed),
        "mesh_npz_path": str(report["mesh_npz_path"]),
        "step_path": str(report["step_path"]),
    }


def verify_geometry_parity_for_run(
    mesh_dir: str | Path,
    output_dir: str | Path | None = None,
    summary_csv_path: str | Path | None = None,
    sample_ids: list[int] | None = None,
    tolerances: GeometryParityTolerances | None = None,
    planform_mesh_size_m: float | None = None,
) -> list[dict[str, object]]:
    """Verify geometry parity for every (or selected) sample in a run's mesh directory."""
    mesh_dir = Path(mesh_dir)
    mesh_files = sorted(mesh_dir.glob("plate3d_*_fenicsx.npz"))
    if sample_ids is not None:
        wanted = {int(sid) for sid in sample_ids}
        mesh_files = [path for path in mesh_files if _extract_sample_id(path) in wanted]
    if not mesh_files:
        raise FileNotFoundError(f"No plate3d_*_fenicsx.npz solver meshes found in {mesh_dir}.")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    if summary_csv_path is None and output_dir is not None:
        summary_csv_path = output_dir / "geometry_parity_summary.csv"

    reports: list[dict[str, object]] = []
    for mesh_path in mesh_files:
        sample_id = _extract_sample_id(mesh_path)
        step_path = mesh_dir / f"plate3d_{sample_id:04d}.step"
        cad_report_path = mesh_dir / f"plate3d_{sample_id:04d}_cad.json"
        report = verify_sample_geometry_parity(
            mesh_npz_path=mesh_path,
            step_path=step_path,
            cad_report_path=cad_report_path if cad_report_path.exists() else None,
            tolerances=tolerances,
            planform_mesh_size_m=planform_mesh_size_m,
        )
        reports.append(report)
        metrics = report["metrics"]
        verdict = "PARITY OK" if report["parity_ok"] else "PARITY FAILED"
        detail = (
            f"planform symdiff {float(metrics['planform_symdiff_rel']):.3e} rel, "
            f"substrate vol diff {float(metrics['substrate_volume_rel_diff']):.3e} rel, "
            f"piezo vol diff {float(metrics['piezo_volume_rel_diff']):.3e} rel"
        )
        if not report["parity_ok"]:
            failed = ", ".join(str(check["name"]) for check in report["checks"] if not check["ok"])
            detail += f"; failed: {failed}"
        print(f"sample {sample_id:04d}: {verdict} ({detail})")
        if output_dir is not None:
            json_path = output_dir / f"sample_{sample_id:04d}_geometry_parity.json"
            json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if summary_csv_path is not None:
        summary_csv_path = Path(summary_csv_path)
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_SUMMARY_CSV_FIELDS)
            writer.writeheader()
            for report in reports:
                writer.writerow(_summary_row(report))
        print(f"Saved geometry parity summary to {summary_csv_path}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify per-sample geometry parity between the surrogate solver mesh "
            "(plate3d_XXXX_fenicsx.npz) and the ANSYS STEP export (plate3d_XXXX.step)."
        ),
    )
    parser.add_argument(
        "--mesh-dir",
        required=True,
        help="Run mesh directory containing plate3d_XXXX_fenicsx.npz / .step / _cad.json files.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        type=int,
        default=[],
        help="Verify only this sample id. Repeat the flag for multiple samples; default is all.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for per-sample geometry-parity JSON reports and the summary CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Explicit summary CSV path. Defaults to <output-dir>/geometry_parity_summary.csv.",
    )
    defaults = GeometryParityTolerances()
    parser.add_argument("--rel-volume-tol", type=float, default=defaults.rel_volume_tol)
    parser.add_argument("--rel-planform-area-tol", type=float, default=defaults.rel_planform_area_tol)
    parser.add_argument("--rel-planform-symdiff-tol", type=float, default=defaults.rel_planform_symdiff_tol)
    parser.add_argument("--bbox-tol-m", type=float, default=defaults.bbox_abs_tol_m)
    parser.add_argument(
        "--planform-mesh-size-m",
        type=float,
        default=None,
        help="Target 2D mesh size for the STEP footprint reconstruction. Defaults to plate_min_extent/16.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when any sample fails a parity check.",
    )
    args = parser.parse_args()

    tolerances = GeometryParityTolerances(
        rel_volume_tol=float(args.rel_volume_tol),
        rel_planform_area_tol=float(args.rel_planform_area_tol),
        rel_planform_symdiff_tol=float(args.rel_planform_symdiff_tol),
        bbox_abs_tol_m=float(args.bbox_tol_m),
    )
    reports = verify_geometry_parity_for_run(
        mesh_dir=args.mesh_dir,
        output_dir=args.output_dir or None,
        summary_csv_path=args.summary_csv or None,
        sample_ids=args.sample_id or None,
        tolerances=tolerances,
        planform_mesh_size_m=args.planform_mesh_size_m,
    )
    failures = [report for report in reports if not report["parity_ok"]]
    if failures:
        failed_ids = ", ".join(f"{int(report['sample_id']):04d}" for report in failures)
        print(f"Geometry parity FAILED for sample(s): {failed_ids}")
        if args.strict:
            raise SystemExit(1)
    else:
        print(f"Geometry parity OK for all {len(reports)} sample(s).")


if __name__ == "__main__":
    main()
