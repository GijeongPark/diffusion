from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peh_inverse_design.geometry_pipeline import GeometryBuildConfig
    from peh_inverse_design.problem_spec import (
        default_problem_spec_path,
        geometry_defaults_from_problem_spec,
        load_problem_spec,
        write_ansys_workbench_handoff,
        write_problem_spec_snapshot,
    )
    from peh_inverse_design.volume_mesh import (
        VolumeMeshConfig,
        mesh_tiled_plate_volume_sample,
    )
else:
    from .geometry_pipeline import GeometryBuildConfig
    from .problem_spec import (
        default_problem_spec_path,
        geometry_defaults_from_problem_spec,
        load_problem_spec,
        write_ansys_workbench_handoff,
        write_problem_spec_snapshot,
    )
    from .volume_mesh import (
        VolumeMeshConfig,
        mesh_tiled_plate_volume_sample,
    )


def _explicit_pair(
    x_value: float | None,
    y_value: float | None,
    x_name: str,
    y_name: str,
) -> tuple[float, float] | None:
    provided = x_value is not None or y_value is not None
    if not provided:
        return None
    if x_value is None or y_value is None:
        raise ValueError(f"{x_name} and {y_name} must be provided together.")
    return float(x_value), float(y_value)


def _explicit_counts(
    x_value: int | None,
    y_value: int | None,
    x_name: str,
    y_name: str,
) -> tuple[int, int] | None:
    provided = x_value is not None or y_value is not None
    if not provided:
        return None
    if x_value is None or y_value is None:
        raise ValueError(f"{x_name} and {y_name} must be provided together.")
    return int(x_value), int(y_value)


def _metadata_pair(data: np.lib.npyio.NpzFile, idx: int, key: str) -> tuple[float, float] | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.ndim == 1 and arr.shape[0] == 2:
        return float(arr[0]), float(arr[1])
    if arr.ndim >= 2 and arr.shape[1] == 2:
        return float(arr[idx, 0]), float(arr[idx, 1])
    raise ValueError(f"{key} must have shape (2,) or (N, 2); got {arr.shape}.")


def _metadata_counts(data: np.lib.npyio.NpzFile, idx: int, key: str) -> tuple[int, int] | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.ndim == 1 and arr.shape[0] == 2:
        return int(arr[0]), int(arr[1])
    if arr.ndim >= 2 and arr.shape[1] == 2:
        return int(arr[idx, 0]), int(arr[idx, 1])
    raise ValueError(f"{key} must have shape (2,) or (N, 2); got {arr.shape}.")


def _resolve_geometry_config_for_sample(
    data: np.lib.npyio.NpzFile,
    idx: int,
    args: argparse.Namespace,
    problem_spec: dict[str, object] | None,
) -> GeometryBuildConfig:
    cell_size_m = _explicit_pair(args.cell_size_x_m, args.cell_size_y_m, "--cell-size-x-m", "--cell-size-y-m")
    tile_counts = _explicit_counts(args.tile_count_x, args.tile_count_y, "--tile-count-x", "--tile-count-y")

    if cell_size_m is None:
        cell_size_m = _metadata_pair(data, idx, "cell_size_m")
    if tile_counts is None:
        tile_counts = _metadata_counts(data, idx, "tile_counts")
    if (cell_size_m is None or tile_counts is None) and problem_spec is not None:
        spec_cell_size_m, spec_tile_counts = geometry_defaults_from_problem_spec(problem_spec)
        if cell_size_m is None:
            cell_size_m = spec_cell_size_m
        if tile_counts is None:
            tile_counts = spec_tile_counts

    if cell_size_m is None or tile_counts is None:
        raise ValueError(
            "Physical plate sizing is not available. Pass --cell-size-x-m/--cell-size-y-m and "
            "--tile-count-x/--tile-count-y, or provide cell_size_m/tile_counts metadata in the input NPZ."
        )

    return GeometryBuildConfig(
        cell_size_m=cell_size_m,
        tile_counts=tile_counts,
        enforce_connected_plate=False,
    )


def _resolve_sample_id(data: np.lib.npyio.NpzFile, idx: int) -> int:
    if "sample_id" in data.files:
        return int(np.asarray(data["sample_id"])[idx])
    if "source_sample_id" in data.files:
        return int(np.asarray(data["source_sample_id"])[idx])
    return int(idx)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build solid STEP geometry for ANSYS plus Python solver meshes for the same piezoelectric plate problem.",
    )
    parser.add_argument(
        "--unit-cell-npz",
        default="data/dataset_100.npz",
        help="Input unit-cell dataset produced by periodic_grf_sdf.ipynb.",
    )
    parser.add_argument(
        "--mesh-dir",
        default="meshes/volumes",
        help="Output directory for STEP geometry, CAD reports, and Python solver mesh files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples.",
    )
    parser.add_argument(
        "--target-ok",
        type=int,
        default=None,
        help="Continue scanning candidate samples until this many successful solid exports are produced.",
    )
    parser.add_argument(
        "--substrate-thickness",
        type=float,
        default=1.0e-3,
        help="Substrate thickness in meters.",
    )
    parser.add_argument(
        "--piezo-thickness",
        type=float,
        default=2.667e-4,
        help="Piezo thickness in meters.",
    )
    parser.add_argument(
        "--mesh-size-scale",
        type=float,
        default=0.06,
        help="Target in-plane solver element size as a fraction of one cell size.",
    )
    parser.add_argument(
        "--cad-reference-size-scale",
        type=float,
        default=0.01,
        help=(
            "Reference size used for CAD feature checks and OCC tolerances as a fraction of one cell size. "
            "Keep this smaller than the solver mesh scale for thin plates."
        ),
    )
    parser.add_argument(
        "--solver-mesh-backend",
        default="layered_tet",
        choices=["layered_tet", "gmsh_volume"],
        help=(
            "Python solver mesh backend. 'layered_tet' is the fast default: export solid STEP for ANSYS, "
            "then build the Python mesh from a partitioned 2D surface extrusion instead of tetrahedralizing the STEP body."
        ),
    )
    parser.add_argument(
        "--ansys-step-strategy",
        default=VolumeMeshConfig().ansys_step_strategy,
        choices=["partitioned_interface", "single_face_assembly"],
        help=(
            "Primary one-file ANSYS STEP strategy. single_face_assembly keeps the piezo bottom continuous; "
            "partitioned_interface keeps a conformal imprinted interface."
        ),
    )
    parser.add_argument(
        "--cad-planform-simplify-scale",
        type=float,
        default=VolumeMeshConfig().cad_planform_simplify_relative_to_reference,
        help=(
            "CAD-only boundary simplification strength, relative to the CAD reference size. "
            "Larger values export a more mesh-friendly STEP without changing the Python solver mesh geometry."
        ),
    )
    parser.add_argument(
        "--cad-min-hole-area-scale",
        type=float,
        default=VolumeMeshConfig().cad_min_hole_area_relative_to_reference_squared,
        help=(
            "CAD-only minimum retained hole area, relative to the squared CAD reference size. "
            "Larger values remove more substrate holes from the STEP export without changing the Python solver mesh geometry."
        ),
    )
    parser.add_argument(
        "--substrate-layers",
        type=int,
        default=2,
        help="Number of swept layers through the substrate thickness for the fast layered_tet solver mesh.",
    )
    parser.add_argument(
        "--piezo-layers",
        type=int,
        default=1,
        help="Number of swept layers through the piezo thickness for the fast layered_tet solver mesh.",
    )
    parser.add_argument(
        "--solver-max-q2-vector-dofs",
        type=int,
        default=VolumeMeshConfig().max_solver_vector_dofs,
        help=(
            "Estimated quadratic vector-DOF cap for the layered_tet solver mesh. "
            "The mesh is coarsened automatically until it falls below this limit."
        ),
    )
    parser.add_argument(
        "--write-native-msh",
        action="store_true",
        help="Keep the native .msh file when using the legacy gmsh_volume solver-mesh backend.",
    )
    parser.add_argument(
        "--write-xdmf",
        action="store_true",
        help="Also write XDMF files when using the legacy gmsh_volume solver-mesh backend.",
    )
    parser.add_argument(
        "--limit-solver-mesh-by-thickness",
        action="store_true",
        help=(
            "Restore the old behavior that caps the solver mesh size by the total thickness. "
            "Disabled by default because it explodes the in-plane element count for thin plates."
        ),
    )
    parser.add_argument(
        "--export-inspection-single-face-step",
        action="store_true",
        help=(
            "Also export the optional inspection-only STEP whose piezo bottom stays a single continuous face. "
            "Disabled by default because it adds an extra OCC export/roundtrip validation pass per sample."
        ),
    )
    parser.add_argument(
        "--repair-cad",
        action="store_true",
        help="Opt in to explicit bridge repair for disconnected substrates instead of rejecting them in exact CAD mode.",
    )
    parser.add_argument(
        "--repair-bridge-width-m",
        type=float,
        default=None,
        help="Explicit bridge width to use in repair CAD mode. Defaults to the geometry config bridge width.",
    )
    parser.add_argument("--cell-size-x-m", type=float, default=None, help="Unit-cell size in the x direction, in meters.")
    parser.add_argument("--cell-size-y-m", type=float, default=None, help="Unit-cell size in the y direction, in meters.")
    parser.add_argument("--tile-count-x", type=int, default=None, help="Number of tiled unit cells along x.")
    parser.add_argument("--tile-count-y", type=int, default=None, help="Number of tiled unit cells along y.")
    parser.add_argument(
        "--problem-spec",
        default="",
        help="Optional shared problem specification YAML. Defaults to configs/peh_inverse_design_spec.yaml when present.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if args.problem_spec:
        problem_spec = load_problem_spec(args.problem_spec, project_root=project_root)
    else:
        default_spec_path = default_problem_spec_path(project_root)
        problem_spec = load_problem_spec(default_spec_path, project_root=project_root) if default_spec_path.exists() else None

    data = np.load(args.unit_cell_npz, allow_pickle=True)
    n_total = int(data["grf"].shape[0])
    n_candidates = n_total if args.limit is None else min(int(args.limit), n_total)
    target_ok = None if args.target_ok is None else int(args.target_ok)
    if target_ok is not None and target_ok <= 0:
        raise ValueError("--target-ok must be strictly positive when provided.")

    volume_config = VolumeMeshConfig(
        substrate_thickness_m=float(args.substrate_thickness),
        piezo_thickness_m=float(args.piezo_thickness),
        mesh_size_relative_to_cell=float(args.mesh_size_scale),
        cad_reference_size_relative_to_cell=float(args.cad_reference_size_scale),
        limit_solver_mesh_by_thickness=bool(args.limit_solver_mesh_by_thickness),
        substrate_layers=int(args.substrate_layers),
        piezo_layers=int(args.piezo_layers),
        solver_mesh_backend=str(args.solver_mesh_backend),
        ansys_step_strategy=str(args.ansys_step_strategy),
        write_native_msh=bool(args.write_native_msh),
        write_xdmf=bool(args.write_xdmf),
        require_connected_substrate=True,
        exact_cad=not bool(args.repair_cad),
        repair_cad=bool(args.repair_cad),
        repair_bridge_width_m=None if args.repair_bridge_width_m is None else float(args.repair_bridge_width_m),
        max_solver_vector_dofs=(
            None if args.solver_max_q2_vector_dofs is None else int(args.solver_max_q2_vector_dofs)
        ),
        export_inspection_single_face_step=bool(args.export_inspection_single_face_step),
        cad_planform_simplify_relative_to_reference=float(args.cad_planform_simplify_scale),
        cad_min_hole_area_relative_to_reference_squared=float(args.cad_min_hole_area_scale),
    )

    mesh_dir = Path(args.mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    if problem_spec is not None:
        write_problem_spec_snapshot(problem_spec, mesh_dir / "problem_spec_used.yaml")

    ok = 0
    fail = 0
    examined = 0
    failure_records: list[dict[str, object]] = []
    selected_source_indices: list[int] = []
    selected_sample_ids: list[int] = []
    last_geometry_signature: tuple[tuple[float, float], tuple[int, int]] | None = None
    for idx in range(n_candidates):
        if target_ok is not None and ok >= target_ok:
            break
        sample_id = _resolve_sample_id(data, idx)
        geometry_config = _resolve_geometry_config_for_sample(data, idx, args, problem_spec)
        geometry_signature = (geometry_config.cell_size_m, geometry_config.tile_counts)
        if geometry_signature != last_geometry_signature:
            plate_lx, plate_ly = geometry_config.plate_size_m
            print(
                "Resolved plate dimensions: "
                f"cell=({geometry_config.cell_size_m[0]:.6g}, {geometry_config.cell_size_m[1]:.6g}) m, "
                f"tiles={geometry_config.tile_counts[0]} x {geometry_config.tile_counts[1]}, "
                f"plate=({plate_lx:.6g}, {plate_ly:.6g}) m",
                flush=True,
            )
            last_geometry_signature = geometry_signature
        examined += 1
        try:
            path = mesh_tiled_plate_volume_sample(
                grf=data["grf"][idx],
                threshold=float(data["threshold"][idx]),
                sample_id=sample_id,
                output_dir=mesh_dir,
                geometry_config=geometry_config,
                volume_config=volume_config,
            )
            if path is None:
                fail += 1
                failure_records.append(
                    {
                        "index": int(idx),
                        "sample_id": int(sample_id),
                        "error": "mesh_tiled_plate_volume_sample returned no output path",
                    }
                )
            else:
                if problem_spec is not None:
                    combined_step_path = mesh_dir / f"plate3d_{sample_id:04d}.step"
                    inspection_step_path = mesh_dir / f"plate3d_{sample_id:04d}_single_face_probe.step"
                    face_selection_manifest_path = mesh_dir / f"plate3d_{sample_id:04d}_ansys_face_groups.json"
                    write_ansys_workbench_handoff(
                        sample_id=sample_id,
                        output_path=mesh_dir / f"plate3d_{sample_id:04d}_ansys_workbench.json",
                        step_path=combined_step_path if combined_step_path.exists() else None,
                        msh_path=None,
                        cad_report_path=mesh_dir / f"plate3d_{sample_id:04d}_cad.json",
                        solver_mesh_path=path,
                        geometry_config=geometry_config,
                        volume_config=volume_config,
                        problem_spec=problem_spec,
                        inspection_single_face_step_path=inspection_step_path if inspection_step_path.exists() else None,
                        face_selection_manifest_path=
                            face_selection_manifest_path if face_selection_manifest_path.exists() else None,
                    )
                ok += 1
                selected_source_indices.append(int(idx))
                selected_sample_ids.append(int(sample_id))
        except Exception as exc:
            fail += 1
            failure_records.append(
                {
                    "index": int(idx),
                    "sample_id": int(sample_id),
                    "error": str(exc),
                }
            )
            print(f"[sample {sample_id:04d}] CAD export failed: {exc}", file=sys.stderr, flush=True)
        print(f"[{idx + 1:4d}/{n_candidates}] ok={ok} fail={fail}", flush=True)

    summary_path = mesh_dir / "mesh_build_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "requested_candidates": int(n_candidates),
                "requested_ok": None if target_ok is None else int(target_ok),
                "examined_candidates": int(examined),
                "ok": int(ok),
                "fail": int(fail),
                "cad_mode": "repair" if volume_config.repair_cad else "exact",
                "solver_mesh_size_relative_to_cell": float(volume_config.mesh_size_relative_to_cell),
                "cad_reference_size_relative_to_cell": float(volume_config.cad_reference_size_relative_to_cell),
                "cad_planform_simplify_relative_to_reference": float(volume_config.cad_planform_simplify_relative_to_reference),
                "cad_min_hole_area_relative_to_reference_squared": float(
                    volume_config.cad_min_hole_area_relative_to_reference_squared
                ),
                "limit_solver_mesh_by_thickness": bool(volume_config.limit_solver_mesh_by_thickness),
                "solver_mesh_backend": str(volume_config.solver_mesh_backend),
                "max_solver_vector_dofs": None
                if volume_config.max_solver_vector_dofs is None
                else int(volume_config.max_solver_vector_dofs),
                "ansys_step_strategy": str(volume_config.ansys_step_strategy),
                "substrate_layers": int(volume_config.substrate_layers),
                "piezo_layers": int(volume_config.piezo_layers),
                "problem_spec_path": "" if problem_spec is None else str(problem_spec.get("_metadata", {}).get("source_path", "")),
                "selected_source_indices": selected_source_indices,
                "selected_sample_ids": selected_sample_ids,
                "failure_records": failure_records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Saved {ok} ANSYS STEP exports and Python solver meshes to {mesh_dir}", flush=True)
    print(f"Saved mesh build summary to {summary_path}", flush=True)
    if target_ok is not None and ok < target_ok:
        raise SystemExit(
            "Could not generate the requested number of solid exports. "
            f"Requested {target_ok}, generated {ok}. Check mesh_build_summary.json for rejection reasons."
        )
    if n_candidates > 0 and ok == 0:
        raise SystemExit(
            "No solver meshes were generated. Check mesh_build_summary.json for the CAD rejection reasons."
        )


if __name__ == "__main__":
    main()
