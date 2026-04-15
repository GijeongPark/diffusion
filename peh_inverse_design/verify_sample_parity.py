from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peh_inverse_design.audit_ansys_alignment import audit_run_sample
    from peh_inverse_design.build_volume_meshes import _resolve_geometry_config_for_sample, _resolve_sample_id
    from peh_inverse_design.problem_spec import (
        build_runtime_defaults,
        default_problem_spec_path,
        load_problem_spec,
        write_ansys_workbench_handoff,
        write_problem_spec_snapshot,
    )
    from peh_inverse_design.volume_mesh import VolumeMeshConfig, mesh_tiled_plate_volume_sample
    from peh_inverse_design.volume_mesh import volume_mesh_preset_overrides
else:
    from .audit_ansys_alignment import audit_run_sample
    from .build_volume_meshes import _resolve_geometry_config_for_sample, _resolve_sample_id
    from .problem_spec import (
        build_runtime_defaults,
        default_problem_spec_path,
        load_problem_spec,
        write_ansys_workbench_handoff,
        write_problem_spec_snapshot,
    )
    from .volume_mesh import VolumeMeshConfig, mesh_tiled_plate_volume_sample, volume_mesh_preset_overrides


def _parse_layer_sweep(value: str) -> list[tuple[int, int]]:
    text = str(value).strip()
    if not text:
        return [(2, 1), (4, 2), (6, 2), (8, 3)]
    parsed: list[tuple[int, int]] = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid layer sweep entry '{item}'. Use substrate_layers:piezo_layers, e.g. 6:2."
            )
        parsed.append((int(parts[0]), int(parts[1])))
    if not parsed:
        raise ValueError("layer sweep must contain at least one substrate_layers:piezo_layers entry.")
    return parsed


def _single_case_for_preset(mesh_preset: str) -> list[tuple[int, int]]:
    normalized = str(mesh_preset).strip().lower()
    if normalized == "ansys_parity":
        return [(8, 3)]
    return [(2, 1)]


def _parse_float_sweep(value: str, default: float) -> list[float]:
    text = str(value).strip()
    if not text:
        return [float(default)]
    parsed = [float(chunk.strip()) for chunk in text.split(",") if chunk.strip()]
    if not parsed:
        raise ValueError("mesh-size sweep must contain at least one numeric value.")
    return parsed


def _find_sample_index(data: np.lib.npyio.NpzFile, sample_id: int) -> int:
    n_total = int(data["grf"].shape[0])
    for idx in range(n_total):
        if int(_resolve_sample_id(data, idx)) == int(sample_id):
            return int(idx)
    raise KeyError(f"sample_id={int(sample_id)} was not found in the input unit-cell NPZ.")


def _fenicsx_available_locally() -> bool:
    try:
        import dolfinx  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _workspace_path(path: Path, project_root: Path) -> str:
    resolved_path = path if path.is_absolute() else (project_root / path).resolve()
    return str(Path("/workspace") / resolved_path.relative_to(project_root))


def _run_solver_case(
    *,
    project_root: Path,
    mesh_path: Path,
    response_dir: Path,
    modal_dir: Path,
    problem_spec_path: Path | None,
    substrate_rho: float,
    piezo_rho: float,
    num_modes: int,
    search_points: int,
    element_order: int,
    store_mode_shapes: bool,
    docker_image: str,
) -> None:
    if _fenicsx_available_locally():
        solver_args = [
            sys.executable,
            "-m",
            "peh_inverse_design.fenicsx_modal_solver",
            "--mesh",
            str(mesh_path),
            "--response-dir",
            str(response_dir),
            "--modes-dir",
            str(modal_dir),
            "--substrate-rho",
            str(float(substrate_rho)),
            "--piezo-rho",
            str(float(piezo_rho)),
            "--num-modes",
            str(int(num_modes)),
            "--search-points",
            str(int(search_points)),
            "--element-order",
            str(int(element_order)),
        ]
        if store_mode_shapes:
            solver_args.append("--store-mode-shapes")
        if problem_spec_path is not None:
            solver_args.extend(["--problem-spec", str(problem_spec_path)])
        subprocess.run(solver_args, cwd=str(project_root), check=True)
        return

    solver_inner = [
        "python3",
        "-m",
        "peh_inverse_design.fenicsx_modal_solver",
        "--mesh",
        _workspace_path(mesh_path, project_root),
        "--response-dir",
        _workspace_path(response_dir, project_root),
        "--modes-dir",
        _workspace_path(modal_dir, project_root),
        "--substrate-rho",
        str(float(substrate_rho)),
        "--piezo-rho",
        str(float(piezo_rho)),
        "--num-modes",
        str(int(num_modes)),
        "--search-points",
        str(int(search_points)),
        "--element-order",
        str(int(element_order)),
    ]
    if store_mode_shapes:
        solver_inner.append("--store-mode-shapes")
    if problem_spec_path is not None:
        solver_inner.extend(["--problem-spec", _workspace_path(problem_spec_path, project_root)])

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{project_root}:/workspace",
        "-w",
        "/workspace",
        str(docker_image),
        "bash",
        "-lc",
        f"pip install -q pyyaml && {' '.join(shlex.quote(value) for value in solver_inner)}",
    ]
    subprocess.run(cmd, cwd=str(project_root), check=True)


def _case_slug(mesh_preset: str, substrate_layers: int, piezo_layers: int, mesh_size_scale: float) -> str:
    scale_token = f"{float(mesh_size_scale):.5f}".rstrip("0").rstrip(".").replace(".", "p")
    return (
        f"{str(mesh_preset).strip().lower()}_sub{int(substrate_layers)}_pz{int(piezo_layers)}_mesh{scale_token}"
    )


def _load_mesh_metadata(mesh_path: Path) -> dict[str, Any]:
    with np.load(mesh_path) as mesh:
        max_solver_vector_dofs = (
            int(np.asarray(mesh["max_solver_vector_dofs"], dtype=np.int64).reshape(-1)[0])
            if "max_solver_vector_dofs" in mesh.files
            else -1
        )
        requested_solver_mesh_size = (
            np.asarray(mesh["requested_solver_mesh_size_m"], dtype=np.float64).reshape(-1)[0]
            if "requested_solver_mesh_size_m" in mesh.files
            else np.asarray(mesh["solver_mesh_size_m"], dtype=np.float64).reshape(-1)[0]
        )
        solver_mesh_coarsening_allowed = (
            int(np.asarray(mesh["solver_mesh_coarsening_allowed"], dtype=np.int32).reshape(-1)[0])
            if "solver_mesh_coarsening_allowed" in mesh.files
            else 1
        )
        mesh_preset = (
            str(np.asarray(mesh["mesh_preset"]).reshape(-1)[0])
            if "mesh_preset" in mesh.files
            else "default"
        )
        return {
            "point_count": int(np.asarray(mesh["points"]).shape[0]),
            "tetra_count": int(np.asarray(mesh["tetra_cells"]).shape[0]),
            "solver_mesh_size_m": float(np.asarray(mesh["solver_mesh_size_m"], dtype=np.float64).reshape(-1)[0]),
            "requested_solver_mesh_size_m": float(requested_solver_mesh_size),
            "estimated_q2_vector_dofs": int(np.asarray(mesh["estimated_q2_vector_dofs"], dtype=np.int64).reshape(-1)[0]),
            "solver_mesh_coarsening_passes": int(
                np.asarray(mesh["solver_mesh_coarsening_passes"], dtype=np.int32).reshape(-1)[0]
            ),
            "solver_mesh_coarsening_allowed": bool(int(solver_mesh_coarsening_allowed)),
            "max_solver_vector_dofs": None if max_solver_vector_dofs < 0 else int(max_solver_vector_dofs),
            "mesh_preset": str(mesh_preset),
        }


def _frequency_match_score(record: dict[str, Any]) -> float:
    candidates: list[float] = []
    for key in ["mode1_vs_ansys_modal_error_percent", "f_peak_vs_ansys_frf_peak_error_percent"]:
        value = float(record.get(key, np.nan))
        if np.isfinite(value):
            candidates.append(abs(value))
    if candidates:
        return max(candidates)
    for key in ["mode1_vs_ambiguous_frequency_error_percent", "f_peak_vs_ambiguous_frequency_error_percent"]:
        value = float(record.get(key, np.nan))
        if np.isfinite(value):
            candidates.append(abs(value))
    return min(candidates) if candidates else float("inf")


def _record_from_summary(
    *,
    case_name: str,
    case_dir: Path,
    substrate_layers: int,
    piezo_layers: int,
    mesh_size_scale: float,
    element_order: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    frequency = summary["frequency_comparison"]
    voltage = summary["voltage_comparison"]
    mesh_materials = summary["mesh_materials"]
    mesh_npz_path = case_dir / "meshes" / "volumes" / f"plate3d_{int(summary['sample_id']):04d}_fenicsx.npz"
    mesh_meta = _load_mesh_metadata(mesh_npz_path)
    return {
        "case_name": str(case_name),
        "case_dir": str(case_dir),
        "status": "ok",
        "sample_id": int(summary["sample_id"]),
        "mesh_preset": str(mesh_meta["mesh_preset"]),
        "substrate_layers": int(substrate_layers),
        "piezo_layers": int(piezo_layers),
        "mesh_size_scale": float(mesh_size_scale),
        "requested_solver_mesh_size_m": float(mesh_meta["requested_solver_mesh_size_m"]),
        "solver_mesh_size_m": float(mesh_meta["solver_mesh_size_m"]),
        "point_count": int(mesh_meta["point_count"]),
        "tetra_count": int(mesh_meta["tetra_count"]),
        "estimated_q2_vector_dofs": int(mesh_meta["estimated_q2_vector_dofs"]),
        "max_solver_vector_dofs": mesh_meta["max_solver_vector_dofs"],
        "solver_mesh_coarsening_allowed": bool(mesh_meta["solver_mesh_coarsening_allowed"]),
        "solver_mesh_coarsening_passes": int(mesh_meta["solver_mesh_coarsening_passes"]),
        "element_order": int(element_order),
        "mode1_frequency_hz": float(frequency["mode1_frequency_hz"]),
        "f_peak_hz": float(frequency["f_peak_hz"]),
        "harmonic_field_frequency_hz": float(frequency["harmonic_field_frequency_hz"]),
        "ansys_modal_reference_hz": float(frequency["ansys_modal_reference_hz"]),
        "ansys_frf_peak_reference_hz": float(frequency["ansys_frf_peak_reference_hz"]),
        "frequency_reference_ambiguous": bool(frequency["frequency_reference_ambiguous"]),
        "mode1_vs_ansys_modal_error_percent": float(frequency["mode1_vs_ansys_modal_error_percent"]),
        "f_peak_vs_ansys_frf_peak_error_percent": float(frequency["f_peak_vs_ansys_frf_peak_error_percent"]),
        "mode1_vs_ambiguous_frequency_error_percent": float(frequency["mode1_vs_ambiguous_frequency_error_percent"]),
        "f_peak_vs_ambiguous_frequency_error_percent": float(frequency["f_peak_vs_ambiguous_frequency_error_percent"]),
        "peak_voltage_peak_v": float(voltage["peak_voltage_peak_v"]),
        "ansys_voltage_reference_v": float(voltage["ansys_voltage_reference_v"]),
        "ansys_voltage_form": str(voltage["ansys_voltage_form"]),
        "selected_voltage_error_percent": float(voltage["selected_voltage_error_percent"]),
        "error_percent_assuming_ansys_peak": float(voltage["error_percent_assuming_ansys_peak"]),
        "modal_theta_mode1": float(summary["electromechanical"]["modal_theta_mode1"]),
        "modal_force_mode1": float(summary["electromechanical"]["modal_force_mode1"]),
        "capacitance_f": float(summary["electromechanical"]["capacitance_f"]),
        "piezo_volume_m3": float(mesh_materials["piezo_volume_m3"]),
        "substrate_volume_m3": float(mesh_materials["substrate_volume_m3"]),
        "frequency_match_score_percent": float(_frequency_match_score(frequency)),
    }


def _failure_record(
    *,
    case_name: str,
    case_dir: Path,
    sample_id: int,
    substrate_layers: int,
    piezo_layers: int,
    mesh_size_scale: float,
    mesh_preset: str,
    element_order: int,
    status: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "case_name": str(case_name),
        "case_dir": str(case_dir),
        "status": str(status),
        "sample_id": int(sample_id),
        "mesh_preset": str(mesh_preset),
        "substrate_layers": int(substrate_layers),
        "piezo_layers": int(piezo_layers),
        "mesh_size_scale": float(mesh_size_scale),
        "element_order": int(element_order),
        "error_message": str(exc),
        "frequency_match_score_percent": float("inf"),
    }


def _write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    if not records:
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def verify_sample_parity(
    *,
    sample_id: int,
    unit_cell_npz: str | Path,
    output_dir: str | Path,
    problem_spec_path: str | Path | None,
    layer_sweep: list[tuple[int, int]],
    mesh_size_scales: list[float],
    mesh_preset: str,
    ansys_modal_hz: float | None,
    ansys_frf_peak_hz: float | None,
    ansys_voltage_v: float | None,
    ansys_voltage_form: str,
    substrate_thickness_m: float | None,
    piezo_thickness_m: float | None,
    substrate_rho: float | None,
    piezo_rho: float | None,
    solver_num_modes: int,
    solver_search_points: int,
    element_order: int,
    store_mode_shapes: bool,
    docker_image: str,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1].resolve()
    if problem_spec_path is not None and str(problem_spec_path).strip():
        problem_spec = load_problem_spec(problem_spec_path, project_root=project_root)
        resolved_problem_spec_path = Path(problem_spec["_metadata"]["source_path"]).resolve()
    else:
        default_spec = default_problem_spec_path(project_root)
        problem_spec = load_problem_spec(default_spec, project_root=project_root) if default_spec.exists() else None
        resolved_problem_spec_path = default_spec.resolve() if default_spec.exists() else None

    runtime_defaults = build_runtime_defaults(problem_spec) if problem_spec is not None else {}
    substrate_thickness_m = float(
        runtime_defaults.get("substrate_thickness_m", 1.0e-3)
        if substrate_thickness_m is None
        else substrate_thickness_m
    )
    piezo_thickness_m = float(
        runtime_defaults.get("piezo_thickness_m", 1.0e-4)
        if piezo_thickness_m is None
        else piezo_thickness_m
    )
    substrate_rho = float(runtime_defaults.get("substrate_rho", 7930.0) if substrate_rho is None else substrate_rho)
    piezo_rho = float(runtime_defaults.get("piezo_rho", 7500.0) if piezo_rho is None else piezo_rho)

    unit_cell_npz = (
        Path(unit_cell_npz).resolve()
        if Path(unit_cell_npz).is_absolute()
        else (project_root / Path(unit_cell_npz)).resolve()
    )
    output_dir = (
        Path(output_dir).resolve()
        if Path(output_dir).is_absolute()
        else (project_root / Path(output_dir)).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(unit_cell_npz, allow_pickle=True) as data:
        sample_idx = _find_sample_index(data, sample_id=int(sample_id))
        geometry_config = _resolve_geometry_config_for_sample(
            data=data,
            idx=sample_idx,
            args=argparse.Namespace(
                cell_size_x_m=None,
                cell_size_y_m=None,
                tile_count_x=None,
                tile_count_y=None,
            ),
            problem_spec=problem_spec,
        )
        grf = np.asarray(data["grf"][sample_idx], dtype=np.float64)
        threshold = float(data["threshold"][sample_idx])

    records: list[dict[str, Any]] = []
    preset_overrides = volume_mesh_preset_overrides(str(mesh_preset))
    for mesh_size_scale in mesh_size_scales:
        for substrate_layers, piezo_layers in layer_sweep:
            case_name = _case_slug(
                mesh_preset=mesh_preset,
                substrate_layers=substrate_layers,
                piezo_layers=piezo_layers,
                mesh_size_scale=mesh_size_scale,
            )
            case_dir = output_dir / case_name
            mesh_dir = case_dir / "meshes" / "volumes"
            response_dir = case_dir / "data" / "fem_responses"
            modal_dir = case_dir / "data" / "modal_data"
            audit_dir = case_dir / "audit"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            response_dir.mkdir(parents=True, exist_ok=True)
            modal_dir.mkdir(parents=True, exist_ok=True)
            if problem_spec is not None:
                write_problem_spec_snapshot(problem_spec, case_dir / "data" / "problem_spec_used.yaml")
                write_problem_spec_snapshot(problem_spec, mesh_dir / "problem_spec_used.yaml")

            volume_config = VolumeMeshConfig(
                substrate_thickness_m=float(substrate_thickness_m),
                piezo_thickness_m=float(piezo_thickness_m),
                mesh_size_relative_to_cell=float(mesh_size_scale),
                cad_reference_size_relative_to_cell=0.01,
                limit_solver_mesh_by_thickness=False,
                substrate_layers=int(substrate_layers),
                piezo_layers=int(piezo_layers),
                solver_mesh_backend="layered_tet",
                require_connected_substrate=True,
                exact_cad=True,
                repair_cad=False,
                max_solver_vector_dofs=(
                    None
                    if preset_overrides["max_solver_vector_dofs"] is None
                    else int(preset_overrides["max_solver_vector_dofs"])
                ),
                allow_solver_mesh_coarsening=bool(preset_overrides["allow_solver_mesh_coarsening"]),
                mesh_preset=str(mesh_preset),
                ansys_step_strategy="single_face_assembly",
            )

            try:
                mesh_path = mesh_tiled_plate_volume_sample(
                    grf=grf,
                    threshold=float(threshold),
                    sample_id=int(sample_id),
                    output_dir=mesh_dir,
                    geometry_config=geometry_config,
                    volume_config=volume_config,
                )
                if mesh_path is None:
                    raise RuntimeError("mesh_tiled_plate_volume_sample returned no mesh path.")
                if problem_spec is not None:
                    step_path = mesh_dir / f"plate3d_{int(sample_id):04d}.step"
                    face_selection_manifest_path = mesh_dir / f"plate3d_{int(sample_id):04d}_ansys_face_groups.json"
                    write_ansys_workbench_handoff(
                        sample_id=int(sample_id),
                        output_path=mesh_dir / f"plate3d_{int(sample_id):04d}_ansys_workbench.json",
                        step_path=step_path if step_path.exists() else None,
                        msh_path=None,
                        cad_report_path=mesh_dir / f"plate3d_{int(sample_id):04d}_cad.json",
                        solver_mesh_path=mesh_path,
                        geometry_config=geometry_config,
                        volume_config=volume_config,
                        problem_spec=problem_spec,
                        inspection_single_face_step_path=None,
                        face_selection_manifest_path=(
                            face_selection_manifest_path if face_selection_manifest_path.exists() else None
                        ),
                    )
            except Exception as exc:
                records.append(
                    _failure_record(
                        case_name=case_name,
                        case_dir=case_dir,
                        sample_id=int(sample_id),
                        substrate_layers=int(substrate_layers),
                        piezo_layers=int(piezo_layers),
                        mesh_size_scale=float(mesh_size_scale),
                        mesh_preset=str(mesh_preset),
                        element_order=int(element_order),
                        status="mesh_failed",
                        exc=exc,
                    )
                )
                print(f"[mesh_failed] {case_name}: {exc}", flush=True)
                continue

            try:
                _run_solver_case(
                    project_root=project_root,
                    mesh_path=mesh_path,
                    response_dir=response_dir,
                    modal_dir=modal_dir,
                    problem_spec_path=resolved_problem_spec_path,
                    substrate_rho=float(substrate_rho),
                    piezo_rho=float(piezo_rho),
                    num_modes=int(solver_num_modes),
                    search_points=int(solver_search_points),
                    element_order=int(element_order),
                    store_mode_shapes=bool(store_mode_shapes),
                    docker_image=str(docker_image),
                )
                summary = audit_run_sample(
                    run_dir=case_dir,
                    sample_id=int(sample_id),
                    output_dir=audit_dir,
                    ansys_modal_hz=ansys_modal_hz,
                    ansys_frf_peak_hz=ansys_frf_peak_hz,
                    ansys_voltage_v=ansys_voltage_v,
                    ansys_voltage_form=ansys_voltage_form,
                )
                record = _record_from_summary(
                    case_name=case_name,
                    case_dir=case_dir,
                    substrate_layers=int(substrate_layers),
                    piezo_layers=int(piezo_layers),
                    mesh_size_scale=float(mesh_size_scale),
                    element_order=int(element_order),
                    summary=summary,
                )
                records.append(record)
                print(
                    f"[ok] {case_name}: mode1={record['mode1_frequency_hz']:.12g} Hz, "
                    f"f_peak={record['f_peak_hz']:.12g} Hz, "
                    f"Vpeak={record['peak_voltage_peak_v']:.12g} V, "
                    f"freq_score={record['frequency_match_score_percent']:.6g}%",
                    flush=True,
                )
            except Exception as exc:
                records.append(
                    _failure_record(
                        case_name=case_name,
                        case_dir=case_dir,
                        sample_id=int(sample_id),
                        substrate_layers=int(substrate_layers),
                        piezo_layers=int(piezo_layers),
                        mesh_size_scale=float(mesh_size_scale),
                        mesh_preset=str(mesh_preset),
                        element_order=int(element_order),
                        status="solve_failed",
                        exc=exc,
                    )
                )
                print(f"[solve_failed] {case_name}: {exc}", flush=True)

    sorted_records = sorted(records, key=_frequency_match_score)
    json_path = output_dir / f"parity_sweep_sample_{int(sample_id):04d}.json"
    csv_path = output_dir / f"parity_sweep_sample_{int(sample_id):04d}.csv"
    summary_payload = {
        "sample_id": int(sample_id),
        "unit_cell_npz": str(unit_cell_npz),
        "problem_spec_path": "" if resolved_problem_spec_path is None else str(resolved_problem_spec_path),
        "mesh_preset": str(mesh_preset),
        "layer_sweep": [[int(sub), int(pz)] for sub, pz in layer_sweep],
        "mesh_size_scales": [float(value) for value in mesh_size_scales],
        "ansys_modal_hz": None if ansys_modal_hz is None else float(ansys_modal_hz),
        "ansys_frf_peak_hz": None if ansys_frf_peak_hz is None else float(ansys_frf_peak_hz),
        "ansys_voltage_v": None if ansys_voltage_v is None else float(ansys_voltage_v),
        "ansys_voltage_form": str(ansys_voltage_form),
        "records": sorted_records,
    }
    json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=False), encoding="utf-8")
    _write_csv(sorted_records, csv_path)

    print("Frequency ranking (best to worst):", flush=True)
    for record in sorted_records:
        if str(record.get("status", "")) != "ok":
            print(
                f"  {record['case_name']}: {record['status']} ({record.get('error_message', '')})",
                flush=True,
            )
            continue
        print(
            f"  {record['case_name']}: "
            f"mode1={record['mode1_frequency_hz']:.12g} Hz, "
            f"f_peak={record['f_peak_hz']:.12g} Hz, "
            f"Vpeak={record['peak_voltage_peak_v']:.12g} V, "
            f"freq_score={record['frequency_match_score_percent']:.6g}%",
            flush=True,
        )
    print(f"Saved parity sweep JSON to {json_path}", flush=True)
    print(f"Saved parity sweep CSV to {csv_path}", flush=True)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-sample ANSYS parity sweep. This keeps the fast reduced modal solver intact, "
            "but sweeps verification mesh settings and audits peak-voltage and modal-vs-FRF comparisons explicitly."
        ),
    )
    parser.add_argument("--sample-id", type=int, required=True, help="Sample id to verify.")
    parser.add_argument("--unit-cell-npz", default="data/unit_cell_dataset.npz", help="Input unit-cell dataset NPZ.")
    parser.add_argument(
        "--output-dir",
        default="tmp/parity_sweeps",
        help="Directory where per-case outputs plus parity_sweep_sample_XXXX.{json,csv} are written.",
    )
    parser.add_argument("--problem-spec", default="", help="Optional shared problem specification YAML.")
    parser.add_argument(
        "--mesh-preset",
        default="ansys_parity",
        choices=["default", "ansys_parity"],
        help="Mesh preset used for the sweep cases.",
    )
    parser.add_argument(
        "--parity-sweep",
        action="store_true",
        help="Run the dedicated parity sweep over multiple through-thickness layer settings.",
    )
    parser.add_argument(
        "--layer-sweep",
        default="",
        help="Comma-separated substrate_layers:piezo_layers cases. Without --parity-sweep this defaults to one case.",
    )
    parser.add_argument(
        "--mesh-size-scales",
        default="0.08",
        help="Comma-separated in-plane mesh-size scales relative to the cell size.",
    )
    parser.add_argument("--ansys-modal-hz", type=float, default=None, help="Optional ANSYS modal reference frequency in Hz.")
    parser.add_argument("--ansys-frf-peak-hz", type=float, default=None, help="Optional ANSYS FRF peak reference in Hz.")
    parser.add_argument("--ansys-voltage-v", type=float, default=None, help="Optional ANSYS voltage reference value.")
    parser.add_argument(
        "--ansys-voltage-form",
        default="unknown",
        choices=["unknown", "peak"],
        help="Interpretation of --ansys-voltage-v. Only peak-voltage comparison is supported.",
    )
    parser.add_argument("--substrate-thickness", type=float, default=None, help="Override substrate thickness in meters.")
    parser.add_argument("--piezo-thickness", type=float, default=None, help="Override piezo thickness in meters.")
    parser.add_argument("--substrate-rho", type=float, default=None, help="Override substrate density in kg/m^3.")
    parser.add_argument("--piezo-rho", type=float, default=None, help="Override piezo density in kg/m^3.")
    parser.add_argument("--solver-num-modes", type=int, default=8, help="Number of structural modes for the reduced solver.")
    parser.add_argument("--solver-search-points", type=int, default=301, help="Coarse FRF search points before refinement.")
    parser.add_argument("--element-order", type=int, default=2, help="Solid displacement interpolation order.")
    parser.add_argument("--docker-image", default="dolfinx/dolfinx:stable", help="Docker image used when FEniCSx is not installed locally.")
    parser.add_argument(
        "--no-store-mode-shapes",
        action="store_true",
        help="Disable extra mode-shape storage. Leave this off only if you need the lightest possible parity run.",
    )
    args = parser.parse_args()
    layer_sweep = (
        _parse_layer_sweep(args.layer_sweep)
        if str(args.layer_sweep).strip()
        else (
            [(2, 1), (4, 2), (6, 2), (8, 3)]
            if bool(args.parity_sweep)
            else _single_case_for_preset(str(args.mesh_preset))
        )
    )

    verify_sample_parity(
        sample_id=int(args.sample_id),
        unit_cell_npz=args.unit_cell_npz,
        output_dir=args.output_dir,
        problem_spec_path=args.problem_spec or None,
        layer_sweep=layer_sweep,
        mesh_size_scales=_parse_float_sweep(args.mesh_size_scales, default=0.08),
        mesh_preset=str(args.mesh_preset),
        ansys_modal_hz=args.ansys_modal_hz,
        ansys_frf_peak_hz=args.ansys_frf_peak_hz,
        ansys_voltage_v=args.ansys_voltage_v,
        ansys_voltage_form=args.ansys_voltage_form,
        substrate_thickness_m=args.substrate_thickness,
        piezo_thickness_m=args.piezo_thickness,
        substrate_rho=args.substrate_rho,
        piezo_rho=args.piezo_rho,
        solver_num_modes=int(args.solver_num_modes),
        solver_search_points=int(args.solver_search_points),
        element_order=int(args.element_order),
        store_mode_shapes=not bool(args.no_store_mode_shapes),
        docker_image=str(args.docker_image),
    )


if __name__ == "__main__":
    main()
