from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .audit_ansys_alignment import audit_run_sample
from .problem_spec import (
    build_runtime_defaults,
    default_problem_spec_path,
    geometry_defaults_from_problem_spec,
    load_problem_spec,
    write_problem_spec_snapshot,
)
from .solver_diagnostics import load_solver_provenance
from .subset_unit_cell_dataset import subset_unit_cell_dataset


@dataclass(frozen=True)
class PipelineConfig:
    source_unit_cell_npz: str | Path
    run_name: str = ""
    output_root: str | Path = "runs"
    limit: int | None = None
    sample_ids: str = ""
    materialize_input_dataset: bool = True
    problem_spec_path: str | Path | None = None
    docker_image: str = "dolfinx/dolfinx:stable"
    substrate_thickness_m: float | None = None
    piezo_thickness_m: float | None = None
    mesh_size_scale: float = 0.08
    cad_reference_size_scale: float = 0.01
    limit_solver_mesh_by_thickness: bool = False
    mesh_preset: str = "default"
    solver_mesh_backend: str = "layered_tet"
    ansys_step_strategy: str = "single_face_assembly"
    substrate_layers: int | None = None
    piezo_layers: int | None = None
    solver_num_modes: int = 8
    solver_search_points: int = 301
    solver_element_order: int = 2
    solver_eigensolver_backend: str = "auto"
    allow_eigensolver_fallback: bool = False
    strict_parity: bool = False
    solver_store_mode_shapes: bool = False
    skip_existing_solver_outputs: bool = True
    solver_max_q2_vector_dofs: int | None = None
    solver_oom_fallback_element_order: int | None = 1
    cell_size_x_m: float | None = None
    cell_size_y_m: float | None = None
    tile_count_x: int | None = None
    tile_count_y: int | None = None
    substrate_rho: float | None = None
    piezo_rho: float | None = None
    exact_cad: bool = True
    repair_cad: bool = False
    repair_bridge_width_m: float | None = None
    export_inspection_single_face_step: bool = False
    create_reports: bool = True
    build_integrated_dataset: bool = True
    audit_ansys_modal_hz: float | None = None
    audit_ansys_frf_peak_hz: float | None = None
    audit_ansys_voltage_v: float | None = None
    audit_ansys_voltage_form: str = "peak"
    audit_sample_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if bool(self.exact_cad) == bool(self.repair_cad):
            raise ValueError("Exactly one CAD mode must be active in PipelineConfig: exact_cad or repair_cad.")
        if self.repair_bridge_width_m is not None and float(self.repair_bridge_width_m) <= 0.0:
            raise ValueError("repair_bridge_width_m must be strictly positive when provided.")
        if self.repair_bridge_width_m is not None and not bool(self.repair_cad):
            raise ValueError(
                "repair_bridge_width_m can only be set when repair_cad=True and exact_cad=False."
            )
        if str(self.solver_mesh_backend).strip().lower() not in {"layered_tet", "gmsh_volume"}:
            raise ValueError("solver_mesh_backend must be 'layered_tet' or 'gmsh_volume'.")
        if str(self.mesh_preset).strip().lower() not in {"default", "ansys_parity"}:
            raise ValueError("mesh_preset must be 'default' or 'ansys_parity'.")
        if str(self.ansys_step_strategy).strip().lower() not in {"partitioned_interface", "single_face_assembly"}:
            raise ValueError("ansys_step_strategy must be 'partitioned_interface' or 'single_face_assembly'.")
        if self.substrate_layers is not None and int(self.substrate_layers) <= 0:
            raise ValueError("substrate_layers must be a positive integer when provided.")
        if self.piezo_layers is not None and int(self.piezo_layers) <= 0:
            raise ValueError("piezo_layers must be a positive integer when provided.")
        if self.solver_max_q2_vector_dofs is not None and int(self.solver_max_q2_vector_dofs) <= 0:
            raise ValueError("solver_max_q2_vector_dofs must be strictly positive when provided.")
        if self.solver_oom_fallback_element_order is not None and int(self.solver_oom_fallback_element_order) <= 0:
            raise ValueError("solver_oom_fallback_element_order must be strictly positive when provided.")
        if str(self.solver_eigensolver_backend).strip().lower() not in {"shift_invert_lu", "iterative_gd", "auto"}:
            raise ValueError("solver_eigensolver_backend must be 'shift_invert_lu', 'iterative_gd', or 'auto'.")
        if bool(self.strict_parity) and bool(self.allow_eigensolver_fallback):
            raise ValueError("allow_eigensolver_fallback cannot be enabled when strict_parity=True.")
        if bool(self.strict_parity) and str(self.solver_eigensolver_backend).strip().lower() == "iterative_gd":
            raise ValueError("strict_parity requires solver_eigensolver_backend='shift_invert_lu' or 'auto'.")
        if str(self.audit_ansys_voltage_form).strip().lower() not in {"unknown", "peak", "rms"}:
            raise ValueError("audit_ansys_voltage_form must be one of: unknown, peak, rms.")


@dataclass(frozen=True)
class PipelineArtifacts:
    project_root: Path
    run_root: Path
    candidate_unit_cell_npz: Path
    unit_cell_npz: Path
    mesh_dir: Path
    response_dir: Path
    modal_dir: Path
    response_dataset_path: Path
    integrated_dataset_path: Path
    integrated_index_csv_path: Path
    audit_dir: Path
    report_dir: Path
    gallery_path: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_python(project_root: Path) -> Path:
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _resolve_path(path: str | Path, base: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def _infer_run_name(config: PipelineConfig, source_path: Path) -> str:
    if config.run_name:
        return str(config.run_name)
    if config.sample_ids.strip():
        return "test_selected"
    if config.limit is not None:
        return f"test_n{int(config.limit):03d}"
    return source_path.stem


def _print_step(title: str) -> None:
    print(flush=True)
    print("=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72, flush=True)


def _mesh_preset_defaults(mesh_preset: str) -> dict[str, object]:
    normalized = str(mesh_preset).strip().lower()
    if normalized == "ansys_parity":
        return {
            "substrate_layers": 8,
            "piezo_layers": 3,
            "solver_max_q2_vector_dofs": None,
            "allow_solver_mesh_coarsening": False,
        }
    if normalized == "default":
        return {
            "substrate_layers": 2,
            "piezo_layers": 1,
            "solver_max_q2_vector_dofs": 5_000_000,
            "allow_solver_mesh_coarsening": True,
        }
    raise ValueError("mesh_preset must be 'default' or 'ansys_parity'.")


def _effective_mesh_builder_settings(config: PipelineConfig) -> dict[str, object]:
    defaults = _mesh_preset_defaults(config.mesh_preset)
    return {
        "mesh_preset": str(config.mesh_preset).strip().lower(),
        "substrate_layers": int(defaults["substrate_layers"] if config.substrate_layers is None else config.substrate_layers),
        "piezo_layers": int(defaults["piezo_layers"] if config.piezo_layers is None else config.piezo_layers),
        "solver_max_q2_vector_dofs": (
            defaults["solver_max_q2_vector_dofs"]
            if config.solver_max_q2_vector_dofs is None
            else int(config.solver_max_q2_vector_dofs)
        ),
        "allow_solver_mesh_coarsening": bool(defaults["allow_solver_mesh_coarsening"]),
    }


def _run_command(args: list[str | Path], cwd: Path, env: dict[str, str] | None = None) -> None:
    rendered = " ".join(shlex.quote(str(arg)) for arg in args)
    print(f"$ {rendered}", flush=True)
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run([str(arg) for arg in args], cwd=str(cwd), env=merged_env, check=True)


def _cli_flag_present(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def _apply_strict_ansys_parity_overrides(
    args: argparse.Namespace,
    *,
    explicit_audit_voltage_form: bool,
) -> argparse.Namespace:
    if not bool(getattr(args, "strict_ansys_parity", False)):
        return args
    args.mesh_preset = "ansys_parity"
    args.strict_parity = True
    args.solver_eigensolver_backend = "shift_invert_lu"
    args.allow_eigensolver_fallback = False
    if not explicit_audit_voltage_form:
        args.audit_ansys_voltage_form = "peak"
    return args


def _build_mesh_command(
    *,
    project_python: Path,
    candidate_unit_cell_npz: Path,
    mesh_dir: Path,
    config: PipelineConfig,
    runtime_problem_spec_path: Path | None,
) -> list[str | Path]:
    mesh_cmd: list[str | Path] = [
        project_python,
        "-m",
        "peh_inverse_design.build_volume_meshes",
        "--unit-cell-npz",
        candidate_unit_cell_npz,
        "--mesh-dir",
        mesh_dir,
        "--substrate-thickness",
        str(config.substrate_thickness_m),
        "--piezo-thickness",
        str(config.piezo_thickness_m),
        "--mesh-size-scale",
        str(config.mesh_size_scale),
        "--cad-reference-size-scale",
        str(config.cad_reference_size_scale),
        "--mesh-preset",
        str(config.mesh_preset),
        "--solver-mesh-backend",
        str(config.solver_mesh_backend),
        "--ansys-step-strategy",
        str(config.ansys_step_strategy),
    ]
    if config.substrate_layers is not None:
        mesh_cmd.extend(["--substrate-layers", str(int(config.substrate_layers))])
    if config.piezo_layers is not None:
        mesh_cmd.extend(["--piezo-layers", str(int(config.piezo_layers))])
    if config.solver_max_q2_vector_dofs is not None:
        mesh_cmd.extend(["--solver-max-q2-vector-dofs", str(int(config.solver_max_q2_vector_dofs))])
    if config.limit_solver_mesh_by_thickness:
        mesh_cmd.append("--limit-solver-mesh-by-thickness")
    if config.repair_cad:
        mesh_cmd.append("--repair-cad")
    if config.repair_bridge_width_m is not None:
        mesh_cmd.extend(["--repair-bridge-width-m", str(float(config.repair_bridge_width_m))])
    if config.export_inspection_single_face_step:
        mesh_cmd.append("--export-inspection-single-face-step")
    if config.cell_size_x_m is not None:
        mesh_cmd.extend(["--cell-size-x-m", str(config.cell_size_x_m)])
    if config.cell_size_y_m is not None:
        mesh_cmd.extend(["--cell-size-y-m", str(config.cell_size_y_m)])
    if config.tile_count_x is not None:
        mesh_cmd.extend(["--tile-count-x", str(int(config.tile_count_x))])
    if config.tile_count_y is not None:
        mesh_cmd.extend(["--tile-count-y", str(int(config.tile_count_y))])
    if runtime_problem_spec_path is not None:
        mesh_cmd.extend(["--problem-spec", runtime_problem_spec_path])
    if config.limit is not None and not config.sample_ids.strip():
        mesh_cmd.extend(["--target-ok", str(int(config.limit))])
    return mesh_cmd


def _audit_is_requested(config: PipelineConfig) -> bool:
    return bool(config.audit_sample_ids) or any(
        value is not None
        for value in [
            config.audit_ansys_modal_hz,
            config.audit_ansys_frf_peak_hz,
            config.audit_ansys_voltage_v,
        ]
    )


def _audit_sample_ids_for_run(config: PipelineConfig, mesh_files: list[Path]) -> list[int]:
    if config.audit_sample_ids:
        unique = sorted({int(sample_id) for sample_id in config.audit_sample_ids})
        return unique
    return sorted({_solver_sample_id_from_mesh(mesh_path) for mesh_path in mesh_files})


def _run_audit_summaries(
    *,
    run_root: Path,
    mesh_files: list[Path],
    config: PipelineConfig,
) -> list[dict[str, object]]:
    audit_dir = run_root / "reports" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for sample_id in _audit_sample_ids_for_run(config, mesh_files):
        summary = audit_run_sample(
            run_dir=run_root,
            sample_id=int(sample_id),
            output_dir=audit_dir,
            ansys_modal_hz=config.audit_ansys_modal_hz,
            ansys_frf_peak_hz=config.audit_ansys_frf_peak_hz,
            ansys_voltage_v=config.audit_ansys_voltage_v,
            ansys_voltage_form=config.audit_ansys_voltage_form,
        )
        summaries.append(summary)
        frequency = summary["frequency_comparison"]
        voltage = summary["voltage_comparison"]
        runtime_inputs = summary.get("runtime_inputs", {})
        print(
            "Audit summary       : "
            f"sample={int(summary['sample_id']):04d}, "
            f"mode1={float(frequency['mode1_frequency_hz']):.12g} Hz, "
            f"f_peak={float(frequency['f_peak_hz']):.12g} Hz, "
            f"Vpeak={float(voltage['peak_voltage_peak_v']):.12g} V, "
            f"Vrms={float(voltage['peak_voltage_rms_v']):.12g} V, "
            f"selected_voltage_error={float(voltage['selected_voltage_error_percent']):.6g}%, "
            f"base_excitation={float(runtime_inputs.get('base_excitation_amplitude_m_per_s2', np.nan)):.6g} m/s^2, "
            f"modal_damping_ratio={float(runtime_inputs.get('modal_damping_ratio', np.nan)):.6g}, "
            f"load_ohm={float(runtime_inputs.get('external_load_resistance_ohm', np.nan)):.6g}"
        )
        solver_provenance = summary.get("solver_provenance", {})
        if bool(config.strict_parity) and (
            not bool(solver_provenance.get("solver_parity_valid", True))
            or bool(solver_provenance.get("diagnostic_only", False))
        ):
            reason = str(solver_provenance.get("parity_invalid_reason", "")).strip() or "strict parity became invalid"
            raise RuntimeError(
                f"Audit for sample {int(summary['sample_id']):04d} detected a strict-parity violation: {reason}"
            )
    return summaries


def _parse_audit_sample_ids(values: list[str] | None) -> tuple[int, ...]:
    parsed: list[int] = []
    for value in values or []:
        for chunk in str(value).split(","):
            token = chunk.strip()
            if token:
                parsed.append(int(token))
    return tuple(parsed)


def _load_mesh_build_summary(mesh_dir: Path) -> dict[str, object] | None:
    summary_path = mesh_dir / "mesh_build_summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cad_failure_guidance(mesh_summary: dict[str, object]) -> str:
    failure_records = mesh_summary.get("failure_records", [])
    if not isinstance(failure_records, list) or not failure_records:
        return ""
    errors = [str(record.get("error", "")) for record in failure_records if isinstance(record, dict)]
    if errors and all("disconnected substrate planform" in error for error in errors):
        return (
            " The current samples are disconnected in exact CAD mode. Either regenerate/filter the unit-cell "
            "dataset so the tiled substrate is one connected component, or rerun with "
            "PipelineConfig(exact_cad=False, repair_cad=True, repair_bridge_width_m=...) to add explicit "
            "connector geometry on purpose."
        )
    if errors and all("under-resolved for CAD export" in error for error in errors):
        return (
            " The current samples are below the exact CAD feature-size tolerance. Regenerate/filter the dataset, "
            "increase the physical feature size, or reduce the CAD strictness only if you intentionally want a "
            "manufacturing-style repair workflow."
        )
    return ""


def _ensure_docker_image(image: str, cwd: Path) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required but was not found in PATH.")
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inspect.returncode != 0:
        _run_command(["docker", "pull", image], cwd=cwd)


def _workspace_path(path: Path, project_root: Path) -> str:
    return str(Path("/workspace") / path.relative_to(project_root))


def _solver_sample_id_from_mesh(mesh_path: Path) -> int:
    matches = re.findall(r"(\d+)", mesh_path.stem)
    if not matches:
        raise ValueError(f"Could not infer sample id from solver mesh path: {mesh_path}")
    return int(matches[-1])


def _solver_response_output_path(response_dir: Path, sample_id: int) -> Path:
    return response_dir / f"sample_{int(sample_id):04d}_response.npz"


def _solver_modal_output_path(modal_dir: Path, sample_id: int) -> Path:
    return modal_dir / f"sample_{int(sample_id):04d}_modal.npz"


def _solver_diagnostic_output_path(response_dir: Path, sample_id: int) -> Path:
    return response_dir / f"sample_{int(sample_id):04d}_solver_diagnostic.json"


def _assert_strict_parity_outputs_valid(
    *,
    mesh_files: list[Path],
    response_dir: Path,
) -> None:
    for mesh_path in mesh_files:
        sample_id = _solver_sample_id_from_mesh(mesh_path)
        response_path = _solver_response_output_path(response_dir, sample_id)
        if not response_path.exists():
            raise FileNotFoundError(
                f"Strict parity run for sample {sample_id:04d} did not produce {response_path}."
            )
        with np.load(response_path, allow_pickle=True) as response:
            provenance = load_solver_provenance(response)
        if bool(provenance.get("solver_parity_valid", True)) and not bool(provenance.get("diagnostic_only", False)):
            continue
        reason = str(provenance.get("parity_invalid_reason", "")).strip() or "strict parity became invalid"
        raise RuntimeError(
            f"Strict parity became invalid for sample {sample_id:04d}: {reason} "
            f"[backend={provenance.get('eigensolver_backend', 'unknown')}, "
            f"used_eigensolver_fallback={bool(provenance.get('used_eigensolver_fallback', False))}, "
            f"used_element_order_fallback={bool(provenance.get('used_element_order_fallback', False))}]"
        )


def _solver_requested_backend(config: PipelineConfig) -> str:
    backend = str(config.solver_eigensolver_backend).strip().lower()
    if bool(config.strict_parity) and backend == "auto":
        return "shift_invert_lu"
    return backend


def _parity_sensitive_solver_run(config: PipelineConfig) -> bool:
    return bool(config.strict_parity) or str(config.mesh_preset).strip().lower() == "ansys_parity" or _audit_is_requested(config)


def _write_pipeline_solver_diagnostic(
    *,
    response_dir: Path,
    mesh_path: Path,
    config: PipelineConfig,
    parity_invalid_reason: str,
    error_message: str,
    actual_element_order: int | None = None,
    used_element_order_fallback: bool = False,
) -> Path:
    sample_id = _solver_sample_id_from_mesh(mesh_path)
    payload = {
        "sample_id": int(sample_id),
        "mesh_path": str(mesh_path),
        "eigensolver_backend": _solver_requested_backend(config),
        "requested_eigensolver_backend": _solver_requested_backend(config),
        "solver_element_order": int(config.solver_element_order if actual_element_order is None else actual_element_order),
        "requested_solver_element_order": int(config.solver_element_order),
        "used_eigensolver_fallback": False,
        "used_element_order_fallback": bool(used_element_order_fallback),
        "solver_parity_valid": False,
        "parity_invalid_reason": str(parity_invalid_reason),
        "strict_parity_requested": bool(config.strict_parity),
        "diagnostic_only": bool(used_element_order_fallback or config.allow_eigensolver_fallback),
        "error_message": str(error_message),
    }
    output_path = _solver_diagnostic_output_path(response_dir, sample_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return output_path


def _solver_outputs_exist(mesh_path: Path, response_dir: Path, modal_dir: Path) -> bool:
    sample_id = _solver_sample_id_from_mesh(mesh_path)
    return (
        _solver_response_output_path(response_dir, sample_id).exists()
        and _solver_modal_output_path(modal_dir, sample_id).exists()
    )


def _build_solver_inner_args(
    *,
    project_root: Path,
    response_dir: Path,
    modal_dir: Path,
    config: PipelineConfig,
    runtime_problem_spec_path: Path | None,
    mesh_path: Path | None = None,
    mesh_dir: Path | None = None,
    element_order: int | None = None,
    requested_element_order: int | None = None,
) -> list[str]:
    actual_element_order = int(config.solver_element_order if element_order is None else element_order)
    requested_solver_element_order = int(
        config.solver_element_order if requested_element_order is None else requested_element_order
    )
    solver_args = ["python3", "-m", "peh_inverse_design.fenicsx_modal_solver"]
    if mesh_path is not None:
        solver_args.extend(["--mesh", _workspace_path(mesh_path, project_root)])
    if mesh_dir is not None:
        solver_args.extend(["--mesh-dir", _workspace_path(mesh_dir, project_root)])
    solver_args.extend(
        [
            "--response-dir",
            _workspace_path(response_dir, project_root),
            "--modes-dir",
            _workspace_path(modal_dir, project_root),
            "--substrate-rho",
            str(config.substrate_rho),
            "--piezo-rho",
            str(config.piezo_rho),
            "--num-modes",
            str(config.solver_num_modes),
            "--search-points",
            str(config.solver_search_points),
            "--element-order",
            str(actual_element_order),
            "--requested-element-order",
            str(requested_solver_element_order),
            "--eigensolver-backend",
            _solver_requested_backend(config),
        ]
    )
    if config.allow_eigensolver_fallback:
        solver_args.append("--allow-eigensolver-fallback")
    if config.strict_parity:
        solver_args.append("--strict-parity")
    if config.solver_store_mode_shapes:
        solver_args.append("--store-mode-shapes")
    if config.skip_existing_solver_outputs:
        solver_args.append("--skip-existing")
    if runtime_problem_spec_path is not None:
        solver_args.extend(["--problem-spec", _workspace_path(runtime_problem_spec_path, project_root)])
    return solver_args


def _build_solver_docker_command(
    *,
    project_root: Path,
    response_dir: Path,
    modal_dir: Path,
    config: PipelineConfig,
    runtime_problem_spec_path: Path | None,
    mesh_path: Path | None = None,
    mesh_dir: Path | None = None,
    element_order: int | None = None,
    requested_element_order: int | None = None,
) -> list[str]:
    solver_args = _build_solver_inner_args(
        project_root=project_root,
        response_dir=response_dir,
        modal_dir=modal_dir,
        config=config,
        runtime_problem_spec_path=runtime_problem_spec_path,
        mesh_path=mesh_path,
        mesh_dir=mesh_dir,
        element_order=element_order,
        requested_element_order=requested_element_order,
    )
    solver_inner = " ".join(shlex.quote(str(arg)) for arg in solver_args)
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{project_root}:/workspace",
        "-w",
        "/workspace",
        config.docker_image,
        "bash",
        "-c",
        f"pip install -q pyyaml && {solver_inner}",
    ]


def _run_solver_with_isolated_retry(
    *,
    mesh_files: list[Path],
    project_root: Path,
    response_dir: Path,
    modal_dir: Path,
    config: PipelineConfig,
    runtime_problem_spec_path: Path | None,
) -> None:
    parity_sensitive = _parity_sensitive_solver_run(config)
    batch_cmd = _build_solver_docker_command(
        project_root=project_root,
        response_dir=response_dir,
        modal_dir=modal_dir,
        config=config,
        runtime_problem_spec_path=runtime_problem_spec_path,
        mesh_dir=mesh_files[0].parent,
    )
    try:
        _run_command(batch_cmd, cwd=project_root)
        return
    except subprocess.CalledProcessError as exc:
        if exc.returncode != 137:
            raise

    remaining = [mesh_path for mesh_path in mesh_files if not _solver_outputs_exist(mesh_path, response_dir, modal_dir)]
    if not remaining:
        return

    print(
        "Batch Docker solve exited with status 137 after partial progress. "
        "Retrying remaining meshes one by one in fresh containers."
    )
    for mesh_path in remaining:
        print(f"Retrying {mesh_path.name} in an isolated Docker run.")
        try:
            _run_command(
                _build_solver_docker_command(
                    project_root=project_root,
                    response_dir=response_dir,
                    modal_dir=modal_dir,
                    config=config,
                    runtime_problem_spec_path=runtime_problem_spec_path,
                    mesh_path=mesh_path,
                ),
                cwd=project_root,
            )
        except subprocess.CalledProcessError as mesh_exc:
            fallback_order = config.solver_oom_fallback_element_order
            if (
                mesh_exc.returncode == 137
                and fallback_order is not None
                and int(fallback_order) < int(config.solver_element_order)
                and not parity_sensitive
            ):
                print(
                    f"{mesh_path.name} hit Docker exit 137 with element_order={config.solver_element_order}. "
                    f"Retrying once with the lower-memory fallback element_order={int(fallback_order)}."
                )
                try:
                    _run_command(
                        _build_solver_docker_command(
                            project_root=project_root,
                            response_dir=response_dir,
                            modal_dir=modal_dir,
                            config=config,
                            runtime_problem_spec_path=runtime_problem_spec_path,
                            mesh_path=mesh_path,
                            element_order=int(fallback_order),
                            requested_element_order=int(config.solver_element_order),
                        ),
                        cwd=project_root,
                    )
                    continue
                except subprocess.CalledProcessError as fallback_exc:
                    mesh_exc = fallback_exc
            if mesh_exc.returncode == 137 and parity_sensitive:
                diagnostic_path = _write_pipeline_solver_diagnostic(
                    response_dir=response_dir,
                    mesh_path=mesh_path,
                    config=config,
                    parity_invalid_reason=(
                        "docker exit 137 occurred, and parity-sensitive runs are not allowed to downgrade element order"
                    ),
                    error_message=(
                        f"Solver container exited with status 137 for {mesh_path.name} at element_order="
                        f"{config.solver_element_order}."
                    ),
                    actual_element_order=int(config.solver_element_order),
                    used_element_order_fallback=False,
                )
                raise RuntimeError(
                    f"Isolated solver retry failed for {mesh_path.name} with exit status 137. "
                    "This run is parity-invalid because element-order fallback is disabled for strict/audit/ansys_parity runs. "
                    f"Diagnostic payload: {diagnostic_path}"
                ) from mesh_exc
            raise RuntimeError(
                f"Isolated solver retry failed for {mesh_path.name} with exit status {mesh_exc.returncode}. "
                "If this persists, reduce the number of samples per run, lower solver_max_q2_vector_dofs, "
                "or lower --solver-element-order."
            ) from mesh_exc


def _resolve_problem_spec_path(config: PipelineConfig, project_root: Path) -> Path | None:
    if config.problem_spec_path is not None and str(config.problem_spec_path).strip():
        path = _resolve_path(config.problem_spec_path, project_root)
        if not path.exists():
            raise FileNotFoundError(f"Problem specification not found: {path}")
        return path

    default_path = default_problem_spec_path(project_root)
    if default_path.exists():
        return default_path
    return None


def _resolve_runtime_config(config: PipelineConfig, problem_spec: dict[str, object] | None) -> PipelineConfig:
    runtime_defaults = build_runtime_defaults(problem_spec) if problem_spec is not None else {}
    return replace(
        config,
        substrate_thickness_m=float(
            runtime_defaults.get("substrate_thickness_m", 1.0e-3)
            if config.substrate_thickness_m is None
            else config.substrate_thickness_m
        ),
        piezo_thickness_m=float(
            runtime_defaults.get("piezo_thickness_m", 1.0e-4)
            if config.piezo_thickness_m is None
            else config.piezo_thickness_m
        ),
        substrate_rho=float(
            runtime_defaults.get("substrate_rho", 7930.0)
            if config.substrate_rho is None
            else config.substrate_rho
        ),
        piezo_rho=float(
            runtime_defaults.get("piezo_rho", 7500.0)
            if config.piezo_rho is None
            else config.piezo_rho
        ),
    )


def _write_run_config_snapshot(
    run_root: Path,
    config: PipelineConfig,
    problem_spec_path: Path | None,
    runtime_problem_spec_path: Path | None,
    geometry_scale_source: str,
    plate_size_m: tuple[float, float],
) -> Path:
    snapshot_path = run_root / "data" / "run_config_used.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "problem_spec_path": "" if problem_spec_path is None else str(problem_spec_path),
        "runtime_problem_spec_path": "" if runtime_problem_spec_path is None else str(runtime_problem_spec_path),
        "geometry_scale_source": str(geometry_scale_source),
        "effective_plate_size_m": [float(plate_size_m[0]), float(plate_size_m[1])],
        "effective_config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(config).items()
        },
    }
    snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return snapshot_path


def _prepare_candidate_unit_cell_dataset(config: PipelineConfig, run_root: Path, project_root: Path) -> Path:
    source_path = _resolve_path(config.source_unit_cell_npz, project_root)
    if not source_path.exists():
        raise FileNotFoundError(f"Unit-cell dataset not found: {source_path}")

    if config.sample_ids.strip():
        target = run_root / "data" / "unit_cell_candidates.npz"
        subset_unit_cell_dataset(
            input_path=source_path,
            output_path=target,
            sample_ids=config.sample_ids,
        )
        return target

    if config.materialize_input_dataset:
        target = run_root / "data" / "unit_cell_candidates.npz"
        subset_unit_cell_dataset(
            input_path=source_path,
            output_path=target,
            limit=None,
        )
        return target

    return source_path


def _materialize_successful_unit_cell_dataset(
    candidate_unit_cell_npz: Path,
    output_path: Path,
    source_indices: list[int] | np.ndarray,
) -> Path:
    if len(source_indices) == 0:
        raise ValueError("source_indices must contain at least one successful candidate.")
    return subset_unit_cell_dataset(
        input_path=candidate_unit_cell_npz,
        output_path=output_path,
        source_indices=source_indices,
    )


def _explicit_float_pair(
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


def _explicit_int_pair(
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


def _metadata_pair(data: np.lib.npyio.NpzFile, key: str) -> tuple[float, float] | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.ndim == 1 and arr.shape[0] == 2:
        return float(arr[0]), float(arr[1])
    if arr.ndim >= 2 and arr.shape[1] == 2:
        first = np.asarray(arr[0], dtype=np.float64)
        values = np.asarray(arr, dtype=np.float64)
        if not np.allclose(values, first, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{key} varies per sample; pass explicit PipelineConfig values instead.")
        return float(first[0]), float(first[1])
    raise ValueError(f"{key} must have shape (2,) or (N, 2); got {arr.shape}.")


def _metadata_counts(data: np.lib.npyio.NpzFile, key: str) -> tuple[int, int] | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.ndim == 1 and arr.shape[0] == 2:
        return int(arr[0]), int(arr[1])
    if arr.ndim >= 2 and arr.shape[1] == 2:
        first = np.asarray(arr[0], dtype=np.int64)
        values = np.asarray(arr, dtype=np.int64)
        if not np.array_equal(values, np.broadcast_to(first, values.shape)):
            raise ValueError(f"{key} varies per sample; pass explicit PipelineConfig values instead.")
        return int(first[0]), int(first[1])
    raise ValueError(f"{key} must have shape (2,) or (N, 2); got {arr.shape}.")


def _resolve_geometry_scale_summary(
    config: PipelineConfig,
    unit_cell_npz: Path,
    problem_spec_path: Path | None,
    project_root: Path,
) -> tuple[tuple[float, float], tuple[int, int], str]:
    explicit_cell_size = _explicit_float_pair(
        config.cell_size_x_m,
        config.cell_size_y_m,
        "PipelineConfig.cell_size_x_m",
        "PipelineConfig.cell_size_y_m",
    )
    explicit_tile_counts = _explicit_int_pair(
        config.tile_count_x,
        config.tile_count_y,
        "PipelineConfig.tile_count_x",
        "PipelineConfig.tile_count_y",
    )

    if explicit_cell_size is not None and explicit_tile_counts is not None:
        return explicit_cell_size, explicit_tile_counts, "PipelineConfig"
    if explicit_cell_size is not None or explicit_tile_counts is not None:
        raise ValueError(
            "Provide all four geometry scale fields together in PipelineConfig: "
            "cell_size_x_m, cell_size_y_m, tile_count_x, tile_count_y."
        )

    data = np.load(unit_cell_npz, allow_pickle=True)
    metadata_cell_size = _metadata_pair(data, "cell_size_m")
    metadata_tile_counts = _metadata_counts(data, "tile_counts")
    if metadata_cell_size is not None and metadata_tile_counts is not None:
        return metadata_cell_size, metadata_tile_counts, unit_cell_npz.name

    if problem_spec_path is not None:
        problem_spec = load_problem_spec(problem_spec_path, project_root=project_root)
        cell_size_m, tile_counts = geometry_defaults_from_problem_spec(problem_spec)
        return cell_size_m, tile_counts, problem_spec_path.name

    raise ValueError(
        "Physical geometry scale is missing. Set PipelineConfig.cell_size_x_m, "
        "PipelineConfig.cell_size_y_m, PipelineConfig.tile_count_x, and "
        "PipelineConfig.tile_count_y, include cell_size_m and tile_counts in the input NPZ, "
        "or provide a shared problem specification YAML."
    )


def run_pipeline(config: PipelineConfig) -> PipelineArtifacts:
    project_root = _project_root()
    project_python = _project_python(project_root)
    source_path = _resolve_path(config.source_unit_cell_npz, project_root)
    run_name = _infer_run_name(config, source_path)
    run_root = _resolve_path(Path(config.output_root) / run_name, project_root)
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Notebook/kernel python : {sys.executable}", flush=True)
    print(f"Pipeline step python   : {project_python}", flush=True)

    problem_spec_path = _resolve_problem_spec_path(config, project_root)
    runtime_problem_spec_path: Path | None = None
    problem_spec: dict[str, object] | None = None
    if problem_spec_path is not None:
        runtime_problem_spec_path = run_root / "data" / "problem_spec_used.yaml"
        problem_spec = load_problem_spec(problem_spec_path, project_root=project_root)
        write_problem_spec_snapshot(problem_spec, runtime_problem_spec_path)
    config = _resolve_runtime_config(config, problem_spec)
    candidate_unit_cell_npz = _prepare_candidate_unit_cell_dataset(config, run_root, project_root)
    cell_size_m, tile_counts, geometry_scale_source = _resolve_geometry_scale_summary(
        config,
        candidate_unit_cell_npz,
        problem_spec_path=runtime_problem_spec_path,
        project_root=project_root,
    )
    success_unit_cell_npz = run_root / "data" / "unit_cell_dataset.npz"
    mesh_dir = run_root / "meshes" / "volumes"
    response_dir = run_root / "data" / "fem_responses"
    modal_dir = run_root / "data" / "modal_data"
    response_dataset_path = run_root / "data" / "response_dataset.npz"
    integrated_dataset_path = run_root / "data" / "integrated_dataset.npz"
    integrated_index_csv_path = integrated_dataset_path.with_suffix(".csv")
    report_dir = run_root / "reports"
    audit_dir = report_dir / "audit"
    gallery_path = report_dir / "gallery.png"
    effective_mesh = _effective_mesh_builder_settings(config)
    total_steps = 5 + int(_audit_is_requested(config))
    step_idx = 1

    plate_size_m = (
        float(cell_size_m[0]) * int(tile_counts[0]),
        float(cell_size_m[1]) * int(tile_counts[1]),
    )
    run_config_path = _write_run_config_snapshot(
        run_root=run_root,
        config=config,
        problem_spec_path=problem_spec_path,
        runtime_problem_spec_path=runtime_problem_spec_path,
        geometry_scale_source=geometry_scale_source,
        plate_size_m=plate_size_m,
    )
    print(
        "Geometry scale       : "
        f"cell=({cell_size_m[0]:.6g}, {cell_size_m[1]:.6g}) m, "
        f"tiles={tile_counts[0]} x {tile_counts[1]}, "
        f"plate=({plate_size_m[0]:.6g}, {plate_size_m[1]:.6g}) m "
        f"[source: {geometry_scale_source}]"
    )
    print(f"Run config snapshot : {run_config_path}")
    cad_mode = "repair" if config.repair_cad else "exact"
    if config.repair_cad and config.repair_bridge_width_m is not None:
        print(
            "CAD export mode     : "
            f"{cad_mode} (bridge width = {float(config.repair_bridge_width_m):.6g} m)"
        )
    else:
        print(f"CAD export mode     : {cad_mode}")
    print(
        "Solver profile      : "
        f"backend={config.solver_mesh_backend}, mesh_preset={effective_mesh['mesh_preset']}, "
        f"layers={effective_mesh['substrate_layers']}+{effective_mesh['piezo_layers']}, "
        f"modes={config.solver_num_modes}, search_points={config.solver_search_points}, "
        f"element_order={config.solver_element_order}, eigensolver_backend={_solver_requested_backend(config)}, "
        f"allow_eigensolver_fallback={config.allow_eigensolver_fallback}, strict_parity={config.strict_parity}, "
        f"skip_existing={config.skip_existing_solver_outputs}, "
        f"mesh_size_scale={config.mesh_size_scale:.6g}, cad_reference_scale={config.cad_reference_size_scale:.6g}, "
        f"max_q2_vector_dofs={effective_mesh['solver_max_q2_vector_dofs']}, "
        f"oom_fallback_order={config.solver_oom_fallback_element_order}, "
        f"ansys_step_strategy={config.ansys_step_strategy}, thickness_limited_mesh={config.limit_solver_mesh_by_thickness}"
    )

    _print_step(f"Step {step_idx}/{total_steps}: Export ANSYS STEP geometry and build Python solver meshes")
    step_idx += 1
    summary_path = mesh_dir / "mesh_build_summary.json"
    if summary_path.exists():
        summary_path.unlink()
    mesh_cmd = _build_mesh_command(
        project_python=project_python,
        candidate_unit_cell_npz=candidate_unit_cell_npz,
        mesh_dir=mesh_dir,
        config=config,
        runtime_problem_spec_path=runtime_problem_spec_path,
    )
    try:
        _run_command(mesh_cmd, cwd=project_root)
    except subprocess.CalledProcessError as exc:
        mesh_summary = _load_mesh_build_summary(mesh_dir)
        if mesh_summary is not None:
            failure_records = mesh_summary.get("failure_records", [])
            detail = ""
            if failure_records:
                first = failure_records[0]
                detail = f" First CAD rejection: sample {int(first['sample_id']):04d}: {first['error']}"
            guidance = _cad_failure_guidance(mesh_summary)
            raise RuntimeError(
                f"Step 1/5 failed while exporting STEP geometry or building solver meshes in {mesh_dir}."
                f"{detail}{guidance} See {summary_path} for the full summary."
            ) from exc
        raise
    mesh_summary = _load_mesh_build_summary(mesh_dir)
    if mesh_summary is not None:
        ok = int(mesh_summary.get("ok", 0))
        fail = int(mesh_summary.get("fail", 0))
        print(f"Mesh build summary   : ok={ok} fail={fail} [mode: {mesh_summary.get('cad_mode', 'unknown')}]")
        if ok == 0:
            failure_records = mesh_summary.get("failure_records", [])
            detail = ""
            if failure_records:
                first = failure_records[0]
                detail = f" First CAD rejection: sample {int(first['sample_id']):04d}: {first['error']}"
            guidance = _cad_failure_guidance(mesh_summary)
            raise RuntimeError(
                f"Step 1 generated no valid STEP/solver-mesh exports in {mesh_dir}."
                f"{detail}{guidance} See {summary_path} for the full summary."
            )
        if fail > 0:
            print(f"Proceeding with {ok} successful mesh exports; {fail} samples were rejected by the CAD validator.")

        selected_source_indices_raw = mesh_summary.get("selected_source_indices", [])
        if not isinstance(selected_source_indices_raw, list) or not selected_source_indices_raw:
            raise RuntimeError(
                f"Step 1 succeeded but did not record selected_source_indices in {summary_path}."
            )
        selected_source_indices = [int(value) for value in selected_source_indices_raw]
        _materialize_successful_unit_cell_dataset(
            candidate_unit_cell_npz=candidate_unit_cell_npz,
            output_path=success_unit_cell_npz,
            source_indices=selected_source_indices,
        )
        unit_cell_npz = success_unit_cell_npz
        print(f"Successful geometry dataset: {unit_cell_npz} ({len(selected_source_indices)} samples)")
    else:
        raise RuntimeError(f"Step 1 completed without writing {summary_path}.")

    _print_step(f"Step {step_idx}/{total_steps}: Ensure Docker image")
    step_idx += 1
    _ensure_docker_image(config.docker_image, cwd=project_root)

    _print_step(f"Step {step_idx}/{total_steps}: Run FEniCSx solver")
    step_idx += 1
    response_dir.mkdir(parents=True, exist_ok=True)
    modal_dir.mkdir(parents=True, exist_ok=True)
    mesh_files = sorted(mesh_dir.glob("plate3d_*_fenicsx.npz"))
    if not mesh_files:
        raise FileNotFoundError(
            f"No FEniCSx mesh files found in {mesh_dir}. "
            f"Check {summary_path} for the CAD export summary."
        )
    print(f"Batch-solving {len(mesh_files)} mesh file(s) in one Docker run.")
    _run_solver_with_isolated_retry(
        mesh_files=mesh_files,
        project_root=project_root,
        response_dir=response_dir,
        modal_dir=modal_dir,
        config=config,
        runtime_problem_spec_path=runtime_problem_spec_path,
    )
    if config.strict_parity:
        _assert_strict_parity_outputs_valid(
            mesh_files=mesh_files,
            response_dir=response_dir,
        )

    _print_step(f"Step {step_idx}/{total_steps}: Aggregate datasets")
    step_idx += 1
    _run_command(
        [
            project_python,
            "-m",
            "peh_inverse_design.build_response_dataset",
            "--response-dir",
            response_dir,
            "--output",
            response_dataset_path,
        ],
        cwd=project_root,
    )
    if config.build_integrated_dataset:
        integrated_cmd: list[str | Path] = [
            project_python,
            "-m",
            "peh_inverse_design.build_integrated_dataset",
            "--unit-cell-npz",
            unit_cell_npz,
            "--response-dir",
            response_dir,
            "--modal-dir",
            modal_dir,
            "--mesh-dir",
            mesh_dir,
            "--output",
            integrated_dataset_path,
        ]
        _run_command(integrated_cmd, cwd=project_root)

    if _audit_is_requested(config):
        _print_step(f"Step {step_idx}/{total_steps}: Audit ANSYS alignment")
        step_idx += 1
        _run_audit_summaries(
            run_root=run_root,
            mesh_files=mesh_files,
            config=config,
        )

    _print_step(f"Step {step_idx}/{total_steps}: Create reports")
    if config.create_reports:
        env = dict(os.environ)
        env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
        _run_command(
            [
                project_python,
                "peh_inverse_design/visualize_run_outputs.py",
                "--dataset",
                unit_cell_npz,
                "--mesh-dir",
                mesh_dir,
                "--response-dir",
                response_dir,
                "--modal-dir",
                modal_dir,
                "--output-dir",
                report_dir,
            ],
            cwd=project_root,
            env=env,
        )

    artifacts = PipelineArtifacts(
        project_root=project_root,
        run_root=run_root,
        candidate_unit_cell_npz=candidate_unit_cell_npz,
        unit_cell_npz=unit_cell_npz,
        mesh_dir=mesh_dir,
        response_dir=response_dir,
        modal_dir=modal_dir,
        response_dataset_path=response_dataset_path,
        integrated_dataset_path=integrated_dataset_path,
        integrated_index_csv_path=integrated_index_csv_path,
        audit_dir=audit_dir,
        report_dir=report_dir,
        gallery_path=gallery_path,
    )
    print()
    print("Pipeline complete.")
    for key, value in artifacts.as_dict().items():
        print(f"{key}: {value}")
    return artifacts



def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the integrated PEH inverse-design pipeline: solid STEP geometry export, "
            "FEniCSx modal solve, dataset aggregation, and reporting."
        ),
    )
    parser.add_argument("--unit-cell-npz", required=True, help="Input unit-cell dataset NPZ.")
    parser.add_argument("--run-name", default="", help="Optional run name. Defaults to a name inferred from the dataset or limit.")
    parser.add_argument("--output-root", default="runs", help="Directory under which the run folder is created.")
    parser.add_argument("--limit", type=int, default=None, help="Requested number of successful solid exports when sample-ids is empty.")
    parser.add_argument("--sample-ids", default="", help="Comma-separated sample ids to process exactly. Overrides limit-based success targeting.")
    parser.add_argument("--problem-spec", default="", help="Optional shared problem specification YAML.")
    parser.add_argument("--docker-image", "--image", dest="docker_image", default="dolfinx/dolfinx:stable", help="Docker image used for the FEniCSx solve.")
    parser.add_argument(
        "--substrate-thickness",
        type=float,
        default=None,
        help="Substrate thickness in meters. Defaults to the shared problem spec when available.",
    )
    parser.add_argument(
        "--piezo-thickness",
        type=float,
        default=None,
        help="Piezo thickness in meters. Defaults to the shared problem spec when available.",
    )
    parser.add_argument("--mesh-size-scale", type=float, default=0.08, help="Target in-plane solver element size as a fraction of one cell size.")
    parser.add_argument("--cad-reference-size-scale", type=float, default=0.01, help="Reference size for CAD feature checks as a fraction of one cell size.")
    parser.add_argument(
        "--mesh-preset",
        default="default",
        choices=["default", "ansys_parity"],
        help="Named mesh preset forwarded to build_volume_meshes. Explicit layer and DOF-cap overrides still win.",
    )
    parser.add_argument("--solver-mesh-backend", default="layered_tet", choices=["layered_tet", "gmsh_volume"], help="Python solver mesh backend. layered_tet is the fast default.")
    parser.add_argument("--ansys-step-strategy", default="single_face_assembly", choices=["partitioned_interface", "single_face_assembly"], help="Primary one-file ANSYS STEP strategy.")
    parser.add_argument("--substrate-layers", type=int, default=None, help="Explicit substrate-layer override. Defaults to the selected mesh preset.")
    parser.add_argument("--piezo-layers", type=int, default=None, help="Explicit piezo-layer override. Defaults to the selected mesh preset.")
    parser.add_argument("--limit-solver-mesh-by-thickness", action="store_true", help="Restore thickness-limited solver meshing. Disabled by default for thin plates.")
    parser.add_argument("--solver-num-modes", type=int, default=8, help="Number of modes retained by the in-house modal solver.")
    parser.add_argument("--solver-search-points", type=int, default=301, help="Coarse FRF search points used before local peak refinement.")
    parser.add_argument("--solver-element-order", type=int, default=2, help="Solid displacement interpolation order for the in-house modal solver.")
    parser.add_argument(
        "--solver-eigensolver-backend",
        default="auto",
        choices=["shift_invert_lu", "iterative_gd", "auto"],
        help="Eigensolver backend forwarded to fenicsx_modal_solver.",
    )
    parser.add_argument(
        "--allow-eigensolver-fallback",
        action="store_true",
        help="Allow automatic shift_invert_lu -> iterative_gd fallback for diagnostic-only solver runs.",
    )
    parser.add_argument(
        "--strict-parity",
        action="store_true",
        help="Disable solver fallbacks that break apples-to-apples parity and forward strict parity to the solver.",
    )
    parser.add_argument(
        "--strict-ansys-parity",
        action="store_true",
        help=(
            "Convenience strict parity mode: mesh_preset=ansys_parity, strict_parity=True, "
            "solver_eigensolver_backend=shift_invert_lu, allow_eigensolver_fallback=False, "
            "and audit_ansys_voltage_form=peak unless you explicitly override it."
        ),
    )
    parser.add_argument(
        "--solver-max-q2-vector-dofs",
        type=int,
        default=None,
        help="Explicit quadratic vector-DOF cap override for layered_tet meshes. Defaults to the selected mesh preset.",
    )
    parser.add_argument(
        "--solver-oom-fallback-element-order",
        type=int,
        default=1,
        help="Retry isolated Docker OOM failures once with this lower displacement order. Set 0 to disable the fallback.",
    )
    parser.add_argument("--solver-store-mode-shapes", action="store_true", help="Store per-mode nodal diagnostics from the in-house solver.")
    parser.add_argument("--no-skip-existing-solver-outputs", action="store_true", help="Recompute responses even when per-sample outputs already exist.")
    parser.add_argument("--cell-size-x-m", type=float, default=None, help="Unit-cell size in x in meters.")
    parser.add_argument("--cell-size-y-m", type=float, default=None, help="Unit-cell size in y in meters.")
    parser.add_argument("--tile-count-x", type=int, default=None, help="Number of tiled unit cells in x.")
    parser.add_argument("--tile-count-y", type=int, default=None, help="Number of tiled unit cells in y.")
    parser.add_argument(
        "--substrate-rho",
        type=float,
        default=None,
        help="Substrate density in kg/m^3. Defaults to the shared problem spec when available.",
    )
    parser.add_argument(
        "--piezo-rho",
        type=float,
        default=None,
        help="Piezo density in kg/m^3. Defaults to the shared problem spec when available.",
    )
    parser.add_argument("--repair-cad", action="store_true", help="Use explicit bridge-repair CAD instead of exact topology-preserving CAD.")
    parser.add_argument("--repair-bridge-width-m", type=float, default=None, help="Explicit bridge width in repair CAD mode.")
    parser.add_argument("--audit-ansys-modal-hz", type=float, default=None, help="Optional ANSYS modal reference frequency in Hz.")
    parser.add_argument("--audit-ansys-frf-peak-hz", type=float, default=None, help="Optional ANSYS FRF peak reference frequency in Hz.")
    parser.add_argument("--audit-ansys-voltage-v", type=float, default=None, help="Optional ANSYS voltage reference value.")
    parser.add_argument(
        "--audit-ansys-voltage-form",
        default="peak",
        choices=["unknown", "peak", "rms"],
        help=(
            "Interpretation of --audit-ansys-voltage-v. Defaults to 'peak'; "
            "use 'unknown' only when you explicitly want both peak and RMS comparisons."
        ),
    )
    parser.add_argument(
        "--audit-sample-id",
        action="append",
        default=[],
        help="Sample id(s) to audit after aggregation. Repeat the flag or pass a comma-separated list.",
    )
    parser.add_argument("--no-reports", action="store_true", help="Skip summary image generation.")
    parser.add_argument("--no-integrated-dataset", action="store_true", help="Skip integrated_dataset.npz generation.")
    parser.add_argument("--no-materialize-input-dataset", action="store_true", help="Use the source NPZ directly instead of writing a run-local candidate NPZ copy.")
    return parser


def main() -> None:
    parser = _cli_parser()
    raw_argv = sys.argv[1:]
    args = parser.parse_args()
    args = _apply_strict_ansys_parity_overrides(
        args,
        explicit_audit_voltage_form=_cli_flag_present(raw_argv, "--audit-ansys-voltage-form"),
    )
    config = PipelineConfig(
        source_unit_cell_npz=args.unit_cell_npz,
        run_name=args.run_name,
        output_root=args.output_root,
        limit=args.limit,
        sample_ids=args.sample_ids,
        materialize_input_dataset=not bool(args.no_materialize_input_dataset),
        problem_spec_path=args.problem_spec or None,
        docker_image=args.docker_image,
        substrate_thickness_m=None if args.substrate_thickness is None else float(args.substrate_thickness),
        piezo_thickness_m=None if args.piezo_thickness is None else float(args.piezo_thickness),
        mesh_size_scale=float(args.mesh_size_scale),
        cad_reference_size_scale=float(args.cad_reference_size_scale),
        limit_solver_mesh_by_thickness=bool(args.limit_solver_mesh_by_thickness),
        mesh_preset=str(args.mesh_preset),
        solver_mesh_backend=str(args.solver_mesh_backend),
        ansys_step_strategy=str(args.ansys_step_strategy),
        substrate_layers=None if args.substrate_layers is None else int(args.substrate_layers),
        piezo_layers=None if args.piezo_layers is None else int(args.piezo_layers),
        solver_num_modes=int(args.solver_num_modes),
        solver_search_points=int(args.solver_search_points),
        solver_element_order=int(args.solver_element_order),
        solver_eigensolver_backend=str(args.solver_eigensolver_backend),
        allow_eigensolver_fallback=bool(args.allow_eigensolver_fallback),
        strict_parity=bool(args.strict_parity),
        solver_max_q2_vector_dofs=args.solver_max_q2_vector_dofs,
        solver_oom_fallback_element_order=(
            None if int(args.solver_oom_fallback_element_order) <= 0 else int(args.solver_oom_fallback_element_order)
        ),
        solver_store_mode_shapes=bool(args.solver_store_mode_shapes),
        skip_existing_solver_outputs=not bool(args.no_skip_existing_solver_outputs),
        cell_size_x_m=args.cell_size_x_m,
        cell_size_y_m=args.cell_size_y_m,
        tile_count_x=args.tile_count_x,
        tile_count_y=args.tile_count_y,
        substrate_rho=None if args.substrate_rho is None else float(args.substrate_rho),
        piezo_rho=None if args.piezo_rho is None else float(args.piezo_rho),
        exact_cad=not bool(args.repair_cad),
        repair_cad=bool(args.repair_cad),
        repair_bridge_width_m=args.repair_bridge_width_m,
        create_reports=not bool(args.no_reports),
        build_integrated_dataset=not bool(args.no_integrated_dataset),
        audit_ansys_modal_hz=args.audit_ansys_modal_hz,
        audit_ansys_frf_peak_hz=args.audit_ansys_frf_peak_hz,
        audit_ansys_voltage_v=args.audit_ansys_voltage_v,
        audit_ansys_voltage_form=str(args.audit_ansys_voltage_form),
        audit_sample_ids=_parse_audit_sample_ids(args.audit_sample_id),
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
