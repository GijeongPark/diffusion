from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .subset_unit_cell_dataset import subset_unit_cell_dataset


@dataclass(frozen=True)
class PipelineConfig:
    source_unit_cell_npz: str | Path
    run_name: str = ""
    output_root: str | Path = "runs"
    limit: int | None = None
    sample_ids: str = ""
    materialize_input_dataset: bool = True
    docker_image: str = "dolfinx/dolfinx:stable"
    substrate_thickness_m: float = 1.0e-3
    piezo_thickness_m: float = 2.667e-4
    mesh_size_scale: float = 0.06
    substrate_rho: float = 7930.0
    piezo_rho: float = 7800.0
    create_reports: bool = True
    build_integrated_dataset: bool = True


@dataclass(frozen=True)
class PipelineArtifacts:
    project_root: Path
    run_root: Path
    unit_cell_npz: Path
    mesh_dir: Path
    response_dir: Path
    modal_dir: Path
    response_dataset_path: Path
    integrated_dataset_path: Path
    integrated_index_csv_path: Path
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
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _run_command(args: list[str | Path], cwd: Path, env: dict[str, str] | None = None) -> None:
    rendered = " ".join(shlex.quote(str(arg)) for arg in args)
    print(f"$ {rendered}")
    subprocess.run([str(arg) for arg in args], cwd=str(cwd), env=env, check=True)


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


def _materialize_unit_cell_dataset(config: PipelineConfig, run_root: Path, project_root: Path) -> tuple[Path, int | None]:
    source_path = _resolve_path(config.source_unit_cell_npz, project_root)
    if not source_path.exists():
        raise FileNotFoundError(f"Unit-cell dataset not found: {source_path}")

    needs_subset = config.limit is not None or config.sample_ids.strip() != ""
    if config.materialize_input_dataset or needs_subset:
        target = run_root / "data" / "unit_cell_dataset.npz"
        subset_unit_cell_dataset(
            input_path=source_path,
            output_path=target,
            limit=config.limit,
            sample_ids=config.sample_ids,
        )
        return target, None
    return source_path, config.limit


def run_pipeline(config: PipelineConfig) -> PipelineArtifacts:
    project_root = _project_root()
    project_python = _project_python(project_root)
    source_path = _resolve_path(config.source_unit_cell_npz, project_root)
    run_name = _infer_run_name(config, source_path)
    run_root = _resolve_path(Path(config.output_root) / run_name, project_root)
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Notebook/kernel python : {sys.executable}")
    print(f"Pipeline step python   : {project_python}")

    unit_cell_npz, downstream_limit = _materialize_unit_cell_dataset(config, run_root, project_root)
    mesh_dir = run_root / "meshes" / "volumes"
    response_dir = run_root / "data" / "fem_responses"
    modal_dir = run_root / "data" / "modal_data"
    response_dataset_path = run_root / "data" / "response_dataset.npz"
    integrated_dataset_path = run_root / "data" / "integrated_dataset.npz"
    integrated_index_csv_path = integrated_dataset_path.with_suffix(".csv")
    report_dir = run_root / "reports"
    gallery_path = report_dir / "gallery.png"

    _print_step("Step 1/5: Build 3D volume meshes")
    mesh_cmd: list[str | Path] = [
        project_python,
        "-m",
        "peh_inverse_design.build_volume_meshes",
        "--unit-cell-npz",
        unit_cell_npz,
        "--mesh-dir",
        mesh_dir,
        "--substrate-thickness",
        str(config.substrate_thickness_m),
        "--piezo-thickness",
        str(config.piezo_thickness_m),
        "--mesh-size-scale",
        str(config.mesh_size_scale),
    ]
    if downstream_limit is not None:
        mesh_cmd.extend(["--limit", str(int(downstream_limit))])
    _run_command(mesh_cmd, cwd=project_root)

    _print_step("Step 2/5: Ensure Docker image")
    _ensure_docker_image(config.docker_image, cwd=project_root)

    _print_step("Step 3/5: Run FEniCSx solver")
    response_dir.mkdir(parents=True, exist_ok=True)
    modal_dir.mkdir(parents=True, exist_ok=True)
    mesh_files = sorted(mesh_dir.glob("plate3d_*_fenicsx.npz"))
    if not mesh_files:
        raise FileNotFoundError(f"No FEniCSx mesh files found in {mesh_dir}.")
    for idx, mesh_file in enumerate(mesh_files, start=1):
        print(f"[{idx}/{len(mesh_files)}] {mesh_file.name}")
        solver_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{project_root}:/workspace",
            "-w",
            "/workspace",
            config.docker_image,
            "python3",
            "-m",
            "peh_inverse_design.fenicsx_modal_solver",
            "--mesh",
            str(Path("/workspace") / mesh_file.relative_to(project_root)),
            "--response-dir",
            str(Path("/workspace") / response_dir.relative_to(project_root)),
            "--modes-dir",
            str(Path("/workspace") / modal_dir.relative_to(project_root)),
            "--substrate-rho",
            str(config.substrate_rho),
            "--piezo-rho",
            str(config.piezo_rho),
        ]
        _run_command(solver_cmd, cwd=project_root)

    _print_step("Step 4/5: Aggregate datasets")
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

    _print_step("Step 5/5: Create reports")
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
        unit_cell_npz=unit_cell_npz,
        mesh_dir=mesh_dir,
        response_dir=response_dir,
        modal_dir=modal_dir,
        response_dataset_path=response_dataset_path,
        integrated_dataset_path=integrated_dataset_path,
        integrated_index_csv_path=integrated_index_csv_path,
        report_dir=report_dir,
        gallery_path=gallery_path,
    )
    print()
    print("Pipeline complete.")
    for key, value in artifacts.as_dict().items():
        print(f"{key}: {value}")
    return artifacts
