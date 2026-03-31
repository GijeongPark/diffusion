from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peh_inverse_design.mesh_tags import FACET_TOP_ELECTRODE_TAG
    from peh_inverse_design.problem_spec import build_runtime_defaults, load_problem_spec
else:
    from .mesh_tags import FACET_TOP_ELECTRODE_TAG
    from .problem_spec import build_runtime_defaults, load_problem_spec


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_problem_spec_paths(run_dir: Path) -> list[Path]:
    return [
        run_dir / "data" / "problem_spec_used.yaml",
        run_dir / "meshes" / "volumes" / "problem_spec_used.yaml",
    ]


def _load_run_problem_spec(run_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    for path in _candidate_problem_spec_paths(run_dir):
        if path.exists():
            return load_problem_spec(path), path
    return None, None


def _load_top_surface(mesh_path: Path, modal_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(mesh_path) as mesh, np.load(modal_path) as modal:
        points = np.asarray(mesh["points"], dtype=np.float64)
        triangles = np.asarray(mesh["triangle_cells"], dtype=np.int64)
        triangle_tags = np.asarray(mesh["triangle_tags"], dtype=np.int32)
        top_triangles = triangles[triangle_tags == FACET_TOP_ELECTRODE_TAG]
        if top_triangles.shape[0] == 0:
            top_triangles = triangles
        strain = np.asarray(modal["top_surface_strain_eqv"], dtype=np.float64).reshape(-1)
    if strain.shape[0] != top_triangles.shape[0]:
        raise ValueError(
            f"Top-surface strain length mismatch for {modal_path}: "
            f"{strain.shape[0]} values for {top_triangles.shape[0]} top triangles."
        )
    centroids = np.mean(points[top_triangles], axis=1)
    return points, top_triangles, centroids, strain


def _export_top_surface_data(
    *,
    sample_id: int,
    output_dir: Path,
    points: np.ndarray,
    top_triangles: np.ndarray,
    centroids: np.ndarray,
    strain: np.ndarray,
    field_frequency_hz: float,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"sample_{sample_id:04d}_top_surface_strain.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "triangle_index",
                "centroid_x_m",
                "centroid_y_m",
                "centroid_z_m",
                "strain_eqv",
            ],
        )
        writer.writeheader()
        for idx, (centroid, value) in enumerate(zip(centroids, strain, strict=True)):
            writer.writerow(
                {
                    "triangle_index": int(idx),
                    "centroid_x_m": f"{float(centroid[0]):.12g}",
                    "centroid_y_m": f"{float(centroid[1]):.12g}",
                    "centroid_z_m": f"{float(centroid[2]):.12g}",
                    "strain_eqv": f"{float(value):.12g}",
                }
            )

    mesh_npz_path = output_dir / f"sample_{sample_id:04d}_top_surface_strain_mesh.npz"
    np.savez_compressed(
        mesh_npz_path,
        sample_id=np.asarray(int(sample_id), dtype=np.int32),
        field_frequency_hz=np.asarray(float(field_frequency_hz), dtype=np.float64),
        points=np.asarray(points, dtype=np.float64),
        top_surface_triangles=np.asarray(top_triangles, dtype=np.int64),
        top_surface_triangle_centroids=np.asarray(centroids, dtype=np.float64),
        top_surface_strain_eqv=np.asarray(strain, dtype=np.float64),
    )
    return csv_path, mesh_npz_path


def audit_run_sample(
    run_dir: str | Path,
    sample_id: int,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    sample_id = int(sample_id)
    mesh_dir = run_dir / "meshes" / "volumes"
    response_path = run_dir / "data" / "fem_responses" / f"sample_{sample_id:04d}_response.npz"
    modal_path = run_dir / "data" / "modal_data" / f"sample_{sample_id:04d}_modal.npz"
    mesh_path = mesh_dir / f"plate3d_{sample_id:04d}_fenicsx.npz"
    handoff_path = mesh_dir / f"plate3d_{sample_id:04d}_ansys_workbench.json"
    face_groups_path = mesh_dir / f"plate3d_{sample_id:04d}_ansys_face_groups.json"

    for path in [response_path, modal_path, mesh_path, handoff_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required audit input is missing: {path}")

    with np.load(response_path) as response, np.load(modal_path) as modal:
        freq_hz = np.asarray(response["freq_hz"], dtype=np.float64).reshape(-1)
        voltage_mag = np.asarray(response["voltage_mag"], dtype=np.float64).reshape(-1)
        f_peak_hz = float(np.asarray(response["f_peak_hz"], dtype=np.float64))
        peak_voltage = float(np.max(voltage_mag))
        argmax_freq_hz = float(freq_hz[int(np.argmax(voltage_mag))])
        eigenfreq_hz = np.asarray(modal["eigenfreq_hz"], dtype=np.float64).reshape(-1)
        field_frequency_hz = float(np.asarray(modal["field_frequency_hz"], dtype=np.float64))

    handoff = _load_json(handoff_path)
    face_groups = _load_json(face_groups_path) if face_groups_path.exists() else None
    problem_spec, problem_spec_path = _load_run_problem_spec(run_dir)
    runtime_defaults = build_runtime_defaults(problem_spec) if problem_spec is not None else {}
    points, top_triangles, centroids, strain = _load_top_surface(mesh_path, modal_path)

    plate_size = np.asarray(handoff["geometry"]["plate_size_m"], dtype=np.float64).reshape(2)
    max_idx = int(np.argmax(strain))
    top_1pct_threshold = float(np.percentile(strain, 99.0))
    top_1pct_mask = strain >= top_1pct_threshold
    root_mask = centroids[:, 0] <= 0.1 * float(plate_size[0])
    tip_mask = centroids[:, 0] >= 0.9 * float(plate_size[0])
    warnings: list[str] = []
    if top_1pct_mask.any():
        top_1pct_x_mean = float(np.mean(centroids[top_1pct_mask, 0]))
        if top_1pct_x_mean > 0.6 * float(plate_size[0]):
            warnings.append(
                "Top 1% of the stored piezo top-surface strain is biased toward the free-edge side; "
                "verify this against ANSYS on the piezo top face at the same field frequency."
            )
    else:
        top_1pct_x_mean = float("nan")

    if abs(argmax_freq_hz - f_peak_hz) > 1.0e-9:
        warnings.append("The saved FRF peak frequency does not match the stored f_peak_hz exactly.")

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "sample_id": int(sample_id),
        "problem_spec_path": "" if problem_spec_path is None else str(problem_spec_path),
        "geometry": {
            "plate_size_m": [float(plate_size[0]), float(plate_size[1])],
            "substrate_thickness_m": float(handoff["geometry"]["substrate_thickness_m"]),
            "piezo_thickness_m": float(handoff["geometry"]["piezo_thickness_m"]),
            "total_thickness_m": float(handoff["geometry"]["total_thickness_m"]),
        },
        "effective_problem_defaults": runtime_defaults,
        "frequency_comparison": {
            "ansys_modal_target_hz": float(eigenfreq_hz[0]) if eigenfreq_hz.size > 0 else float("nan"),
            "f_peak_hz": float(f_peak_hz),
            "field_frequency_hz": float(field_frequency_hz),
            "saved_voltage_peak_freq_hz": float(argmax_freq_hz),
        },
        "voltage_comparison": {
            "peak_voltage_v": float(peak_voltage),
            "external_load_resistance_ohm": float(runtime_defaults.get("resistance_ohm", np.nan)),
        },
        "top_surface_strain": {
            "triangle_count": int(strain.shape[0]),
            "max_strain_eqv": float(strain[max_idx]),
            "max_strain_centroid_m": [float(value) for value in centroids[max_idx]],
            "top_1pct_x_mean_m": float(top_1pct_x_mean),
            "root_mean_strain_eqv": float(np.mean(strain[root_mask])) if root_mask.any() else float("nan"),
            "tip_mean_strain_eqv": float(np.mean(strain[tip_mask])) if tip_mask.any() else float("nan"),
            "root_max_strain_eqv": float(np.max(strain[root_mask])) if root_mask.any() else float("nan"),
            "tip_max_strain_eqv": float(np.max(strain[tip_mask])) if tip_mask.any() else float("nan"),
        },
        "ansys_face_groups": {
            "path": str(face_groups_path) if face_groups_path.exists() else "",
            "piezo_bottom_expected_region_count": (
                int(face_groups["named_selection_recipes"]["piezo_bottom_electrode"]["expected_region_count"])
                if face_groups is not None
                else None
            ),
            "selection_strategy": "" if face_groups is None else str(face_groups.get("selection_strategy", "")),
        },
        "warnings": warnings,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        csv_path, mesh_npz_path = _export_top_surface_data(
            sample_id=sample_id,
            output_dir=output_dir,
            points=points,
            top_triangles=top_triangles,
            centroids=centroids,
            strain=strain,
            field_frequency_hz=field_frequency_hz,
        )
        summary["exports"] = {
            "top_surface_csv_path": str(csv_path),
            "top_surface_mesh_npz_path": str(mesh_npz_path),
        }
        summary_path = output_dir / f"sample_{sample_id:04d}_audit_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False), encoding="utf-8")
        summary["exports"]["audit_summary_path"] = str(summary_path)

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"run_dir: {summary['run_dir']}")
    print(f"sample_id: {summary['sample_id']}")
    print(
        "geometry: "
        f"plate={tuple(summary['geometry']['plate_size_m'])} m, "
        f"substrate_thickness={summary['geometry']['substrate_thickness_m']:.6g} m, "
        f"piezo_thickness={summary['geometry']['piezo_thickness_m']:.6g} m"
    )
    print(
        "frequency_comparison: "
        f"eigenfreq_hz[0]={summary['frequency_comparison']['ansys_modal_target_hz']:.12g}, "
        f"f_peak_hz={summary['frequency_comparison']['f_peak_hz']:.12g}, "
        f"field_frequency_hz={summary['frequency_comparison']['field_frequency_hz']:.12g}"
    )
    print(
        "voltage_comparison: "
        f"peak_voltage_v={summary['voltage_comparison']['peak_voltage_v']:.12g}, "
        f"load_ohm={summary['voltage_comparison']['external_load_resistance_ohm']:.12g}"
    )
    print(
        "top_surface_strain: "
        f"max={summary['top_surface_strain']['max_strain_eqv']:.12g} "
        f"at {tuple(summary['top_surface_strain']['max_strain_centroid_m'])}, "
        f"root_mean={summary['top_surface_strain']['root_mean_strain_eqv']:.12g}, "
        f"tip_mean={summary['top_surface_strain']['tip_mean_strain_eqv']:.12g}"
    )
    print(
        "ansys_face_groups: "
        f"path={summary['ansys_face_groups']['path']}, "
        f"piezo_bottom_expected_region_count={summary['ansys_face_groups']['piezo_bottom_expected_region_count']}"
    )
    for warning in summary.get("warnings", []):
        print(f"warning: {warning}")
    exports = summary.get("exports", {})
    for key, value in exports.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one run/sample for strict ANSYS-vs-FEniCS comparison. "
            "This reports the correct modal vs FRF frequencies and can export the piezo top-surface strain field."
        ),
    )
    parser.add_argument("--run-dir", required=True, help="Run directory such as runs/0329TEST.")
    parser.add_argument("--sample-id", type=int, required=True, help="Sample id to audit.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory for exported top-surface strain CSV/NPZ and the JSON summary.",
    )
    args = parser.parse_args()

    summary = audit_run_sample(
        run_dir=args.run_dir,
        sample_id=int(args.sample_id),
        output_dir=args.output_dir or None,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
