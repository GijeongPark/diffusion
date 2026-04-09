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
    from peh_inverse_design.mesh_tags import FACET_TOP_ELECTRODE_TAG, VOLUME_PIEZO_TAG, VOLUME_SUBSTRATE_TAG
    from peh_inverse_design.modal_surface_fields import available_surface_strain_fields, preferred_surface_strain_field
    from peh_inverse_design.problem_spec import build_runtime_defaults, load_problem_spec
else:
    from .mesh_tags import FACET_TOP_ELECTRODE_TAG, VOLUME_PIEZO_TAG, VOLUME_SUBSTRATE_TAG
    from .modal_surface_fields import available_surface_strain_fields, preferred_surface_strain_field
    from .problem_spec import build_runtime_defaults, load_problem_spec


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percent_error_percent(value: float, reference: float) -> float:
    value = float(value)
    reference = float(reference)
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) <= 0.0:
        return float("nan")
    return 100.0 * (value - reference) / reference


def _as_reference_or_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _normalize_voltage_form(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"unknown", "peak", "rms"}:
        raise ValueError("ansys_voltage_form must be one of: unknown, peak, rms.")
    return normalized


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


def _load_top_surface(mesh_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(mesh_path) as mesh:
        points = np.asarray(mesh["points"], dtype=np.float64)
        triangles = np.asarray(mesh["triangle_cells"], dtype=np.int64)
        triangle_tags = np.asarray(mesh["triangle_tags"], dtype=np.int32)
        top_triangles = triangles[triangle_tags == FACET_TOP_ELECTRODE_TAG]
        if top_triangles.shape[0] == 0:
            top_triangles = triangles
    centroids = np.mean(points[top_triangles], axis=1)
    return points, top_triangles, centroids


def _surface_strain_summary(
    *,
    strain: np.ndarray,
    centroids: np.ndarray,
    plate_size_x_m: float,
    label: str,
    field_name: str,
    frequency_hz: float,
) -> tuple[dict[str, Any], list[str]]:
    max_idx = int(np.argmax(strain))
    top_1pct_threshold = float(np.percentile(strain, 99.0))
    top_1pct_mask = strain >= top_1pct_threshold
    root_mask = centroids[:, 0] <= 0.1 * float(plate_size_x_m)
    tip_mask = centroids[:, 0] >= 0.9 * float(plate_size_x_m)
    warnings: list[str] = []
    if top_1pct_mask.any():
        top_1pct_x_mean = float(np.mean(centroids[top_1pct_mask, 0]))
        if top_1pct_x_mean > 0.6 * float(plate_size_x_m):
            warnings.append(
                f"{label} is biased toward the free-edge side; verify this against ANSYS on the matching piezo face."
            )
    else:
        top_1pct_x_mean = float("nan")

    summary = {
        "field_name": field_name,
        "label": label,
        "frequency_hz": float(frequency_hz),
        "triangle_count": int(strain.shape[0]),
        "max_strain_eqv": float(strain[max_idx]),
        "max_strain_centroid_m": [float(value) for value in centroids[max_idx]],
        "top_1pct_x_mean_m": float(top_1pct_x_mean),
        "root_mean_strain_eqv": float(np.mean(strain[root_mask])) if root_mask.any() else float("nan"),
        "tip_mean_strain_eqv": float(np.mean(strain[tip_mask])) if tip_mask.any() else float("nan"),
        "root_max_strain_eqv": float(np.max(strain[root_mask])) if root_mask.any() else float("nan"),
        "tip_max_strain_eqv": float(np.max(strain[tip_mask])) if tip_mask.any() else float("nan"),
    }
    return summary, warnings


def _load_surface_fields(
    mesh_path: Path,
    modal_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    points, top_triangles, centroids = _load_top_surface(mesh_path)
    with np.load(modal_path) as modal:
        available_fields = [
            {
                "key": field.key,
                "kind": field.kind,
                "label": field.label,
                "frequency_hz": field.frequency_hz,
                "strain": np.asarray(field.strain, dtype=np.float64),
            }
            for field in available_surface_strain_fields(modal, top_triangles.shape[0])
        ]
        preferred = preferred_surface_strain_field(modal, top_triangles.shape[0])
    if preferred is None:
        raise ValueError(f"No compatible top-surface strain field was found in {modal_path}.")
    preferred_summary = {
        "key": preferred.key,
        "kind": preferred.kind,
        "label": preferred.label,
        "frequency_hz": preferred.frequency_hz,
        "strain": np.asarray(preferred.strain, dtype=np.float64),
    }
    return points, top_triangles, centroids, preferred_summary, available_fields


def _mesh_material_summary(mesh_path: Path) -> dict[str, Any]:
    with np.load(mesh_path) as mesh:
        points = np.asarray(mesh["points"], dtype=np.float64)
        tetra_cells = np.asarray(mesh["tetra_cells"], dtype=np.int64)
        tetra_tags = np.asarray(mesh["tetra_tags"], dtype=np.int32)
    tet_points = points[tetra_cells]
    edge_a = tet_points[:, 1] - tet_points[:, 0]
    edge_b = tet_points[:, 2] - tet_points[:, 0]
    edge_c = tet_points[:, 3] - tet_points[:, 0]
    volumes = np.abs(np.einsum("ij,ij->i", edge_a, np.cross(edge_b, edge_c))) / 6.0
    substrate_mask = tetra_tags == VOLUME_SUBSTRATE_TAG
    piezo_mask = tetra_tags == VOLUME_PIEZO_TAG
    return {
        "substrate_volume_m3": float(np.sum(volumes[substrate_mask])),
        "piezo_volume_m3": float(np.sum(volumes[piezo_mask])),
        "substrate_cell_count": int(np.count_nonzero(substrate_mask)),
        "piezo_cell_count": int(np.count_nonzero(piezo_mask)),
    }


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
    ansys_modal_hz: float | None = None,
    ansys_frf_peak_hz: float | None = None,
    ansys_voltage_v: float | None = None,
    ansys_voltage_form: str = "unknown",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    sample_id = int(sample_id)
    ansys_voltage_form = _normalize_voltage_form(ansys_voltage_form)
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
        peak_voltage_peak_v = float(np.max(voltage_mag))
        peak_voltage_rms_v = float(peak_voltage_peak_v / np.sqrt(2.0))
        argmax_freq_hz = float(freq_hz[int(np.argmax(voltage_mag))])
        eigenfreq_hz = np.asarray(modal["eigenfreq_hz"], dtype=np.float64).reshape(-1)
        mode1_frequency_hz = (
            float(np.asarray(modal["mode1_frequency_hz"], dtype=np.float64))
            if "mode1_frequency_hz" in modal
            else (float(eigenfreq_hz[0]) if eigenfreq_hz.size > 0 else float("nan"))
        )
        harmonic_field_frequency_hz = (
            float(np.asarray(modal["harmonic_field_frequency_hz"], dtype=np.float64))
            if "harmonic_field_frequency_hz" in modal
            else (float(np.asarray(modal["field_frequency_hz"], dtype=np.float64)) if "field_frequency_hz" in modal else float("nan"))
        )
        modal_theta = np.asarray(modal["modal_theta"], dtype=np.float64).reshape(-1) if "modal_theta" in modal else np.zeros(0, dtype=np.float64)
        modal_force = np.asarray(modal["modal_force"], dtype=np.float64).reshape(-1) if "modal_force" in modal else np.zeros(0, dtype=np.float64)
        modal_mass = np.asarray(modal["modal_mass"], dtype=np.float64).reshape(-1) if "modal_mass" in modal else np.zeros(0, dtype=np.float64)
        capacitance_f = (
            float(np.asarray(modal["capacitance_f"], dtype=np.float64).reshape(-1)[0])
            if "capacitance_f" in modal
            else float("nan")
        )

    handoff = _load_json(handoff_path)
    face_groups = _load_json(face_groups_path) if face_groups_path.exists() else None
    problem_spec, problem_spec_path = _load_run_problem_spec(run_dir)
    runtime_defaults = build_runtime_defaults(problem_spec) if problem_spec is not None else {}
    points, top_triangles, centroids, preferred_surface_field, available_surface_fields = _load_surface_fields(mesh_path, modal_path)
    mesh_materials = _mesh_material_summary(mesh_path)

    plate_size = np.asarray(handoff["geometry"]["plate_size_m"], dtype=np.float64).reshape(2)
    warnings: list[str] = []
    if abs(argmax_freq_hz - f_peak_hz) > 1.0e-9:
        warnings.append("The saved FRF peak frequency does not match the stored f_peak_hz exactly.")
    ansys_modal_reference_hz = None if ansys_modal_hz is None else float(ansys_modal_hz)
    ansys_frf_peak_reference_hz = None if ansys_frf_peak_hz is None else float(ansys_frf_peak_hz)
    supplied_frequency_references = [
        reference
        for reference in [ansys_modal_reference_hz, ansys_frf_peak_reference_hz]
        if reference is not None and np.isfinite(reference)
    ]
    ambiguous_frequency_reference_hz = (
        float(supplied_frequency_references[0]) if len(supplied_frequency_references) == 1 else float("nan")
    )
    ambiguous_frequency_comparison = len(supplied_frequency_references) == 1
    if ambiguous_frequency_comparison:
        warnings.append(
            "Only one ANSYS frequency reference was provided; the comparison is ambiguous, so both "
            "mode1_frequency_hz and f_peak_hz are reported against that same ANSYS value."
        )

    mode1_vs_ansys_modal_error_percent = (
        _percent_error_percent(mode1_frequency_hz, ansys_modal_reference_hz)
        if ansys_modal_reference_hz is not None
        else float("nan")
    )
    f_peak_vs_ansys_frf_peak_error_percent = (
        _percent_error_percent(f_peak_hz, ansys_frf_peak_reference_hz)
        if ansys_frf_peak_reference_hz is not None
        else float("nan")
    )
    mode1_vs_ambiguous_reference_error_percent = (
        _percent_error_percent(mode1_frequency_hz, ambiguous_frequency_reference_hz)
        if ambiguous_frequency_comparison
        else float("nan")
    )
    f_peak_vs_ambiguous_reference_error_percent = (
        _percent_error_percent(f_peak_hz, ambiguous_frequency_reference_hz)
        if ambiguous_frequency_comparison
        else float("nan")
    )

    voltage_convention_tolerance_percent = 2.0
    voltage_error_if_ansys_peak = (
        _percent_error_percent(peak_voltage_peak_v, ansys_voltage_v)
        if ansys_voltage_v is not None
        else float("nan")
    )
    voltage_error_if_ansys_rms = (
        _percent_error_percent(peak_voltage_rms_v, ansys_voltage_v)
        if ansys_voltage_v is not None
        else float("nan")
    )
    selected_voltage_error_percent = float("nan")
    selected_in_house_voltage_v = float("nan")
    if ansys_voltage_v is not None:
        if ansys_voltage_form == "peak":
            selected_voltage_error_percent = float(voltage_error_if_ansys_peak)
            selected_in_house_voltage_v = float(peak_voltage_peak_v)
        elif ansys_voltage_form == "rms":
            selected_voltage_error_percent = float(voltage_error_if_ansys_rms)
            selected_in_house_voltage_v = float(peak_voltage_rms_v)
    voltage_convention_mismatch_likely = bool(
        ansys_voltage_v is not None
        and ansys_voltage_form == "unknown"
        and np.isfinite(voltage_error_if_ansys_rms)
        and abs(voltage_error_if_ansys_rms) <= voltage_convention_tolerance_percent
        and (
            not np.isfinite(voltage_error_if_ansys_peak)
            or abs(voltage_error_if_ansys_peak) > voltage_convention_tolerance_percent
        )
    )
    if voltage_convention_mismatch_likely:
        warnings.append(
            "The ANSYS-vs-in-house voltage gap is likely explained by an RMS-vs-peak convention mismatch."
        )

    top_surface_strain, surface_warnings = _surface_strain_summary(
        strain=preferred_surface_field["strain"],
        centroids=centroids,
        plate_size_x_m=float(plate_size[0]),
        label=str(preferred_surface_field["label"]),
        field_name=str(preferred_surface_field["key"]),
        frequency_hz=float(preferred_surface_field["frequency_hz"]),
    )
    warnings.extend(surface_warnings)
    harmonic_surface_strain = None
    for field in available_surface_fields:
        if field["kind"] != "harmonic":
            continue
        if str(field["key"]) == str(preferred_surface_field["key"]):
            break
        harmonic_surface_strain, harmonic_warnings = _surface_strain_summary(
            strain=np.asarray(field["strain"], dtype=np.float64),
            centroids=centroids,
            plate_size_x_m=float(plate_size[0]),
            label=str(field["label"]),
            field_name=str(field["key"]),
            frequency_hz=float(field["frequency_hz"]),
        )
        warnings.extend(harmonic_warnings)
        break

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
            "ansys_modal_target_hz": (
                float(ansys_modal_reference_hz)
                if ansys_modal_reference_hz is not None
                else (float(eigenfreq_hz[0]) if eigenfreq_hz.size > 0 else float("nan"))
            ),
            "mode1_frequency_hz": float(mode1_frequency_hz),
            "f_peak_hz": float(f_peak_hz),
            "harmonic_field_frequency_hz": float(harmonic_field_frequency_hz),
            "field_frequency_hz": float(harmonic_field_frequency_hz),
            "saved_voltage_peak_freq_hz": float(argmax_freq_hz),
            "ansys_modal_reference_hz": _as_reference_or_nan(ansys_modal_reference_hz),
            "ansys_frf_peak_reference_hz": _as_reference_or_nan(ansys_frf_peak_reference_hz),
            "frequency_reference_ambiguous": bool(ambiguous_frequency_comparison),
            "ambiguous_frequency_reference_hz": float(ambiguous_frequency_reference_hz),
            "mode1_vs_ansys_modal_error_percent": float(mode1_vs_ansys_modal_error_percent),
            "f_peak_vs_ansys_frf_peak_error_percent": float(f_peak_vs_ansys_frf_peak_error_percent),
            "mode1_vs_ambiguous_frequency_error_percent": float(mode1_vs_ambiguous_reference_error_percent),
            "f_peak_vs_ambiguous_frequency_error_percent": float(f_peak_vs_ambiguous_reference_error_percent),
        },
        "voltage_comparison": {
            "peak_voltage_v": float(peak_voltage_peak_v),
            "peak_voltage_peak_v": float(peak_voltage_peak_v),
            "peak_voltage_rms_v": float(peak_voltage_rms_v),
            "external_load_resistance_ohm": float(runtime_defaults.get("resistance_ohm", np.nan)),
            "ansys_voltage_reference_v": _as_reference_or_nan(ansys_voltage_v),
            "ansys_voltage_form": str(ansys_voltage_form),
            "selected_in_house_voltage_v": float(selected_in_house_voltage_v),
            "selected_voltage_error_percent": float(selected_voltage_error_percent),
            "error_percent_assuming_ansys_peak": float(voltage_error_if_ansys_peak),
            "error_percent_assuming_ansys_rms": float(voltage_error_if_ansys_rms),
            "rms_equivalent_error_percent": float(voltage_error_if_ansys_rms),
            "voltage_convention_mismatch_likely": bool(voltage_convention_mismatch_likely),
            "voltage_convention_tolerance_percent": float(voltage_convention_tolerance_percent),
        },
        "electromechanical": {
            "modal_theta_mode1": float(modal_theta[0]) if modal_theta.size > 0 else float("nan"),
            "modal_force_mode1": float(modal_force[0]) if modal_force.size > 0 else float("nan"),
            "modal_mass_mode1": float(modal_mass[0]) if modal_mass.size > 0 else float("nan"),
            "capacitance_f": float(capacitance_f),
        },
        "mesh_materials": mesh_materials,
        "top_surface_strain": top_surface_strain,
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
    if harmonic_surface_strain is not None:
        summary["harmonic_top_surface_strain"] = harmonic_surface_strain

    if output_dir is not None:
        output_dir = Path(output_dir)
        csv_path, mesh_npz_path = _export_top_surface_data(
            sample_id=sample_id,
            output_dir=output_dir,
            points=points,
            top_triangles=top_triangles,
            centroids=centroids,
            strain=np.asarray(preferred_surface_field["strain"], dtype=np.float64),
            field_frequency_hz=float(preferred_surface_field["frequency_hz"]),
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
    frequency_comparison = summary["frequency_comparison"]
    voltage_comparison = summary["voltage_comparison"]
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
        f"mode1_frequency_hz={frequency_comparison['mode1_frequency_hz']:.12g}, "
        f"f_peak_hz={frequency_comparison['f_peak_hz']:.12g}, "
        f"harmonic_field_frequency_hz={frequency_comparison['harmonic_field_frequency_hz']:.12g}, "
        f"ansys_modal_reference_hz={frequency_comparison['ansys_modal_reference_hz']:.12g}, "
        f"ansys_frf_peak_reference_hz={frequency_comparison['ansys_frf_peak_reference_hz']:.12g}"
    )
    if bool(frequency_comparison.get("frequency_reference_ambiguous", False)):
        print(
            "frequency_reference_ambiguity: "
            f"ansys_reference_hz={frequency_comparison['ambiguous_frequency_reference_hz']:.12g}, "
            f"mode1_error_percent={frequency_comparison['mode1_vs_ambiguous_frequency_error_percent']:.12g}, "
            f"f_peak_error_percent={frequency_comparison['f_peak_vs_ambiguous_frequency_error_percent']:.12g}"
        )
    else:
        if np.isfinite(float(frequency_comparison.get("mode1_vs_ansys_modal_error_percent", np.nan))):
            print(
                "modal_frequency_error: "
                f"mode1_vs_ansys_modal_error_percent={frequency_comparison['mode1_vs_ansys_modal_error_percent']:.12g}"
            )
        if np.isfinite(float(frequency_comparison.get("f_peak_vs_ansys_frf_peak_error_percent", np.nan))):
            print(
                "frf_frequency_error: "
                f"f_peak_vs_ansys_frf_peak_error_percent={frequency_comparison['f_peak_vs_ansys_frf_peak_error_percent']:.12g}"
            )
    print(
        "voltage_comparison: "
        f"peak_voltage_peak_v={voltage_comparison['peak_voltage_peak_v']:.12g}, "
        f"peak_voltage_rms_v={voltage_comparison['peak_voltage_rms_v']:.12g}, "
        f"load_ohm={voltage_comparison['external_load_resistance_ohm']:.12g}"
    )
    if np.isfinite(float(voltage_comparison.get("ansys_voltage_reference_v", np.nan))):
        print(
            "voltage_reference: "
            f"ansys_voltage_v={voltage_comparison['ansys_voltage_reference_v']:.12g}, "
            f"ansys_voltage_form={voltage_comparison['ansys_voltage_form']}"
        )
        if str(voltage_comparison.get("ansys_voltage_form", "unknown")) == "unknown":
            print(
                "voltage_reference_ambiguity: "
                f"error_percent_assuming_ansys_peak={voltage_comparison['error_percent_assuming_ansys_peak']:.12g}, "
                f"error_percent_assuming_ansys_rms={voltage_comparison['error_percent_assuming_ansys_rms']:.12g}, "
                f"voltage_convention_mismatch_likely={bool(voltage_comparison['voltage_convention_mismatch_likely'])}"
            )
        else:
            print(
                "voltage_error: "
                f"selected_voltage_error_percent={voltage_comparison['selected_voltage_error_percent']:.12g}"
            )
    print(
        "electromechanical: "
        f"modal_theta_mode1={summary['electromechanical']['modal_theta_mode1']:.12g}, "
        f"modal_force_mode1={summary['electromechanical']['modal_force_mode1']:.12g}, "
        f"capacitance_f={summary['electromechanical']['capacitance_f']:.12g}"
    )
    print(
        "mesh_materials: "
        f"substrate_volume_m3={summary['mesh_materials']['substrate_volume_m3']:.12g}, "
        f"piezo_volume_m3={summary['mesh_materials']['piezo_volume_m3']:.12g}, "
        f"substrate_cells={summary['mesh_materials']['substrate_cell_count']}, "
        f"piezo_cells={summary['mesh_materials']['piezo_cell_count']}"
    )
    print(
        "top_surface_strain: "
        f"{summary['top_surface_strain']['label']} "
        f"max={summary['top_surface_strain']['max_strain_eqv']:.12g} "
        f"at {tuple(summary['top_surface_strain']['max_strain_centroid_m'])}, "
        f"root_mean={summary['top_surface_strain']['root_mean_strain_eqv']:.12g}, "
        f"tip_mean={summary['top_surface_strain']['tip_mean_strain_eqv']:.12g}, "
        f"top_1pct_x_mean_m={summary['top_surface_strain']['top_1pct_x_mean_m']:.12g}"
    )
    if "harmonic_top_surface_strain" in summary:
        print(
            "harmonic_top_surface_strain: "
            f"max={summary['harmonic_top_surface_strain']['max_strain_eqv']:.12g}, "
            f"root_mean={summary['harmonic_top_surface_strain']['root_mean_strain_eqv']:.12g}, "
            f"tip_mean={summary['harmonic_top_surface_strain']['tip_mean_strain_eqv']:.12g}"
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
    parser.add_argument("--ansys-modal-hz", type=float, default=None, help="Optional ANSYS modal reference frequency in Hz.")
    parser.add_argument("--ansys-frf-peak-hz", type=float, default=None, help="Optional ANSYS FRF peak reference frequency in Hz.")
    parser.add_argument("--ansys-voltage-v", type=float, default=None, help="Optional ANSYS voltage reference value.")
    parser.add_argument(
        "--ansys-voltage-form",
        default="unknown",
        choices=["unknown", "peak", "rms"],
        help="Interpretation of --ansys-voltage-v. Use 'unknown' to compare against both peak and RMS.",
    )
    args = parser.parse_args()

    summary = audit_run_sample(
        run_dir=args.run_dir,
        sample_id=int(args.sample_id),
        output_dir=args.output_dir or None,
        ansys_modal_hz=args.ansys_modal_hz,
        ansys_frf_peak_hz=args.ansys_frf_peak_hz,
        ansys_voltage_v=args.ansys_voltage_v,
        ansys_voltage_form=args.ansys_voltage_form,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
