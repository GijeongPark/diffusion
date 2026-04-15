from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ProblemSpec = dict[str, Any]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_problem_spec_path(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else _project_root()
    return root / "configs" / "peh_inverse_design_spec.yaml"


def load_problem_spec(
    problem_spec_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> ProblemSpec:
    path = Path(problem_spec_path) if problem_spec_path is not None else default_problem_spec_path(project_root)
    if not path.exists():
        raise FileNotFoundError(f"Problem specification not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Problem specification at {path} must parse to a mapping.")

    spec = copy.deepcopy(raw)
    spec.setdefault("_metadata", {})
    spec["_metadata"]["source_path"] = str(path.resolve())
    return spec


def geometry_defaults_from_problem_spec(problem_spec: Mapping[str, Any]) -> tuple[tuple[float, float], tuple[int, int]]:
    geometry = problem_spec.get("geometry", {})
    if not isinstance(geometry, Mapping):
        raise ValueError("Problem specification key 'geometry' must be a mapping.")

    cell_size = geometry.get("unit_cell_size_m")
    tile_counts = geometry.get("tile_counts")
    if cell_size is None or tile_counts is None:
        raise ValueError("Problem specification must define geometry.unit_cell_size_m and geometry.tile_counts.")
    if len(cell_size) != 2 or len(tile_counts) != 2:
        raise ValueError("geometry.unit_cell_size_m and geometry.tile_counts must both contain two entries.")

    return (
        (float(cell_size[0]), float(cell_size[1])),
        (int(tile_counts[0]), int(tile_counts[1])),
    )


def build_mechanical_config_kwargs(problem_spec: Mapping[str, Any]) -> dict[str, float]:
    materials = problem_spec.get("materials", {})
    substrate = materials.get("substrate", {}) if isinstance(materials, Mapping) else {}
    mechanics = problem_spec.get("mechanics", {})
    damping = mechanics.get("damping", {}) if isinstance(mechanics, Mapping) else {}
    excitation = mechanics.get("base_excitation", {}) if isinstance(mechanics, Mapping) else {}
    piezoelectric = materials.get("piezoelectric", {}) if isinstance(materials, Mapping) else {}
    geometry = problem_spec.get("geometry", {})

    piezo_density = piezoelectric.get("density_kg_per_m3", 7500.0)
    if isinstance(geometry, Mapping) and "piezo_density_kg_per_m3" in geometry:
        piezo_density = geometry["piezo_density_kg_per_m3"]

    return {
        "substrate_E_pa": float(substrate.get("youngs_modulus_pa", 1.9305e11)),
        "substrate_nu": float(substrate.get("poisson_ratio", 0.30)),
        "substrate_rho": float(substrate.get("density_kg_per_m3", 7930.0)),
        "piezo_rho": float(piezo_density),
        "damping_ratio": float(damping.get("modal_damping_ratio", 0.025)),
        "base_acceleration_m_per_s2": float(excitation.get("amplitude_m_per_s2", 2.5)),
    }


def build_piezo_config_kwargs(problem_spec: Mapping[str, Any]) -> dict[str, Any]:
    geometry = problem_spec.get("geometry", {})
    materials = problem_spec.get("materials", {})
    electrical = problem_spec.get("electrical", {})
    piezoelectric = materials.get("piezoelectric", {}) if isinstance(materials, Mapping) else {}
    full_3d = piezoelectric.get("full_3d_constants", {}) if isinstance(piezoelectric, Mapping) else {}
    reduced_plate = piezoelectric.get("reduced_plate_constants", {}) if isinstance(piezoelectric, Mapping) else {}

    eps_matrix = full_3d.get("permittivity_epsS_f_per_m")
    if eps_matrix is None or len(eps_matrix) < 3 or len(eps_matrix[2]) < 3:
        raise ValueError("Problem specification must define materials.piezoelectric.full_3d_constants.permittivity_epsS_f_per_m.")
    e_matrix = full_3d.get("piezoelectric_e_c_per_m2")
    if e_matrix is None:
        raise ValueError("Problem specification must define materials.piezoelectric.full_3d_constants.piezoelectric_e_c_per_m2.")
    stiffness = full_3d.get("stiffness_cE_pa")
    if stiffness is None:
        raise ValueError("Problem specification must define materials.piezoelectric.full_3d_constants.stiffness_cE_pa.")

    full_3d_eps33s_f_per_m = float(eps_matrix[2][2])
    capacitance_eps33s_f_per_m = float(
        reduced_plate.get("eps33s_f_per_m", full_3d_eps33s_f_per_m)
    )

    return {
        "thickness_m": float(geometry.get("piezo_patch_thickness_m", 1.0e-4)),
        "resistance_ohm": float(electrical.get("external_load_resistance_ohm", 1.0e4)),
        "eps33s_f_per_m": full_3d_eps33s_f_per_m,
        "capacitance_eps33s_f_per_m": capacitance_eps33s_f_per_m,
        "e_matrix_c_per_m2": tuple(tuple(float(value) for value in row) for row in e_matrix),
        "stiffness_cE_pa": tuple(tuple(float(value) for value in row) for row in stiffness),
    }


def build_runtime_defaults(problem_spec: Mapping[str, Any]) -> dict[str, float | str]:
    geometry = problem_spec.get("geometry", {})
    materials = problem_spec.get("materials", {})
    substrate = materials.get("substrate", {}) if isinstance(materials, Mapping) else {}
    mechanics = build_mechanical_config_kwargs(problem_spec)
    electrical = problem_spec.get("electrical", {})
    house_voltage_amplitude_convention = str(
        electrical.get("house_voltage_amplitude_convention", "peak")
    ).strip().lower()
    if house_voltage_amplitude_convention != "peak":
        raise ValueError(
            "electrical.house_voltage_amplitude_convention must be 'peak'. RMS handling was removed."
        )

    return {
        "substrate_thickness_m": float(geometry.get("substrate_thickness_m", 1.0e-3)),
        "piezo_thickness_m": float(geometry.get("piezo_patch_thickness_m", 1.0e-4)),
        "substrate_rho": float(substrate.get("density_kg_per_m3", mechanics["substrate_rho"])),
        "piezo_rho": float(mechanics["piezo_rho"]),
        "resistance_ohm": float(electrical.get("external_load_resistance_ohm", 1.0e4)),
        "house_voltage_amplitude_convention": house_voltage_amplitude_convention,
    }


def summarize_problem_spec(problem_spec: Mapping[str, Any]) -> ProblemSpec:
    keys = ["project", "geometry", "materials", "mechanics", "electrical", "frf"]
    summary: ProblemSpec = {}
    for key in keys:
        if key in problem_spec:
            summary[key] = copy.deepcopy(problem_spec[key])
    metadata = problem_spec.get("_metadata")
    if isinstance(metadata, Mapping):
        summary["_metadata"] = copy.deepcopy(dict(metadata))
    return summary


def write_problem_spec_snapshot(problem_spec: Mapping[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(summarize_problem_spec(problem_spec), sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def write_ansys_workbench_handoff(
    sample_id: int,
    output_path: str | Path,
    step_path: str | Path | None,
    msh_path: str | Path | None,
    cad_report_path: str | Path | None,
    solver_mesh_path: str | Path | None,
    geometry_config: Any,
    volume_config: Any,
    problem_spec: Mapping[str, Any],
    inspection_single_face_step_path: str | Path | None = None,
    face_selection_manifest_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    substrate_thickness_m = float(getattr(volume_config, "substrate_thickness_m"))
    piezo_thickness_m = float(getattr(volume_config, "piezo_thickness_m"))
    total_thickness_m = substrate_thickness_m + piezo_thickness_m
    plate_size_m = getattr(geometry_config, "plate_size_m")
    cell_size_m = getattr(geometry_config, "cell_size_m")
    tile_counts = getattr(geometry_config, "tile_counts")

    ansys_step_strategy = str(getattr(volume_config, "ansys_step_strategy", "single_face_assembly")).strip().lower()
    files_payload = {
        "step_path": "" if step_path is None else str(Path(step_path)),
        "inspection_single_face_step_path": ""
        if inspection_single_face_step_path is None
        else str(Path(inspection_single_face_step_path)),
        "face_selection_manifest_path": ""
        if face_selection_manifest_path is None
        else str(Path(face_selection_manifest_path)),
        "gmsh_volume_mesh_path": "" if msh_path is None else str(Path(msh_path)),
        "python_solver_mesh_path": "" if solver_mesh_path is None else str(Path(solver_mesh_path)),
        "cad_report_path": "" if cad_report_path is None else str(Path(cad_report_path)),
    }
    geometry_import_payload = {
        "import_via": "Geometry cell",
        "geometry_format": "STEP",
        "recommended_step_variant": "step_path",
        "expected_body_type": "solid",
        "expected_solid_body_count": 2,
        "allow_surface_bodies": False,
        "allow_line_bodies": False,
        "body_roles": [
            {
                "name": "substrate",
                "kind": "solid",
                "z_range_m": [0.0, substrate_thickness_m],
            },
            {
                "name": "piezo",
                "kind": "solid",
                "z_range_m": [substrate_thickness_m, total_thickness_m],
            },
        ],
    }
    if ansys_step_strategy == "partitioned_interface":
        geometry_import_payload["recommended_import_sequence"] = [
            "Import files.step_path into the Geometry cell as one combined CAD file.",
            "Use the combined, conformal interface STEP for meshing; this keeps the substrate and piezo interface imprinted into matching CAD regions.",
            "If you need electrode or interface scoping, use files.face_selection_manifest_path as the face-group recipe instead of manual clicking.",
        ]
    else:
        geometry_import_payload["recommended_import_sequence"] = [
            "Import files.step_path into the Geometry cell as one combined CAD file.",
            "Use the single-file, two-body assembly STEP whose piezo bottom stays continuous; this is intended for Workbench cases where the partitioned interface makes the piezo body fail meshing.",
            "If you need electrode or interface scoping, use files.face_selection_manifest_path as the face-group recipe instead of manual clicking.",
        ]

    payload = {
        "sample_id": int(sample_id),
        "geometry": {
            "unit_cell_size_m": [float(cell_size_m[0]), float(cell_size_m[1])],
            "tile_counts": [int(tile_counts[0]), int(tile_counts[1])],
            "plate_size_m": [float(plate_size_m[0]), float(plate_size_m[1])],
            "substrate_thickness_m": substrate_thickness_m,
            "piezo_thickness_m": piezo_thickness_m,
            "total_thickness_m": total_thickness_m,
        },
        "files": files_payload,
        "ansys_workbench_geometry_import": geometry_import_payload,
        "shared_problem_definition": {
            "mechanical_boundary_condition": {
                "type": "cantilever_clamped_edge",
                "location": {"axis": "x", "side": "min", "value_m": 0.0},
            },
            "base_excitation": {
                "type": "harmonic_support_acceleration",
                "direction": "z",
                "location": "clamped_edge",
                "implementation_intent": "same as FEniCS modal-reduction path",
            },
            "electrodes": {
                "top_electrode": {
                    "surface": "piezo_top_face",
                    "z_value_m": total_thickness_m,
                },
                "bottom_electrode": {
                    "surface": "substrate_piezo_interface",
                    "z_value_m": substrate_thickness_m,
                },
            },
            "response": {
                "quantity": "voltage_magnitude",
                "frequency_window": "normalized about fundamental peak",
            },
        },
        "problem_spec": summarize_problem_spec(problem_spec),
        "notes": [],
    }
    payload["notes"] = [
        "Use files.step_path as the recommended one-file Workbench geometry handoff.",
    ]
    if ansys_step_strategy == "partitioned_interface":
        payload["notes"].append(
            "The combined STEP keeps a conformal, meshable substrate/piezo interface. That can fragment the piezo bottom into multiple CAD faces, so use files.face_selection_manifest_path to build electrode/interface groups without manual clicking."
        )
    else:
        payload["notes"].append(
            "The combined STEP keeps the piezo bottom continuous as a one-file, two-body assembly. This is intended for Workbench cases where the conformal partitioned interface causes the piezo body to fail meshing."
        )
        payload["notes"].append(
            "If Workbench tries to merge or imprint the two bodies automatically, keep the imported substrate and piezo as separate solid bodies in the same Geometry cell and scope electrodes/interfaces through files.face_selection_manifest_path."
        )
    if inspection_single_face_step_path is not None:
        payload["notes"].append(
            "files.inspection_single_face_step_path is an optional alternate STEP variant for comparison during Workbench debugging."
        )
    payload["notes"].extend([
        "The Python path uses its own Gmsh-derived solver mesh; ANSYS should import the combined STEP assembly directly and mesh it inside Workbench.",
        "Keep the ANSYS setup synchronized with the FEniCS path through the shared problem specification embedded in this handoff file.",
    ])
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return output_path
