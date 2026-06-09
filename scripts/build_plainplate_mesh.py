"""Build a single void-free plain-plate solver mesh (Task 2, mesh stage).

This bypasses the SDF/GRF metaplate geometry path and builds a full rectangular
two-layer plate (steel substrate tag 11 + PZT layer tag 12) using the additive
``mesh_plain_plate_volume_sample`` helper, which reuses the production
``_build_layered_tet_solver_mesh`` routine. All mesh knobs are kept at the
current production/notebook values so the only difference from a metaplate run is
the void-free in-plane geometry.

Run with the project venv (gmsh + shapely + meshio):

    ./.venv/bin/python scripts/build_plainplate_mesh.py

Writes:  runs/plainplate/meshes/volumes/plate3d_9999_fenicsx.npz

The FEM solve is a separate Docker step (scripts/solve_plainplate_voltage.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peh_inverse_design.core.problem_spec import (
    build_runtime_defaults,
    default_problem_spec_path,
    geometry_defaults_from_problem_spec,
    load_problem_spec,
)
from peh_inverse_design.geometry.geometry_pipeline import GeometryBuildConfig
from peh_inverse_design.meshing.volume_mesh import (
    VolumeMeshConfig,
    mesh_plain_plate_volume_sample,
    volume_mesh_preset_overrides,
)

# Distinctive numeric id so the mesh file is plate3d_9999_fenicsx.npz.
SAMPLE_ID = 9999

# Current production / notebook mesh + layering knobs (only geometry differs).
MESH_PRESET = "default"            # 2 substrate + 1 piezo layers
MESH_SIZE_SCALE = 0.08             # notebook MESH_SIZE_SCALE
CAD_REFERENCE_SIZE_SCALE = 0.01    # notebook CAD_REFERENCE_SIZE_SCALE


def main() -> None:
    spec = load_problem_spec(default_problem_spec_path(PROJECT_ROOT), project_root=PROJECT_ROOT)
    runtime = build_runtime_defaults(spec)
    cell_size_m, tile_counts = geometry_defaults_from_problem_spec(spec)

    geometry_config = GeometryBuildConfig(
        cell_size_m=cell_size_m,
        tile_counts=tile_counts,
        enforce_connected_plate=False,
    )

    preset_overrides = volume_mesh_preset_overrides(MESH_PRESET)
    volume_config = VolumeMeshConfig(
        substrate_thickness_m=float(runtime["substrate_thickness_m"]),
        piezo_thickness_m=float(runtime["piezo_thickness_m"]),
        mesh_size_relative_to_cell=MESH_SIZE_SCALE,
        cad_reference_size_relative_to_cell=CAD_REFERENCE_SIZE_SCALE,
        substrate_layers=int(preset_overrides["substrate_layers"]),
        piezo_layers=int(preset_overrides["piezo_layers"]),
        solver_mesh_backend="layered_tet",
        max_solver_vector_dofs=preset_overrides["max_solver_vector_dofs"],
        allow_solver_mesh_coarsening=bool(preset_overrides["allow_solver_mesh_coarsening"]),
        mesh_preset=MESH_PRESET,
        exact_cad=True,
        repair_cad=False,
    )

    plate_lx, plate_ly = geometry_config.plate_size_m
    mesh_dir = PROJECT_ROOT / "runs" / "plainplate" / "meshes" / "volumes"
    print(
        "Plain-plate mesh build: "
        f"plate=({plate_lx:.6g}, {plate_ly:.6g}) m, "
        f"substrate={volume_config.substrate_thickness_m:.6g} m (tag 11), "
        f"piezo={volume_config.piezo_thickness_m:.6g} m (tag 12), "
        f"layers={volume_config.substrate_layers}+{volume_config.piezo_layers}, "
        f"mesh_size_scale={MESH_SIZE_SCALE}, preset={MESH_PRESET}",
        flush=True,
    )

    mesh_path = mesh_plain_plate_volume_sample(
        sample_id=SAMPLE_ID,
        output_dir=mesh_dir,
        geometry_config=geometry_config,
        volume_config=volume_config,
    )
    if mesh_path is None:
        raise SystemExit("Plain-plate mesh build returned no output path.")
    print(f"Wrote plain-plate solver mesh: {mesh_path}", flush=True)


if __name__ == "__main__":
    main()
