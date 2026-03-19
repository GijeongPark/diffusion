from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gmsh
import meshio
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from .geometry_pipeline import (
    GeometryBuildConfig,
    _iter_polygons,
    build_unit_cell_solid_polygons,
    tile_unit_cell_polygons,
)
from .mesh_tags import (
    FACET_BOTTOM_ELECTRODE_TAG,
    FACET_BOTTOM_PLATE_TAG,
    FACET_CLAMPED_TAG,
    FACET_FREE_X_MAX_TAG,
    FACET_FREE_Y_MAX_TAG,
    FACET_FREE_Y_MIN_TAG,
    FACET_TOP_ELECTRODE_TAG,
    VOLUME_PIEZO_TAG,
    VOLUME_SUBSTRATE_TAG,
)


@dataclass(frozen=True)
class VolumeMeshConfig:
    substrate_thickness_m: float = 1.0e-3
    piezo_thickness_m: float = 2.667e-4
    mesh_size_relative_to_cell: float = 0.06
    substrate_layers: int = 1
    piezo_layers: int = 1

    @property
    def total_thickness_m(self) -> float:
        return float(self.substrate_thickness_m) + float(self.piezo_thickness_m)


def _build_occ_surface_from_polygon(
    polygon: Polygon,
    mesh_size_m: float,
    point_cache: dict[tuple[float, float, float], int],
) -> int | None:
    def add_point(x: float, y: float, z: float = 0.0) -> int:
        key = (round(x, 9), round(y, 9), round(z, 9))
        if key not in point_cache:
            point_cache[key] = gmsh.model.occ.addPoint(x, y, z, mesh_size_m)
        return point_cache[key]

    exterior_coords = np.asarray(polygon.exterior.coords, dtype=np.float64)
    exterior_points = [add_point(float(x), float(y)) for x, y in exterior_coords[:-1]]
    if len(exterior_points) < 3:
        return None

    exterior_lines = []
    for idx in range(len(exterior_points)):
        p0 = exterior_points[idx]
        p1 = exterior_points[(idx + 1) % len(exterior_points)]
        if p0 == p1:
            continue
        exterior_lines.append(gmsh.model.occ.addLine(p0, p1))
    if len(exterior_lines) < 3:
        return None
    exterior_loop = gmsh.model.occ.addCurveLoop(exterior_lines)

    hole_loops: list[int] = []
    for interior in polygon.interiors:
        coords = np.asarray(interior.coords, dtype=np.float64)
        points = [add_point(float(x), float(y)) for x, y in coords[:-1]]
        if len(points) < 3:
            continue
        lines = []
        for idx in range(len(points)):
            p0 = points[idx]
            p1 = points[(idx + 1) % len(points)]
            if p0 == p1:
                continue
            lines.append(gmsh.model.occ.addLine(p0, p1))
        if len(lines) < 3:
            continue
        hole_loops.append(gmsh.model.occ.addCurveLoop(lines))

    return gmsh.model.occ.addPlaneSurface([exterior_loop] + hole_loops)


def _collect_surfaces_at_z(surface_tags: Iterable[int], z_value: float, tol: float) -> list[int]:
    out: list[int] = []
    for tag in surface_tags:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        if abs(zmin - z_value) < tol and abs(zmax - z_value) < tol:
            out.append(tag)
    return out


def _classify_boundary_surfaces(
    plate_size_m: tuple[float, float],
    total_thickness_m: float,
    substrate_thickness_m: float,
    tol: float,
) -> dict[str, list[int]]:
    lx, ly = plate_size_m
    groups = {
        "clamped": [],
        "free_x_max": [],
        "free_y_min": [],
        "free_y_max": [],
        "top_electrode": [],
        "bottom_electrode": [],
        "bottom_plate": [],
    }
    for dim, tag in gmsh.model.getEntities(2):
        if dim != 2:
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        if abs(xmin) < tol and abs(xmax) < tol:
            groups["clamped"].append(tag)
        elif abs(xmin - lx) < tol and abs(xmax - lx) < tol:
            groups["free_x_max"].append(tag)
        elif abs(ymin) < tol and abs(ymax) < tol:
            groups["free_y_min"].append(tag)
        elif abs(ymin - ly) < tol and abs(ymax - ly) < tol:
            groups["free_y_max"].append(tag)

        if abs(zmin - total_thickness_m) < tol and abs(zmax - total_thickness_m) < tol:
            groups["top_electrode"].append(tag)
        elif abs(zmin - substrate_thickness_m) < tol and abs(zmax - substrate_thickness_m) < tol:
            groups["bottom_electrode"].append(tag)
        elif abs(zmin) < tol and abs(zmax) < tol:
            groups["bottom_plate"].append(tag)
    return groups


def mesh_tiled_plate_volume_sample(
    grf: np.ndarray,
    threshold: float,
    sample_id: int,
    output_dir: str | Path,
    geometry_config: GeometryBuildConfig | None = None,
    volume_config: VolumeMeshConfig | None = None,
) -> Path | None:
    geometry_config = geometry_config or GeometryBuildConfig()
    volume_config = volume_config or VolumeMeshConfig()

    unit_cell_polygons = build_unit_cell_solid_polygons(
        grf=np.asarray(grf, dtype=np.float64),
        threshold=float(threshold),
        resolution=int(grf.shape[0]),
        decimate_tol=geometry_config.contour_decimate_tol,
    )
    tiled_polygons = tile_unit_cell_polygons(
        unit_cell_polygons=unit_cell_polygons,
        cell_size_m=geometry_config.cell_size_m,
        tile_counts=geometry_config.tile_counts,
    )
    polygons = [poly for geom in tiled_polygons for poly in _iter_polygons(geom)]
    if not polygons:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    msh_path = output_dir / f"plate3d_{int(sample_id):04d}.msh"

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("plate3d")
    try:
        point_cache: dict[tuple[float, float, float], int] = {}
        surface_tags: list[int] = []
        mesh_size_m = geometry_config.cell_size_m[0] * volume_config.mesh_size_relative_to_cell
        for polygon in polygons:
            surface_tag = _build_occ_surface_from_polygon(
                polygon=polygon,
                mesh_size_m=mesh_size_m,
                point_cache=point_cache,
            )
            if surface_tag is not None:
                surface_tags.append(surface_tag)
        if not surface_tags:
            return None

        gmsh.model.occ.synchronize()
        extruded_substrate: list[tuple[int, int]] = []
        for surface_tag in surface_tags:
            extruded_substrate.extend(
                gmsh.model.occ.extrude(
                    [(2, surface_tag)],
                    0.0,
                    0.0,
                    volume_config.substrate_thickness_m,
                    numElements=[int(volume_config.substrate_layers)],
                    heights=[1.0],
                    recombine=False,
                )
            )
        gmsh.model.occ.synchronize()

        substrate_volume_tags = [tag for dim, tag in extruded_substrate if dim == 3]
        candidate_top_surfaces = [tag for dim, tag in extruded_substrate if dim == 2]
        z_tol = max(
            1.0e-7,
            mesh_size_m * 1.0e-2,
            volume_config.substrate_thickness_m * 1.0e-2,
        )
        substrate_top_surfaces = _collect_surfaces_at_z(
            candidate_top_surfaces,
            z_value=float(volume_config.substrate_thickness_m),
            tol=z_tol,
        )
        if not substrate_top_surfaces:
            return None

        extruded_piezo: list[tuple[int, int]] = []
        for surface_tag in substrate_top_surfaces:
            extruded_piezo.extend(
                gmsh.model.occ.extrude(
                    [(2, surface_tag)],
                    0.0,
                    0.0,
                    volume_config.piezo_thickness_m,
                    numElements=[int(volume_config.piezo_layers)],
                    heights=[1.0],
                    recombine=False,
                )
            )
        gmsh.model.occ.synchronize()

        piezo_volume_tags = [tag for dim, tag in extruded_piezo if dim == 3]
        if not substrate_volume_tags or not piezo_volume_tags:
            return None

        gmsh.model.addPhysicalGroup(3, substrate_volume_tags, tag=VOLUME_SUBSTRATE_TAG)
        gmsh.model.setPhysicalName(3, VOLUME_SUBSTRATE_TAG, "substrate")
        gmsh.model.addPhysicalGroup(3, piezo_volume_tags, tag=VOLUME_PIEZO_TAG)
        gmsh.model.setPhysicalName(3, VOLUME_PIEZO_TAG, "piezo")

        tol = max(
            1.0e-7,
            mesh_size_m * 1.0e-2,
            min(
                geometry_config.plate_size_m[0],
                geometry_config.plate_size_m[1],
                volume_config.total_thickness_m,
            )
            * 1.0e-6,
        )
        surface_groups = _classify_boundary_surfaces(
            plate_size_m=geometry_config.plate_size_m,
            total_thickness_m=volume_config.total_thickness_m,
            substrate_thickness_m=volume_config.substrate_thickness_m,
            tol=tol,
        )

        physical_map = {
            FACET_CLAMPED_TAG: ("clamped", surface_groups["clamped"]),
            FACET_FREE_X_MAX_TAG: ("free_x_max", surface_groups["free_x_max"]),
            FACET_FREE_Y_MIN_TAG: ("free_y_min", surface_groups["free_y_min"]),
            FACET_FREE_Y_MAX_TAG: ("free_y_max", surface_groups["free_y_max"]),
            FACET_TOP_ELECTRODE_TAG: ("top_electrode", surface_groups["top_electrode"]),
            FACET_BOTTOM_ELECTRODE_TAG: ("bottom_electrode", surface_groups["bottom_electrode"]),
            FACET_BOTTOM_PLATE_TAG: ("bottom_plate", surface_groups["bottom_plate"]),
        }
        for tag, (name, entities) in physical_map.items():
            unique_entities = sorted(set(entities))
            if not unique_entities:
                continue
            gmsh.model.addPhysicalGroup(2, unique_entities, tag=tag)
            gmsh.model.setPhysicalName(2, tag, name)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(msh_path))
        return msh_path
    finally:
        gmsh.finalize()


def _stack_cell_blocks(mesh: meshio.Mesh, cell_type: str, data_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    cell_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []
    if data_name not in mesh.cell_data:
        return None
    for block, values in zip(mesh.cells, mesh.cell_data[data_name]):
        if block.type != cell_type:
            continue
        cell_blocks.append(np.asarray(block.data, dtype=np.int64))
        data_blocks.append(np.asarray(values, dtype=np.int32))
    if not cell_blocks:
        return None
    return np.vstack(cell_blocks), np.concatenate(data_blocks)


def convert_volume_msh_to_xdmf(msh_path: str | Path) -> tuple[Path | None, Path | None]:
    msh_path = Path(msh_path)
    try:
        mesh = meshio.read(msh_path)
    except Exception:
        return None, None

    tetra = _stack_cell_blocks(mesh, "tetra", "gmsh:physical")
    triangles = _stack_cell_blocks(mesh, "triangle", "gmsh:physical")
    if tetra is None or triangles is None:
        return None, None

    tetra_cells, tetra_tags = tetra
    tri_cells, tri_tags = triangles

    volume_mesh = meshio.Mesh(
        points=np.asarray(mesh.points, dtype=np.float64),
        cells=[("tetra", tetra_cells)],
        cell_data={"name_to_read": [tetra_tags]},
    )
    facet_mesh = meshio.Mesh(
        points=np.asarray(mesh.points, dtype=np.float64),
        cells=[("triangle", tri_cells)],
        cell_data={"name_to_read": [tri_tags]},
    )

    volume_xdmf = msh_path.with_suffix(".xdmf")
    facet_xdmf = msh_path.with_name(f"{msh_path.stem}_facets.xdmf")
    meshio.write(volume_xdmf, volume_mesh)
    meshio.write(facet_xdmf, facet_mesh)
    return volume_xdmf, facet_xdmf


def convert_volume_msh_to_fenicsx_npz(msh_path: str | Path) -> Path | None:
    msh_path = Path(msh_path)
    try:
        mesh = meshio.read(msh_path)
    except Exception:
        return None

    tetra = _stack_cell_blocks(mesh, "tetra", "gmsh:physical")
    triangles = _stack_cell_blocks(mesh, "triangle", "gmsh:physical")
    if tetra is None or triangles is None:
        return None

    tetra_cells, tetra_tags = tetra
    tri_cells, tri_tags = triangles
    npz_path = msh_path.with_name(f"{msh_path.stem}_fenicsx.npz")
    np.savez_compressed(
        npz_path,
        points=np.asarray(mesh.points, dtype=np.float64),
        tetra_cells=np.asarray(tetra_cells, dtype=np.int64),
        tetra_tags=np.asarray(tetra_tags, dtype=np.int32),
        triangle_cells=np.asarray(tri_cells, dtype=np.int32),
        triangle_tags=np.asarray(tri_tags, dtype=np.int32),
    )
    return npz_path
