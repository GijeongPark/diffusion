from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import gmsh
import meshio
import numpy as np
from shapely import set_precision
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

from .geometry_pipeline import (
    GeometryBuildConfig,
    _connect_plate_components,
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
    cad_reference_size_relative_to_cell: float = 0.01
    limit_solver_mesh_by_thickness: bool = False
    substrate_layers: int = 2
    piezo_layers: int = 1
    solver_mesh_backend: str = "layered_tet"
    write_native_msh: bool = False
    write_xdmf: bool = False
    occ_heal_tolerance_m: float = 1.0e-8
    require_connected_substrate: bool = False
    exact_cad: bool = True
    repair_cad: bool = False
    repair_bridge_width_m: float | None = None
    min_planform_feature_relative_to_mesh: float = 0.25
    volume_relative_tolerance: float = 5.0e-4
    max_solver_vector_dofs: int | None = 5_000_000
    solver_mesh_growth_factor: float = 1.15
    solver_mesh_limit_retries: int = 5
    ansys_step_strategy: str = "single_face_assembly"
    export_inspection_single_face_step: bool = False
    cad_planform_simplify_relative_to_reference: float = 1.5
    cad_min_hole_area_relative_to_reference_squared: float = 100.0

    @property
    def total_thickness_m(self) -> float:
        return float(self.substrate_thickness_m) + float(self.piezo_thickness_m)

    @property
    def target_min_elements_through_thickness(self) -> float:
        return 2.5

    def __post_init__(self) -> None:
        if float(self.substrate_thickness_m) <= 0.0:
            raise ValueError("substrate_thickness_m must be strictly positive.")
        if float(self.piezo_thickness_m) <= 0.0:
            raise ValueError("piezo_thickness_m must be strictly positive.")
        if float(self.mesh_size_relative_to_cell) <= 0.0:
            raise ValueError("mesh_size_relative_to_cell must be strictly positive.")
        if float(self.cad_reference_size_relative_to_cell) <= 0.0:
            raise ValueError("cad_reference_size_relative_to_cell must be strictly positive.")
        if float(self.occ_heal_tolerance_m) <= 0.0:
            raise ValueError("occ_heal_tolerance_m must be strictly positive.")
        if int(self.substrate_layers) <= 0:
            raise ValueError("substrate_layers must be a positive integer.")
        if int(self.piezo_layers) <= 0:
            raise ValueError("piezo_layers must be a positive integer.")
        backend = str(self.solver_mesh_backend).strip().lower()
        if backend not in {"layered_tet", "gmsh_volume"}:
            raise ValueError("solver_mesh_backend must be 'layered_tet' or 'gmsh_volume'.")
        ansys_step_strategy = str(self.ansys_step_strategy).strip().lower()
        if ansys_step_strategy not in {"partitioned_interface", "single_face_assembly"}:
            raise ValueError(
                "ansys_step_strategy must be 'partitioned_interface' or 'single_face_assembly'."
            )
        if bool(self.exact_cad) == bool(self.repair_cad):
            raise ValueError("Exactly one CAD mode must be active: exact_cad or repair_cad.")
        if self.repair_bridge_width_m is not None and float(self.repair_bridge_width_m) <= 0.0:
            raise ValueError("repair_bridge_width_m must be strictly positive when provided.")
        if float(self.min_planform_feature_relative_to_mesh) <= 0.0:
            raise ValueError("min_planform_feature_relative_to_mesh must be strictly positive.")
        if float(self.volume_relative_tolerance) <= 0.0:
            raise ValueError("volume_relative_tolerance must be strictly positive.")
        if self.max_solver_vector_dofs is not None and int(self.max_solver_vector_dofs) <= 0:
            raise ValueError("max_solver_vector_dofs must be strictly positive when provided.")
        if float(self.solver_mesh_growth_factor) <= 1.0:
            raise ValueError("solver_mesh_growth_factor must be greater than 1.0.")
        if int(self.solver_mesh_limit_retries) < 0:
            raise ValueError("solver_mesh_limit_retries must be non-negative.")
        if float(self.cad_planform_simplify_relative_to_reference) < 0.0:
            raise ValueError("cad_planform_simplify_relative_to_reference must be non-negative.")
        if float(self.cad_min_hole_area_relative_to_reference_squared) < 0.0:
            raise ValueError("cad_min_hole_area_relative_to_reference_squared must be non-negative.")


@dataclass(frozen=True)
class _CadPlanform:
    polygon: Polygon
    was_repaired: bool
    initial_component_count: int
    hole_count: int
    area_m2: float


@dataclass(frozen=True)
class _CadValidationReport:
    stage: str
    substrate_tag: int
    piezo_tag: int
    solid_body_count: int
    stray_surface_tags: tuple[int, ...]
    stray_curve_tags: tuple[int, ...]
    internal_vertical_substrate_face_tags: tuple[int, ...]
    piezo_bottom_face_count: int
    substrate_volume_m3: float
    piezo_volume_m3: float
    expected_substrate_volume_m3: float
    expected_piezo_volume_m3: float


@dataclass(frozen=True)
class _SingleBodyCadValidationReport:
    stage: str
    body_role: str
    body_tag: int
    solid_body_count: int
    stray_surface_tags: tuple[int, ...]
    stray_curve_tags: tuple[int, ...]
    horizontal_face_count: int
    volume_m3: float
    expected_volume_m3: float
    bounding_box_xyzxyz_m: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class _CadExportArtifacts:
    solver_mesh_path: Path
    step_path: Path | None
    cad_report_path: Path
    planform: _CadPlanform
    selection_manifest_path: Path | None = None
    pre_export_report: _CadValidationReport | None = None
    roundtrip_report: _CadValidationReport | None = None
    substrate_step_path: Path | None = None
    piezo_step_path: Path | None = None
    substrate_pre_export_report: _SingleBodyCadValidationReport | None = None
    substrate_roundtrip_report: _SingleBodyCadValidationReport | None = None
    piezo_pre_export_report: _SingleBodyCadValidationReport | None = None
    piezo_roundtrip_report: _SingleBodyCadValidationReport | None = None
    inspection_step_path: Path | None = None
    inspection_pre_export_report: _CadValidationReport | None = None
    inspection_roundtrip_report: _CadValidationReport | None = None


@contextmanager
def _silence_native_output() -> Iterable[None]:
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


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

    exterior_lines: list[int] = []
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
        lines: list[int] = []
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


def _normalize_step_length_units_to_metre(step_path: Path) -> None:
    try:
        payload = step_path.read_text(encoding="utf-8")
    except OSError:
        return
    normalized = payload.replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT($,.METRE.)")
    if normalized != payload:
        step_path.write_text(normalized, encoding="utf-8")


def _export_occ_geometry_to_step(step_path: Path) -> Path | None:
    step_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        gmsh.model.occ.synchronize()
        with _silence_native_output():
            gmsh.write(str(step_path))
    except Exception:
        return None
    return step_path


def _reload_occ_geometry_from_step(step_path: Path, model_name: str = "plate3d_mesh") -> None:
    gmsh.clear()
    gmsh.model.add(model_name)
    gmsh.model.occ.importShapes(str(step_path))
    gmsh.model.occ.synchronize()


def _heal_current_occ_model(tolerance_m: float) -> None:
    gmsh.model.occ.synchronize()
    # Some gmsh/OpenCASCADE builds interpret an empty dimTags list as "heal and replace
    # the whole model", which can drop the current 3D entities after fragment/fuse. Heal
    # the explicit solids instead so the substrate/piezo volume tags remain available.
    dimtags = gmsh.model.getEntities(3)
    if not dimtags:
        dimtags = gmsh.model.getEntities()
    if not dimtags:
        return
    try:
        gmsh.model.occ.healShapes(dimtags, float(tolerance_m), True, True, True, True, True)
    except TypeError:
        gmsh.model.occ.healShapes(
            dimTags=dimtags,
            tolerance=float(tolerance_m),
            fixDegenerated=True,
            fixSmallEdges=True,
            fixSmallFaces=True,
            sewFaces=True,
            makeSolids=True,
        )
    except AttributeError:
        # Older gmsh builds can lack healShapes; in that case keep the boolean result as-is.
        pass
    gmsh.model.occ.synchronize()


def _set_mesh_size_on_all_points(mesh_size_m: float) -> None:
    point_dimtags = gmsh.model.getEntities(0)
    if point_dimtags:
        gmsh.model.mesh.setSize(point_dimtags, float(mesh_size_m))


def _volume_z_tolerance(mesh_size_m: float, volume_config: VolumeMeshConfig) -> float:
    return max(
        1.0e-8,
        1.0e-3 * float(mesh_size_m),
        1.0e-4 * max(float(volume_config.substrate_thickness_m), float(volume_config.piezo_thickness_m)),
    )


def _resolve_cad_reference_size_m(
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
) -> float:
    in_plane_cell_m = float(min(geometry_config.cell_size_m))
    return in_plane_cell_m * float(volume_config.cad_reference_size_relative_to_cell)


def _resolve_solver_mesh_size_m(
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
) -> float:
    in_plane_cell_m = float(min(geometry_config.cell_size_m))
    in_plane_target_m = in_plane_cell_m * float(volume_config.mesh_size_relative_to_cell)
    if not bool(volume_config.limit_solver_mesh_by_thickness):
        return in_plane_target_m
    thickness_target_m = float(volume_config.total_thickness_m) / float(
        volume_config.target_min_elements_through_thickness
    )
    return min(in_plane_target_m, thickness_target_m)


def _ansys_step_strategy(volume_config: VolumeMeshConfig) -> str:
    return str(getattr(volume_config, "ansys_step_strategy", "single_face_assembly")).strip().lower()


def _classify_volume_entities_by_z(
    substrate_thickness_m: float,
    total_thickness_m: float,
    tol: float,
) -> dict[str, list[int]]:
    groups = {"substrate": [], "piezo": [], "unknown": []}
    for dim, tag in gmsh.model.getEntities(3):
        if dim != 3:
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, tag)
        if zmin >= -tol and zmax <= float(substrate_thickness_m) + tol:
            groups["substrate"].append(tag)
        elif zmin >= float(substrate_thickness_m) - tol and zmax <= float(total_thickness_m) + tol:
            groups["piezo"].append(tag)
        else:
            groups["unknown"].append(tag)
    return groups


def _classify_boundary_surfaces(
    plate_size_m: tuple[float, float],
    total_thickness_m: float,
    substrate_thickness_m: float,
    tol: float,
) -> dict[str, list[int]]:
    lx, ly = float(plate_size_m[0]), float(plate_size_m[1])
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


def _surface_classification_tolerance(
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    mesh_size_m: float,
) -> float:
    return max(
        _volume_z_tolerance(mesh_size_m, volume_config),
        min(
            float(geometry_config.plate_size_m[0]),
            float(geometry_config.plate_size_m[1]),
            float(volume_config.total_thickness_m),
        )
        * 1.0e-6,
    )


def _safe_set_entity_name(dim: int, tag: int, name: str) -> None:
    try:
        gmsh.model.setEntityName(int(dim), int(tag), str(name))
    except Exception:
        pass


def _safe_set_color(dimtags: Iterable[tuple[int, int]], rgb: tuple[int, int, int]) -> None:
    dimtags = [(int(dim), int(tag)) for dim, tag in dimtags]
    if not dimtags:
        return
    try:
        gmsh.model.setColor(dimtags, int(rgb[0]), int(rgb[1]), int(rgb[2]))
    except Exception:
        pass


def _annotate_current_occ_entities_for_ansys(
    report: _CadValidationReport,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    mesh_size_m: float,
) -> None:
    _safe_set_entity_name(3, int(report.substrate_tag), "substrate")
    _safe_set_entity_name(3, int(report.piezo_tag), "piezo")
    _safe_set_color([(3, int(report.substrate_tag))], (120, 160, 220))
    _safe_set_color([(3, int(report.piezo_tag))], (235, 185, 90))

    tol = _surface_classification_tolerance(geometry_config, volume_config, mesh_size_m)
    surface_groups = _classify_boundary_surfaces(
        plate_size_m=geometry_config.plate_size_m,
        total_thickness_m=volume_config.total_thickness_m,
        substrate_thickness_m=volume_config.substrate_thickness_m,
        tol=tol,
    )
    named_surfaces: dict[str, list[int]] = {
        "clamped_edge": surface_groups["clamped"],
        "free_x_max": surface_groups["free_x_max"],
        "free_y_min": surface_groups["free_y_min"],
        "free_y_max": surface_groups["free_y_max"],
        "piezo_top_electrode": surface_groups["top_electrode"],
        "bottom_plate": surface_groups["bottom_plate"],
    }
    piezo_bottom_faces = _collect_horizontal_face_tags_for_volume(
        volume_tag=int(report.piezo_tag),
        z_value_m=volume_config.substrate_thickness_m,
        tol=tol,
    )
    named_surfaces["piezo_bottom_electrode"] = list(piezo_bottom_faces)

    for name, tags in named_surfaces.items():
        for tag in sorted(set(int(value) for value in tags)):
            _safe_set_entity_name(2, int(tag), name)

    _safe_set_color([(2, int(tag)) for tag in surface_groups["top_electrode"]], (255, 210, 110))
    _safe_set_color([(2, int(tag)) for tag in piezo_bottom_faces], (255, 180, 90))
    _safe_set_color([(2, int(tag)) for tag in surface_groups["clamped"]], (180, 70, 70))


def _resolve_repair_bridge_width_m(
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
) -> float:
    if volume_config.repair_bridge_width_m is not None:
        return float(volume_config.repair_bridge_width_m)
    return float(geometry_config.connectivity_bridge_width_m)


def _minimum_planform_feature_size_m(
    mesh_size_m: float,
    volume_config: VolumeMeshConfig,
) -> float:
    return max(
        10.0 * float(volume_config.occ_heal_tolerance_m),
        float(volume_config.min_planform_feature_relative_to_mesh) * float(mesh_size_m),
    )


def _normalize_planform_precision(
    geometry: Polygon | MultiPolygon,
    mesh_size_m: float,
) -> Polygon | MultiPolygon:
    grid_size_m = max(1.0e-12, min(1.0e-9, 1.0e-6 * float(mesh_size_m)))
    normalized = set_precision(geometry, grid_size=grid_size_m)
    return normalized.buffer(0)


def _cad_planform_simplify_tolerance_m(
    mesh_size_m: float,
    volume_config: VolumeMeshConfig,
) -> float:
    return max(
        0.0,
        float(volume_config.cad_planform_simplify_relative_to_reference) * float(mesh_size_m),
    )


def _minimum_cad_hole_area_m2(
    mesh_size_m: float,
    volume_config: VolumeMeshConfig,
) -> float:
    return max(
        0.0,
        float(volume_config.cad_min_hole_area_relative_to_reference_squared) * float(mesh_size_m) ** 2,
    )


def _prune_small_holes(
    polygon: Polygon,
    min_hole_area_m2: float,
) -> Polygon:
    if min_hole_area_m2 <= 0.0 or not polygon.interiors:
        return polygon
    kept_holes: list[list[tuple[float, float]]] = []
    for interior in polygon.interiors:
        hole_poly = Polygon(interior)
        if hole_poly.is_empty:
            continue
        if float(hole_poly.area) >= float(min_hole_area_m2):
            kept_holes.append([(float(x), float(y)) for x, y in interior.coords])
    return Polygon(polygon.exterior.coords, holes=kept_holes)


def _simplify_planform_for_cad(
    planform: Polygon,
    mesh_size_m: float,
    volume_config: VolumeMeshConfig,
) -> Polygon:
    simplify_tol_m = _cad_planform_simplify_tolerance_m(mesh_size_m, volume_config)
    min_hole_area_m2 = _minimum_cad_hole_area_m2(mesh_size_m, volume_config)

    working: Polygon | MultiPolygon = planform
    if simplify_tol_m > 0.0:
        working = working.simplify(float(simplify_tol_m), preserve_topology=True)
        working = _normalize_planform_precision(working, mesh_size_m)

    simplified = _require_single_polygon(working, context="Substrate planform")
    simplified = _prune_small_holes(simplified, min_hole_area_m2)
    simplified = _require_single_polygon(
        _normalize_planform_precision(simplified, mesh_size_m),
        context="Substrate planform",
    )
    return simplified


def _finalize_planform(
    polygon: Polygon,
    *,
    was_repaired: bool,
    initial_component_count: int,
    mesh_size_m: float,
    volume_config: VolumeMeshConfig,
    invalid_message: str,
) -> _CadPlanform:
    if not polygon.is_valid:
        raise RuntimeError(invalid_message)

    minimum_clearance = float(polygon.minimum_clearance)
    min_feature_size_m = _minimum_planform_feature_size_m(mesh_size_m, volume_config)
    if np.isfinite(minimum_clearance) and minimum_clearance < min_feature_size_m:
        raise RuntimeError(
            "The substrate planform is under-resolved for CAD export: "
            f"minimum clearance {minimum_clearance:.6g} m is below the "
            f"{min_feature_size_m:.6g} m tolerance."
        )

    return _CadPlanform(
        polygon=polygon,
        was_repaired=was_repaired,
        initial_component_count=initial_component_count,
        hole_count=len(polygon.interiors),
        area_m2=float(polygon.area),
    )


def _prepare_planform_for_cad_export(
    planform: _CadPlanform,
    mesh_size_m: float,
    volume_config: VolumeMeshConfig,
) -> _CadPlanform:
    simplified = _simplify_planform_for_cad(
        planform=planform.polygon,
        mesh_size_m=mesh_size_m,
        volume_config=volume_config,
    )
    return _finalize_planform(
        polygon=simplified,
        was_repaired=planform.was_repaired,
        initial_component_count=planform.initial_component_count,
        mesh_size_m=mesh_size_m,
        volume_config=volume_config,
        invalid_message="The CAD-simplified substrate planform is invalid and cannot be exported as CAD.",
    )


def _require_single_polygon(
    geometry: Polygon | MultiPolygon,
    context: str,
) -> Polygon:
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        raise RuntimeError(f"{context} is disconnected ({len(geometry.geoms)} separate components).")
    polygons = list(_iter_polygons(geometry))
    if len(polygons) == 1:
        return polygons[0]
    raise RuntimeError(f"{context} did not resolve to a single polygon; got {geometry.geom_type}.")


def _build_substrate_planform(
    polygons: Iterable[Polygon],
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    mesh_size_m: float,
) -> _CadPlanform:
    if geometry_config.enforce_connected_plate and volume_config.exact_cad:
        raise RuntimeError(
            "Exact CAD export does not allow GeometryBuildConfig.enforce_connected_plate because that changes the "
            "substrate topology. Leave it disabled to reject disconnected samples, or enable VolumeMeshConfig."
            "repair_cad to add explicit bridge geometry."
        )

    plate_box = box(0.0, 0.0, float(geometry_config.plate_size_m[0]), float(geometry_config.plate_size_m[1]))
    merged = unary_union(list(polygons)).intersection(plate_box)
    if merged.is_empty:
        raise RuntimeError("The tiled substrate planform is empty after clipping to the plate domain.")

    cleaned = merged.buffer(0)
    if cleaned.is_empty:
        raise RuntimeError("The tiled substrate planform became empty while cleaning invalid topology.")
    cleaned = _normalize_planform_precision(cleaned, mesh_size_m)

    initial_components = list(_iter_polygons(cleaned))
    if not initial_components:
        raise RuntimeError("The tiled substrate planform did not contain any polygonal solids.")

    working = cleaned
    was_repaired = False
    if volume_config.exact_cad:
        if len(initial_components) != 1:
            raise RuntimeError(
                "Exact CAD export rejected a disconnected substrate planform. "
                "The STEP path preserves the intended topology instead of silently bridging components."
            )
    else:
        if len(initial_components) > 1:
            repaired_components = _connect_plate_components(
                polygons=initial_components,
                plate_size_m=geometry_config.plate_size_m,
                bridge_width_m=_resolve_repair_bridge_width_m(geometry_config, volume_config),
            )
            working = unary_union(repaired_components).intersection(plate_box).buffer(0)
            working = _normalize_planform_precision(working, mesh_size_m)
            was_repaired = True
            if len(list(_iter_polygons(working))) != 1:
                raise RuntimeError(
                    "Repair CAD mode could not connect the substrate into one solid with explicit bridges."
                )

    planform = _require_single_polygon(working, context="Substrate planform")
    planform = _require_single_polygon(_normalize_planform_precision(planform, mesh_size_m), context="Substrate planform")
    return _finalize_planform(
        polygon=planform,
        was_repaired=was_repaired,
        initial_component_count=len(initial_components),
        mesh_size_m=mesh_size_m,
        volume_config=volume_config,
        invalid_message="The cleaned substrate planform is invalid and cannot be exported as CAD.",
    )


def _find_stray_surface_tags() -> tuple[int, ...]:
    surfaces = gmsh.model.getEntities(2)
    if not surfaces:
        return ()
    boundary_surfaces = {
        tag
        for dim, tag in gmsh.model.getBoundary(gmsh.model.getEntities(3), combined=False, oriented=False, recursive=False)
        if dim == 2
    }
    return tuple(sorted(tag for dim, tag in surfaces if dim == 2 and tag not in boundary_surfaces))


def _find_stray_curve_tags() -> tuple[int, ...]:
    curves = gmsh.model.getEntities(1)
    if not curves:
        return ()
    boundary_curves = {
        tag
        for dim, tag in gmsh.model.getBoundary(gmsh.model.getEntities(2), combined=False, oriented=False, recursive=False)
        if dim == 1
    }
    return tuple(sorted(tag for dim, tag in curves if dim == 1 and tag not in boundary_curves))


def _collect_horizontal_face_tags_for_volume(
    volume_tag: int,
    z_value_m: float,
    tol: float,
) -> tuple[int, ...]:
    face_tags: set[int] = set()
    for dim, tag in gmsh.model.getBoundary([(3, int(volume_tag))], combined=False, oriented=False, recursive=False):
        if dim != 2:
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        if abs(zmin - float(z_value_m)) <= tol and abs(zmax - float(z_value_m)) <= tol:
            face_tags.add(int(tag))
    return tuple(sorted(face_tags))


def _remove_stray_occ_entities() -> None:
    stray_surface_tags = _find_stray_surface_tags()
    if stray_surface_tags:
        gmsh.model.occ.remove([(2, int(tag)) for tag in stray_surface_tags], recursive=True)
        gmsh.model.occ.synchronize()

    stray_curve_tags = _find_stray_curve_tags()
    if stray_curve_tags:
        gmsh.model.occ.remove([(1, int(tag)) for tag in stray_curve_tags], recursive=True)
        gmsh.model.occ.synchronize()


def _collect_internal_vertical_substrate_face_tags(
    substrate_tag: int,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    tol: float,
) -> tuple[int, ...]:
    plate_lx, plate_ly = geometry_config.plate_size_m
    face_tags: set[int] = set()
    for dim, tag in gmsh.model.getBoundary([(3, int(substrate_tag))], combined=False, oriented=False, recursive=False):
        if dim != 2:
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        spans_substrate_thickness = (
            zmin <= tol and zmax >= float(volume_config.substrate_thickness_m) - tol
        )
        is_vertical = (zmax - zmin) >= float(volume_config.substrate_thickness_m) - tol
        on_outer_boundary = (
            (abs(xmin) <= tol and abs(xmax) <= tol)
            or (abs(xmin - float(plate_lx)) <= tol and abs(xmax - float(plate_lx)) <= tol)
            or (abs(ymin) <= tol and abs(ymax) <= tol)
            or (abs(ymin - float(plate_ly)) <= tol and abs(ymax - float(plate_ly)) <= tol)
        )
        if spans_substrate_thickness and is_vertical and not on_outer_boundary:
            face_tags.add(int(tag))
    return tuple(sorted(face_tags))


def _volume_tolerance_m3(
    expected_volume_m3: float,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
) -> float:
    plate_area_m2 = float(geometry_config.plate_size_m[0]) * float(geometry_config.plate_size_m[1])
    return max(
        1.0e-15,
        float(expected_volume_m3) * float(volume_config.volume_relative_tolerance),
        plate_area_m2 * float(volume_config.occ_heal_tolerance_m),
    )


def _validate_current_occ_model(
    stage: str,
    planform: _CadPlanform,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    mesh_size_m: float,
) -> _CadValidationReport:
    tol = _volume_z_tolerance(mesh_size_m, volume_config)
    groups = _classify_volume_entities_by_z(
        substrate_thickness_m=volume_config.substrate_thickness_m,
        total_thickness_m=volume_config.total_thickness_m,
        tol=tol,
    )
    solids = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
    if len(solids) != 2:
        raise RuntimeError(f"{stage}: expected exactly 2 solid bodies, found {len(solids)}.")
    if groups["unknown"]:
        raise RuntimeError(f"{stage}: could not classify OCC volumes by z-range: {groups['unknown']}")
    if len(groups["substrate"]) != 1:
        raise RuntimeError(f"{stage}: expected exactly 1 substrate solid, found {len(groups['substrate'])}.")
    if len(groups["piezo"]) != 1:
        raise RuntimeError(f"{stage}: expected exactly 1 piezo solid, found {len(groups['piezo'])}.")

    substrate_tag = int(groups["substrate"][0])
    piezo_tag = int(groups["piezo"][0])
    stray_surface_tags = _find_stray_surface_tags()
    stray_curve_tags = _find_stray_curve_tags()
    if stray_surface_tags:
        raise RuntimeError(f"{stage}: found stray surface bodies {list(stray_surface_tags)}.")
    if stray_curve_tags:
        raise RuntimeError(f"{stage}: found stray line bodies {list(stray_curve_tags)}.")

    substrate_volume_m3 = float(gmsh.model.occ.getMass(3, substrate_tag))
    piezo_volume_m3 = float(gmsh.model.occ.getMass(3, piezo_tag))
    expected_substrate_volume_m3 = float(planform.area_m2) * float(volume_config.substrate_thickness_m)
    expected_piezo_volume_m3 = (
        float(geometry_config.plate_size_m[0])
        * float(geometry_config.plate_size_m[1])
        * float(volume_config.piezo_thickness_m)
    )

    substrate_volume_tol = _volume_tolerance_m3(
        expected_volume_m3=expected_substrate_volume_m3,
        geometry_config=geometry_config,
        volume_config=volume_config,
    )
    piezo_volume_tol = _volume_tolerance_m3(
        expected_volume_m3=expected_piezo_volume_m3,
        geometry_config=geometry_config,
        volume_config=volume_config,
    )
    if abs(substrate_volume_m3 - expected_substrate_volume_m3) > substrate_volume_tol:
        raise RuntimeError(
            f"{stage}: substrate volume {substrate_volume_m3:.8e} m^3 does not match the expected "
            f"{expected_substrate_volume_m3:.8e} m^3."
        )
    if abs(piezo_volume_m3 - expected_piezo_volume_m3) > piezo_volume_tol:
        raise RuntimeError(
            f"{stage}: piezo volume {piezo_volume_m3:.8e} m^3 does not match the expected "
            f"{expected_piezo_volume_m3:.8e} m^3."
        )

    plate_lx, plate_ly = geometry_config.plate_size_m
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, piezo_tag)
    if not (
        abs(xmin) <= tol
        and abs(ymin) <= tol
        and abs(zmin - float(volume_config.substrate_thickness_m)) <= tol
        and abs(xmax - float(plate_lx)) <= tol
        and abs(ymax - float(plate_ly)) <= tol
        and abs(zmax - float(volume_config.total_thickness_m)) <= tol
    ):
        raise RuntimeError(f"{stage}: the piezo solid is not the expected fully covered top layer.")

    internal_vertical_faces = _collect_internal_vertical_substrate_face_tags(
        substrate_tag=substrate_tag,
        geometry_config=geometry_config,
        volume_config=volume_config,
        tol=_surface_classification_tolerance(geometry_config, volume_config, mesh_size_m),
    )
    if planform.hole_count > 0 and not internal_vertical_faces:
        raise RuntimeError(f"{stage}: substrate holes disappeared; no internal vertical faces were found.")
    piezo_bottom_faces = _collect_horizontal_face_tags_for_volume(
        volume_tag=piezo_tag,
        z_value_m=volume_config.substrate_thickness_m,
        tol=_surface_classification_tolerance(geometry_config, volume_config, mesh_size_m),
    )
    if not piezo_bottom_faces:
        raise RuntimeError(f"{stage}: could not identify any piezo bottom faces on the interface plane.")

    return _CadValidationReport(
        stage=stage,
        substrate_tag=substrate_tag,
        piezo_tag=piezo_tag,
        solid_body_count=len(solids),
        stray_surface_tags=stray_surface_tags,
        stray_curve_tags=stray_curve_tags,
        internal_vertical_substrate_face_tags=internal_vertical_faces,
        piezo_bottom_face_count=len(piezo_bottom_faces),
        substrate_volume_m3=substrate_volume_m3,
        piezo_volume_m3=piezo_volume_m3,
        expected_substrate_volume_m3=expected_substrate_volume_m3,
        expected_piezo_volume_m3=expected_piezo_volume_m3,
    )


def _validate_single_body_occ_model(
    stage: str,
    body_role: str,
    planform: _CadPlanform,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    mesh_size_m: float,
) -> _SingleBodyCadValidationReport:
    solids = [tag for dim, tag in gmsh.model.getEntities(3) if dim == 3]
    if len(solids) != 1:
        raise RuntimeError(f"{stage}: expected exactly 1 solid body for the {body_role} export, found {len(solids)}.")

    stray_surface_tags = _find_stray_surface_tags()
    stray_curve_tags = _find_stray_curve_tags()
    if stray_surface_tags:
        raise RuntimeError(f"{stage}: found stray surface bodies {list(stray_surface_tags)}.")
    if stray_curve_tags:
        raise RuntimeError(f"{stage}: found stray line bodies {list(stray_curve_tags)}.")

    body_tag = int(solids[0])
    bbox = tuple(float(value) for value in gmsh.model.getBoundingBox(3, body_tag))
    volume_m3 = float(gmsh.model.occ.getMass(3, body_tag))
    tol = _surface_classification_tolerance(geometry_config, volume_config, mesh_size_m)
    horizontal_faces = _collect_horizontal_face_tags_for_volume(
        volume_tag=body_tag,
        z_value_m=bbox[2],
        tol=tol,
    )

    plate_lx, plate_ly = geometry_config.plate_size_m
    if str(body_role) == "substrate":
        expected_volume_m3 = float(planform.area_m2) * float(volume_config.substrate_thickness_m)
        if not (
            abs(bbox[0]) <= tol
            and abs(bbox[1]) <= tol
            and abs(bbox[2]) <= tol
            and abs(bbox[3] - float(plate_lx)) <= tol
            and abs(bbox[4] - float(plate_ly)) <= tol
            and abs(bbox[5] - float(volume_config.substrate_thickness_m)) <= tol
        ):
            raise RuntimeError(f"{stage}: the standalone substrate STEP has an unexpected bounding box {bbox}.")
    elif str(body_role) == "piezo":
        expected_volume_m3 = (
            float(geometry_config.plate_size_m[0])
            * float(geometry_config.plate_size_m[1])
            * float(volume_config.piezo_thickness_m)
        )
        if not (
            abs(bbox[0]) <= tol
            and abs(bbox[1]) <= tol
            and abs(bbox[2] - float(volume_config.substrate_thickness_m)) <= tol
            and abs(bbox[3] - float(plate_lx)) <= tol
            and abs(bbox[4] - float(plate_ly)) <= tol
            and abs(bbox[5] - float(volume_config.total_thickness_m)) <= tol
        ):
            raise RuntimeError(f"{stage}: the standalone piezo STEP has an unexpected bounding box {bbox}.")
    else:
        raise ValueError(f"Unknown body_role for standalone STEP export: {body_role}")

    volume_tol = _volume_tolerance_m3(
        expected_volume_m3=expected_volume_m3,
        geometry_config=geometry_config,
        volume_config=volume_config,
    )
    if abs(volume_m3 - expected_volume_m3) > volume_tol:
        raise RuntimeError(
            f"{stage}: {body_role} volume {volume_m3:.8e} m^3 does not match the expected {expected_volume_m3:.8e} m^3."
        )

    return _SingleBodyCadValidationReport(
        stage=stage,
        body_role=str(body_role),
        body_tag=body_tag,
        solid_body_count=len(solids),
        stray_surface_tags=stray_surface_tags,
        stray_curve_tags=stray_curve_tags,
        horizontal_face_count=len(horizontal_faces),
        volume_m3=volume_m3,
        expected_volume_m3=expected_volume_m3,
        bounding_box_xyzxyz_m=bbox,
    )


def _prepare_single_body_occ_model(
    body_role: str,
    planform: _CadPlanform,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    mesh_size_m: float,
) -> _SingleBodyCadValidationReport:
    point_cache: dict[tuple[float, float, float], int] = {}
    if str(body_role) == "substrate":
        surface_tag = _build_occ_surface_from_polygon(
            polygon=planform.polygon,
            mesh_size_m=mesh_size_m,
            point_cache=point_cache,
        )
        if surface_tag is None:
            raise RuntimeError("No OCC substrate surface could be created from the cleaned planform polygon.")
        extruded = gmsh.model.occ.extrude([(2, int(surface_tag))], 0.0, 0.0, float(volume_config.substrate_thickness_m))
        volume_tags = [tag for dim, tag in extruded if dim == 3]
        if len(volume_tags) != 1:
            raise RuntimeError(
                f"Expected one standalone substrate solid from extrusion, but OpenCASCADE produced {len(volume_tags)}."
            )
    elif str(body_role) == "piezo":
        plate_lx, plate_ly = geometry_config.plate_size_m
        gmsh.model.occ.addBox(
            0.0,
            0.0,
            float(volume_config.substrate_thickness_m),
            float(plate_lx),
            float(plate_ly),
            float(volume_config.piezo_thickness_m),
        )
    else:
        raise ValueError(f"Unknown body_role for standalone STEP export: {body_role}")

    gmsh.model.occ.synchronize()
    _heal_current_occ_model(volume_config.occ_heal_tolerance_m)
    _remove_stray_occ_entities()
    _set_mesh_size_on_all_points(mesh_size_m)
    return _validate_single_body_occ_model(
        stage="pre_export",
        body_role=str(body_role),
        planform=planform,
        geometry_config=geometry_config,
        volume_config=volume_config,
        mesh_size_m=mesh_size_m,
    )


def _prepare_cad_occ_model(
    planform: _CadPlanform,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    mesh_size_m: float,
    partition_interface: bool,
) -> _CadValidationReport:
    point_cache: dict[tuple[float, float, float], int] = {}
    surface_tag = _build_occ_surface_from_polygon(
        polygon=planform.polygon,
        mesh_size_m=mesh_size_m,
        point_cache=point_cache,
    )
    if surface_tag is None:
        raise RuntimeError("No OCC substrate surface could be created from the cleaned planform polygon.")

    extruded = gmsh.model.occ.extrude([(2, int(surface_tag))], 0.0, 0.0, float(volume_config.substrate_thickness_m))
    substrate_volume_tags = [tag for dim, tag in extruded if dim == 3]
    if len(substrate_volume_tags) != 1:
        raise RuntimeError(
            f"Expected one substrate solid from extrusion, but OpenCASCADE produced {len(substrate_volume_tags)}."
        )

    plate_lx, plate_ly = geometry_config.plate_size_m
    piezo_tag = gmsh.model.occ.addBox(
        0.0,
        0.0,
        float(volume_config.substrate_thickness_m),
        float(plate_lx),
        float(plate_ly),
        float(volume_config.piezo_thickness_m),
    )
    gmsh.model.occ.synchronize()
    if partition_interface:
        gmsh.model.occ.fragment([(3, int(substrate_volume_tags[0]))], [(3, int(piezo_tag))], -1, True, True)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
    _heal_current_occ_model(volume_config.occ_heal_tolerance_m)
    _remove_stray_occ_entities()
    _set_mesh_size_on_all_points(mesh_size_m)
    return _validate_current_occ_model(
        stage="pre_export",
        planform=planform,
        geometry_config=geometry_config,
        volume_config=volume_config,
        mesh_size_m=mesh_size_m,
    )


def _write_cad_report(
    report_path: Path,
    sample_id: int,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    planform: _CadPlanform,
    pre_export_report: _CadValidationReport,
    roundtrip_report: _CadValidationReport,
    inspection_step_path: Path | None = None,
    inspection_pre_export_report: _CadValidationReport | None = None,
    inspection_roundtrip_report: _CadValidationReport | None = None,
) -> Path:
    payload = {
        "sample_id": int(sample_id),
        "cad_mode": "repair" if volume_config.repair_cad else "exact",
        "repair_applied": bool(planform.was_repaired),
        "initial_component_count": int(planform.initial_component_count),
        "hole_count": int(planform.hole_count),
        "planform_area_m2": float(planform.area_m2),
        "plate_size_m": [float(geometry_config.plate_size_m[0]), float(geometry_config.plate_size_m[1])],
        "substrate_thickness_m": float(volume_config.substrate_thickness_m),
        "piezo_thickness_m": float(volume_config.piezo_thickness_m),
        "solver_mesh_size_relative_to_cell": float(volume_config.mesh_size_relative_to_cell),
        "cad_reference_size_relative_to_cell": float(volume_config.cad_reference_size_relative_to_cell),
        "solver_mesh_backend": str(volume_config.solver_mesh_backend),
        "substrate_layers": int(volume_config.substrate_layers),
        "piezo_layers": int(volume_config.piezo_layers),
        "limit_solver_mesh_by_thickness": bool(volume_config.limit_solver_mesh_by_thickness),
        "cad_planform_simplify_relative_to_reference": float(volume_config.cad_planform_simplify_relative_to_reference),
        "cad_min_hole_area_relative_to_reference_squared": float(
            volume_config.cad_min_hole_area_relative_to_reference_squared
        ),
        "ansys_step_strategy": _ansys_step_strategy(volume_config),
        "pre_export": asdict(pre_export_report),
        "step_roundtrip": asdict(roundtrip_report),
    }
    if inspection_step_path is not None:
        payload["inspection_single_face_step_path"] = str(inspection_step_path)
    if inspection_pre_export_report is not None:
        payload["inspection_pre_export"] = asdict(inspection_pre_export_report)
    if inspection_roundtrip_report is not None:
        payload["inspection_single_face_step_roundtrip"] = asdict(inspection_roundtrip_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _write_ansys_face_selection_manifest(
    manifest_path: Path,
    sample_id: int,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    roundtrip_report: _CadValidationReport,
) -> Path:
    tol_m = _surface_classification_tolerance(
        geometry_config=geometry_config,
        volume_config=volume_config,
        mesh_size_m=_resolve_cad_reference_size_m(geometry_config, volume_config),
    )
    ansys_step_strategy = _ansys_step_strategy(volume_config)
    piezo_bottom_region_count = int(roundtrip_report.piezo_bottom_face_count)
    if ansys_step_strategy == "partitioned_interface":
        notes = [
            "The combined STEP keeps a conformal, meshable interface. The piezo bottom can therefore be split into multiple CAD faces.",
            "Instead of clicking every fragment manually in Workbench, build a Named Selection by filtering faces on the piezo body whose centroid or bounding-box z value equals the interface plane within the tolerance below.",
        ]
    else:
        notes = [
            "The combined STEP keeps the piezo bottom continuous as a one-file, two-body assembly, so the piezo-bottom electrode should resolve to one face on the piezo body.",
            "If Workbench re-imprints or otherwise changes the imported topology, rebuild the Named Selection by filtering faces on the piezo body whose centroid or bounding-box z value equals the interface plane within the tolerance below.",
        ]
    payload = {
        "sample_id": int(sample_id),
        "selection_strategy": "worksheet_by_body_and_z_plane",
        "notes": notes,
        "geometry": {
            "plate_size_m": [float(geometry_config.plate_size_m[0]), float(geometry_config.plate_size_m[1])],
            "substrate_thickness_m": float(volume_config.substrate_thickness_m),
            "piezo_thickness_m": float(volume_config.piezo_thickness_m),
            "total_thickness_m": float(volume_config.total_thickness_m),
            "selection_tolerance_m": float(tol_m),
        },
        "named_selection_recipes": {
            "piezo_body": {
                "entity_kind": "body",
                "body_role": "piezo",
                "expected_solid_count": 1,
            },
            "substrate_body": {
                "entity_kind": "body",
                "body_role": "substrate",
                "expected_solid_count": 1,
            },
            "piezo_bottom_electrode": {
                "entity_kind": "face",
                "body_role": "piezo",
                "z_plane_m": float(volume_config.substrate_thickness_m),
                "tolerance_m": float(tol_m),
                "expected_region_count": piezo_bottom_region_count,
            },
            "piezo_top_electrode": {
                "entity_kind": "face",
                "body_role": "piezo",
                "z_plane_m": float(volume_config.total_thickness_m),
                "tolerance_m": float(tol_m),
                "expected_region_count": 1,
            },
            "clamped_edge": {
                "entity_kind": "face",
                "x_plane_m": 0.0,
                "tolerance_m": float(tol_m),
            },
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _write_split_body_cad_report(
    report_path: Path,
    sample_id: int,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    planform: _CadPlanform,
    substrate_step_path: Path,
    substrate_pre_export_report: _SingleBodyCadValidationReport,
    substrate_roundtrip_report: _SingleBodyCadValidationReport,
    piezo_step_path: Path,
    piezo_pre_export_report: _SingleBodyCadValidationReport,
    piezo_roundtrip_report: _SingleBodyCadValidationReport,
    inspection_step_path: Path | None = None,
    inspection_pre_export_report: _CadValidationReport | None = None,
    inspection_roundtrip_report: _CadValidationReport | None = None,
) -> Path:
    payload = {
        "sample_id": int(sample_id),
        "cad_mode": "repair" if volume_config.repair_cad else "exact",
        "repair_applied": bool(planform.was_repaired),
        "initial_component_count": int(planform.initial_component_count),
        "hole_count": int(planform.hole_count),
        "planform_area_m2": float(planform.area_m2),
        "plate_size_m": [float(geometry_config.plate_size_m[0]), float(geometry_config.plate_size_m[1])],
        "substrate_thickness_m": float(volume_config.substrate_thickness_m),
        "piezo_thickness_m": float(volume_config.piezo_thickness_m),
        "solver_mesh_size_relative_to_cell": float(volume_config.mesh_size_relative_to_cell),
        "cad_reference_size_relative_to_cell": float(volume_config.cad_reference_size_relative_to_cell),
        "solver_mesh_backend": str(volume_config.solver_mesh_backend),
        "substrate_layers": int(volume_config.substrate_layers),
        "piezo_layers": int(volume_config.piezo_layers),
        "limit_solver_mesh_by_thickness": bool(volume_config.limit_solver_mesh_by_thickness),
        "ansys_step_strategy": "split_body_steps",
        "substrate_step_path": str(substrate_step_path),
        "piezo_step_path": str(piezo_step_path),
        "substrate_pre_export": asdict(substrate_pre_export_report),
        "substrate_step_roundtrip": asdict(substrate_roundtrip_report),
        "piezo_pre_export": asdict(piezo_pre_export_report),
        "piezo_step_roundtrip": asdict(piezo_roundtrip_report),
    }
    if inspection_step_path is not None:
        payload["legacy_single_face_probe_step_path"] = str(inspection_step_path)
    if inspection_pre_export_report is not None:
        payload["legacy_single_face_probe_pre_export"] = asdict(inspection_pre_export_report)
    if inspection_roundtrip_report is not None:
        payload["legacy_single_face_probe_step_roundtrip"] = asdict(inspection_roundtrip_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _build_step_export_variant(
    step_path: Path,
    planform: _CadPlanform,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    mesh_size_m: float,
    partition_interface: bool,
    require_single_piezo_bottom_face: bool = False,
) -> tuple[_CadValidationReport, _CadValidationReport]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("plate3d")
    try:
        pre_export_report = _prepare_cad_occ_model(
            planform=planform,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=mesh_size_m,
            partition_interface=partition_interface,
        )
        _annotate_current_occ_entities_for_ansys(
            report=pre_export_report,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=mesh_size_m,
        )
        if _export_occ_geometry_to_step(step_path) is None:
            raise RuntimeError(f"Could not export STEP geometry to {step_path}.")
        _reload_occ_geometry_from_step(step_path)
        roundtrip_report = _validate_current_occ_model(
            stage="step_roundtrip",
            planform=planform,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=mesh_size_m,
        )
        if require_single_piezo_bottom_face and int(roundtrip_report.piezo_bottom_face_count) != 1:
            raise RuntimeError(
                "step_roundtrip: expected exactly 1 continuous piezo bottom face for the single-face probe STEP, "
                f"found {int(roundtrip_report.piezo_bottom_face_count)}."
            )
        _normalize_step_length_units_to_metre(step_path)
        return pre_export_report, roundtrip_report
    finally:
        gmsh.finalize()


def _build_single_body_step_export_variant(
    step_path: Path,
    body_role: str,
    planform: _CadPlanform,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
    mesh_size_m: float,
) -> tuple[_SingleBodyCadValidationReport, _SingleBodyCadValidationReport]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add(f"plate3d_{body_role}")
    try:
        pre_export_report = _prepare_single_body_occ_model(
            body_role=str(body_role),
            planform=planform,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=mesh_size_m,
        )
        if _export_occ_geometry_to_step(step_path) is None:
            raise RuntimeError(f"Could not export {body_role} STEP geometry to {step_path}.")
        _reload_occ_geometry_from_step(step_path, model_name=f"plate3d_{body_role}_mesh")
        roundtrip_report = _validate_single_body_occ_model(
            stage="step_roundtrip",
            body_role=str(body_role),
            planform=planform,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=mesh_size_m,
        )
        _normalize_step_length_units_to_metre(step_path)
        return pre_export_report, roundtrip_report
    finally:
        gmsh.finalize()


def _extract_linear_triangles_from_current_gmsh_model() -> tuple[np.ndarray, np.ndarray]:
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    if len(node_tags) == 0:
        raise RuntimeError("gmsh did not return any 2D mesh nodes.")

    node_tags = np.asarray(node_tags, dtype=np.int64)
    node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(node_tags)
    node_tags = node_tags[order]
    node_coords = node_coords[order]

    triangles: list[np.ndarray] = []
    elem_types, _, elem_nodes = gmsh.model.mesh.getElements(2)
    for elem_type, node_block in zip(elem_types, elem_nodes):
        props = gmsh.model.mesh.getElementProperties(int(elem_type))
        name = str(props[0]).lower()
        dim = int(props[1])
        num_nodes = int(props[3])
        if dim != 2 or not name.startswith("triangle"):
            continue
        block = np.asarray(node_block, dtype=np.int64).reshape(-1, num_nodes)
        block = block[:, :3]
        local = np.searchsorted(node_tags, block)
        if np.any(local >= node_tags.shape[0]) or not np.array_equal(node_tags[local], block):
            raise RuntimeError("Could not map gmsh node tags to local indices for the 2D surface mesh.")
        triangles.append(np.asarray(local, dtype=np.int64))
    if not triangles:
        raise RuntimeError("gmsh did not create any triangular surface elements.")
    tri = np.vstack(triangles)
    xy = np.asarray(node_coords[:, :2], dtype=np.float64)
    orient = (
        (xy[tri[:, 1], 0] - xy[tri[:, 0], 0]) * (xy[tri[:, 2], 1] - xy[tri[:, 0], 1])
        - (xy[tri[:, 1], 1] - xy[tri[:, 0], 1]) * (xy[tri[:, 2], 0] - xy[tri[:, 0], 0])
    )
    negative = orient < 0.0
    if np.any(negative):
        tri = tri.copy()
        tmp = tri[negative, 1].copy()
        tri[negative, 1] = tri[negative, 2]
        tri[negative, 2] = tmp
    return xy, tri



def _mesh_partitioned_full_plate_triangles(
    planform: _CadPlanform,
    geometry_config: GeometryBuildConfig,
    mesh_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plate_box = box(0.0, 0.0, float(geometry_config.plate_size_m[0]), float(geometry_config.plate_size_m[1]))

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("plate2d_solver")
    try:
        point_cache: dict[tuple[float, float, float], int] = {}
        substrate_surface = _build_occ_surface_from_polygon(
            polygon=planform.polygon,
            mesh_size_m=mesh_size_m,
            point_cache=point_cache,
        )
        plate_surface = _build_occ_surface_from_polygon(
            polygon=plate_box,
            mesh_size_m=mesh_size_m,
            point_cache=point_cache,
        )
        if substrate_surface is None or plate_surface is None:
            raise RuntimeError("Could not build the partitioned 2D surfaces for the solver mesh.")

        gmsh.model.occ.synchronize()
        gmsh.model.occ.fragment([(2, int(plate_surface))], [(2, int(substrate_surface))], -1, True, True)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        _set_mesh_size_on_all_points(mesh_size_m)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(2)
        xy, triangles = _extract_linear_triangles_from_current_gmsh_model()
    finally:
        gmsh.finalize()

    prepared = prep(planform.polygon.buffer(max(1.0e-12, 1.0e-9 * float(mesh_size_m))))
    centroids = np.mean(xy[triangles], axis=1)
    in_substrate = np.fromiter(
        (prepared.covers(Point(float(x), float(y))) for x, y in centroids),
        dtype=bool,
        count=triangles.shape[0],
    )
    if not np.any(in_substrate):
        raise RuntimeError("The solver surface mesh did not contain any substrate triangles after partitioning.")
    return xy, triangles, in_substrate



def _solver_mesh_z_levels(volume_config: VolumeMeshConfig) -> tuple[np.ndarray, np.ndarray]:
    substrate_layers = int(volume_config.substrate_layers)
    piezo_layers = int(volume_config.piezo_layers)
    z_levels = [0.0]
    layer_tags: list[int] = []
    for idx in range(substrate_layers):
        z_levels.append(float(volume_config.substrate_thickness_m) * float(idx + 1) / float(substrate_layers))
        layer_tags.append(VOLUME_SUBSTRATE_TAG)
    for idx in range(piezo_layers):
        z_levels.append(
            float(volume_config.substrate_thickness_m)
            + float(volume_config.piezo_thickness_m) * float(idx + 1) / float(piezo_layers)
        )
        layer_tags.append(VOLUME_PIEZO_TAG)
    return np.asarray(z_levels, dtype=np.float64), np.asarray(layer_tags, dtype=np.int32)



def _tetra_signed_volumes(points: np.ndarray, tetra_cells: np.ndarray) -> np.ndarray:
    p = np.asarray(points[tetra_cells], dtype=np.float64)
    return np.einsum(
        "ij,ij->i",
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
        p[:, 3] - p[:, 0],
    ) / 6.0


def _estimate_quadratic_vector_dofs(points: np.ndarray, tetra_cells: np.ndarray) -> int:
    n_points = int(np.asarray(points).shape[0])
    if n_points <= 0 or np.asarray(tetra_cells).size == 0:
        return 3 * max(n_points, 0)

    tet = np.asarray(tetra_cells, dtype=np.int64)
    edge_blocks = np.vstack(
        [
            tet[:, [0, 1]],
            tet[:, [0, 2]],
            tet[:, [0, 3]],
            tet[:, [1, 2]],
            tet[:, [1, 3]],
            tet[:, [2, 3]],
        ]
    )
    lower = np.minimum(edge_blocks[:, 0], edge_blocks[:, 1]).astype(np.uint64, copy=False)
    upper = np.maximum(edge_blocks[:, 0], edge_blocks[:, 1]).astype(np.uint64, copy=False)
    edge_keys = lower * np.uint64(n_points) + upper
    unique_edges = int(np.unique(edge_keys).size)
    return 3 * (n_points + unique_edges)



def _classify_outer_boundary_edge(
    p0: np.ndarray,
    p1: np.ndarray,
    plate_size_m: tuple[float, float],
    tol: float,
) -> int | None:
    lx, ly = float(plate_size_m[0]), float(plate_size_m[1])
    if abs(float(p0[0])) <= tol and abs(float(p1[0])) <= tol:
        return FACET_CLAMPED_TAG
    if abs(float(p0[0]) - lx) <= tol and abs(float(p1[0]) - lx) <= tol:
        return FACET_FREE_X_MAX_TAG
    if abs(float(p0[1])) <= tol and abs(float(p1[1])) <= tol:
        return FACET_FREE_Y_MIN_TAG
    if abs(float(p0[1]) - ly) <= tol and abs(float(p1[1]) - ly) <= tol:
        return FACET_FREE_Y_MAX_TAG
    return None



def _build_layered_tet_solver_mesh(
    planform: _CadPlanform,
    sample_id: int,
    output_dir: Path,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
) -> Path:
    requested_solver_mesh_size_m = _resolve_solver_mesh_size_m(
        geometry_config=geometry_config,
        volume_config=volume_config,
    )
    solver_mesh_size_m = float(requested_solver_mesh_size_m)
    max_solver_vector_dofs = None if volume_config.max_solver_vector_dofs is None else int(volume_config.max_solver_vector_dofs)
    growth_factor = float(volume_config.solver_mesh_growth_factor)
    max_retries = int(volume_config.solver_mesh_limit_retries)
    estimated_q2_vector_dofs = 0
    coarsening_pass = 0

    while True:
        xy, surface_triangles, substrate_mask = _mesh_partitioned_full_plate_triangles(
            planform=planform,
            geometry_config=geometry_config,
            mesh_size_m=solver_mesh_size_m,
        )
        n_surface_points = int(xy.shape[0])
        if n_surface_points == 0:
            raise RuntimeError("The layered solver mesh has no 2D surface nodes.")

        z_levels, layer_tags = _solver_mesh_z_levels(volume_config)
        n_planes = int(z_levels.shape[0])
        plane_offsets = np.arange(n_planes, dtype=np.int64) * n_surface_points
        points = np.column_stack([
            np.tile(xy[:, 0], n_planes),
            np.tile(xy[:, 1], n_planes),
            np.repeat(z_levels, n_surface_points),
        ]).astype(np.float64)

        tetra_blocks: list[np.ndarray] = []
        tetra_tags: list[np.ndarray] = []
        sorted_surface_triangles = np.sort(np.asarray(surface_triangles, dtype=np.int64), axis=1)
        for layer_idx, cell_tag in enumerate(layer_tags.tolist()):
            tri_source = sorted_surface_triangles if int(cell_tag) == VOLUME_PIEZO_TAG else sorted_surface_triangles[substrate_mask]
            if tri_source.shape[0] == 0:
                if int(cell_tag) == VOLUME_SUBSTRATE_TAG:
                    raise RuntimeError("The substrate layer would be empty after surface classification.")
                continue
            lower = tri_source + int(plane_offsets[layer_idx])
            upper = tri_source + int(plane_offsets[layer_idx + 1])
            tet = np.vstack([
                np.column_stack([lower[:, 0], lower[:, 1], lower[:, 2], upper[:, 2]]),
                np.column_stack([lower[:, 0], lower[:, 1], upper[:, 1], upper[:, 2]]),
                np.column_stack([lower[:, 0], upper[:, 0], upper[:, 1], upper[:, 2]]),
            ]).astype(np.int64)
            signed = _tetra_signed_volumes(points, tet)
            negative = signed < 0.0
            if np.any(negative):
                tet = tet.copy()
                tmp = tet[negative, 2].copy()
                tet[negative, 2] = tet[negative, 3]
                tet[negative, 3] = tmp
                signed = _tetra_signed_volumes(points, tet)
            if np.any(signed <= 0.0):
                raise RuntimeError("The layered tetrahedral solver mesh contains non-positive tetra volumes.")
            tetra_blocks.append(tet)
            tetra_tags.append(np.full(tet.shape[0], int(cell_tag), dtype=np.int32))

        tetra_cells = np.vstack(tetra_blocks)
        tetra_cell_tags = np.concatenate(tetra_tags)
        estimated_q2_vector_dofs = _estimate_quadratic_vector_dofs(points, tetra_cells)
        if (
            max_solver_vector_dofs is not None
            and estimated_q2_vector_dofs > max_solver_vector_dofs
            and coarsening_pass < max_retries
        ):
            coarsening_pass += 1
            next_solver_mesh_size_m = solver_mesh_size_m * growth_factor
            print(
                f"Coarsening layered solver mesh for sample {int(sample_id):04d}: "
                f"estimated quadratic vector DOFs {estimated_q2_vector_dofs} exceed the "
                f"{max_solver_vector_dofs} limit; retrying with in-plane mesh size "
                f"{next_solver_mesh_size_m:.6g} m."
            )
            solver_mesh_size_m = next_solver_mesh_size_m
            continue
        break

    if max_solver_vector_dofs is not None and estimated_q2_vector_dofs > max_solver_vector_dofs:
        raise RuntimeError(
            f"Layered solver mesh for sample {int(sample_id):04d} still exceeds the quadratic-vector-DOF limit "
            f"after {max_retries} coarsening retry/retries: {estimated_q2_vector_dofs} > {max_solver_vector_dofs}."
        )

    top_triangles = np.asarray(surface_triangles + int(plane_offsets[-1]), dtype=np.int64)
    bottom_triangles = np.asarray(surface_triangles[substrate_mask][:, [0, 2, 1]] + int(plane_offsets[0]), dtype=np.int64)
    interface_triangles = np.asarray(surface_triangles + int(plane_offsets[int(volume_config.substrate_layers)]), dtype=np.int64)
    triangle_cells: list[np.ndarray] = [top_triangles, interface_triangles, bottom_triangles]
    triangle_tags: list[np.ndarray] = [
        np.full(top_triangles.shape[0], FACET_TOP_ELECTRODE_TAG, dtype=np.int32),
        np.full(interface_triangles.shape[0], FACET_BOTTOM_ELECTRODE_TAG, dtype=np.int32),
        np.full(bottom_triangles.shape[0], FACET_BOTTOM_PLATE_TAG, dtype=np.int32),
    ]

    edge_tol = _surface_classification_tolerance(geometry_config, volume_config, solver_mesh_size_m)
    for layer_idx in range(n_planes - 1):
        lower_offset = int(plane_offsets[layer_idx])
        upper_offset = int(plane_offsets[layer_idx + 1])
        source_triangles = surface_triangles if int(layer_tags[layer_idx]) == VOLUME_PIEZO_TAG else surface_triangles[substrate_mask]
        if source_triangles.shape[0] == 0:
            continue
        all_edges = np.sort(
            np.vstack([
                source_triangles[:, [0, 1]],
                source_triangles[:, [1, 2]],
                source_triangles[:, [0, 2]],
            ]),
            axis=1,
        )
        boundary_edges, counts = np.unique(all_edges, axis=0, return_counts=True)
        boundary_edges = boundary_edges[counts == 1]
        side_triangles_layer: list[np.ndarray] = []
        side_tags_layer: list[int] = []
        for edge in boundary_edges:
            tag = _classify_outer_boundary_edge(
                p0=xy[int(edge[0])],
                p1=xy[int(edge[1])],
                plate_size_m=geometry_config.plate_size_m,
                tol=edge_tol,
            )
            if tag is None:
                continue
            i0 = int(edge[0])
            i1 = int(edge[1])
            lower0 = lower_offset + i0
            lower1 = lower_offset + i1
            upper0 = upper_offset + i0
            upper1 = upper_offset + i1
            side_triangles_layer.append(np.asarray([lower0, lower1, upper1], dtype=np.int64))
            side_triangles_layer.append(np.asarray([lower0, upper1, upper0], dtype=np.int64))
            side_tags_layer.extend([tag, tag])
        if side_triangles_layer:
            triangle_cells.append(np.vstack(side_triangles_layer))
            triangle_tags.append(np.asarray(side_tags_layer, dtype=np.int32))

    tri_cells = np.vstack(triangle_cells)
    tri_tags = np.concatenate(triangle_tags)

    substrate_expected = float(planform.area_m2) * float(volume_config.substrate_thickness_m)
    piezo_expected = (
        float(geometry_config.plate_size_m[0])
        * float(geometry_config.plate_size_m[1])
        * float(volume_config.piezo_thickness_m)
    )
    cell_volumes = np.abs(_tetra_signed_volumes(points, tetra_cells))
    substrate_actual = float(np.sum(cell_volumes[tetra_cell_tags == VOLUME_SUBSTRATE_TAG]))
    piezo_actual = float(np.sum(cell_volumes[tetra_cell_tags == VOLUME_PIEZO_TAG]))
    if abs(substrate_actual - substrate_expected) > _volume_tolerance_m3(substrate_expected, geometry_config, volume_config):
        raise RuntimeError(
            f"Layered solver mesh substrate volume {substrate_actual:.8e} m^3 does not match the expected {substrate_expected:.8e} m^3."
        )
    if abs(piezo_actual - piezo_expected) > _volume_tolerance_m3(piezo_expected, geometry_config, volume_config):
        raise RuntimeError(
            f"Layered solver mesh piezo volume {piezo_actual:.8e} m^3 does not match the expected {piezo_expected:.8e} m^3."
        )

    solver_mesh_path = output_dir / f"plate3d_{int(sample_id):04d}_fenicsx.npz"
    np.savez_compressed(
        solver_mesh_path,
        points=points,
        tetra_cells=np.asarray(tetra_cells, dtype=np.int64),
        tetra_tags=np.asarray(tetra_cell_tags, dtype=np.int32),
        triangle_cells=np.asarray(tri_cells, dtype=np.int64),
        triangle_tags=np.asarray(tri_tags, dtype=np.int32),
        solver_mesh_size_m=np.asarray([solver_mesh_size_m], dtype=np.float64),
        estimated_q2_vector_dofs=np.asarray([estimated_q2_vector_dofs], dtype=np.int64),
        solver_mesh_coarsening_passes=np.asarray([coarsening_pass], dtype=np.int32),
    )
    return solver_mesh_path



def _mesh_polygons_volume_sample_gmsh_volume(
    polygons: Iterable[Polygon],
    sample_id: int,
    output_dir: Path,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
) -> _CadExportArtifacts | None:
    msh_path = output_dir / f"plate3d_{int(sample_id):04d}.msh"
    step_path = output_dir / f"plate3d_{int(sample_id):04d}.step"
    cad_report_path = output_dir / f"plate3d_{int(sample_id):04d}_cad.json"
    selection_manifest_path = output_dir / f"plate3d_{int(sample_id):04d}_ansys_face_groups.json"
    solver_mesh_path = output_dir / f"plate3d_{int(sample_id):04d}_fenicsx.npz"

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("plate3d")
    try:
        cad_reference_size_m = _resolve_cad_reference_size_m(
            geometry_config=geometry_config,
            volume_config=volume_config,
        )
        solver_mesh_size_m = _resolve_solver_mesh_size_m(
            geometry_config=geometry_config,
            volume_config=volume_config,
        )
        planform = _build_substrate_planform(
            polygons=polygons,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=cad_reference_size_m,
        )
        cad_planform = _prepare_planform_for_cad_export(
            planform=planform,
            mesh_size_m=cad_reference_size_m,
            volume_config=volume_config,
        )
        pre_export_report = _prepare_cad_occ_model(
            planform=cad_planform,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=cad_reference_size_m,
            partition_interface=True,
        )
        _annotate_current_occ_entities_for_ansys(
            report=pre_export_report,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=cad_reference_size_m,
        )

        if _export_occ_geometry_to_step(step_path) is None:
            return None

        _reload_occ_geometry_from_step(step_path)
        _set_mesh_size_on_all_points(solver_mesh_size_m)
        roundtrip_report = _validate_current_occ_model(
            stage="step_roundtrip",
            planform=cad_planform,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=cad_reference_size_m,
        )

        gmsh.model.addPhysicalGroup(3, [roundtrip_report.substrate_tag], tag=VOLUME_SUBSTRATE_TAG)
        gmsh.model.setPhysicalName(3, VOLUME_SUBSTRATE_TAG, "substrate")
        gmsh.model.addPhysicalGroup(3, [roundtrip_report.piezo_tag], tag=VOLUME_PIEZO_TAG)
        gmsh.model.setPhysicalName(3, VOLUME_PIEZO_TAG, "piezo")

        surface_groups = _classify_boundary_surfaces(
            plate_size_m=geometry_config.plate_size_m,
            total_thickness_m=volume_config.total_thickness_m,
            substrate_thickness_m=volume_config.substrate_thickness_m,
            tol=_surface_classification_tolerance(geometry_config, volume_config, cad_reference_size_m),
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
    finally:
        gmsh.finalize()

    if volume_config.write_xdmf:
        convert_volume_msh_to_xdmf(msh_path)
    converted_npz = convert_volume_msh_to_fenicsx_npz(msh_path)
    if converted_npz is None:
        raise RuntimeError(f"Could not convert {msh_path} to the FEniCSx NPZ format.")
    if converted_npz != solver_mesh_path:
        converted_npz.replace(solver_mesh_path)
    if not volume_config.write_native_msh and msh_path.exists():
        msh_path.unlink()

    _write_cad_report(
        report_path=cad_report_path,
        sample_id=sample_id,
        geometry_config=geometry_config,
        volume_config=volume_config,
        planform=cad_planform,
        pre_export_report=pre_export_report,
        roundtrip_report=roundtrip_report,
    )
    _write_ansys_face_selection_manifest(
        manifest_path=selection_manifest_path,
        sample_id=sample_id,
        geometry_config=geometry_config,
        volume_config=volume_config,
        roundtrip_report=roundtrip_report,
    )
    return _CadExportArtifacts(
        solver_mesh_path=solver_mesh_path,
        step_path=step_path,
        cad_report_path=cad_report_path,
        planform=cad_planform,
        selection_manifest_path=selection_manifest_path,
        pre_export_report=pre_export_report,
        roundtrip_report=roundtrip_report,
    )



def _mesh_polygons_volume_sample_layered_tet(
    polygons: Iterable[Polygon],
    sample_id: int,
    output_dir: Path,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
) -> _CadExportArtifacts | None:
    step_path = output_dir / f"plate3d_{int(sample_id):04d}.step"
    inspection_step_path = None
    cad_report_path = output_dir / f"plate3d_{int(sample_id):04d}_cad.json"
    selection_manifest_path = output_dir / f"plate3d_{int(sample_id):04d}_ansys_face_groups.json"

    cad_reference_size_m = _resolve_cad_reference_size_m(
        geometry_config=geometry_config,
        volume_config=volume_config,
    )
    solver_planform = _build_substrate_planform(
        polygons=polygons,
        geometry_config=geometry_config,
        volume_config=volume_config,
        mesh_size_m=cad_reference_size_m,
    )
    cad_planform = _prepare_planform_for_cad_export(
        planform=solver_planform,
        mesh_size_m=cad_reference_size_m,
        volume_config=volume_config,
    )
    primary_strategy = _ansys_step_strategy(volume_config)
    primary_partition_interface = primary_strategy == "partitioned_interface"
    pre_export_report, roundtrip_report = _build_step_export_variant(
        step_path=step_path,
        planform=cad_planform,
        geometry_config=geometry_config,
        volume_config=volume_config,
        mesh_size_m=cad_reference_size_m,
        partition_interface=primary_partition_interface,
        require_single_piezo_bottom_face=not primary_partition_interface,
    )
    inspection_pre_export_report = None
    inspection_roundtrip_report = None
    if bool(volume_config.export_inspection_single_face_step) and primary_partition_interface:
        inspection_step_path = output_dir / f"plate3d_{int(sample_id):04d}_single_face_probe.step"
        inspection_pre_export_report, inspection_roundtrip_report = _build_step_export_variant(
            step_path=inspection_step_path,
            planform=cad_planform,
            geometry_config=geometry_config,
            volume_config=volume_config,
            mesh_size_m=cad_reference_size_m,
            partition_interface=False,
            require_single_piezo_bottom_face=True,
        )

    solver_mesh_path = _build_layered_tet_solver_mesh(
        planform=solver_planform,
        sample_id=sample_id,
        output_dir=output_dir,
        geometry_config=geometry_config,
        volume_config=volume_config,
    )

    _write_cad_report(
        report_path=cad_report_path,
        sample_id=sample_id,
        geometry_config=geometry_config,
        volume_config=volume_config,
        planform=cad_planform,
        pre_export_report=pre_export_report,
        roundtrip_report=roundtrip_report,
        inspection_step_path=inspection_step_path,
        inspection_pre_export_report=inspection_pre_export_report,
        inspection_roundtrip_report=inspection_roundtrip_report,
    )
    _write_ansys_face_selection_manifest(
        manifest_path=selection_manifest_path,
        sample_id=sample_id,
        geometry_config=geometry_config,
        volume_config=volume_config,
        roundtrip_report=roundtrip_report,
    )
    return _CadExportArtifacts(
        solver_mesh_path=solver_mesh_path,
        step_path=step_path,
        cad_report_path=cad_report_path,
        planform=cad_planform,
        selection_manifest_path=selection_manifest_path,
        pre_export_report=pre_export_report,
        roundtrip_report=roundtrip_report,
        inspection_step_path=inspection_step_path,
        inspection_pre_export_report=inspection_pre_export_report,
        inspection_roundtrip_report=inspection_roundtrip_report,
    )



def _mesh_polygons_volume_sample(
    polygons: Iterable[Polygon],
    sample_id: int,
    output_dir: str | Path,
    geometry_config: GeometryBuildConfig,
    volume_config: VolumeMeshConfig,
) -> _CadExportArtifacts | None:
    polygons = [poly for geom in polygons for poly in _iter_polygons(geom)]
    if not polygons:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = str(volume_config.solver_mesh_backend).strip().lower()
    if backend == "gmsh_volume":
        return _mesh_polygons_volume_sample_gmsh_volume(
            polygons=polygons,
            sample_id=sample_id,
            output_dir=output_dir,
            geometry_config=geometry_config,
            volume_config=volume_config,
        )
    return _mesh_polygons_volume_sample_layered_tet(
        polygons=polygons,
        sample_id=sample_id,
        output_dir=output_dir,
        geometry_config=geometry_config,
        volume_config=volume_config,
    )



def mesh_tiled_plate_volume_sample(
    grf: np.ndarray,
    threshold: float,
    sample_id: int,
    output_dir: str | Path,
    geometry_config: GeometryBuildConfig | None = None,
    volume_config: VolumeMeshConfig | None = None,
) -> Path | None:
    if geometry_config is None:
        raise ValueError(
            "geometry_config must be provided explicitly so volume meshes are built at the intended physical size."
        )
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
        enforce_connectivity=False,
        connectivity_bridge_width_m=geometry_config.connectivity_bridge_width_m,
    )
    artifacts = _mesh_polygons_volume_sample(
        polygons=tiled_polygons,
        sample_id=sample_id,
        output_dir=output_dir,
        geometry_config=geometry_config,
        volume_config=volume_config,
    )
    if artifacts is None:
        return None
    return artifacts.solver_mesh_path


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
