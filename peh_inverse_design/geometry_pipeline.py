from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import gmsh
import meshio
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, box
from shapely.ops import polygonize, unary_union
from skimage.measure import find_contours
from skimage.transform import resize

ArrayF64 = NDArray[np.float64]
ArrayBool = NDArray[np.bool_]

DEFAULT_CELL_SIZE_M = (0.10, 0.10)
DEFAULT_TILE_COUNTS = (10, 10)
DEFAULT_SDF_ML_RESOLUTION = (128, 128)
DEFAULT_MESH_SIZE_RELATIVE_TO_CELL = 0.04
DEFAULT_CONTOUR_DECIMATE_TOL = 0.005
DEFAULT_SPLIT_SEED = 42

_CELL_CORNERS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class GeometryBuildConfig:
    cell_size_m: tuple[float, float] = DEFAULT_CELL_SIZE_M
    tile_counts: tuple[int, int] = DEFAULT_TILE_COUNTS
    sdf_ml_resolution: tuple[int, int] = DEFAULT_SDF_ML_RESOLUTION
    mesh_size_relative_to_cell: float = DEFAULT_MESH_SIZE_RELATIVE_TO_CELL
    contour_decimate_tol: float = DEFAULT_CONTOUR_DECIMATE_TOL
    split_seed: int = DEFAULT_SPLIT_SEED
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)

    @property
    def mesh_size_m(self) -> float:
        return self.mesh_size_relative_to_cell * float(self.cell_size_m[0])

    @property
    def plate_size_m(self) -> tuple[float, float]:
        return (
            float(self.cell_size_m[0]) * int(self.tile_counts[0]),
            float(self.cell_size_m[1]) * int(self.tile_counts[1]),
        )


def _near_any_corner(point: NDArray[np.float64], tol: float = 0.025) -> bool:
    for corner in _CELL_CORNERS:
        if np.linalg.norm(point - corner) < tol:
            return True
    return False


def extract_periodic_contours(
    grf: ArrayF64,
    threshold: float,
    resolution: int | None = None,
) -> list[ArrayF64]:
    """Extract unit-cell contours with periodic wrapping at the cell edges."""
    if resolution is None:
        resolution = int(grf.shape[0])

    grf_padded = np.pad(grf, ((0, 1), (0, 1)), mode="wrap")
    raw_contours = find_contours(grf_padded, level=float(threshold), positive_orientation="high")
    contours = [np.asarray(contour, dtype=np.float64) / float(resolution) for contour in raw_contours]

    merged = True
    result: list[ArrayF64 | None] = [contour.copy() for contour in contours]
    while merged:
        merged = False
        for i, contour_i in enumerate(result):
            if contour_i is None or not _near_any_corner(contour_i[-1]):
                continue
            for j, contour_j in enumerate(result):
                if i == j or contour_j is None:
                    continue
                if _near_any_corner(contour_j[0]):
                    result[i] = np.concatenate([contour_i, contour_j], axis=0)
                    result[j] = None
                    merged = True
                    break
            if merged:
                break

    filtered: list[ArrayF64] = []
    for contour in result:
        if contour is None:
            continue
        if len(contour) <= 4 and _near_any_corner(contour[0]) and _near_any_corner(contour[-1]):
            continue
        filtered.append(contour)
    return filtered


def _build_periodic_interpolator(grf: ArrayF64) -> RegularGridInterpolator:
    x_grf = np.linspace(0.0, 1.0, grf.shape[0], endpoint=False)
    grf_padded = np.pad(grf, ((0, 1), (0, 1)), mode="wrap")
    x_padded = np.append(x_grf, 1.0)
    return RegularGridInterpolator(
        (x_padded, x_padded),
        grf_padded,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


def build_unit_cell_solid_polygons(
    grf: ArrayF64,
    threshold: float,
    resolution: int | None = None,
    decimate_tol: float = DEFAULT_CONTOUR_DECIMATE_TOL,
) -> list[Polygon]:
    """Convert a unit-cell GRF sample into solid-region polygons on [0, 1]^2."""
    if resolution is None:
        resolution = int(grf.shape[0])

    contours = extract_periodic_contours(grf=grf, threshold=threshold, resolution=resolution)
    interpolator = _build_periodic_interpolator(grf)
    unit_cell = box(0.0, 0.0, 1.0, 1.0)

    def is_solid(point: tuple[float, float]) -> bool:
        value = float(interpolator(np.asarray(point, dtype=np.float64).reshape(1, -1))[0])
        return value >= float(threshold)

    simplified_lines: list[LineString] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        line = LineString(contour)
        simplified = line.simplify(decimate_tol, preserve_topology=True)
        simplified_lines.append(simplified)

    if not simplified_lines:
        return [unit_cell] if is_solid((0.5, 0.5)) else []

    merged_lines = unary_union([unit_cell.boundary] + simplified_lines)
    polygons = list(polygonize(merged_lines))
    if not polygons:
        return [unit_cell] if is_solid((0.5, 0.5)) else []

    solid_polygons: list[Polygon] = []
    for polygon in polygons:
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area < 1e-6:
            continue
        point = polygon.representative_point()
        if is_solid((point.x, point.y)):
            solid_polygons.extend(_iter_polygons(polygon))
    return solid_polygons


def _iter_polygons(geometry: Polygon | MultiPolygon | GeometryCollection) -> Iterator[Polygon]:
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            yield from _iter_polygons(polygon)
        return
    if isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            if isinstance(geom, (Polygon, MultiPolygon, GeometryCollection)):
                yield from _iter_polygons(geom)


def tile_unit_cell_polygons(
    unit_cell_polygons: Iterable[Polygon],
    cell_size_m: tuple[float, float] = DEFAULT_CELL_SIZE_M,
    tile_counts: tuple[int, int] = DEFAULT_TILE_COUNTS,
) -> list[Polygon]:
    """Scale one unit cell to physical units and tile it across the finite plate."""
    scaled_polygons = [
        affinity.scale(
            polygon,
            xfact=float(cell_size_m[0]),
            yfact=float(cell_size_m[1]),
            origin=(0.0, 0.0),
        )
        for polygon in unit_cell_polygons
    ]
    tiled: list[Polygon] = []
    for ix in range(int(tile_counts[0])):
        for iy in range(int(tile_counts[1])):
            xoff = ix * float(cell_size_m[0])
            yoff = iy * float(cell_size_m[1])
            for polygon in scaled_polygons:
                tiled.append(affinity.translate(polygon, xoff=xoff, yoff=yoff))

    merged = unary_union(tiled)
    return list(_iter_polygons(merged))


def _classify_outer_plate_edge(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    plate_size_m: tuple[float, float],
    tol: float,
) -> str | None:
    plate_x, plate_y = float(plate_size_m[0]), float(plate_size_m[1])
    if abs(start[0]) < tol and abs(end[0]) < tol:
        return "clamped"
    if abs(start[0] - plate_x) < tol and abs(end[0] - plate_x) < tol:
        return "free_x_max"
    if abs(start[1]) < tol and abs(end[1]) < tol:
        return "free_y_min"
    if abs(start[1] - plate_y) < tol and abs(end[1] - plate_y) < tol:
        return "free_y_max"
    return None


def _mesh_polygons_with_boundary_groups(
    polygons: Iterable[Polygon],
    msh_path: Path,
    mesh_size_m: float,
    plate_size_m: tuple[float, float],
) -> Path | None:
    polygons = list(polygons)
    if not polygons:
        return None

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("tiled_plate")

    try:
        point_cache: dict[tuple[float, float], int] = {}
        boundary_line_tags: dict[str, list[int]] = {
            "clamped": [],
            "free_x_max": [],
            "free_y_min": [],
            "free_y_max": [],
        }
        surface_tags: list[int] = []
        tol = max(1e-9, mesh_size_m * 1e-3)

        def add_point(x: float, y: float) -> int:
            key = (round(x, 9), round(y, 9))
            if key not in point_cache:
                point_cache[key] = gmsh.model.geo.addPoint(x, y, 0.0, mesh_size_m)
            return point_cache[key]

        for polygon in polygons:
            for geom in _iter_polygons(polygon):
                exterior_coords = np.asarray(geom.exterior.coords, dtype=np.float64)
                exterior_point_tags = [
                    add_point(float(x), float(y))
                    for x, y in exterior_coords[:-1]
                ]
                if len(exterior_point_tags) < 3:
                    continue

                exterior_line_tags: list[int] = []
                n_exterior = len(exterior_point_tags)
                for idx in range(n_exterior):
                    start_tag = exterior_point_tags[idx]
                    end_tag = exterior_point_tags[(idx + 1) % n_exterior]
                    if start_tag == end_tag:
                        continue

                    line_tag = gmsh.model.geo.addLine(start_tag, end_tag)
                    exterior_line_tags.append(line_tag)

                    start = exterior_coords[idx]
                    end = exterior_coords[(idx + 1) % n_exterior]
                    group_name = _classify_outer_plate_edge(start, end, plate_size_m, tol)
                    if group_name is not None:
                        boundary_line_tags[group_name].append(line_tag)

                if len(exterior_line_tags) < 3:
                    continue

                try:
                    outer_loop = gmsh.model.geo.addCurveLoop(exterior_line_tags)
                except Exception:
                    continue

                hole_loops: list[int] = []
                for interior in geom.interiors:
                    interior_coords = np.asarray(interior.coords, dtype=np.float64)
                    interior_point_tags = [
                        add_point(float(x), float(y))
                        for x, y in interior_coords[:-1]
                    ]
                    interior_line_tags: list[int] = []
                    n_interior = len(interior_point_tags)
                    for idx in range(n_interior):
                        start_tag = interior_point_tags[idx]
                        end_tag = interior_point_tags[(idx + 1) % n_interior]
                        if start_tag == end_tag:
                            continue
                        interior_line_tags.append(gmsh.model.geo.addLine(start_tag, end_tag))
                    if len(interior_line_tags) >= 3:
                        try:
                            hole_loops.append(gmsh.model.geo.addCurveLoop(interior_line_tags))
                        except Exception:
                            pass

                try:
                    surface_tags.append(gmsh.model.geo.addPlaneSurface([outer_loop] + hole_loops))
                except Exception:
                    continue

        if not surface_tags:
            return None

        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, surface_tags, tag=1)
        gmsh.model.setPhysicalName(2, 1, "solid")

        physical_line_tags = {
            "clamped": 101,
            "free_x_max": 102,
            "free_y_min": 103,
            "free_y_max": 104,
        }
        for name, tag in physical_line_tags.items():
            unique_lines = sorted(set(boundary_line_tags[name]))
            if not unique_lines:
                continue
            gmsh.model.addPhysicalGroup(1, unique_lines, tag=tag)
            gmsh.model.setPhysicalName(1, tag, name)

        gmsh.model.mesh.generate(2)
        msh_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(msh_path))
        return msh_path
    finally:
        gmsh.finalize()


def convert_msh_to_xdmf(msh_path: str | Path) -> Path | None:
    msh_path = Path(msh_path)
    try:
        mesh = meshio.read(msh_path)
    except Exception:
        return None

    triangle_cells = [cells for cells in mesh.cells if cells.type == "triangle"]
    if not triangle_cells:
        return None

    points_2d = mesh.points[:, :2]
    xdmf_path = msh_path.with_suffix(".xdmf")
    triangle_mesh = meshio.Mesh(points=points_2d, cells=triangle_cells)
    meshio.xdmf.write(xdmf_path, triangle_mesh)
    return xdmf_path


def mesh_tiled_plate_sample(
    grf: ArrayF64,
    threshold: float,
    sample_id: int,
    output_dir: str | Path,
    config: GeometryBuildConfig,
) -> tuple[Path | None, Path | None]:
    unit_cell_polygons = build_unit_cell_solid_polygons(
        grf=grf,
        threshold=threshold,
        resolution=int(grf.shape[0]),
        decimate_tol=config.contour_decimate_tol,
    )
    tiled_polygons = tile_unit_cell_polygons(
        unit_cell_polygons=unit_cell_polygons,
        cell_size_m=config.cell_size_m,
        tile_counts=config.tile_counts,
    )

    output_dir = Path(output_dir)
    msh_path = output_dir / f"plate_{sample_id:04d}.msh"
    msh_path = _mesh_polygons_with_boundary_groups(
        polygons=tiled_polygons,
        msh_path=msh_path,
        mesh_size_m=config.mesh_size_m,
        plate_size_m=config.plate_size_m,
    )
    if msh_path is None:
        return None, None
    xdmf_path = convert_msh_to_xdmf(msh_path)
    return msh_path, xdmf_path


def _resample_sdf_batch(
    sdf_batch: NDArray[np.floating],
    target_shape: tuple[int, int],
) -> NDArray[np.float32]:
    n_samples = int(sdf_batch.shape[0])
    out = np.empty((n_samples, target_shape[0], target_shape[1]), dtype=np.float32)
    for idx in range(n_samples):
        out[idx] = resize(
            sdf_batch[idx],
            output_shape=target_shape,
            order=1,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.float32)
    return out


def _assign_splits(
    n_samples: int,
    ratios: tuple[float, float, float],
    seed: int,
) -> list[str]:
    if n_samples <= 0:
        return []
    ratios_arr = np.asarray(ratios, dtype=np.float64)
    ratios_arr = ratios_arr / ratios_arr.sum()

    indices = np.arange(n_samples)
    order = np.random.default_rng(seed).permutation(indices)

    n_train = int(np.floor(ratios_arr[0] * n_samples))
    n_val = int(np.floor(ratios_arr[1] * n_samples))
    n_test = n_samples - n_train - n_val

    split_names = np.empty(n_samples, dtype=object)
    split_names[order[:n_train]] = "train"
    split_names[order[n_train:n_train + n_val]] = "val"
    split_names[order[n_train + n_val:n_train + n_val + n_test]] = "test"
    return [str(name) for name in split_names.tolist()]


def _write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "mesh_path",
        "xdmf_path",
        "geometry_npz_key",
        "response_npz_key",
        "mesh_ok",
        "fem_ok",
        "split",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_geometry_dataset(
    unit_cell_dataset_path: str | Path,
    geometry_output_path: str | Path,
    manifest_path: str | Path,
    mesh_output_dir: str | Path,
    config: GeometryBuildConfig | None = None,
    limit: int | None = None,
    build_meshes: bool = True,
) -> dict[str, NDArray]:
    """Create the ML geometry dataset and the FEM mesh manifest."""
    config = config or GeometryBuildConfig()

    unit_cell_dataset_path = Path(unit_cell_dataset_path)
    geometry_output_path = Path(geometry_output_path)
    manifest_path = Path(manifest_path)
    mesh_output_dir = Path(mesh_output_dir)

    source = np.load(unit_cell_dataset_path, allow_pickle=True)
    n_total = int(source["grf"].shape[0])
    n_samples = n_total if limit is None else min(int(limit), n_total)
    indices = np.arange(n_samples, dtype=np.int32)

    sample_ids = (
        source["sample_id"][:n_samples].astype(np.int32)
        if "sample_id" in source.files
        else indices
    )
    grf = source["grf"][:n_samples].astype(np.float32)
    binary = source["binary"][:n_samples].astype(bool)
    sdf = source["sdf"][:n_samples].astype(np.float32)
    threshold = source["threshold"][:n_samples].astype(np.float32)
    volume_fraction = source["volume_fraction"][:n_samples].astype(np.float32)
    sdf_ml = _resample_sdf_batch(sdf, config.sdf_ml_resolution)

    geometry_dataset = {
        "sample_id": sample_ids,
        "grf": grf,
        "binary": binary,
        "sdf": sdf,
        "sdf_ml": sdf_ml,
        "volume_fraction": volume_fraction,
        "threshold": threshold,
        "cell_size_m": np.tile(np.asarray(config.cell_size_m, dtype=np.float32), (n_samples, 1)),
        "tile_counts": np.tile(np.asarray(config.tile_counts, dtype=np.int32), (n_samples, 1)),
    }
    geometry_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(geometry_output_path, **geometry_dataset)

    split_names = _assign_splits(
        n_samples=n_samples,
        ratios=config.split_ratios,
        seed=config.split_seed,
    )

    manifest_rows: list[dict[str, str]] = []
    for local_idx, sample_id in enumerate(sample_ids.tolist()):
        mesh_path = ""
        xdmf_path = ""
        mesh_ok = 0
        if build_meshes:
            msh_path, xdmf = mesh_tiled_plate_sample(
                grf=grf[local_idx].astype(np.float64),
                threshold=float(threshold[local_idx]),
                sample_id=int(sample_id),
                output_dir=mesh_output_dir,
                config=config,
            )
            if msh_path is not None:
                mesh_ok = 1
                mesh_path = str(msh_path)
            if xdmf is not None:
                xdmf_path = str(xdmf)

        manifest_rows.append(
            {
                "sample_id": str(int(sample_id)),
                "mesh_path": mesh_path,
                "xdmf_path": xdmf_path,
                "geometry_npz_key": str(int(sample_id)),
                "response_npz_key": "",
                "mesh_ok": str(mesh_ok),
                "fem_ok": "0",
                "split": split_names[local_idx],
            }
        )

    _write_manifest(manifest_rows, manifest_path)
    return geometry_dataset
