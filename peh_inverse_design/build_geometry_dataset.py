from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peh_inverse_design.geometry_pipeline import GeometryBuildConfig, build_geometry_dataset
else:
    from .geometry_pipeline import GeometryBuildConfig, build_geometry_dataset


def _explicit_pair(
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


def _explicit_counts(
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
        values = np.asarray(arr, dtype=np.float64)
        first = values[0]
        if not np.allclose(values, first, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{key} varies per sample; pass explicit CLI values instead of relying on metadata.")
        return float(first[0]), float(first[1])
    raise ValueError(f"{key} must have shape (2,) or (N, 2); got {arr.shape}.")


def _metadata_counts(data: np.lib.npyio.NpzFile, key: str) -> tuple[int, int] | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.ndim == 1 and arr.shape[0] == 2:
        return int(arr[0]), int(arr[1])
    if arr.ndim >= 2 and arr.shape[1] == 2:
        values = np.asarray(arr, dtype=np.int64)
        first = values[0]
        if not np.array_equal(values, np.broadcast_to(first, values.shape)):
            raise ValueError(f"{key} varies per sample; pass explicit CLI values instead of relying on metadata.")
        return int(first[0]), int(first[1])
    raise ValueError(f"{key} must have shape (2,) or (N, 2); got {arr.shape}.")


def _resolve_geometry_config(args: argparse.Namespace) -> GeometryBuildConfig:
    data = np.load(args.unit_cell_npz, allow_pickle=True)
    cell_size_m = _explicit_pair(args.cell_size_x_m, args.cell_size_y_m, "--cell-size-x-m", "--cell-size-y-m")
    tile_counts = _explicit_counts(args.tile_count_x, args.tile_count_y, "--tile-count-x", "--tile-count-y")

    if cell_size_m is None:
        cell_size_m = _metadata_pair(data, "cell_size_m")
    if tile_counts is None:
        tile_counts = _metadata_counts(data, "tile_counts")

    if cell_size_m is None or tile_counts is None:
        raise ValueError(
            "Physical plate sizing is not available. Pass --cell-size-x-m/--cell-size-y-m and "
            "--tile-count-x/--tile-count-y, or provide cell_size_m/tile_counts metadata in the input NPZ."
        )

    return GeometryBuildConfig(
        cell_size_m=cell_size_m,
        tile_counts=tile_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build geometry_dataset.npz and full-plate meshes from unit-cell samples.",
    )
    parser.add_argument(
        "--unit-cell-npz",
        default="data/dataset_100.npz",
        help="Input unit-cell dataset produced by periodic_grf_sdf.ipynb.",
    )
    parser.add_argument(
        "--geometry-output",
        default="data/geometry_dataset.npz",
        help="Output NPZ for ML-ready geometry tensors.",
    )
    parser.add_argument(
        "--manifest",
        default="data/samples.csv",
        help="Output CSV manifest for geometry/FEM bookkeeping.",
    )
    parser.add_argument(
        "--mesh-dir",
        default="meshes/plates",
        help="Directory for full-plate .msh/.xdmf files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples.",
    )
    parser.add_argument(
        "--skip-meshes",
        action="store_true",
        help="Only build geometry_dataset.npz and the manifest.",
    )
    parser.add_argument("--cell-size-x-m", type=float, default=None, help="Unit-cell size in the x direction, in meters.")
    parser.add_argument("--cell-size-y-m", type=float, default=None, help="Unit-cell size in the y direction, in meters.")
    parser.add_argument("--tile-count-x", type=int, default=None, help="Number of tiled unit cells along x.")
    parser.add_argument("--tile-count-y", type=int, default=None, help="Number of tiled unit cells along y.")
    args = parser.parse_args()

    config = _resolve_geometry_config(args)
    build_geometry_dataset(
        unit_cell_dataset_path=args.unit_cell_npz,
        geometry_output_path=args.geometry_output,
        manifest_path=args.manifest,
        mesh_output_dir=args.mesh_dir,
        config=config,
        limit=args.limit,
        build_meshes=not args.skip_meshes,
    )
    print(f"Saved geometry dataset to {args.geometry_output}")
    print(f"Saved manifest to {args.manifest}")
    if not args.skip_meshes:
        print(f"Saved meshes to {args.mesh_dir}")


if __name__ == "__main__":
    main()
