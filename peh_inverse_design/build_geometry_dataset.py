from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peh_inverse_design.geometry_pipeline import GeometryBuildConfig, build_geometry_dataset
else:
    from .geometry_pipeline import GeometryBuildConfig, build_geometry_dataset


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
    args = parser.parse_args()

    config = GeometryBuildConfig()
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
