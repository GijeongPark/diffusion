from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peh_inverse_design.geometry_pipeline import GeometryBuildConfig
    from peh_inverse_design.volume_mesh import (
        VolumeMeshConfig,
        convert_volume_msh_to_fenicsx_npz,
        convert_volume_msh_to_xdmf,
        mesh_tiled_plate_volume_sample,
    )
else:
    from .geometry_pipeline import GeometryBuildConfig
    from .volume_mesh import (
        VolumeMeshConfig,
        convert_volume_msh_to_fenicsx_npz,
        convert_volume_msh_to_xdmf,
        mesh_tiled_plate_volume_sample,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 3D layered substrate+piezo meshes for FEniCSx.",
    )
    parser.add_argument(
        "--unit-cell-npz",
        default="data/dataset_100.npz",
        help="Input unit-cell dataset produced by periodic_grf_sdf.ipynb.",
    )
    parser.add_argument(
        "--mesh-dir",
        default="meshes/volumes",
        help="Output directory for 3D .msh files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples.",
    )
    parser.add_argument(
        "--substrate-thickness",
        type=float,
        default=1.0e-3,
        help="Substrate thickness in meters.",
    )
    parser.add_argument(
        "--piezo-thickness",
        type=float,
        default=2.667e-4,
        help="Piezo thickness in meters.",
    )
    parser.add_argument(
        "--mesh-size-scale",
        type=float,
        default=0.06,
        help="Target in-plane element size as a fraction of one cell size.",
    )
    args = parser.parse_args()

    data = np.load(args.unit_cell_npz, allow_pickle=True)
    n_total = int(data["grf"].shape[0])
    n_samples = n_total if args.limit is None else min(int(args.limit), n_total)

    geometry_config = GeometryBuildConfig()
    volume_config = VolumeMeshConfig(
        substrate_thickness_m=float(args.substrate_thickness),
        piezo_thickness_m=float(args.piezo_thickness),
        mesh_size_relative_to_cell=float(args.mesh_size_scale),
    )

    mesh_dir = Path(args.mesh_dir)
    ok = 0
    fail = 0
    for idx in range(n_samples):
        path = mesh_tiled_plate_volume_sample(
            grf=data["grf"][idx],
            threshold=float(data["threshold"][idx]),
            sample_id=idx,
            output_dir=mesh_dir,
            geometry_config=geometry_config,
            volume_config=volume_config,
        )
        if path is None:
            fail += 1
        else:
            convert_volume_msh_to_xdmf(path)
            convert_volume_msh_to_fenicsx_npz(path)
            ok += 1
        print(f"[{idx + 1:4d}/{n_samples}] ok={ok} fail={fail}")

    print(f"Saved {ok} 3D meshes to {mesh_dir}")


if __name__ == "__main__":
    main()
