from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TOP_ELECTRODE_TAG = 105


def _infer_sample_ids(response_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in sorted(response_dir.glob("sample_*_response.npz")):
        matches = re.findall(r"(\d+)", path.stem)
        if matches:
            ids.append(int(matches[-1]))
    return ids


def _load_dataset_row(dataset_path: Path, sample_id: int) -> dict[str, np.ndarray]:
    data = np.load(dataset_path, allow_pickle=True)
    return {
        "binary": np.asarray(data["binary"][sample_id], dtype=bool),
        "sdf": np.asarray(data["sdf"][sample_id], dtype=np.float64),
        "threshold": np.asarray(data["threshold"][sample_id]),
        "volume_fraction": np.asarray(data["volume_fraction"][sample_id]),
    }


def _load_mesh(mesh_npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(mesh_npz_path)
    return {
        "points": np.asarray(data["points"], dtype=np.float64),
        "tetra_cells": np.asarray(data["tetra_cells"], dtype=np.int64),
        "tetra_tags": np.asarray(data["tetra_tags"], dtype=np.int32),
        "triangle_cells": np.asarray(data["triangle_cells"], dtype=np.int32),
        "triangle_tags": np.asarray(data["triangle_tags"], dtype=np.int32),
    }


def _load_response(response_path: Path) -> dict[str, np.ndarray]:
    data = np.load(response_path)
    return {
        "sample_id": np.asarray(data["sample_id"]),
        "f_peak_hz": np.asarray(data["f_peak_hz"]),
        "freq_hz": np.asarray(data["freq_hz"], dtype=np.float64),
        "voltage_mag": np.asarray(data["voltage_mag"], dtype=np.float64),
        "quality_flag": np.asarray(data["quality_flag"]),
    }


def _load_modal(modal_path: Path | None) -> dict[str, np.ndarray] | None:
    if modal_path is None or not modal_path.exists():
        return None
    data = np.load(modal_path)
    return {key: np.asarray(data[key]) for key in data.files}


def _plot_unit_cell(ax: plt.Axes, binary: np.ndarray) -> None:
    ax.imshow(binary.T, origin="lower", extent=[0, 1, 0, 1], cmap="gray_r")
    ax.set_title("Unit Cell Geometry")
    ax.set_xlabel("x / a")
    ax.set_ylabel("y / a")
    ax.set_aspect("equal")


def _plot_tiled_geometry(ax: plt.Axes, binary: np.ndarray, repeat: tuple[int, int] = (10, 10)) -> None:
    tiled = np.tile(binary, repeat)
    ax.imshow(tiled.T, origin="lower", extent=[0, repeat[0], 0, repeat[1]], cmap="gray_r")
    ax.set_title("Repeated Plate Geometry")
    ax.set_xlabel("cell index x")
    ax.set_ylabel("cell index y")
    ax.set_aspect("equal")


def _extract_top_surface_mesh(mesh: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    points = mesh["points"]
    tri = mesh["triangle_cells"]
    tri_tags = mesh["triangle_tags"]
    top_tri = tri[tri_tags == TOP_ELECTRODE_TAG]
    if top_tri.shape[0] == 0:
        top_tri = tri
    used_nodes = np.unique(top_tri.reshape(-1))
    local_points = points[used_nodes]
    node_map = np.full(points.shape[0], -1, dtype=np.int64)
    node_map[used_nodes] = np.arange(used_nodes.shape[0], dtype=np.int64)
    local_tri = node_map[top_tri]
    return local_points, local_tri


def _add_mesh_zoom_inset(
    parent_ax: plt.Axes,
    points: np.ndarray,
    triangles: np.ndarray,
    strain: np.ndarray | None,
) -> None:
    span_x = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    span_y = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    center = np.array([np.min(points[:, 0]) + 0.60 * span_x, np.min(points[:, 1]) + 0.60 * span_y], dtype=np.float64)
    half_window = 0.08 * min(span_x, span_y)

    tri_points = points[triangles]
    centroids = np.mean(tri_points[:, :, :2], axis=1)
    mask = (
        (np.abs(centroids[:, 0] - center[0]) <= half_window)
        & (np.abs(centroids[:, 1] - center[1]) <= half_window)
    )
    zoom_triangles = triangles[mask]
    if zoom_triangles.shape[0] == 0:
        return

    used_nodes = np.unique(zoom_triangles.reshape(-1))
    local_points = points[used_nodes]
    node_map = np.full(points.shape[0], -1, dtype=np.int64)
    node_map[used_nodes] = np.arange(used_nodes.shape[0], dtype=np.int64)
    local_triangles = node_map[zoom_triangles]
    triangulation = mtri.Triangulation(local_points[:, 0], local_points[:, 1], local_triangles)

    inset = parent_ax.inset_axes([0.60, 0.04, 0.36, 0.36])
    if strain is not None and strain.shape[0] == triangles.shape[0]:
        zoom_strain = strain[mask]
        vmax = float(np.percentile(zoom_strain, 99.0))
        if vmax <= 0.0:
            vmax = float(np.max(zoom_strain))
        if vmax <= 0.0:
            vmax = 1.0
        inset.tripcolor(
            triangulation,
            facecolors=zoom_strain,
            shading="flat",
            cmap="inferno",
            norm=mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax),
        )
    else:
        inset.tripcolor(
            triangulation,
            facecolors=np.full(local_triangles.shape[0], 1.0, dtype=np.float64),
            shading="flat",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )
    inset.triplot(triangulation, color=(0.95, 0.95, 0.95, 0.70), linewidth=0.22)
    inset.set_xlim(center[0] - half_window, center[0] + half_window)
    inset.set_ylim(center[1] - half_window, center[1] + half_window)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("mesh detail", fontsize=8)
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(1.0)


def _plot_surface_mesh_or_strain(
    fig: plt.Figure,
    ax: plt.Axes,
    mesh: dict[str, np.ndarray],
    modal: dict[str, np.ndarray] | None,
) -> float:
    points, triangles = _extract_top_surface_mesh(mesh)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    strain = None
    if modal is not None and "top_surface_strain_eqv" in modal:
        candidate = np.asarray(modal["top_surface_strain_eqv"], dtype=np.float64).reshape(-1)
        if candidate.shape[0] == triangles.shape[0]:
            strain = candidate

    if strain is not None and strain.size > 0:
        vmax = float(np.percentile(strain, 99.5))
        if vmax <= 0.0:
            vmax = float(np.max(strain))
        if vmax <= 0.0:
            vmax = 1.0
        collection = ax.tripcolor(
            triangulation,
            facecolors=strain,
            shading="flat",
            cmap="inferno",
            norm=mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax),
        )
        ax.triplot(triangulation, color=(1.0, 1.0, 1.0, 0.16), linewidth=0.10)
        f_field = float(np.asarray(modal["field_frequency_hz"], dtype=np.float64)) if "field_frequency_hz" in modal else np.nan
        title = "Top-Surface Equivalent Strain"
        if np.isfinite(f_field):
            title += f" @ {f_field:.3f} Hz"
        ax.set_title(title)
        cbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\varepsilon_{eq}$")
        strain_max = float(np.max(strain))
        _add_mesh_zoom_inset(ax, points, triangles, strain)
    else:
        collection = ax.tripcolor(
            triangulation,
            facecolors=np.full(triangles.shape[0], 1.0, dtype=np.float64),
            shading="flat",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )
        ax.triplot(triangulation, color=(0.15, 0.25, 0.55, 0.45), linewidth=0.12)
        collection.set_alpha(0.95)
        ax.set_title("Top Surface FEM Mesh")
        strain_max = float("nan")
        _add_mesh_zoom_inset(ax, points, triangles, None)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0.0, float(np.max(points[:, 0])))
    ax.set_ylim(0.0, float(np.max(points[:, 1])))
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    return strain_max


def _plot_frf(ax: plt.Axes, response: dict[str, np.ndarray], modal: dict[str, np.ndarray] | None) -> None:
    f_peak = float(response["f_peak_hz"])
    freq_ratio = response["freq_hz"] / f_peak
    ax.plot(freq_ratio, response["voltage_mag"], color="tab:red", lw=1.8)
    ax.axvline(1.0, color="black", lw=1.0, ls="--", alpha=0.6)

    if modal is not None and "eigenfreq_hz" in modal:
        mode_ratios = np.asarray(modal["eigenfreq_hz"], dtype=np.float64) / f_peak
        for mode_idx, ratio in enumerate(mode_ratios[:6], start=1):
            if 0.9 <= ratio <= 1.1:
                ax.axvline(ratio, color="tab:blue", lw=0.8, ls=":", alpha=0.7)
                ax.text(ratio, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.0, f"m{mode_idx}",
                        color="tab:blue", fontsize=7, ha="center", va="top")

    ax.set_title("Voltage FRF")
    ax.set_xlabel(r"$f / f_{peak}$")
    ax.set_ylabel(r"$|V(f)|$")
    ax.grid(alpha=0.25)


def _plot_stats(ax: plt.Axes, sample_id: int, dataset_row: dict[str, np.ndarray], mesh: dict[str, np.ndarray],
                response: dict[str, np.ndarray], modal: dict[str, np.ndarray] | None, strain_max: float) -> dict[str, float]:
    n_nodes = int(mesh["points"].shape[0])
    n_tetra = int(mesh["tetra_cells"].shape[0])
    f_peak = float(response["f_peak_hz"])
    vmax = float(np.max(response["voltage_mag"]))
    vfrac = float(dataset_row["volume_fraction"])
    n_modes = int(len(modal["eigenfreq_hz"])) if modal is not None and "eigenfreq_hz" in modal else 0

    stats = {
        "sample_id": float(sample_id),
        "volume_fraction": vfrac,
        "f_peak_hz": f_peak,
        "peak_voltage": vmax,
        "n_nodes": float(n_nodes),
        "n_tetra": float(n_tetra),
        "n_modes": float(n_modes),
        "max_top_surface_strain": strain_max,
    }
    lines = [
        f"sample_id: {sample_id}",
        f"volume fraction: {vfrac:.3f}",
        f"f_peak: {f_peak:.4f} Hz",
        f"peak |V|: {vmax:.4f}",
        f"nodes: {n_nodes:,}",
        f"tetra: {n_tetra:,}",
        f"modes saved: {n_modes}",
    ]
    if np.isfinite(strain_max):
        lines.append(f"max top strain: {strain_max:.4e}")
    ax.axis("off")
    ax.set_title("Run Summary")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    return stats


def _plot_sample_summary(
    sample_id: int,
    dataset_path: Path,
    mesh_npz_path: Path,
    response_path: Path,
    modal_path: Path | None,
    output_path: Path,
) -> dict[str, float]:
    dataset_row = _load_dataset_row(dataset_path, sample_id)
    mesh = _load_mesh(mesh_npz_path)
    response = _load_response(response_path)
    modal = _load_modal(modal_path)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.25], height_ratios=[1.0, 1.0])
    ax_unit = fig.add_subplot(gs[0, 0])
    ax_tiled = fig.add_subplot(gs[0, 1])
    ax_mesh = fig.add_subplot(gs[:, 2])
    ax_frf = fig.add_subplot(gs[1, 0])
    ax_stats = fig.add_subplot(gs[1, 1])

    _plot_unit_cell(ax_unit, dataset_row["binary"])
    _plot_tiled_geometry(ax_tiled, dataset_row["binary"])
    strain_max = _plot_surface_mesh_or_strain(fig, ax_mesh, mesh, modal)
    _plot_frf(ax_frf, response, modal)
    stats = _plot_stats(ax_stats, sample_id, dataset_row, mesh, response, modal, strain_max)

    fig.suptitle(f"PEH Sample {sample_id:04d}", fontsize=16, y=0.98)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stats


def _write_summary_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "volume_fraction",
        "f_peak_hz",
        "peak_voltage",
        "n_nodes",
        "n_tetra",
        "n_modes",
        "max_top_surface_strain",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_gallery(image_paths: list[Path], output_path: Path) -> None:
    if not image_paths:
        return
    n = len(image_paths)
    fig, axes = plt.subplots(n, 1, figsize=(14, 7 * n))
    if n == 1:
        axes = [axes]
    for ax, path in zip(axes, image_paths):
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(path.stem)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create human-readable figures for geometry, mesh/strain, and FEM outputs.",
    )
    parser.add_argument("--dataset", default="data/dataset_100.npz", help="Source unit-cell dataset.")
    parser.add_argument("--mesh-dir", default="meshes/volumes", help="Directory with plate3d_*_fenicsx.npz meshes.")
    parser.add_argument("--response-dir", default="data/fem_responses", help="Directory with sample response files.")
    parser.add_argument("--modal-dir", default="data/modal_data", help="Directory with modal diagnostics.")
    parser.add_argument("--output-dir", default="reports/run_outputs", help="Directory for PNG reports.")
    parser.add_argument("--sample-ids", default="", help="Comma-separated sample ids. Empty means infer from response files.")
    parser.add_argument("--limit", type=int, default=None, help="Only visualize the first N inferred samples.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    mesh_dir = Path(args.mesh_dir)
    response_dir = Path(args.response_dir)
    modal_dir = Path(args.modal_dir)
    output_dir = Path(args.output_dir)

    if args.sample_ids.strip():
        sample_ids = [int(item.strip()) for item in args.sample_ids.split(",") if item.strip()]
    else:
        sample_ids = _infer_sample_ids(response_dir)
    if args.limit is not None:
        sample_ids = sample_ids[: int(args.limit)]
    if not sample_ids:
        raise FileNotFoundError(f"No sample ids could be inferred from {response_dir}.")

    rows: list[dict[str, float]] = []
    image_paths: list[Path] = []
    for sample_id in sample_ids:
        mesh_npz = mesh_dir / f"plate3d_{sample_id:04d}_fenicsx.npz"
        response_npz = response_dir / f"sample_{sample_id:04d}_response.npz"
        modal_npz = modal_dir / f"sample_{sample_id:04d}_modal.npz"
        if not mesh_npz.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_npz}")
        if not response_npz.exists():
            raise FileNotFoundError(f"Response file not found: {response_npz}")

        output_path = output_dir / f"sample_{sample_id:04d}_summary.png"
        row = _plot_sample_summary(
            sample_id=sample_id,
            dataset_path=dataset_path,
            mesh_npz_path=mesh_npz,
            response_path=response_npz,
            modal_path=modal_npz if modal_npz.exists() else None,
            output_path=output_path,
        )
        rows.append(row)
        image_paths.append(output_path)
        print(f"Saved {output_path}")

    _build_gallery(image_paths, output_dir / "gallery.png")
    _write_summary_csv(rows, output_dir / "summary.csv")
    print(f"Saved {output_dir / 'gallery.png'}")
    print(f"Saved {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
