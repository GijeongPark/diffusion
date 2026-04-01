from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peh_inverse_design.mesh_tags import FACET_TOP_ELECTRODE_TAG
else:
    from .mesh_tags import FACET_TOP_ELECTRODE_TAG


def _infer_sample_ids_from_source(data: np.lib.npyio.NpzFile, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if "sample_id" in data.files:
        sample_id = np.asarray(data["sample_id"][:n_samples], dtype=np.int32)
    else:
        sample_id = np.arange(n_samples, dtype=np.int32)

    if "source_sample_id" in data.files:
        source_sample_id = np.asarray(data["source_sample_id"][:n_samples], dtype=np.int32)
    else:
        source_sample_id = sample_id.copy()
    return sample_id, source_sample_id


def _extract_sample_id_from_path(path: Path) -> int:
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        raise ValueError(f"Could not infer sample id from {path}.")
    return int(matches[-1])


def _copy_source_fields(data: np.lib.npyio.NpzFile, n_total: int, n_samples: int) -> dict[str, np.ndarray]:
    copied: dict[str, np.ndarray] = {}
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim >= 1 and int(arr.shape[0]) == n_total:
            copied[key] = arr[:n_samples]
        else:
            copied[key] = arr
    return copied


def _load_response_records(response_dir: Path) -> tuple[dict[int, dict[str, np.ndarray | float | int]], int]:
    records: dict[int, dict[str, np.ndarray | float | int]] = {}
    n_freq = 0
    for path in sorted(response_dir.glob("sample_*_response.npz")):
        data = np.load(path, allow_pickle=True)
        sample_id = int(data["sample_id"]) if "sample_id" in data.files else _extract_sample_id_from_path(path)
        freq_hz = np.asarray(data["freq_hz"], dtype=np.float64).reshape(-1)
        voltage_mag = np.asarray(data["voltage_mag"], dtype=np.float64).reshape(-1)
        n_freq = max(n_freq, int(freq_hz.shape[0]))
        records[sample_id] = {
            "f_peak_hz": float(data["f_peak_hz"]),
            "freq_hz": freq_hz,
            "voltage_mag": voltage_mag,
            "quality_flag": int(data["quality_flag"]) if "quality_flag" in data.files else 1,
            "path": str(path),
        }
    return records, n_freq


def _load_modal_records(modal_dir: Path) -> dict[int, dict[str, np.ndarray | str | float]]:
    records: dict[int, dict[str, np.ndarray | str | float]] = {}
    for path in sorted(modal_dir.glob("sample_*_modal.npz")):
        data = np.load(path, allow_pickle=True)
        sample_id = int(data["sample_id"]) if "sample_id" in data.files else _extract_sample_id_from_path(path)
        record: dict[str, np.ndarray | str | float] = {"path": str(path)}
        for key in data.files:
            record[key] = np.asarray(data[key])
        records[sample_id] = record
    return records


def _max_shape_for_key(
    records: dict[int, dict[str, np.ndarray | str | float]],
    key: str,
) -> tuple[int, ...] | None:
    shapes = [np.asarray(record[key]).shape for record in records.values() if key in record]
    if not shapes:
        return None
    ndims = {len(shape) for shape in shapes}
    if len(ndims) != 1:
        return None
    ndim = next(iter(ndims))
    if ndim == 0:
        return ()
    return tuple(max(shape[axis] for shape in shapes) for axis in range(ndim))


def _assign_padded(target: np.ndarray, idx: int, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        target[idx] = values
        return
    slices = tuple(slice(0, dim) for dim in values.shape)
    target[(idx,) + slices] = values


def _extract_top_surface_mesh(mesh_npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(mesh_npz_path)
    points = np.asarray(raw["points"], dtype=np.float64)
    triangles = np.asarray(raw["triangle_cells"], dtype=np.int64)
    triangle_tags = np.asarray(raw["triangle_tags"], dtype=np.int32)

    top_triangles = triangles[triangle_tags == FACET_TOP_ELECTRODE_TAG]
    if top_triangles.shape[0] == 0:
        top_triangles = triangles
    used_nodes = np.unique(top_triangles.reshape(-1))
    local_points = points[used_nodes]
    node_map = np.full(points.shape[0], -1, dtype=np.int64)
    node_map[used_nodes] = np.arange(used_nodes.shape[0], dtype=np.int64)
    local_triangles = node_map[top_triangles]
    return local_points, local_triangles


def build_integrated_dataset(
    unit_cell_npz: str | Path,
    response_dir: str | Path,
    output_path: str | Path,
    modal_dir: str | Path | None = None,
    mesh_dir: str | Path | None = None,
    limit: int | None = None,
    index_csv_path: str | Path | None = None,
    embed_top_surface_mesh: bool = False,
) -> Path:
    unit_cell_npz = Path(unit_cell_npz)
    response_dir = Path(response_dir)
    output_path = Path(output_path)
    modal_dir = Path(modal_dir) if modal_dir is not None else None
    mesh_dir = Path(mesh_dir) if mesh_dir is not None else None

    source = np.load(unit_cell_npz, allow_pickle=True)
    n_total = int(source["grf"].shape[0])
    n_samples = n_total if limit is None else min(int(limit), n_total)
    sample_id, source_sample_id = _infer_sample_ids_from_source(source, n_samples)
    integrated = _copy_source_fields(source, n_total, n_samples)
    integrated["sample_id"] = sample_id
    integrated["source_sample_id"] = source_sample_id

    response_records, n_freq = _load_response_records(response_dir)
    response_ok = np.zeros(n_samples, dtype=np.int32)
    f_peak_hz = np.full(n_samples, np.nan, dtype=np.float64)
    peak_voltage = np.full(n_samples, np.nan, dtype=np.float64)
    quality_flag = np.zeros(n_samples, dtype=np.int32)
    response_npz_path = np.full(n_samples, "", dtype=object)
    if n_freq > 0:
        freq_hz = np.full((n_samples, n_freq), np.nan, dtype=np.float64)
        freq_ratio = np.full((n_samples, n_freq), np.nan, dtype=np.float64)
        voltage_mag = np.full((n_samples, n_freq), np.nan, dtype=np.float64)
    else:
        freq_hz = np.zeros((n_samples, 0), dtype=np.float64)
        freq_ratio = np.zeros((n_samples, 0), dtype=np.float64)
        voltage_mag = np.zeros((n_samples, 0), dtype=np.float64)

    for idx, sid in enumerate(sample_id.tolist()):
        record = response_records.get(int(sid))
        if record is None:
            continue
        freq = np.asarray(record["freq_hz"], dtype=np.float64)
        voltage = np.asarray(record["voltage_mag"], dtype=np.float64)
        f_peak = float(record["f_peak_hz"])
        response_ok[idx] = 1
        f_peak_hz[idx] = f_peak
        peak_voltage[idx] = float(np.nanmax(voltage))
        quality_flag[idx] = int(record["quality_flag"])
        freq_hz[idx, : freq.shape[0]] = freq
        freq_ratio[idx, : freq.shape[0]] = freq / f_peak
        voltage_mag[idx, : voltage.shape[0]] = voltage
        response_npz_path[idx] = str(record["path"])

    integrated["response_ok"] = response_ok
    integrated["f_peak_hz"] = f_peak_hz
    integrated["peak_voltage"] = peak_voltage
    integrated["quality_flag"] = quality_flag
    integrated["freq_hz"] = freq_hz
    integrated["freq_ratio"] = freq_ratio
    integrated["voltage_mag"] = voltage_mag
    integrated["response_npz_path"] = response_npz_path

    modal_ok = np.zeros(n_samples, dtype=np.int32)
    modal_npz_path = np.full(n_samples, "", dtype=object)
    field_frequency_hz = np.full(n_samples, np.nan, dtype=np.float64)
    mode1_frequency_hz = np.full(n_samples, np.nan, dtype=np.float64)
    harmonic_field_frequency_hz = np.full(n_samples, np.nan, dtype=np.float64)
    max_top_surface_strain = np.full(n_samples, np.nan, dtype=np.float64)
    eigenfreq_hz = None
    modal_force = None
    modal_theta = None
    modal_mass = None
    capacitance_f = None
    top_surface_strain_eqv = np.empty(n_samples, dtype=object)
    top_surface_strain_eqv[:] = None
    mode1_top_surface_strain_eqv = np.empty(n_samples, dtype=object)
    mode1_top_surface_strain_eqv[:] = None
    harmonic_top_surface_strain_eqv = np.empty(n_samples, dtype=object)
    harmonic_top_surface_strain_eqv[:] = None
    mesh_npz_path = np.full(n_samples, "", dtype=object)
    top_surface_points = np.empty(n_samples, dtype=object) if embed_top_surface_mesh else None
    top_surface_triangles = np.empty(n_samples, dtype=object) if embed_top_surface_mesh else None
    if embed_top_surface_mesh:
        top_surface_points[:] = None
        top_surface_triangles[:] = None

    modal_records = _load_modal_records(modal_dir) if modal_dir is not None and modal_dir.exists() else {}
    eigen_shape = _max_shape_for_key(modal_records, "eigenfreq_hz")
    force_shape = _max_shape_for_key(modal_records, "modal_force")
    theta_shape = _max_shape_for_key(modal_records, "modal_theta")
    mass_shape = _max_shape_for_key(modal_records, "modal_mass")
    cap_shape = _max_shape_for_key(modal_records, "capacitance_f")
    if eigen_shape is not None:
        eigenfreq_hz = np.full((n_samples,) + eigen_shape, np.nan, dtype=np.float64)
    if force_shape is not None:
        modal_force = np.full((n_samples,) + force_shape, np.nan, dtype=np.float64)
    if theta_shape is not None:
        modal_theta = np.full((n_samples,) + theta_shape, np.nan, dtype=np.float64)
    if mass_shape is not None:
        modal_mass = np.full((n_samples,) + mass_shape, np.nan, dtype=np.float64)
    if cap_shape is not None:
        capacitance_f = np.full((n_samples,) + cap_shape, np.nan, dtype=np.float64)

    for idx, sid in enumerate(sample_id.tolist()):
        if mesh_dir is not None:
            mesh_path = mesh_dir / f"plate3d_{int(sid):04d}_fenicsx.npz"
            if mesh_path.exists():
                mesh_npz_path[idx] = str(mesh_path)
                if embed_top_surface_mesh and top_surface_points is not None and top_surface_triangles is not None:
                    points, triangles = _extract_top_surface_mesh(mesh_path)
                    top_surface_points[idx] = points
                    top_surface_triangles[idx] = triangles

        modal = modal_records.get(int(sid))
        if modal is None:
            continue
        modal_ok[idx] = 1
        modal_npz_path[idx] = str(modal["path"])
        if eigenfreq_hz is not None and "eigenfreq_hz" in modal:
            _assign_padded(eigenfreq_hz, idx, np.asarray(modal["eigenfreq_hz"], dtype=np.float64))
        if modal_force is not None and "modal_force" in modal:
            _assign_padded(modal_force, idx, np.asarray(modal["modal_force"], dtype=np.float64))
        if modal_theta is not None and "modal_theta" in modal:
            _assign_padded(modal_theta, idx, np.asarray(modal["modal_theta"], dtype=np.float64))
        if modal_mass is not None and "modal_mass" in modal:
            _assign_padded(modal_mass, idx, np.asarray(modal["modal_mass"], dtype=np.float64))
        if capacitance_f is not None and "capacitance_f" in modal:
            _assign_padded(capacitance_f, idx, np.asarray(modal["capacitance_f"], dtype=np.float64))
        if "mode1_frequency_hz" in modal:
            mode1_frequency_hz[idx] = float(np.asarray(modal["mode1_frequency_hz"], dtype=np.float64))
        elif "eigenfreq_hz" in modal:
            eigen = np.asarray(modal["eigenfreq_hz"], dtype=np.float64).reshape(-1)
            if eigen.size > 0:
                mode1_frequency_hz[idx] = float(eigen[0])
        if "harmonic_field_frequency_hz" in modal:
            harmonic_field_frequency_hz[idx] = float(np.asarray(modal["harmonic_field_frequency_hz"], dtype=np.float64))
        if "field_frequency_hz" in modal:
            field_frequency_hz[idx] = float(np.asarray(modal["field_frequency_hz"], dtype=np.float64))
        if np.isnan(harmonic_field_frequency_hz[idx]) and np.isfinite(field_frequency_hz[idx]):
            harmonic_field_frequency_hz[idx] = field_frequency_hz[idx]
        if "mode1_top_surface_strain_eqv" in modal:
            strain = np.asarray(modal["mode1_top_surface_strain_eqv"], dtype=np.float64)
            mode1_top_surface_strain_eqv[idx] = strain
            if strain.size > 0:
                max_top_surface_strain[idx] = float(np.max(strain))
        if "harmonic_top_surface_strain_eqv" in modal:
            strain = np.asarray(modal["harmonic_top_surface_strain_eqv"], dtype=np.float64)
            harmonic_top_surface_strain_eqv[idx] = strain
            top_surface_strain_eqv[idx] = strain
        elif "top_surface_strain_eqv" in modal:
            strain = np.asarray(modal["top_surface_strain_eqv"], dtype=np.float64)
            harmonic_top_surface_strain_eqv[idx] = strain
            top_surface_strain_eqv[idx] = strain
            if np.isnan(max_top_surface_strain[idx]) and strain.size > 0:
                max_top_surface_strain[idx] = float(np.max(strain))

    integrated["modal_ok"] = modal_ok
    integrated["modal_npz_path"] = modal_npz_path
    integrated["mesh_npz_path"] = mesh_npz_path
    integrated["mode1_frequency_hz"] = mode1_frequency_hz
    integrated["harmonic_field_frequency_hz"] = harmonic_field_frequency_hz
    integrated["field_frequency_hz"] = field_frequency_hz
    integrated["max_top_surface_strain"] = max_top_surface_strain
    integrated["top_surface_strain_eqv"] = top_surface_strain_eqv
    integrated["mode1_top_surface_strain_eqv"] = mode1_top_surface_strain_eqv
    integrated["harmonic_top_surface_strain_eqv"] = harmonic_top_surface_strain_eqv
    if eigenfreq_hz is not None:
        integrated["eigenfreq_hz"] = eigenfreq_hz
    if modal_force is not None:
        integrated["modal_force"] = modal_force
    if modal_theta is not None:
        integrated["modal_theta"] = modal_theta
    if modal_mass is not None:
        integrated["modal_mass"] = modal_mass
    if capacitance_f is not None:
        integrated["capacitance_f"] = capacitance_f
    if embed_top_surface_mesh and top_surface_points is not None and top_surface_triangles is not None:
        integrated["top_surface_points"] = top_surface_points
        integrated["top_surface_triangles"] = top_surface_triangles

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **integrated)

    if index_csv_path is None:
        index_csv_path = output_path.with_suffix(".csv")
    index_csv_path = Path(index_csv_path)
    index_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, sid in enumerate(sample_id.tolist()):
        rows.append(
            {
                "sample_id": str(int(sid)),
                "source_sample_id": str(int(source_sample_id[idx])),
                "response_ok": str(int(response_ok[idx])),
                "modal_ok": str(int(modal_ok[idx])),
                "f_peak_hz": "" if np.isnan(f_peak_hz[idx]) else f"{float(f_peak_hz[idx]):.12g}",
                "peak_voltage": "" if np.isnan(peak_voltage[idx]) else f"{float(peak_voltage[idx]):.12g}",
                "max_top_surface_strain": (
                    "" if np.isnan(max_top_surface_strain[idx]) else f"{float(max_top_surface_strain[idx]):.12g}"
                ),
                "mesh_npz_path": str(mesh_npz_path[idx]),
                "response_npz_path": str(response_npz_path[idx]),
                "modal_npz_path": str(modal_npz_path[idx]),
            }
        )
    with index_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "source_sample_id",
                "response_ok",
                "modal_ok",
                "f_peak_hz",
                "peak_voltage",
                "max_top_surface_strain",
                "mesh_npz_path",
                "response_npz_path",
                "modal_npz_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one integrated dataset NPZ from unit-cell geometry, FEM responses, and modal outputs.",
    )
    parser.add_argument("--unit-cell-npz", required=True, help="Notebook output or subset unit-cell dataset NPZ.")
    parser.add_argument("--response-dir", required=True, help="Directory with sample_XXXX_response.npz files.")
    parser.add_argument("--output", required=True, help="Output NPZ path for the integrated dataset.")
    parser.add_argument("--modal-dir", default="", help="Directory with sample_XXXX_modal.npz files.")
    parser.add_argument("--mesh-dir", default="", help="Directory with plate3d_XXXX_fenicsx.npz files.")
    parser.add_argument("--limit", type=int, default=None, help="Only keep the first N samples from the source dataset.")
    parser.add_argument("--index-csv", default="", help="Optional CSV summary path. Defaults to output path with .csv.")
    parser.add_argument(
        "--embed-top-surface-mesh",
        action="store_true",
        help="Embed top-surface mesh points/triangles into the integrated dataset.",
    )
    args = parser.parse_args()

    output_path = build_integrated_dataset(
        unit_cell_npz=args.unit_cell_npz,
        response_dir=args.response_dir,
        modal_dir=args.modal_dir or None,
        mesh_dir=args.mesh_dir or None,
        output_path=args.output,
        limit=args.limit,
        index_csv_path=args.index_csv or None,
        embed_top_surface_mesh=bool(args.embed_top_surface_mesh),
    )
    print(f"Saved integrated dataset to {output_path}")
    print(f"Saved integrated index to {Path(args.index_csv) if args.index_csv else output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
