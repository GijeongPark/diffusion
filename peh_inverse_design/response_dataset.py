from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peh_inverse_design.solver_diagnostics import load_modal_diagnostics, load_solver_provenance
else:
    from .solver_diagnostics import load_modal_diagnostics, load_solver_provenance


def save_fem_response(
    sample_id: int,
    f_peak_hz: float,
    freq_hz: NDArray[np.floating],
    voltage_mag: NDArray[np.floating],
    output_dir: str | Path,
    quality_flag: int = 1,
    solver_provenance: dict[str, np.ndarray] | None = None,
    modal_diagnostics: dict[str, np.ndarray] | None = None,
) -> Path:
    """Persist one FEM response in the standard on-disk format."""
    freq_hz = np.asarray(freq_hz, dtype=np.float64).reshape(-1)
    voltage_mag = np.asarray(voltage_mag, dtype=np.float64).reshape(-1)
    if freq_hz.shape != voltage_mag.shape:
        raise ValueError("freq_hz and voltage_mag must have the same shape.")
    if freq_hz.ndim != 1:
        raise ValueError("freq_hz and voltage_mag must be 1-D arrays.")
    if f_peak_hz <= 0.0:
        raise ValueError("f_peak_hz must be positive.")
    peak_voltage_peak_v = float(np.nanmax(voltage_mag))
    peak_voltage_rms_v = float(peak_voltage_peak_v / np.sqrt(2.0))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sample_{int(sample_id):04d}_response.npz"
    payload: dict[str, np.ndarray] = {
        "sample_id": np.asarray(int(sample_id), dtype=np.int32),
        "f_peak_hz": np.asarray(float(f_peak_hz), dtype=np.float64),
        "freq_hz": freq_hz,
        "voltage_mag": voltage_mag,
        "peak_voltage": np.asarray(peak_voltage_peak_v, dtype=np.float64),
        "peak_voltage_peak_v": np.asarray(peak_voltage_peak_v, dtype=np.float64),
        "peak_voltage_rms_v": np.asarray(peak_voltage_rms_v, dtype=np.float64),
        "peak_voltage_form": np.asarray("peak"),
        "quality_flag": np.asarray(int(quality_flag), dtype=np.int32),
    }
    if solver_provenance is not None:
        payload.update(solver_provenance)
    if modal_diagnostics is not None:
        payload.update(modal_diagnostics)
    np.savez_compressed(path, **payload)
    return path


def _load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_response_directory(
    response_dir: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, NDArray]:
    """Aggregate per-sample FEM outputs into response_dataset.npz."""
    response_dir = Path(response_dir)
    output_path = Path(output_path)
    response_files = sorted(response_dir.glob("sample_*_response.npz"))
    if not response_files:
        raise FileNotFoundError(f"No response files found in {response_dir}.")

    loaded: dict[int, dict[str, NDArray | float | int | str | bool]] = {}
    n_freq: int | None = None
    low_mode_count = 0

    for path in response_files:
        data = np.load(path, allow_pickle=True)
        sample_id = int(data["sample_id"]) if "sample_id" in data.files else int(path.stem.split("_")[1])
        f_peak_hz = float(data["f_peak_hz"])
        freq_hz = np.asarray(data["freq_hz"], dtype=np.float64).reshape(-1)
        voltage_mag = np.asarray(data["voltage_mag"], dtype=np.float64).reshape(-1)
        quality_flag = int(data["quality_flag"]) if "quality_flag" in data.files else 1
        peak_voltage_peak_v = (
            float(np.asarray(data["peak_voltage_peak_v"], dtype=np.float64))
            if "peak_voltage_peak_v" in data.files
            else (
                float(np.asarray(data["peak_voltage"], dtype=np.float64))
                if "peak_voltage" in data.files
                else float(np.nanmax(voltage_mag))
            )
        )
        peak_voltage_rms_v = (
            float(np.asarray(data["peak_voltage_rms_v"], dtype=np.float64))
            if "peak_voltage_rms_v" in data.files
            else float(peak_voltage_peak_v / np.sqrt(2.0))
        )
        peak_voltage_form = (
            str(np.asarray(data["peak_voltage_form"]).reshape(-1)[0])
            if "peak_voltage_form" in data.files
            else "peak"
        )
        provenance = load_solver_provenance(data)
        diagnostics = load_modal_diagnostics(data)
        low_mode_count = max(
            low_mode_count,
            int(np.asarray(diagnostics["low_mode_eigenfreq_hz"], dtype=np.float64).reshape(-1).shape[0]),
        )

        if freq_hz.shape != voltage_mag.shape:
            raise ValueError(f"Mismatched frequency/response shapes in {path}.")
        if f_peak_hz <= 0.0:
            raise ValueError(f"Non-positive f_peak_hz found in {path}.")
        if n_freq is None:
            n_freq = int(freq_hz.shape[0])
        elif int(freq_hz.shape[0]) != n_freq:
            raise ValueError("All FEM responses must use the same number of frequency samples.")

        loaded[sample_id] = {
            "f_peak_hz": f_peak_hz,
            "freq_hz": freq_hz,
            "voltage_mag": voltage_mag,
            "peak_voltage_peak_v": peak_voltage_peak_v,
            "peak_voltage_rms_v": peak_voltage_rms_v,
            "peak_voltage_form": peak_voltage_form,
            "quality_flag": quality_flag,
            "eigensolver_backend": str(provenance["eigensolver_backend"]),
            "solver_element_order": int(provenance["solver_element_order"]),
            "requested_solver_element_order": int(provenance["requested_solver_element_order"]),
            "requested_eigensolver_backend": str(provenance["requested_eigensolver_backend"]),
            "used_eigensolver_fallback": bool(provenance["used_eigensolver_fallback"]),
            "used_element_order_fallback": bool(provenance["used_element_order_fallback"]),
            "solver_parity_valid": bool(provenance["solver_parity_valid"]),
            "parity_invalid_reason": str(provenance["parity_invalid_reason"]),
            "strict_parity_requested": bool(provenance["strict_parity_requested"]),
            "diagnostic_only": bool(provenance["diagnostic_only"]),
            "low_mode_eigenfreq_hz": np.asarray(diagnostics["low_mode_eigenfreq_hz"], dtype=np.float64),
            "low_mode_modal_force": np.asarray(diagnostics["low_mode_modal_force"], dtype=np.float64),
            "low_mode_modal_theta": np.asarray(diagnostics["low_mode_modal_theta"], dtype=np.float64),
            "drive_coupling_score": np.asarray(diagnostics["drive_coupling_score"], dtype=np.float64),
            "dominant_drive_coupling_mode_index": int(diagnostics["dominant_drive_coupling_mode_index"]),
            "dominant_drive_coupling_mode_frequency_hz": float(
                diagnostics["dominant_drive_coupling_mode_frequency_hz"]
            ),
            "suspect_mode_ordering": bool(diagnostics["suspect_mode_ordering"]),
        }

    if n_freq is None:
        raise RuntimeError("Failed to infer the frequency-grid length from the response files.")

    manifest_rows: list[dict[str, str]] | None = None
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
    if manifest_path is not None and manifest_path.exists():
        manifest_rows = _load_manifest(manifest_path)
        sample_ids = [int(row["sample_id"]) for row in manifest_rows]
    else:
        manifest_rows = None
        sample_ids = sorted(loaded)

    n_samples = len(sample_ids)
    f_peak_hz = np.full(n_samples, np.nan, dtype=np.float64)
    freq_hz = np.full((n_samples, n_freq), np.nan, dtype=np.float64)
    freq_ratio = np.full((n_samples, n_freq), np.nan, dtype=np.float64)
    voltage_mag = np.full((n_samples, n_freq), np.nan, dtype=np.float64)
    peak_voltage = np.full(n_samples, np.nan, dtype=np.float64)
    peak_voltage_peak_v = np.full(n_samples, np.nan, dtype=np.float64)
    peak_voltage_rms_v = np.full(n_samples, np.nan, dtype=np.float64)
    peak_voltage_form = np.full(n_samples, "", dtype="<U16")
    quality_flag = np.zeros(n_samples, dtype=np.int32)
    eigensolver_backend = np.full(n_samples, "", dtype="<U32")
    solver_element_order = np.full(n_samples, -1, dtype=np.int32)
    requested_solver_element_order = np.full(n_samples, -1, dtype=np.int32)
    requested_eigensolver_backend = np.full(n_samples, "", dtype="<U32")
    used_eigensolver_fallback = np.zeros(n_samples, dtype=np.int32)
    used_element_order_fallback = np.zeros(n_samples, dtype=np.int32)
    solver_parity_valid = np.ones(n_samples, dtype=np.int32)
    parity_invalid_reason = np.full(n_samples, "", dtype=object)
    strict_parity_requested = np.zeros(n_samples, dtype=np.int32)
    diagnostic_only = np.zeros(n_samples, dtype=np.int32)
    low_mode_eigenfreq_hz = np.full((n_samples, low_mode_count), np.nan, dtype=np.float64)
    low_mode_modal_force = np.full((n_samples, low_mode_count), np.nan, dtype=np.float64)
    low_mode_modal_theta = np.full((n_samples, low_mode_count), np.nan, dtype=np.float64)
    drive_coupling_score = np.full((n_samples, low_mode_count), np.nan, dtype=np.float64)
    dominant_drive_coupling_mode_index = np.full(n_samples, -1, dtype=np.int32)
    dominant_drive_coupling_mode_frequency_hz = np.full(n_samples, np.nan, dtype=np.float64)
    suspect_mode_ordering = np.zeros(n_samples, dtype=np.int32)

    for idx, sample_id in enumerate(sample_ids):
        record = loaded.get(int(sample_id))
        if record is None:
            continue
        f_peak = float(record["f_peak_hz"])
        freq = np.asarray(record["freq_hz"], dtype=np.float64)
        voltage = np.asarray(record["voltage_mag"], dtype=np.float64)
        qflag = int(record["quality_flag"])

        f_peak_hz[idx] = f_peak
        freq_hz[idx] = freq
        freq_ratio[idx] = freq / f_peak
        voltage_mag[idx] = voltage
        peak_voltage_peak_v[idx] = float(record["peak_voltage_peak_v"])
        peak_voltage_rms_v[idx] = float(record["peak_voltage_rms_v"])
        peak_voltage_form[idx] = str(record["peak_voltage_form"])
        peak_voltage[idx] = float(record["peak_voltage_peak_v"])
        quality_flag[idx] = qflag
        eigensolver_backend[idx] = str(record["eigensolver_backend"])
        solver_element_order[idx] = int(record["solver_element_order"])
        requested_solver_element_order[idx] = int(record["requested_solver_element_order"])
        requested_eigensolver_backend[idx] = str(record["requested_eigensolver_backend"])
        used_eigensolver_fallback[idx] = int(bool(record["used_eigensolver_fallback"]))
        used_element_order_fallback[idx] = int(bool(record["used_element_order_fallback"]))
        solver_parity_valid[idx] = int(bool(record["solver_parity_valid"]))
        parity_invalid_reason[idx] = str(record["parity_invalid_reason"])
        strict_parity_requested[idx] = int(bool(record["strict_parity_requested"]))
        diagnostic_only[idx] = int(bool(record["diagnostic_only"]))
        low_freq = np.asarray(record["low_mode_eigenfreq_hz"], dtype=np.float64).reshape(-1)
        low_force = np.asarray(record["low_mode_modal_force"], dtype=np.float64).reshape(-1)
        low_theta = np.asarray(record["low_mode_modal_theta"], dtype=np.float64).reshape(-1)
        low_score = np.asarray(record["drive_coupling_score"], dtype=np.float64).reshape(-1)
        low_mode_eigenfreq_hz[idx, : low_freq.shape[0]] = low_freq
        low_mode_modal_force[idx, : low_force.shape[0]] = low_force
        low_mode_modal_theta[idx, : low_theta.shape[0]] = low_theta
        drive_coupling_score[idx, : low_score.shape[0]] = low_score
        dominant_drive_coupling_mode_index[idx] = int(record["dominant_drive_coupling_mode_index"])
        dominant_drive_coupling_mode_frequency_hz[idx] = float(record["dominant_drive_coupling_mode_frequency_hz"])
        suspect_mode_ordering[idx] = int(bool(record["suspect_mode_ordering"]))

    response_dataset = {
        "sample_id": np.asarray(sample_ids, dtype=np.int32),
        "f_peak_hz": f_peak_hz,
        "freq_ratio": freq_ratio,
        "freq_hz": freq_hz,
        "voltage_mag": voltage_mag,
        "peak_voltage": peak_voltage,
        "peak_voltage_peak_v": peak_voltage_peak_v,
        "peak_voltage_rms_v": peak_voltage_rms_v,
        "peak_voltage_form": peak_voltage_form,
        "quality_flag": quality_flag,
        "eigensolver_backend": eigensolver_backend,
        "solver_element_order": solver_element_order,
        "requested_solver_element_order": requested_solver_element_order,
        "requested_eigensolver_backend": requested_eigensolver_backend,
        "used_eigensolver_fallback": used_eigensolver_fallback,
        "used_element_order_fallback": used_element_order_fallback,
        "solver_parity_valid": solver_parity_valid,
        "parity_invalid_reason": parity_invalid_reason,
        "strict_parity_requested": strict_parity_requested,
        "diagnostic_only": diagnostic_only,
        "low_mode_eigenfreq_hz": low_mode_eigenfreq_hz,
        "low_mode_modal_force": low_mode_modal_force,
        "low_mode_modal_theta": low_mode_modal_theta,
        "drive_coupling_score": drive_coupling_score,
        "dominant_drive_coupling_mode_index": dominant_drive_coupling_mode_index,
        "dominant_drive_coupling_mode_frequency_hz": dominant_drive_coupling_mode_frequency_hz,
        "suspect_mode_ordering": suspect_mode_ordering,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **response_dataset)

    if manifest_rows is not None and manifest_path is not None:
        output_key = output_path.name
        for row in manifest_rows:
            sample_id = int(row["sample_id"])
            ok = int(sample_id in loaded and int(loaded[sample_id]["quality_flag"]) != 0)
            row["response_npz_key"] = output_key if ok else row.get("response_npz_key", "")
            row["fem_ok"] = str(ok)
        _write_manifest(manifest_rows, manifest_path)

    return response_dataset
