from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def save_fem_response(
    sample_id: int,
    f_peak_hz: float,
    freq_hz: NDArray[np.floating],
    voltage_mag: NDArray[np.floating],
    output_dir: str | Path,
    quality_flag: int = 1,
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sample_{int(sample_id):04d}_response.npz"
    np.savez_compressed(
        path,
        sample_id=np.asarray(int(sample_id), dtype=np.int32),
        f_peak_hz=np.asarray(float(f_peak_hz), dtype=np.float64),
        freq_hz=freq_hz,
        voltage_mag=voltage_mag,
        quality_flag=np.asarray(int(quality_flag), dtype=np.int32),
    )
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

    loaded: dict[int, dict[str, NDArray | float | int]] = {}
    n_freq: int | None = None

    for path in response_files:
        data = np.load(path, allow_pickle=True)
        sample_id = int(data["sample_id"]) if "sample_id" in data.files else int(path.stem.split("_")[1])
        f_peak_hz = float(data["f_peak_hz"])
        freq_hz = np.asarray(data["freq_hz"], dtype=np.float64).reshape(-1)
        voltage_mag = np.asarray(data["voltage_mag"], dtype=np.float64).reshape(-1)
        quality_flag = int(data["quality_flag"]) if "quality_flag" in data.files else 1

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
            "quality_flag": quality_flag,
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
    quality_flag = np.zeros(n_samples, dtype=np.int32)

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
        peak_voltage[idx] = float(np.nanmax(voltage))
        quality_flag[idx] = qflag

    response_dataset = {
        "sample_id": np.asarray(sample_ids, dtype=np.int32),
        "f_peak_hz": f_peak_hz,
        "freq_ratio": freq_ratio,
        "freq_hz": freq_hz,
        "voltage_mag": voltage_mag,
        "peak_voltage": peak_voltage,
        "quality_flag": quality_flag,
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
