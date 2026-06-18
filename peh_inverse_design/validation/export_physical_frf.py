"""Export per-sample voltage FRFs as CSV in physical units over an absolute sweep range.

The in-house solver saves each sample's FRF over a normalized window around its
own fundamental peak (f/f_peak in e.g. [0.9, 1.1]), which is the right shape for
the surrogate dataset but cannot be compared point-by-point against an ANSYS
harmonic sweep. This tool re-evaluates the stored reduced modal model on an
absolute frequency grid (Hz) and writes one CSV per sample with the voltage in
peak volts — the same physical units and amplitude convention as the ANSYS
voltage-probe export.

Numpy-only: runs on the host Python environment, no Docker required.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from peh_inverse_design.solver.modal_frf import evaluate_voltage_frf, load_modal_model
else:
    from ..solver.modal_frf import evaluate_voltage_frf, load_modal_model


def _extract_sample_id(path: Path) -> int:
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        raise ValueError(f"Could not infer sample id from {path}.")
    return int(matches[-1])


def build_physical_frequency_grid(
    freq_min_hz: float,
    freq_max_hz: float,
    num_points: int,
) -> np.ndarray:
    freq_min_hz = float(freq_min_hz)
    freq_max_hz = float(freq_max_hz)
    num_points = int(num_points)
    if freq_min_hz < 0.0:
        raise ValueError("freq_min_hz must be non-negative.")
    if freq_max_hz <= freq_min_hz:
        raise ValueError("freq_max_hz must be strictly greater than freq_min_hz.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    return np.linspace(freq_min_hz, freq_max_hz, num_points, dtype=np.float64)


def compute_physical_voltage_frf(
    modal_npz_path: str | Path,
    frequencies_hz: np.ndarray,
    damping_ratio: float | None = None,
    resistance_ohm: float | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Evaluate one sample's voltage FRF magnitude (peak volts) on a physical grid.

    Damping and load resistance default to the values stored with the modal data,
    i.e. the exact configuration the FEM solve used.
    """
    modal_model = load_modal_model(modal_npz_path)
    if damping_ratio is None:
        if "damping_ratio" not in modal_model:
            raise KeyError(f"{modal_npz_path} does not store damping_ratio; pass it explicitly.")
        damping_ratio = float(modal_model["damping_ratio"])
    if resistance_ohm is None:
        if "resistance_ohm" not in modal_model:
            raise KeyError(f"{modal_npz_path} does not store resistance_ohm; pass it explicitly.")
        resistance_ohm = float(modal_model["resistance_ohm"])

    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    eigenfreq_hz = np.asarray(modal_model["eigenfreq_hz"], dtype=np.float64).reshape(-1)
    max_retained_hz = float(np.max(eigenfreq_hz))
    if float(np.max(frequencies_hz)) > max_retained_hz:
        warnings.warn(
            f"Sweep extends to {float(np.max(frequencies_hz)):.6g} Hz but the retained modal basis "
            f"tops out at {max_retained_hz:.6g} Hz ({eigenfreq_hz.shape[0]} mode(s)); the FRF above "
            "the last retained mode is truncated. Increase solver_num_modes for wide sweeps.",
            stacklevel=2,
        )

    voltage = evaluate_voltage_frf(
        frequencies_hz=frequencies_hz,
        modal_model=modal_model,
        damping_ratio=float(damping_ratio),
        resistance_ohm=float(resistance_ohm),
    )
    info: dict[str, float | int] = {
        "sample_id": int(modal_model.get("sample_id", _extract_sample_id(Path(modal_npz_path)))),
        "damping_ratio": float(damping_ratio),
        "resistance_ohm": float(resistance_ohm),
        "n_modes": int(eigenfreq_hz.shape[0]),
        "max_retained_eigenfreq_hz": max_retained_hz,
    }
    if "base_acceleration_m_per_s2" in modal_model:
        info["base_acceleration_m_per_s2"] = float(modal_model["base_acceleration_m_per_s2"])
    return np.abs(voltage), info


def export_physical_voltage_frf_csv(
    modal_npz_path: str | Path,
    output_csv_path: str | Path,
    freq_min_hz: float,
    freq_max_hz: float,
    num_points: int,
    damping_ratio: float | None = None,
    resistance_ohm: float | None = None,
) -> Path:
    """Write one sample's FRF as CSV with columns frequency_hz, voltage_v (peak amplitude)."""
    frequencies_hz = build_physical_frequency_grid(freq_min_hz, freq_max_hz, num_points)
    voltage_mag, info = compute_physical_voltage_frf(
        modal_npz_path=modal_npz_path,
        frequencies_hz=frequencies_hz,
        damping_ratio=damping_ratio,
        resistance_ohm=resistance_ohm,
    )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(f"# sample_id = {int(info['sample_id'])}\n")
        handle.write("# voltage_amplitude_convention = peak\n")
        handle.write(f"# damping_ratio = {info['damping_ratio']:.12g}\n")
        handle.write(f"# resistance_ohm = {info['resistance_ohm']:.12g}\n")
        if "base_acceleration_m_per_s2" in info:
            handle.write(f"# base_acceleration_m_per_s2 = {info['base_acceleration_m_per_s2']:.12g}\n")
        handle.write(f"# n_modes = {int(info['n_modes'])}\n")
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "voltage_v"])
        for f_hz, v in zip(frequencies_hz, voltage_mag):
            writer.writerow([f"{float(f_hz):.12g}", f"{float(v):.12g}"])
    return output_csv_path


def export_run_physical_frf(
    modal_dir: str | Path,
    output_dir: str | Path,
    freq_min_hz: float,
    freq_max_hz: float,
    num_points: int,
    damping_ratio: float | None = None,
    resistance_ohm: float | None = None,
) -> list[Path]:
    """Export physical-unit FRF CSVs for every sample_XXXX_modal.npz in a run."""
    modal_dir = Path(modal_dir)
    output_dir = Path(output_dir)
    modal_files = sorted(modal_dir.glob("sample_*_modal.npz"))
    if not modal_files:
        raise FileNotFoundError(f"No sample_*_modal.npz files found in {modal_dir}.")

    written: list[Path] = []
    for modal_path in modal_files:
        sample_id = _extract_sample_id(modal_path)
        output_csv = output_dir / f"sample_{sample_id:04d}_voltage_frf_hz.csv"
        written.append(
            export_physical_voltage_frf_csv(
                modal_npz_path=modal_path,
                output_csv_path=output_csv,
                freq_min_hz=freq_min_hz,
                freq_max_hz=freq_max_hz,
                num_points=num_points,
                damping_ratio=damping_ratio,
                resistance_ohm=resistance_ohm,
            )
        )
        print(f"Exported physical FRF CSV: {output_csv}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export voltage FRFs in physical units (Hz, peak volts) over an absolute sweep range, "
            "one CSV per sample, from the stored modal data."
        ),
    )
    parser.add_argument(
        "--modal-dir",
        default="data/modal_data",
        help="Directory containing sample_XXXX_modal.npz files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for sample_XXXX_voltage_frf_hz.csv files.",
    )
    parser.add_argument("--freq-min-hz", type=float, required=True, help="Sweep start in Hz (inclusive).")
    parser.add_argument("--freq-max-hz", type=float, required=True, help="Sweep end in Hz (inclusive).")
    parser.add_argument("--points", type=int, default=201, help="Number of evenly spaced sweep points.")
    parser.add_argument(
        "--damping-ratio",
        type=float,
        default=None,
        help="Optional modal damping override. Defaults to the value stored with each sample.",
    )
    parser.add_argument(
        "--resistance-ohm",
        type=float,
        default=None,
        help="Optional load resistance override. Defaults to the value stored with each sample.",
    )
    args = parser.parse_args()

    written = export_run_physical_frf(
        modal_dir=args.modal_dir,
        output_dir=args.output_dir,
        freq_min_hz=float(args.freq_min_hz),
        freq_max_hz=float(args.freq_max_hz),
        num_points=int(args.points),
        damping_ratio=args.damping_ratio,
        resistance_ohm=args.resistance_ohm,
    )
    print(f"Exported {len(written)} physical FRF CSV file(s) to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
