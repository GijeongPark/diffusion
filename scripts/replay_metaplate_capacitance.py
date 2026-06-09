"""Replay metaplate V_peak for raw vs plate-condensed capacitance (no FEM rerun).

Capacitance only enters the reduced electrical circuit, not the structural FEM
eigensolve, so the effect of switching the permittivity from raw eps_S to the
plate-condensed eps_bar can be evaluated directly from already-stored modal data
(<run>/data/modal_data/sample_XXXX_modal.npz) -- no Docker / dolfinx rerun needed.

For each sample it rebuilds the reduced modal model, swaps in C(eps) = eps *
piezo_volume / thickness^2, and recomputes the FRF peak with the current solver
numerics. A cross-check column replays with the capacitance actually stored in the
modal file (which should reproduce the run's integrated_dataset peak_voltage).

Usage (project venv, pure numpy -- no dolfinx):
    ./.venv/bin/python scripts/replay_metaplate_capacitance.py [RUN_DIR]
Default RUN_DIR = runs/0604
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peh_inverse_design.solver.fenicsx_modal_solver import (
    _evaluate_voltage_frf,
    _inject_exact_frequency,
    _refine_peak_frequency,
    _search_peak_with_adaptive_window,
)

RAW_EPS33_F_PER_M = 1.26934e-08
CONDENSED_EPS33_F_PER_M = 1.72926e-08

# Current notebook/config solver numerics.
SEARCH_SCALE = (0.5, 2.0)
SEARCH_POINTS = 101
FRF_POINTS = 256
NORMALIZED_RANGE = (0.9, 1.1)


def _scalar(d, key, default=None):
    if key not in d.files:
        return default
    return float(np.asarray(d[key], dtype=np.float64).reshape(-1)[0])


def _peak_voltage(modal_model, capacitance_f, damping_ratio, resistance_ohm):
    replay = dict(modal_model)
    replay["capacitance_f"] = np.asarray([float(capacitance_f)], dtype=np.float64)
    search_freq, search_voltage, _, _ = _search_peak_with_adaptive_window(
        modal_model=replay, damping_ratio=damping_ratio, resistance_ohm=resistance_ohm,
        search_scale=SEARCH_SCALE, search_points=SEARCH_POINTS,
    )
    f_peak = _refine_peak_frequency(
        search_freq_hz=search_freq, search_voltage=search_voltage, modal_model=replay,
        damping_ratio=damping_ratio, resistance_ohm=resistance_ohm,
    )
    freq = _inject_exact_frequency(
        np.linspace(NORMALIZED_RANGE[0] * f_peak, NORMALIZED_RANGE[1] * f_peak, FRF_POINTS, dtype=np.float64),
        f_peak,
    )
    voltage = _evaluate_voltage_frf(
        frequencies_hz=freq, modal_model=replay, damping_ratio=damping_ratio, resistance_ohm=resistance_ohm,
    )
    return f_peak, float(np.max(np.abs(voltage)))


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "runs" / "0604"
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    modal_dir = run_dir / "data" / "modal_data"
    modal_files = sorted(modal_dir.glob("sample_*_modal.npz"))
    if not modal_files:
        raise SystemExit(f"No modal files in {modal_dir}")

    # Existing (as-run) peak voltages for cross-check, if available.
    existing = {}
    ds_path = run_dir / "data" / "integrated_dataset.npz"
    if ds_path.exists():
        ds = np.load(ds_path, allow_pickle=True)
        if "sample_id" in ds.files and "peak_voltage" in ds.files:
            for i, sid in enumerate(ds["sample_id"]):
                existing[int(sid)] = float(ds["peak_voltage"][i])

    print(f"Metaplate capacitance replay (no FEM rerun) -- run: {run_dir.name}")
    print(f"  raw eps_S      = {RAW_EPS33_F_PER_M:.6e} F/m")
    print(f"  condensed eps_bar = {CONDENSED_EPS33_F_PER_M:.6e} F/m")
    print(f"  numerics: search_points={SEARCH_POINTS}, frf_points={FRF_POINTS}, range={NORMALIZED_RANGE}\n")

    header = (f"{'sample':>7}{'f_1[Hz]':>11}{'theta_1[N/V]':>15}"
              f"{'C_raw[F]':>13}{'Vpk_raw[V]':>13}{'C_cond[F]':>13}{'Vpk_cond[V]':>13}"
              f"{'cond/raw':>10}{'asrun_xref':>12}")
    print(header)
    print("-" * len(header))

    csv_rows = ["sample_id,f_1_hz,theta_1_n_per_v,C_raw_f,Vpeak_raw_v,C_condensed_f,Vpeak_condensed_v,Vpeak_ratio_cond_over_raw,asrun_peak_voltage_v"]
    for mf in modal_files:
        d = np.load(mf, allow_pickle=True)
        sid = int(np.asarray(d["sample_id"]).reshape(-1)[0]) if "sample_id" in d.files else int(
            "".join(ch for ch in mf.stem if ch.isdigit()) or "-1")
        eig = np.asarray(d["eigenfreq_hz"], dtype=np.float64).reshape(-1)
        modal_model = {
            "eigenfreq_hz": eig,
            "modal_theta": np.asarray(d["modal_theta"], dtype=np.float64).reshape(-1),
            "modal_force": np.asarray(d["modal_force"], dtype=np.float64).reshape(-1),
            "modal_mass": np.asarray(d["modal_mass"], dtype=np.float64).reshape(-1),
            "capacitance_f": np.asarray(d["capacitance_f"], dtype=np.float64).reshape(-1),
        }
        piezo_vol = _scalar(d, "piezo_volume_m3")
        thickness = _scalar(d, "piezo_thickness_m")
        damping = _scalar(d, "damping_ratio")
        resistance = _scalar(d, "resistance_ohm")
        as_run_C = _scalar(d, "capacitance_f")

        c_raw = RAW_EPS33_F_PER_M * piezo_vol / thickness ** 2
        c_cond = CONDENSED_EPS33_F_PER_M * piezo_vol / thickness ** 2
        _, vpk_raw = _peak_voltage(modal_model, c_raw, damping, resistance)
        _, vpk_cond = _peak_voltage(modal_model, c_cond, damping, resistance)
        # cross-check using the capacitance actually used in the run
        _, vpk_asrun = _peak_voltage(modal_model, as_run_C, damping, resistance)
        xref = existing.get(sid)
        xref_str = "n/a" if xref is None else f"{xref:.4f}"

        print(f"{sid:>7d}{eig[0]:>11.5f}{float(modal_model['modal_theta'][0]):>15.6e}"
              f"{c_raw:>13.5e}{vpk_raw:>13.4f}{c_cond:>13.5e}{vpk_cond:>13.4f}"
              f"{vpk_cond / vpk_raw:>10.4f}{xref_str:>12}")
        csv_rows.append(
            f"{sid},{eig[0]:.8f},{float(modal_model['modal_theta'][0]):.8e},"
            f"{c_raw:.8e},{vpk_raw:.6f},{c_cond:.8e},{vpk_cond:.6f},"
            f"{vpk_cond / vpk_raw:.6f},{'' if xref is None else f'{xref:.6f}'}"
        )
        # tiny self-consistency note: replay@as-run-C should match the stored dataset peak
        if xref is not None and abs(vpk_asrun - xref) / max(abs(xref), 1e-12) > 1e-3:
            print(f"        [note] replay@as-run-C={vpk_asrun:.4f} V differs from stored {xref:.4f} V "
                  f"(as-run eps33={_scalar(d, 'capacitance_eps33s_f_per_m'):.6e})")

    print("-" * len(header))
    print("  asrun_xref = peak_voltage stored in integrated_dataset.npz (the run's as-solved value).")

    csv_path = run_dir / "data" / "capacitance_replay.csv"
    csv_path.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    print(f"\nSaved comparison table: {csv_path}")


if __name__ == "__main__":
    main()
