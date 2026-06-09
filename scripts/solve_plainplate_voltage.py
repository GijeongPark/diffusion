"""Solve the plain-plate modal voltage FRF and A/B the capacitance (Task 2, solve stage).

Runs the void-free plain-plate mesh through the *existing* FEniCSx modal voltage
solver with all physics/numerics kept at the current config values (BC, R, damping,
base acceleration, frequency grid, element order, eigensolver). Geometry (a solid
rectangle instead of a perforated metaplate) is the only variable.

Then, from the single stored modal model (modal replay only -- no FEM rerun), it
recomputes V_peak for BOTH the raw and the plate-condensed capacitance permittivity
and prints a comparison table.

This stage needs dolfinx, so it is meant to run inside the project Docker image:

    docker run --rm -v "$PWD":/workspace -w /workspace dolfinx/dolfinx:stable \
        bash -c "pip install -q pyyaml && python3 scripts/solve_plainplate_voltage.py"

Writes:  runs/plainplate/sample_plainplate_modal.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peh_inverse_design.core.problem_spec import (
    build_mechanical_config_kwargs,
    build_piezo_config_kwargs,
    default_problem_spec_path,
    load_problem_spec,
)
from peh_inverse_design.solver.fenicsx_modal_solver import (
    MechanicalConfig,
    PiezoConfig,
    _assemble_modal_model,
    _evaluate_voltage_frf,
    _inject_exact_frequency,
    _refine_peak_frequency,
    _search_peak_with_adaptive_window,
)

# Raw (clamped eps_S) vs plate plane-stress condensed (eps_bar) permittivity [F/m].
RAW_EPS33_F_PER_M = 1.26934e-08
CONDENSED_EPS33_F_PER_M = 1.72926e-08

# Solver numerics kept at the current notebook/config production values.
NUM_MODES = 3
ELEMENT_ORDER = 2
EIGENSOLVER_BACKEND = "shift_invert_cholesky"
SEARCH_SCALE = (0.5, 2.0)
SEARCH_POINTS = 101          # notebook SOLVER_SEARCH_POINTS
FRF_POINTS = 256             # notebook FRF_POINTS
NORMALIZED_RANGE = (0.9, 1.1)

MESH_PATH = PROJECT_ROOT / "runs" / "plainplate" / "meshes" / "volumes" / "plate3d_9999_fenicsx.npz"
OUTPUT_PATH = PROJECT_ROOT / "runs" / "plainplate" / "sample_plainplate_modal.npz"


def _peak_voltage_for_capacitance(
    modal_model: dict,
    capacitance_f: float,
    damping_ratio: float,
    resistance_ohm: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Modal replay: substitute the lumped capacitance and recompute the FRF peak.

    Returns (f_peak_hz, v_peak, freq_hz, voltage_mag) using the SAME modal model
    (eigenfrequencies, theta, modal force) -- no FEM rerun.
    """
    replay = dict(modal_model)
    replay["capacitance_f"] = np.asarray([float(capacitance_f)], dtype=np.float64)

    search_freq, search_voltage, _, _ = _search_peak_with_adaptive_window(
        modal_model=replay,
        damping_ratio=damping_ratio,
        resistance_ohm=resistance_ohm,
        search_scale=SEARCH_SCALE,
        search_points=SEARCH_POINTS,
    )
    f_peak_hz = _refine_peak_frequency(
        search_freq_hz=search_freq,
        search_voltage=search_voltage,
        modal_model=replay,
        damping_ratio=damping_ratio,
        resistance_ohm=resistance_ohm,
    )
    freq_hz = _inject_exact_frequency(
        np.linspace(NORMALIZED_RANGE[0] * f_peak_hz, NORMALIZED_RANGE[1] * f_peak_hz, FRF_POINTS, dtype=np.float64),
        f_peak_hz,
    )
    voltage = _evaluate_voltage_frf(
        frequencies_hz=freq_hz,
        modal_model=replay,
        damping_ratio=damping_ratio,
        resistance_ohm=resistance_ohm,
    )
    voltage_mag = np.abs(voltage)
    v_peak = float(np.max(voltage_mag))
    return f_peak_hz, v_peak, freq_hz, voltage_mag


def main() -> None:
    if not MESH_PATH.exists():
        raise SystemExit(f"Plain-plate mesh not found: {MESH_PATH}. Run scripts/build_plainplate_mesh.py first.")

    spec = load_problem_spec(default_problem_spec_path(PROJECT_ROOT), project_root=PROJECT_ROOT)
    mechanical = MechanicalConfig(**build_mechanical_config_kwargs(spec))
    piezo = PiezoConfig(**build_piezo_config_kwargs(spec))

    damping_ratio = float(mechanical.damping_ratio)
    resistance_ohm = float(piezo.resistance_ohm)
    thickness_m = float(piezo.thickness_m)

    print(
        "Assembling modal model (single FEM solve): "
        f"num_modes={NUM_MODES}, element_order={ELEMENT_ORDER}, backend={EIGENSOLVER_BACKEND}, "
        f"R={resistance_ohm:g} ohm, damping={damping_ratio:g}, base_accel={mechanical.base_acceleration_m_per_s2:g} m/s^2",
        flush=True,
    )
    modal_model = _assemble_modal_model(
        mesh_path=MESH_PATH,
        num_modes=NUM_MODES,
        mechanical=mechanical,
        piezo=piezo,
        element_order=ELEMENT_ORDER,
        store_mode_shapes=False,
        eigensolver_backend=EIGENSOLVER_BACKEND,
    )

    eigenfreq_hz = np.asarray(modal_model["eigenfreq_hz"], dtype=np.float64).reshape(-1)
    modal_theta = np.asarray(modal_model["modal_theta"], dtype=np.float64).reshape(-1)
    modal_force = np.asarray(modal_model["modal_force"], dtype=np.float64).reshape(-1)
    modal_mass = np.asarray(modal_model["modal_mass"], dtype=np.float64).reshape(-1)
    piezo_volume_m3 = float(np.asarray(modal_model["piezo_volume_m3"], dtype=np.float64).reshape(-1)[0])
    substrate_volume_m3 = float(np.asarray(modal_model["substrate_volume_m3"], dtype=np.float64).reshape(-1)[0])

    f1_hz = float(eigenfreq_hz[0])
    theta1 = float(modal_theta[0])

    # Capacitance for each permittivity (identical formula the solver uses):
    #   C = eps33 * piezo_volume / thickness^2
    cap_raw = RAW_EPS33_F_PER_M * piezo_volume_m3 / (thickness_m ** 2)
    cap_condensed = CONDENSED_EPS33_F_PER_M * piezo_volume_m3 / (thickness_m ** 2)

    # Modal replay for both capacitances (no FEM rerun).
    f_peak_raw, v_peak_raw, freq_raw, volt_raw = _peak_voltage_for_capacitance(
        modal_model, cap_raw, damping_ratio, resistance_ohm
    )
    f_peak_cond, v_peak_cond, freq_cond, volt_cond = _peak_voltage_for_capacitance(
        modal_model, cap_condensed, damping_ratio, resistance_ohm
    )

    # The condensed value is the one the updated config uses -> treat it as primary.
    primary_cap = cap_condensed
    primary_eps = CONDENSED_EPS33_F_PER_M
    f_peak_hz, v_peak, freq_hz, voltage_mag = f_peak_cond, v_peak_cond, freq_cond, volt_cond

    np.savez_compressed(
        OUTPUT_PATH,
        # geometry / provenance
        sample_label=np.asarray(["plainplate"]),
        mesh_path=np.asarray([str(MESH_PATH)]),
        element_order=np.asarray([ELEMENT_ORDER], dtype=np.int32),
        num_modes=np.asarray([NUM_MODES], dtype=np.int32),
        eigensolver_backend=np.asarray([EIGENSOLVER_BACKEND]),
        # structural / modal results (capacitance-independent)
        f1_hz=np.asarray([f1_hz], dtype=np.float64),
        first3_eigenfreq_hz=np.asarray(eigenfreq_hz[:3], dtype=np.float64),
        eigenfreq_hz=eigenfreq_hz,
        theta1=np.asarray([theta1], dtype=np.float64),
        modal_theta=modal_theta,
        modal_force=modal_force,
        modal_mass=modal_mass,
        substrate_volume_m3=np.asarray([substrate_volume_m3], dtype=np.float64),
        piezo_volume_m3=np.asarray([piezo_volume_m3], dtype=np.float64),
        piezo_thickness_m=np.asarray([thickness_m], dtype=np.float64),
        resistance_ohm=np.asarray([resistance_ohm], dtype=np.float64),
        damping_ratio=np.asarray([damping_ratio], dtype=np.float64),
        base_acceleration_m_per_s2=np.asarray([mechanical.base_acceleration_m_per_s2], dtype=np.float64),
        # primary (condensed) capacitance + FRF
        capacitance_eps33s_f_per_m=np.asarray([primary_eps], dtype=np.float64),
        capacitance_f=np.asarray([primary_cap], dtype=np.float64),
        f_peak_hz=np.asarray([f_peak_hz], dtype=np.float64),
        v_peak=np.asarray([v_peak], dtype=np.float64),
        freq_hz=freq_hz,
        voltage_mag=voltage_mag,
        # raw-vs-condensed A/B comparison (modal replay)
        raw_eps33_f_per_m=np.asarray([RAW_EPS33_F_PER_M], dtype=np.float64),
        condensed_eps33_f_per_m=np.asarray([CONDENSED_EPS33_F_PER_M], dtype=np.float64),
        capacitance_raw_f=np.asarray([cap_raw], dtype=np.float64),
        capacitance_condensed_f=np.asarray([cap_condensed], dtype=np.float64),
        f_peak_raw_hz=np.asarray([f_peak_raw], dtype=np.float64),
        f_peak_condensed_hz=np.asarray([f_peak_cond], dtype=np.float64),
        v_peak_raw=np.asarray([v_peak_raw], dtype=np.float64),
        v_peak_condensed=np.asarray([v_peak_cond], dtype=np.float64),
        voltage_mag_raw=volt_raw,
        voltage_mag_condensed=volt_cond,
        freq_hz_raw=freq_raw,
        freq_hz_condensed=freq_cond,
    )

    print(f"\nSaved plain-plate modal results: {OUTPUT_PATH}", flush=True)
    print("\n=== Plain rectangular plate: modal voltage results ===", flush=True)
    print(f"  plate                : 1.0 x 1.0 m, substrate 1e-3 m (steel, tag 11) + piezo 1e-4 m (PZT, tag 12), void-free")
    print(f"  first 3 modes f (Hz) : {np.array2string(eigenfreq_hz[:3], precision=6, separator=', ')}")
    print(f"  f_1 (Hz)             : {f1_hz:.6f}")
    print(f"  theta_1 (N/V)        : {theta1:.8e}")
    print(f"  piezo_volume (m^3)   : {piezo_volume_m3:.8e}")
    print()
    header = f"{'permittivity':<26}{'eps33 [F/m]':>16}{'C [F]':>16}{'f_peak [Hz]':>16}{'V_peak [V]':>16}"
    print(header)
    print("-" * len(header))
    print(f"{'raw (eps_S, clamped)':<26}{RAW_EPS33_F_PER_M:>16.6e}{cap_raw:>16.6e}{f_peak_raw:>16.6f}{v_peak_raw:>16.6e}")
    print(f"{'condensed (eps_bar)':<26}{CONDENSED_EPS33_F_PER_M:>16.6e}{cap_condensed:>16.6e}{f_peak_cond:>16.6f}{v_peak_cond:>16.6e}")
    print("-" * len(header))
    print(f"  V_peak ratio (condensed / raw): {v_peak_cond / v_peak_raw:.6f}")


if __name__ == "__main__":
    main()
