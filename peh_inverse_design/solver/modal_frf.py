"""Reduced-order modal voltage FRF evaluation shared by the solver and host tools.

The FEniCSx modal solve stores the reduced electromechanical model per sample in
``sample_XXXX_modal.npz``: undamped eigenfrequencies, modal electromechanical
coupling, modal base-excitation forces, and the electrode capacitance, together
with the damping ratio and load resistance used for the run. Those quantities
fully define the modal-superposition voltage FRF, so the FRF can be re-evaluated
on any frequency grid in physical units (Hz, peak volts) without dolfinx.

This module is numpy-only on purpose so it can run on the host Python
environment as well as inside the dolfinx Docker image.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


REQUIRED_MODAL_MODEL_KEYS = ("eigenfreq_hz", "modal_theta", "modal_force", "capacitance_f")

_OPTIONAL_SCALAR_KEYS = (
    "sample_id",
    "damping_ratio",
    "resistance_ohm",
    "base_acceleration_m_per_s2",
    "piezo_thickness_m",
    "mode1_frequency_hz",
    "harmonic_field_frequency_hz",
    "element_order",
)


def solve_reduced_system(
    omega: float,
    modal_model: dict[str, np.ndarray],
    damping_ratio: float,
    resistance_ohm: float,
) -> tuple[np.ndarray, complex]:
    """Solve the coupled modal + electrical circuit system at one angular frequency.

    Returns the complex modal coordinates and the complex electrode voltage for a
    unit harmonic base excitation already baked into ``modal_force``.
    """
    freq_n = modal_model["eigenfreq_hz"]
    theta = modal_model["modal_theta"]
    force = modal_model["modal_force"]
    capacitance = float(np.asarray(modal_model["capacitance_f"], dtype=np.float64).reshape(-1)[0])

    n_modes = len(freq_n)
    A = np.zeros((n_modes + 1, n_modes + 1), dtype=np.complex128)
    b = np.zeros(n_modes + 1, dtype=np.complex128)
    for mode_idx in range(n_modes):
        omega_n = 2.0 * math.pi * float(freq_n[mode_idx])
        A[mode_idx, mode_idx] = omega_n ** 2 - omega ** 2 + 2j * damping_ratio * omega_n * omega
        A[mode_idx, -1] = -theta[mode_idx]
        b[mode_idx] = force[mode_idx]
    A[-1, :-1] = 1j * omega * theta
    A[-1, -1] = (1.0 / resistance_ohm) + 1j * omega * capacitance
    solution = np.linalg.solve(A, b)
    return solution[:-1], solution[-1]


def evaluate_voltage_frf(
    frequencies_hz: NDArray[np.floating],
    modal_model: dict[str, np.ndarray],
    damping_ratio: float,
    resistance_ohm: float,
) -> NDArray[np.complex128]:
    """Evaluate the complex electrode voltage at each driving frequency in Hz.

    Voltages are peak amplitudes in volts for the base acceleration the modal
    model was assembled with.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
    flat_frequencies = frequencies_hz.reshape(-1)
    flat_voltage = np.zeros(flat_frequencies.shape[0], dtype=np.complex128)
    for idx, f_hz in enumerate(flat_frequencies):
        _, flat_voltage[idx] = solve_reduced_system(
            omega=2.0 * math.pi * float(f_hz),
            modal_model=modal_model,
            damping_ratio=damping_ratio,
            resistance_ohm=resistance_ohm,
        )
    return flat_voltage.reshape(frequencies_hz.shape)


def load_modal_model(modal_npz_path: str | Path) -> dict[str, np.ndarray | float | int]:
    """Load one ``sample_XXXX_modal.npz`` into the dict shape ``evaluate_voltage_frf`` expects.

    The returned dict contains the four reduced-model arrays plus the scalar run
    settings stored alongside them (``damping_ratio``, ``resistance_ohm``,
    ``base_acceleration_m_per_s2``, ...) so callers can default to the exact
    configuration the FEM solve used.
    """
    modal_npz_path = Path(modal_npz_path)
    with np.load(modal_npz_path, allow_pickle=True) as data:
        missing = [key for key in REQUIRED_MODAL_MODEL_KEYS if key not in data.files]
        if missing:
            raise KeyError(
                f"Modal data file {modal_npz_path} is missing required reduced-model keys: {missing}."
            )
        model: dict[str, np.ndarray | float | int] = {
            "eigenfreq_hz": np.asarray(data["eigenfreq_hz"], dtype=np.float64).reshape(-1),
            "modal_theta": np.asarray(data["modal_theta"], dtype=np.float64).reshape(-1),
            "modal_force": np.asarray(data["modal_force"], dtype=np.float64).reshape(-1),
            "capacitance_f": np.asarray(data["capacitance_f"], dtype=np.float64).reshape(-1),
        }
        for key in _OPTIONAL_SCALAR_KEYS:
            if key not in data.files:
                continue
            value = np.asarray(data[key]).reshape(-1)
            if value.size == 0:
                continue
            if key in ("sample_id", "element_order"):
                model[key] = int(value[0])
            else:
                model[key] = float(value[0])

    n_modes = int(np.asarray(model["eigenfreq_hz"]).shape[0])
    for key in ("modal_theta", "modal_force"):
        if int(np.asarray(model[key]).shape[0]) != n_modes:
            raise ValueError(
                f"Modal data file {modal_npz_path} has inconsistent mode counts: "
                f"{key} has {np.asarray(model[key]).shape[0]} entries for {n_modes} eigenfrequencies."
            )
    if n_modes == 0:
        raise ValueError(f"Modal data file {modal_npz_path} does not contain any modes.")
    return model
