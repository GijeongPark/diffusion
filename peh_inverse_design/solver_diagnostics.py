from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


LOW_MODE_DIAGNOSTIC_COUNT = 8


def _string_array(value: str) -> np.ndarray:
    return np.asarray([str(value)])


def _int_array(value: int) -> np.ndarray:
    return np.asarray([int(value)], dtype=np.int32)


def _float_array(value: float) -> np.ndarray:
    return np.asarray([float(value)], dtype=np.float64)


def _bool_array(value: bool) -> np.ndarray:
    return np.asarray([bool(value)], dtype=np.bool_)


def build_solver_provenance_arrays(
    *,
    eigensolver_backend: str,
    solver_element_order: int,
    requested_solver_element_order: int | None = None,
    requested_eigensolver_backend: str = "",
    used_eigensolver_fallback: bool = False,
    used_element_order_fallback: bool = False,
    solver_parity_valid: bool = True,
    parity_invalid_reason: str = "",
    strict_parity_requested: bool = False,
    diagnostic_only: bool = False,
) -> dict[str, np.ndarray]:
    requested_order = int(solver_element_order) if requested_solver_element_order is None else int(requested_solver_element_order)
    return {
        "eigensolver_backend": _string_array(str(eigensolver_backend)),
        "solver_element_order": _int_array(int(solver_element_order)),
        "requested_solver_element_order": _int_array(int(requested_order)),
        "requested_eigensolver_backend": _string_array(str(requested_eigensolver_backend)),
        "used_eigensolver_fallback": _bool_array(bool(used_eigensolver_fallback)),
        "used_element_order_fallback": _bool_array(bool(used_element_order_fallback)),
        "solver_parity_valid": _bool_array(bool(solver_parity_valid)),
        "parity_invalid_reason": _string_array(str(parity_invalid_reason)),
        "strict_parity_requested": _bool_array(bool(strict_parity_requested)),
        "diagnostic_only": _bool_array(bool(diagnostic_only)),
    }


def compute_drive_coupling_diagnostics(
    *,
    eigenfreq_hz: np.ndarray,
    modal_force: np.ndarray,
    modal_theta: np.ndarray,
    low_mode_count: int = LOW_MODE_DIAGNOSTIC_COUNT,
) -> dict[str, np.ndarray]:
    eigenfreq_hz = np.asarray(eigenfreq_hz, dtype=np.float64).reshape(-1)
    modal_force = np.asarray(modal_force, dtype=np.float64).reshape(-1)
    modal_theta = np.asarray(modal_theta, dtype=np.float64).reshape(-1)
    n = min(int(low_mode_count), eigenfreq_hz.shape[0], modal_force.shape[0], modal_theta.shape[0])
    low_mode_eigenfreq_hz = np.asarray(eigenfreq_hz[:n], dtype=np.float64)
    low_mode_modal_force = np.asarray(modal_force[:n], dtype=np.float64)
    low_mode_modal_theta = np.asarray(modal_theta[:n], dtype=np.float64)
    drive_coupling_score = np.abs(low_mode_modal_force * low_mode_modal_theta)

    dominant_index = -1
    dominant_frequency_hz = float("nan")
    suspect_mode_ordering = False
    if n > 0 and np.any(np.isfinite(drive_coupling_score)):
        dominant_index = int(np.nanargmax(drive_coupling_score))
        dominant_frequency_hz = float(low_mode_eigenfreq_hz[dominant_index])
        smallest_index = int(np.nanargmin(low_mode_eigenfreq_hz))
        suspect_mode_ordering = bool(smallest_index != dominant_index)

    return {
        "low_mode_eigenfreq_hz": low_mode_eigenfreq_hz,
        "low_mode_modal_force": low_mode_modal_force,
        "low_mode_modal_theta": low_mode_modal_theta,
        "drive_coupling_score": np.asarray(drive_coupling_score, dtype=np.float64),
        "dominant_drive_coupling_mode_index": _int_array(int(dominant_index)),
        "dominant_drive_coupling_mode_frequency_hz": _float_array(float(dominant_frequency_hz)),
        "suspect_mode_ordering": _bool_array(bool(suspect_mode_ordering)),
    }


def npz_string_scalar(data: Mapping[str, Any], key: str, default: str = "") -> str:
    if key not in data:
        return str(default)
    return str(np.asarray(data[key]).reshape(-1)[0])


def npz_float_scalar(data: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    if key not in data:
        return float(default)
    return float(np.asarray(data[key], dtype=np.float64).reshape(-1)[0])


def npz_int_scalar(data: Mapping[str, Any], key: str, default: int = -1) -> int:
    if key not in data:
        return int(default)
    return int(np.asarray(data[key], dtype=np.int64).reshape(-1)[0])


def npz_bool_scalar(data: Mapping[str, Any], key: str, default: bool = False) -> bool:
    if key not in data:
        return bool(default)
    return bool(np.asarray(data[key]).reshape(-1)[0])


def load_solver_provenance(data: Mapping[str, Any]) -> dict[str, Any]:
    solver_element_order = npz_int_scalar(
        data,
        "solver_element_order",
        default=npz_int_scalar(data, "element_order", default=-1),
    )
    requested_solver_element_order = npz_int_scalar(
        data,
        "requested_solver_element_order",
        default=solver_element_order,
    )
    requested_eigensolver_backend = npz_string_scalar(
        data,
        "requested_eigensolver_backend",
        default=npz_string_scalar(data, "eigensolver_backend", default="unknown"),
    )
    eigensolver_backend = npz_string_scalar(
        data,
        "eigensolver_backend",
        default=requested_eigensolver_backend or "unknown",
    )
    used_element_order_fallback = npz_bool_scalar(
        data,
        "used_element_order_fallback",
        default=(
            solver_element_order >= 0
            and requested_solver_element_order >= 0
            and solver_element_order != requested_solver_element_order
        ),
    )
    return {
        "eigensolver_backend": str(eigensolver_backend),
        "solver_element_order": int(solver_element_order),
        "requested_solver_element_order": int(requested_solver_element_order),
        "requested_eigensolver_backend": str(requested_eigensolver_backend),
        "used_eigensolver_fallback": bool(npz_bool_scalar(data, "used_eigensolver_fallback", default=False)),
        "used_element_order_fallback": bool(used_element_order_fallback),
        "solver_parity_valid": bool(npz_bool_scalar(data, "solver_parity_valid", default=True)),
        "parity_invalid_reason": str(npz_string_scalar(data, "parity_invalid_reason", default="")),
        "strict_parity_requested": bool(npz_bool_scalar(data, "strict_parity_requested", default=False)),
        "diagnostic_only": bool(npz_bool_scalar(data, "diagnostic_only", default=False)),
    }


def load_modal_diagnostics(data: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = compute_drive_coupling_diagnostics(
        eigenfreq_hz=np.asarray(data.get("eigenfreq_hz", []), dtype=np.float64),
        modal_force=np.asarray(data.get("modal_force", []), dtype=np.float64),
        modal_theta=np.asarray(data.get("modal_theta", []), dtype=np.float64),
    )
    if "low_mode_eigenfreq_hz" in data:
        diagnostics["low_mode_eigenfreq_hz"] = np.asarray(data["low_mode_eigenfreq_hz"], dtype=np.float64).reshape(-1)
    if "low_mode_modal_force" in data:
        diagnostics["low_mode_modal_force"] = np.asarray(data["low_mode_modal_force"], dtype=np.float64).reshape(-1)
    if "low_mode_modal_theta" in data:
        diagnostics["low_mode_modal_theta"] = np.asarray(data["low_mode_modal_theta"], dtype=np.float64).reshape(-1)
    if "drive_coupling_score" in data:
        diagnostics["drive_coupling_score"] = np.asarray(data["drive_coupling_score"], dtype=np.float64).reshape(-1)
    diagnostics["dominant_drive_coupling_mode_index"] = npz_int_scalar(
        data,
        "dominant_drive_coupling_mode_index",
        default=int(np.asarray(diagnostics["dominant_drive_coupling_mode_index"], dtype=np.int32).reshape(-1)[0]),
    )
    diagnostics["dominant_drive_coupling_mode_frequency_hz"] = npz_float_scalar(
        data,
        "dominant_drive_coupling_mode_frequency_hz",
        default=float(np.asarray(diagnostics["dominant_drive_coupling_mode_frequency_hz"], dtype=np.float64).reshape(-1)[0]),
    )
    diagnostics["suspect_mode_ordering"] = npz_bool_scalar(
        data,
        "suspect_mode_ordering",
        default=bool(np.asarray(diagnostics["suspect_mode_ordering"]).reshape(-1)[0]),
    )
    return diagnostics
