from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SurfaceStrainField:
    key: str
    kind: str
    label: str
    frequency_key: str
    frequency_hz: float
    strain: np.ndarray


def _extract_strain_array(modal: Any, key: str, triangle_count: int) -> np.ndarray | None:
    if modal is None or key not in modal:
        return None
    strain = np.asarray(modal[key], dtype=np.float64).reshape(-1)
    if triangle_count >= 0 and strain.shape[0] != int(triangle_count):
        return None
    if strain.size == 0 or not np.isfinite(strain).any():
        return None
    return strain


def _extract_frequency(modal: Any, key: str) -> float:
    if modal is None or key not in modal:
        return float("nan")
    values = np.asarray(modal[key], dtype=np.float64).reshape(-1)
    if values.size == 0:
        return float("nan")
    return float(values[0])


def available_surface_strain_fields(modal: Any, triangle_count: int) -> list[SurfaceStrainField]:
    fields: list[SurfaceStrainField] = []

    mode1 = _extract_strain_array(modal, "mode1_top_surface_strain_eqv", triangle_count)
    if mode1 is not None:
        fields.append(
            SurfaceStrainField(
                key="mode1_top_surface_strain_eqv",
                kind="modal",
                label="Mode 1 Modal Top-Surface Equivalent Strain",
                frequency_key="mode1_frequency_hz",
                frequency_hz=_extract_frequency(modal, "mode1_frequency_hz"),
                strain=mode1,
            )
        )

    harmonic = _extract_strain_array(modal, "harmonic_top_surface_strain_eqv", triangle_count)
    if harmonic is None:
        harmonic = _extract_strain_array(modal, "top_surface_strain_eqv", triangle_count)
        harmonic_key = "top_surface_strain_eqv"
        harmonic_label = "Legacy Harmonic Peak Top-Surface Equivalent Strain"
    else:
        harmonic_key = "harmonic_top_surface_strain_eqv"
        harmonic_label = "Harmonic Peak Top-Surface Equivalent Strain"
    if harmonic is not None:
        frequency_key = "harmonic_field_frequency_hz" if harmonic_key == "harmonic_top_surface_strain_eqv" else "field_frequency_hz"
        fields.append(
            SurfaceStrainField(
                key=harmonic_key,
                kind="harmonic",
                label=harmonic_label,
                frequency_key=frequency_key,
                frequency_hz=_extract_frequency(modal, frequency_key),
                strain=harmonic,
            )
        )

    return fields


def preferred_surface_strain_field(modal: Any, triangle_count: int) -> SurfaceStrainField | None:
    fields = available_surface_strain_fields(modal, triangle_count)
    for field in fields:
        if field.kind == "modal":
            return field
    return fields[0] if fields else None


def has_explicit_surface_strain_fields(modal: Any, triangle_count: int = -1) -> bool:
    return (
        _extract_strain_array(modal, "mode1_top_surface_strain_eqv", triangle_count) is not None
        and _extract_strain_array(modal, "harmonic_top_surface_strain_eqv", triangle_count) is not None
    )
