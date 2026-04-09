from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peh_inverse_design.mesh_tags import FACET_TOP_ELECTRODE_TAG, VOLUME_PIEZO_TAG, VOLUME_SUBSTRATE_TAG
    from peh_inverse_design.modal_surface_fields import has_explicit_surface_strain_fields
    from peh_inverse_design.problem_spec import (
        build_runtime_defaults,
        build_mechanical_config_kwargs,
        build_piezo_config_kwargs,
        default_problem_spec_path,
        load_problem_spec,
    )
    from peh_inverse_design.response_dataset import save_fem_response
    from peh_inverse_design.solver_diagnostics import build_solver_provenance_arrays, compute_drive_coupling_diagnostics
else:
    from .mesh_tags import FACET_TOP_ELECTRODE_TAG, VOLUME_PIEZO_TAG, VOLUME_SUBSTRATE_TAG
    from .modal_surface_fields import has_explicit_surface_strain_fields
    from .problem_spec import (
        build_runtime_defaults,
        build_mechanical_config_kwargs,
        build_piezo_config_kwargs,
        default_problem_spec_path,
        load_problem_spec,
    )
    from .response_dataset import save_fem_response
    from .solver_diagnostics import build_solver_provenance_arrays, compute_drive_coupling_diagnostics


@dataclass(frozen=True)
class MechanicalConfig:
    substrate_E_pa: float = 1.9305e11
    substrate_nu: float = 0.30
    substrate_rho: float = 7930.0
    piezo_rho: float = 7500.0
    damping_ratio: float = 0.025
    base_acceleration_m_per_s2: float = 2.5


@dataclass(frozen=True)
class PiezoConfig:
    thickness_m: float = 1.0e-4
    resistance_ohm: float = 1.0e4
    eps33s_f_per_m: float = 1.26934e-08
    e_matrix_c_per_m2: tuple[tuple[float, ...], ...] = (
        (0.0, 0.0, -6.622811852),
        (0.0, 0.0, -6.622811852),
        (0.0, 0.0, 23.24031303),
        (0.0, 0.0, 0.0),
        (0.0, 17.03448276, 0.0),
        (17.03448276, 0.0, 0.0),
    )
    stiffness_cE_pa: tuple[tuple[float, ...], ...] = (
        (1.27205e11, 8.0212213391e10, 8.4670187076e10, 0.0, 0.0, 0.0),
        (8.0212213391e10, 1.27205e11, 8.4670187076e10, 0.0, 0.0, 0.0),
        (8.4670187076e10, 8.4670187076e10, 1.17436e11, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 2.3474178404e10, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 2.30e10, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 2.30e10),
    )


class SolverParityError(RuntimeError):
    def __init__(self, message: str, *, diagnostic_payload: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.diagnostic_payload = diagnostic_payload or {}


def _extract_sample_id(mesh_path: Path) -> int:
    matches = re.findall(r"(\d+)", mesh_path.stem)
    if not matches:
        raise ValueError(f"Could not infer sample id from {mesh_path}.")
    return int(matches[-1])


def _solver_diagnostic_output_path(output_dir: str | Path, sample_id: int) -> Path:
    return Path(output_dir) / f"sample_{int(sample_id):04d}_solver_diagnostic.json"


def _write_solver_diagnostic_payload(
    *,
    output_dir: str | Path,
    sample_id: int,
    payload: dict[str, object],
) -> Path:
    output_path = _solver_diagnostic_output_path(output_dir, sample_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return output_path


def _build_top_surface_point_cell_map(
    raw_points: np.ndarray,
    raw_tetra_cells: np.ndarray,
    raw_triangle_cells: np.ndarray,
    raw_triangle_tags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point_cells = np.full(raw_points.shape[0], -1, dtype=np.int32)
    top_triangles = np.asarray(raw_triangle_cells[raw_triangle_tags == FACET_TOP_ELECTRODE_TAG], dtype=np.int64)
    if raw_points.size == 0 or raw_tetra_cells.size == 0 or top_triangles.size == 0:
        return np.zeros((0,), dtype=np.int64), point_cells

    top_point_ids = np.unique(top_triangles.reshape(-1))
    remaining = np.zeros(raw_points.shape[0], dtype=bool)
    remaining[top_point_ids] = True
    unassigned = int(top_point_ids.size)

    for cell_idx, tetra in enumerate(np.asarray(raw_tetra_cells, dtype=np.int64)):
        active = remaining[tetra] & (point_cells[tetra] < 0)
        if not np.any(active):
            continue
        matched = tetra[active]
        point_cells[matched] = cell_idx
        remaining[matched] = False
        unassigned -= matched.size
        if unassigned == 0:
            break

    if unassigned > 0:
        missing_point_id = int(top_point_ids[point_cells[top_point_ids] < 0][0])
        raise KeyError(f"Could not match top-surface point {raw_points[missing_point_id]} to any tetra cell.")
    return top_point_ids, point_cells


def _cell_vertex_signature(vertices: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in np.asarray(vertices, dtype=np.int64).reshape(-1)))


def _build_created_to_raw_cell_index_map(
    raw_tetra_cells: np.ndarray,
    created_cell_geometry_dofs: np.ndarray,
    input_global_indices: np.ndarray,
) -> np.ndarray:
    raw_lookup: dict[tuple[int, ...], int] = {}
    raw_tetra_cells = np.asarray(raw_tetra_cells, dtype=np.int64)
    created_cell_geometry_dofs = np.asarray(created_cell_geometry_dofs, dtype=np.int64)
    input_global_indices = np.asarray(input_global_indices, dtype=np.int64).reshape(-1)
    for raw_cell_idx, raw_cell in enumerate(raw_tetra_cells):
        signature = _cell_vertex_signature(raw_cell)
        if signature in raw_lookup:
            raise ValueError(f"Duplicate raw tetra connectivity detected for signature {signature}.")
        raw_lookup[signature] = int(raw_cell_idx)

    created_to_raw = np.full(created_cell_geometry_dofs.shape[0], -1, dtype=np.int64)
    for created_cell_idx, geometry_dofs in enumerate(created_cell_geometry_dofs):
        signature = _cell_vertex_signature(input_global_indices[geometry_dofs])
        raw_cell_idx = raw_lookup.get(signature)
        if raw_cell_idx is None:
            raise KeyError(
                "Could not match a created tetra cell back to the raw NPZ cell ordering: "
                f"created_cell={created_cell_idx}, raw_vertices={signature}."
            )
        created_to_raw[created_cell_idx] = int(raw_cell_idx)

    unique_raw = np.unique(created_to_raw)
    if unique_raw.shape[0] != raw_tetra_cells.shape[0]:
        raise ValueError(
            "Created-to-raw tetra mapping is incomplete or duplicated: "
            f"matched={unique_raw.shape[0]} raw_cells={raw_tetra_cells.shape[0]}."
        )
    return created_to_raw


def _mode_to_nodal_displacement(
    mode,
    raw_points: np.ndarray,
    top_point_ids: np.ndarray,
    top_point_cells: np.ndarray,
) -> np.ndarray:
    nodal = np.zeros((raw_points.shape[0], 3), dtype=np.float64)
    if top_point_ids.size == 0:
        return nodal
    nodal[top_point_ids] = np.asarray(
        mode.eval(raw_points[top_point_ids], top_point_cells[top_point_ids]),
        dtype=np.float64,
    )
    return nodal


def _remap_raw_cell_tags_to_created_mesh(
    raw_tetra_cells: np.ndarray,
    raw_tetra_tags: np.ndarray,
    created_cell_geometry_dofs: np.ndarray,
    input_global_indices: np.ndarray,
) -> np.ndarray:
    created_to_raw = _build_created_to_raw_cell_index_map(
        raw_tetra_cells=raw_tetra_cells,
        created_cell_geometry_dofs=created_cell_geometry_dofs,
        input_global_indices=input_global_indices,
    )
    return np.asarray(raw_tetra_tags, dtype=np.int32)[created_to_raw]


def _invert_created_to_raw_cell_index_map(created_to_raw: np.ndarray, n_raw_cells: int) -> np.ndarray:
    created_to_raw = np.asarray(created_to_raw, dtype=np.int64).reshape(-1)
    raw_to_created = np.full(int(n_raw_cells), -1, dtype=np.int64)
    for created_cell_idx, raw_cell_idx in enumerate(created_to_raw):
        if raw_cell_idx < 0 or raw_cell_idx >= int(n_raw_cells):
            raise IndexError(f"Raw cell index {raw_cell_idx} is out of bounds for n_raw_cells={n_raw_cells}.")
        if raw_to_created[raw_cell_idx] >= 0:
            raise ValueError(f"Raw cell {raw_cell_idx} maps to multiple created cells.")
        raw_to_created[raw_cell_idx] = int(created_cell_idx)
    if np.any(raw_to_created < 0):
        missing = np.flatnonzero(raw_to_created < 0)
        raise ValueError(f"Missing created-cell mapping for raw tetra cell(s): {missing[:8].tolist()}.")
    return raw_to_created


def _build_top_surface_triangle_cell_map(
    raw_tetra_cells: np.ndarray,
    raw_triangle_cells: np.ndarray,
    raw_triangle_tags: np.ndarray,
) -> np.ndarray:
    top_triangles = np.asarray(raw_triangle_cells[raw_triangle_tags == FACET_TOP_ELECTRODE_TAG], dtype=np.int64)
    raw_tetra_cells = np.asarray(raw_tetra_cells, dtype=np.int64)
    if top_triangles.shape[0] == 0 or raw_tetra_cells.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    n_points = int(max(np.max(raw_tetra_cells), np.max(top_triangles))) + 1
    point_is_top = np.zeros(n_points, dtype=bool)
    point_is_top[np.unique(top_triangles.reshape(-1))] = True
    point_cells: list[list[int]] = [[] for _ in range(n_points)]
    for cell_idx, tetra in enumerate(raw_tetra_cells):
        for point_id in tetra:
            if point_is_top[int(point_id)]:
                point_cells[int(point_id)].append(int(cell_idx))

    point_cell_sets = {point_id: set(point_cells[point_id]) for point_id in np.flatnonzero(point_is_top)}
    triangle_cell_ids = np.full(top_triangles.shape[0], -1, dtype=np.int64)
    for triangle_idx, triangle in enumerate(top_triangles):
        vertex_ids = [int(vertex) for vertex in triangle]
        shortest_vertex = min(vertex_ids, key=lambda vertex: len(point_cells[vertex]))
        candidate_cells = point_cells[shortest_vertex]
        other_vertices = [vertex for vertex in vertex_ids if vertex != shortest_vertex]
        for cell_idx in candidate_cells:
            if all(cell_idx in point_cell_sets[vertex] for vertex in other_vertices):
                triangle_cell_ids[triangle_idx] = int(cell_idx)
                break
        if triangle_cell_ids[triangle_idx] < 0:
            raise KeyError(
                "Could not match a top-surface triangle to an owning tetra cell: "
                f"triangle={triangle.tolist()}."
            )
    return triangle_cell_ids


def _equivalent_volume_strain_from_displacement(
    points: np.ndarray,
    tetra: np.ndarray,
    displacement: np.ndarray,
) -> float:
    xyz = np.asarray(points[tetra], dtype=np.float64)
    uvw = np.asarray(displacement[tetra], dtype=np.complex128)
    A = np.column_stack([np.ones(4, dtype=np.float64), xyz])
    coefficients = np.linalg.solve(A, uvw)
    grad = np.asarray(coefficients[1:, :].T, dtype=np.complex128)
    eps = 0.5 * (grad + grad.T)

    eq_sq = (
        0.5
        * (
            (eps[0, 0] - eps[1, 1]) * np.conj(eps[0, 0] - eps[1, 1])
            + (eps[1, 1] - eps[2, 2]) * np.conj(eps[1, 1] - eps[2, 2])
            + (eps[2, 2] - eps[0, 0]) * np.conj(eps[2, 2] - eps[0, 0])
        )
        + 3.0
        * (
            eps[0, 1] * np.conj(eps[0, 1])
            + eps[1, 2] * np.conj(eps[1, 2])
            + eps[0, 2] * np.conj(eps[0, 2])
        )
    )
    return float(np.sqrt(max(float(np.real(eq_sq)), 0.0)))


def _compute_top_surface_cellwise_strain(
    points: np.ndarray,
    tetra_cells: np.ndarray,
    triangle_cells: np.ndarray,
    triangle_tags: np.ndarray,
    nodal_displacement: np.ndarray,
    top_triangle_cell_ids: np.ndarray | None = None,
) -> np.ndarray:
    top_triangles = np.asarray(triangle_cells[triangle_tags == FACET_TOP_ELECTRODE_TAG], dtype=np.int64)
    if top_triangles.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    if top_triangle_cell_ids is None:
        top_triangle_cell_ids = _build_top_surface_triangle_cell_map(
            raw_tetra_cells=tetra_cells,
            raw_triangle_cells=triangle_cells,
            raw_triangle_tags=triangle_tags,
        )
    top_triangle_cell_ids = np.asarray(top_triangle_cell_ids, dtype=np.int64).reshape(-1)
    if top_triangle_cell_ids.shape[0] != top_triangles.shape[0]:
        raise ValueError(
            "Top-surface triangle-to-cell mapping length mismatch: "
            f"{top_triangle_cell_ids.shape[0]} cells for {top_triangles.shape[0]} triangles."
        )
    unique_cell_ids, inverse = np.unique(top_triangle_cell_ids, return_inverse=True)
    cell_strain = np.asarray(
        [
            _equivalent_volume_strain_from_displacement(points, np.asarray(tetra_cells[cell_id], dtype=np.int64), nodal_displacement)
            for cell_id in unique_cell_ids
        ],
        dtype=np.float64,
    )
    return cell_strain[inverse]


def _equivalent_volume_strain_from_tensor_values(strain_tensor: np.ndarray) -> np.ndarray:
    strain_tensor = np.asarray(strain_tensor, dtype=np.complex128)
    eq_sq = (
        0.5
        * (
            (strain_tensor[..., 0, 0] - strain_tensor[..., 1, 1]) * np.conj(strain_tensor[..., 0, 0] - strain_tensor[..., 1, 1])
            + (strain_tensor[..., 1, 1] - strain_tensor[..., 2, 2]) * np.conj(strain_tensor[..., 1, 1] - strain_tensor[..., 2, 2])
            + (strain_tensor[..., 2, 2] - strain_tensor[..., 0, 0]) * np.conj(strain_tensor[..., 2, 2] - strain_tensor[..., 0, 0])
        )
        + 3.0
        * (
            strain_tensor[..., 0, 1] * np.conj(strain_tensor[..., 0, 1])
            + strain_tensor[..., 1, 2] * np.conj(strain_tensor[..., 1, 2])
            + strain_tensor[..., 0, 2] * np.conj(strain_tensor[..., 0, 2])
        )
    )
    return np.sqrt(np.maximum(np.real(eq_sq), 0.0))


def _compute_top_surface_fe_strain_from_mode_dofs(
    *,
    raw_points: np.ndarray,
    raw_tetra_cells: np.ndarray,
    raw_triangle_cells: np.ndarray,
    raw_triangle_tags: np.ndarray,
    mode_dof_vectors: np.ndarray,
    element_order: int,
    modal_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    raw_points = np.asarray(raw_points, dtype=np.float64)
    raw_tetra_cells = np.asarray(raw_tetra_cells, dtype=np.int64)
    raw_triangle_cells = np.asarray(raw_triangle_cells, dtype=np.int32)
    raw_triangle_tags = np.asarray(raw_triangle_tags, dtype=np.int32)
    mode_dof_vectors = np.asarray(mode_dof_vectors, dtype=np.float64)
    if raw_points.size == 0 or raw_tetra_cells.size == 0 or mode_dof_vectors.size == 0:
        return np.zeros(0, dtype=np.float64)

    top_triangle_raw_cell_ids = _build_top_surface_triangle_cell_map(
        raw_tetra_cells=raw_tetra_cells,
        raw_triangle_cells=raw_triangle_cells,
        raw_triangle_tags=raw_triangle_tags,
    )
    if top_triangle_raw_cell_ids.size == 0:
        return np.zeros(0, dtype=np.float64)

    MPI, PETSc, _, fem, _, _, _, ufl = _load_fenicsx()
    import basix.ufl  # type: ignore
    import dolfinx.mesh as dmesh  # type: ignore

    comm = MPI.COMM_WORLD
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,), dtype=np.float64))
    mesh = dmesh.create_mesh(comm, raw_tetra_cells, domain, raw_points)
    created_cell_geometry_dofs = np.asarray(mesh.geometry.dofmap, dtype=np.int64)
    input_global_indices = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
    created_to_raw = _build_created_to_raw_cell_index_map(
        raw_tetra_cells=raw_tetra_cells,
        created_cell_geometry_dofs=created_cell_geometry_dofs,
        input_global_indices=input_global_indices,
    )
    raw_to_created = _invert_created_to_raw_cell_index_map(created_to_raw, n_raw_cells=raw_tetra_cells.shape[0])
    top_triangle_created_cell_ids = np.asarray(raw_to_created[top_triangle_raw_cell_ids], dtype=np.int32)
    unique_created_cells, inverse = np.unique(top_triangle_created_cell_ids, return_inverse=True)

    V = fem.functionspace(mesh, ("Lagrange", int(element_order), (mesh.geometry.dim,)))
    reference_points = np.asarray([[0.25, 0.25, 0.25]], dtype=np.float64)
    strain_expression = lambda field: fem.Expression(ufl.sym(ufl.grad(field)), reference_points)

    if modal_coefficients is None:
        displacement = fem.Function(V)
        displacement.x.array[:] = np.asarray(mode_dof_vectors[0], dtype=np.float64)
        strain_values = np.asarray(strain_expression(displacement).eval(mesh, unique_created_cells), dtype=np.float64)
        equivalent = _equivalent_volume_strain_from_tensor_values(strain_values[:, 0, :, :])
        return np.asarray(equivalent, dtype=np.float64)[inverse]

    modal_coefficients = np.asarray(modal_coefficients, dtype=np.complex128).reshape(-1)
    if modal_coefficients.shape[0] != mode_dof_vectors.shape[0]:
        raise ValueError(
            "modal_coefficients length does not match the number of stored mode vectors: "
            f"{modal_coefficients.shape[0]} vs {mode_dof_vectors.shape[0]}."
        )
    displacement_real = fem.Function(V)
    displacement_imag = fem.Function(V)
    displacement_real.x.array[:] = np.tensordot(np.real(modal_coefficients), mode_dof_vectors, axes=(0, 0))
    displacement_imag.x.array[:] = np.tensordot(np.imag(modal_coefficients), mode_dof_vectors, axes=(0, 0))
    strain_real = np.asarray(strain_expression(displacement_real).eval(mesh, unique_created_cells), dtype=np.float64)
    strain_imag = np.asarray(strain_expression(displacement_imag).eval(mesh, unique_created_cells), dtype=np.float64)
    strain_complex = strain_real[:, 0, :, :] + 1j * strain_imag[:, 0, :, :]
    equivalent = _equivalent_volume_strain_from_tensor_values(strain_complex)
    return np.asarray(equivalent, dtype=np.float64)[inverse]


def _load_fenicsx():
    from mpi4py import MPI  # type: ignore
    from petsc4py import PETSc  # type: ignore
    from slepc4py import SLEPc  # type: ignore

    import dolfinx.fem as fem  # type: ignore
    import dolfinx.fem.petsc as fem_petsc  # type: ignore
    import dolfinx.io as io  # type: ignore
    from dolfinx.io import gmsh as io_gmsh  # type: ignore
    import ufl  # type: ignore

    return MPI, PETSc, SLEPc, fem, fem_petsc, io, io_gmsh, ufl


def _isotropic_stiffness_matrix(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.array(
        [
            [lam + 2.0 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=np.float64,
    )
    return C


def _destroy_petsc_object(obj) -> None:
    destroy = getattr(obj, "destroy", None)
    if callable(destroy):
        try:
            destroy()
        except Exception:
            pass


def _is_mumps_factorization_failure(exc: Exception) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        text = str(current)
        if "MUMPS error in numerical factorization" in text or "INFOG(1)=-13" in text:
            return True
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return False


def _build_eps_solver(
    *,
    comm,
    K,
    M,
    num_modes: int,
    PETSc,
    SLEPc,
    backend: str,
):
    eps_solver = SLEPc.EPS().create(comm)
    eps_solver.setOperators(K, M)
    eps_solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps_solver.setDimensions(num_modes)

    normalized_backend = str(backend).strip().lower()
    if normalized_backend == "shift_invert_lu":
        eps_solver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps_solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps_solver.setTarget(0.0)
        st = eps_solver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        ksp = st.getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        eps_solver.setFromOptions()
        return eps_solver, "shift_invert_lu"

    if normalized_backend == "iterative_gd":
        eps_solver.setType(SLEPc.EPS.Type.GD)
        eps_solver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        st = eps_solver.getST()
        st.setType(SLEPc.ST.Type.PRECOND)
        ksp = st.getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.GAMG)
        eps_solver.setTolerances(tol=1.0e-8, max_it=500)
        eps_solver.setFromOptions()
        return eps_solver, "iterative_gd"

    raise ValueError(f"Unsupported eigensolver backend: {backend}")


def _normalize_eigensolver_backend(
    *,
    eigensolver_backend: str,
    strict_parity: bool,
) -> str:
    normalized_backend = str(eigensolver_backend).strip().lower()
    if normalized_backend not in {"shift_invert_lu", "iterative_gd", "auto"}:
        raise ValueError(f"Unsupported eigensolver backend: {eigensolver_backend}")
    if strict_parity and normalized_backend == "auto":
        return "shift_invert_lu"
    if strict_parity and normalized_backend != "shift_invert_lu":
        raise ValueError("--strict-parity requires --eigensolver-backend shift_invert_lu (or the implicit strict default).")
    return normalized_backend


def _normalize_peak_search_seed(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"auto", "f1", "dominant_coupling"}:
        raise ValueError("peak_search_seed must be one of: auto, f1, dominant_coupling.")
    return normalized


def _successful_solver_parity_state(
    *,
    requested_backend: str,
    actual_backend: str,
    allow_eigensolver_fallback: bool,
    strict_parity: bool,
    used_eigensolver_fallback: bool,
    used_element_order_fallback: bool,
) -> tuple[bool, bool, str]:
    diagnostic_only = bool(
        allow_eigensolver_fallback
        or requested_backend == "iterative_gd"
        or actual_backend == "iterative_gd"
        or used_eigensolver_fallback
        or used_element_order_fallback
    )
    solver_parity_valid = bool(not diagnostic_only)
    if used_element_order_fallback:
        return (
            solver_parity_valid,
            True,
            "element order fallback changed the actual solver order from the requested parity order",
        )
    if used_eigensolver_fallback:
        return (
            solver_parity_valid,
            True,
            "automatic eigensolver fallback from shift_invert_lu to iterative_gd was used",
        )
    if requested_backend == "iterative_gd":
        return (
            solver_parity_valid,
            True,
            "iterative_gd eigensolver backend was requested; run is diagnostic only",
        )
    if allow_eigensolver_fallback:
        return (
            solver_parity_valid,
            True,
            "automatic eigensolver fallback was enabled for this run; treat results as diagnostic only",
        )
    return (solver_parity_valid, False, "")


def _assemble_modal_model(
    mesh_path: Path,
    num_modes: int,
    mechanical: MechanicalConfig,
    piezo: PiezoConfig,
    element_order: int = 1,
    requested_element_order: int | None = None,
    store_mode_shapes: bool = False,
    eigensolver_backend: str = "auto",
    allow_eigensolver_fallback: bool = False,
    strict_parity: bool = False,
) -> dict[str, np.ndarray]:
    MPI, PETSc, SLEPc, fem, fem_petsc, io, io_gmsh, ufl = _load_fenicsx()
    import basix.ufl  # type: ignore
    import dolfinx.mesh as dmesh  # type: ignore

    comm = MPI.COMM_WORLD
    mesh_path = Path(mesh_path)
    sample_id = _extract_sample_id(mesh_path)
    requested_backend = _normalize_eigensolver_backend(
        eigensolver_backend=eigensolver_backend,
        strict_parity=strict_parity,
    )
    requested_element_order = int(element_order) if requested_element_order is None else int(requested_element_order)
    if strict_parity and allow_eigensolver_fallback:
        raise ValueError("--allow-eigensolver-fallback cannot be combined with --strict-parity.")
    raw_points = np.zeros((0, 3), dtype=np.float64)
    raw_tetra_cells = np.zeros((0, 4), dtype=np.int64)
    raw_tetra_tags = np.zeros((0,), dtype=np.int32)
    raw_triangle_cells = np.zeros((0, 3), dtype=np.int32)
    raw_triangle_tags = np.zeros((0,), dtype=np.int32)
    created_to_raw_cell_map = np.zeros((0,), dtype=np.int64)
    if mesh_path.suffix == ".npz":
        raw = np.load(mesh_path)
        raw_points = np.asarray(raw["points"], dtype=np.float64)
        raw_tetra_cells = np.asarray(raw["tetra_cells"], dtype=np.int64)
        raw_tetra_tags = np.asarray(raw["tetra_tags"], dtype=np.int32)
        raw_triangle_cells = np.asarray(raw["triangle_cells"], dtype=np.int32)
        raw_triangle_tags = np.asarray(raw["triangle_tags"], dtype=np.int32)

        domain = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,), dtype=np.float64))
        mesh = dmesh.create_mesh(comm, raw_tetra_cells, domain, raw_points)
        created_cell_geometry_dofs = np.asarray(mesh.geometry.dofmap, dtype=np.int64)
        input_global_indices = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
        created_to_raw_cell_map = _build_created_to_raw_cell_index_map(
            raw_tetra_cells=raw_tetra_cells,
            created_cell_geometry_dofs=created_cell_geometry_dofs,
            input_global_indices=input_global_indices,
        )
        remapped_cell_tags = np.asarray(raw_tetra_tags, dtype=np.int32)[created_to_raw_cell_map]
        cell_entities = np.arange(remapped_cell_tags.shape[0], dtype=np.int32)
        cell_tags = dmesh.meshtags(mesh, mesh.topology.dim, cell_entities, remapped_cell_tags)
    elif mesh_path.suffix == ".xdmf":
        with io.XDMFFile(comm, str(mesh_path), "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            cell_tags = xdmf.read_meshtags(mesh, name="Grid")
    else:
        mesh, cell_tags, _ = io_gmsh.read_from_msh(str(mesh_path), comm, 0, gdim=3)
    gdim = mesh.geometry.dim
    x_coords = np.asarray(mesh.geometry.x, dtype=np.float64)
    V = fem.functionspace(mesh, ("Lagrange", int(element_order), (gdim,)))
    store_surface_mode_shapes = (
        bool(store_mode_shapes)
        and raw_points.size > 0
        and raw_tetra_cells.size > 0
        and raw_triangle_cells.size > 0
    )
    plate_dimensions_m = np.asarray(
        [
            float(np.max(x_coords[:, 0]) - np.min(x_coords[:, 0])),
            float(np.max(x_coords[:, 1]) - np.min(x_coords[:, 1])),
        ],
        dtype=np.float64,
    )
    total_thickness_m = float(np.max(x_coords[:, 2]) - np.min(x_coords[:, 2]))

    fdim = mesh.topology.dim - 1
    x_tol = max(1.0e-9, float(np.max(x_coords[:, 0])) * 1.0e-8)
    clamped_facets = dmesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: np.isclose(x[0], 0.0, atol=x_tol),
    )
    clamped_dofs = fem.locate_dofs_topological(V, fdim, clamped_facets)
    zero = np.zeros(gdim, dtype=PETSc.ScalarType)
    bcs = [fem.dirichletbc(zero, clamped_dofs, V)]

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

    C_sub = ufl.as_matrix(_isotropic_stiffness_matrix(mechanical.substrate_E_pa, mechanical.substrate_nu).tolist())
    C_pz = ufl.as_matrix(np.asarray(piezo.stiffness_cE_pa, dtype=np.float64).tolist())

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def strain_voigt(w):
        eps = ufl.sym(ufl.grad(w))
        return ufl.as_vector(
            [eps[0, 0], eps[1, 1], eps[2, 2], 2.0 * eps[1, 2], 2.0 * eps[0, 2], 2.0 * eps[0, 1]]
        )

    def stress_from_voigt(C, w):
        epsv = strain_voigt(w)
        sigv = ufl.dot(C, epsv)
        return ufl.as_tensor(
            [
                [sigv[0], sigv[5], sigv[4]],
                [sigv[5], sigv[1], sigv[3]],
                [sigv[4], sigv[3], sigv[2]],
            ]
        )

    rho_sub = PETSc.ScalarType(mechanical.substrate_rho)
    rho_pz = PETSc.ScalarType(mechanical.piezo_rho)

    a_form = fem.form(
        ufl.inner(stress_from_voigt(C_sub, u), ufl.sym(ufl.grad(v))) * dx(VOLUME_SUBSTRATE_TAG)
        + ufl.inner(stress_from_voigt(C_pz, u), ufl.sym(ufl.grad(v))) * dx(VOLUME_PIEZO_TAG)
    )
    m_form = fem.form(
        rho_sub * ufl.dot(u, v) * dx(VOLUME_SUBSTRATE_TAG)
        + rho_pz * ufl.dot(u, v) * dx(VOLUME_PIEZO_TAG)
    )

    K = None
    M = None
    eps_solver = None
    vr = None
    vi = None
    used_eigensolver_fallback = False
    actual_eigensolver_backend = "shift_invert_lu"
    try:
        K = fem_petsc.assemble_matrix(a_form, bcs=bcs, diag=1.0)
        M = fem_petsc.assemble_matrix(m_form, bcs=bcs, diag=0.0)
        K.assemble()
        M.assemble()

        initial_backend = "shift_invert_lu" if requested_backend == "auto" else requested_backend
        eps_solver, actual_eigensolver_backend = _build_eps_solver(
            comm=comm,
            K=K,
            M=M,
            num_modes=num_modes,
            PETSc=PETSc,
            SLEPc=SLEPc,
            backend=initial_backend,
        )
        try:
            eps_solver.solve()
        except Exception as exc:
            if initial_backend != "shift_invert_lu" or not _is_mumps_factorization_failure(exc):
                raise
            reason = (
                "strict parity requested, but shift_invert_lu failed and automatic eigensolver fallback is disallowed"
                if strict_parity
                else "shift_invert_lu failed and automatic eigensolver fallback is disallowed for this run"
            )
            if not allow_eigensolver_fallback:
                raise SolverParityError(
                    reason,
                    diagnostic_payload={
                        "sample_id": int(sample_id),
                        "mesh_path": str(mesh_path),
                        "error_message": str(exc),
                        "eigensolver_backend": "shift_invert_lu",
                        "requested_eigensolver_backend": str(requested_backend),
                        "solver_element_order": int(element_order),
                        "requested_solver_element_order": int(requested_element_order),
                        "used_eigensolver_fallback": False,
                        "used_element_order_fallback": False,
                        "solver_parity_valid": False,
                        "parity_invalid_reason": reason,
                        "strict_parity_requested": bool(strict_parity),
                        "diagnostic_only": bool(allow_eigensolver_fallback or requested_backend == "iterative_gd"),
                    },
                ) from exc
            print(
                "Eigen solve fallback: shift-invert LU/MUMPS ran out of factorization memory; "
                "retrying with the iterative generalized-Davidson backend.",
                flush=True,
            )
            _destroy_petsc_object(eps_solver)
            eps_solver, actual_eigensolver_backend = _build_eps_solver(
                comm=comm,
                K=K,
                M=M,
                num_modes=num_modes,
                PETSc=PETSc,
                SLEPc=SLEPc,
                backend="iterative_gd",
            )
            try:
                eps_solver.solve()
            except Exception as fallback_exc:
                raise SolverParityError(
                    "iterative_gd eigensolver fallback failed after shift_invert_lu failed",
                    diagnostic_payload={
                        "sample_id": int(sample_id),
                        "mesh_path": str(mesh_path),
                        "error_message": str(fallback_exc),
                        "eigensolver_backend": "iterative_gd",
                        "requested_eigensolver_backend": str(requested_backend),
                        "solver_element_order": int(element_order),
                        "requested_solver_element_order": int(requested_element_order),
                        "used_eigensolver_fallback": True,
                        "used_element_order_fallback": False,
                        "solver_parity_valid": False,
                        "parity_invalid_reason": "iterative_gd eigensolver fallback failed after shift_invert_lu failed",
                        "strict_parity_requested": bool(strict_parity),
                        "diagnostic_only": True,
                    },
                ) from fallback_exc
            used_eigensolver_fallback = True

        nconv = eps_solver.getConverged()
        if nconv <= 0:
            raise RuntimeError("SLEPc did not converge any eigenpairs.")

        e_col3 = np.asarray(piezo.e_matrix_c_per_m2, dtype=np.float64)[:, 2]
        e_col3_constant = ufl.as_vector(np.asarray(e_col3, dtype=np.float64).tolist())
        one = PETSc.ScalarType(1.0)

        piezo_volume_local = fem.assemble_scalar(fem.form(one * dx(VOLUME_PIEZO_TAG)))
        piezo_volume = comm.allreduce(piezo_volume_local, op=MPI.SUM)
        substrate_volume_local = fem.assemble_scalar(fem.form(one * dx(VOLUME_SUBSTRATE_TAG)))
        substrate_volume = comm.allreduce(substrate_volume_local, op=MPI.SUM)
        capacitance = piezo.eps33s_f_per_m * piezo_volume / (piezo.thickness_m ** 2)
        cell_tag_values = np.asarray(cell_tags.values, dtype=np.int32).reshape(-1)
        substrate_cell_count = int(np.count_nonzero(cell_tag_values == VOLUME_SUBSTRATE_TAG))
        piezo_cell_count = int(np.count_nonzero(cell_tag_values == VOLUME_PIEZO_TAG))

        eigenfreq_hz: list[float] = []
        modal_force: list[float] = []
        modal_theta: list[float] = []
        modal_mass: list[float] = []
        mode_nodal_displacements: list[np.ndarray] = []
        mode_dof_vectors: list[np.ndarray] = []

        vr, _ = K.getVecs()
        vi, _ = K.getVecs()
        for mode_idx in range(min(nconv, num_modes)):
            eigval = eps_solver.getEigenpair(mode_idx, vr, vi)
            if eigval <= 0.0:
                continue
            mode = fem.Function(V)
            mode.x.array[:] = vr.array_r

            mass_local = fem.assemble_scalar(
                fem.form(
                    rho_sub * ufl.dot(mode, mode) * dx(VOLUME_SUBSTRATE_TAG)
                    + rho_pz * ufl.dot(mode, mode) * dx(VOLUME_PIEZO_TAG)
                )
            )
            mass_value = comm.allreduce(mass_local, op=MPI.SUM)
            if mass_value <= 0.0:
                continue
            scale = 1.0 / math.sqrt(mass_value)
            mode.x.array[:] *= scale

            gamma_local = fem.assemble_scalar(
                fem.form(
                    rho_sub * mode[2] * dx(VOLUME_SUBSTRATE_TAG)
                    + rho_pz * mode[2] * dx(VOLUME_PIEZO_TAG)
                )
            )
            gamma_value = comm.allreduce(gamma_local, op=MPI.SUM)

            theta_local = fem.assemble_scalar(
                fem.form(ufl.dot(e_col3_constant, strain_voigt(mode)) * dx(VOLUME_PIEZO_TAG))
            )
            theta_value = comm.allreduce(theta_local, op=MPI.SUM) / piezo.thickness_m

            eigenfreq_hz.append(math.sqrt(float(eigval)) / (2.0 * math.pi))
            modal_force.append(-gamma_value * mechanical.base_acceleration_m_per_s2)
            modal_theta.append(theta_value)
            modal_mass.append(1.0)
            if store_surface_mode_shapes:
                mode_dof_vectors.append(np.asarray(mode.x.array, dtype=np.float64).copy())

        if not eigenfreq_hz:
            raise RuntimeError("No positive eigenfrequencies were extracted from the mesh.")

        used_element_order_fallback = bool(int(element_order) != int(requested_element_order))
        solver_parity_valid, diagnostic_only, parity_invalid_reason = _successful_solver_parity_state(
            requested_backend=requested_backend,
            actual_backend=actual_eigensolver_backend,
            allow_eigensolver_fallback=allow_eigensolver_fallback,
            strict_parity=strict_parity,
            used_eigensolver_fallback=used_eigensolver_fallback,
            used_element_order_fallback=used_element_order_fallback,
        )
        modal_diagnostics = compute_drive_coupling_diagnostics(
            eigenfreq_hz=np.asarray(eigenfreq_hz, dtype=np.float64),
            modal_force=np.asarray(modal_force, dtype=np.float64),
            modal_theta=np.asarray(modal_theta, dtype=np.float64),
        )

        payload = {
            "element_order": np.asarray([int(element_order)], dtype=np.int32),
            "eigenfreq_hz": np.asarray(eigenfreq_hz, dtype=np.float64),
            "modal_force": np.asarray(modal_force, dtype=np.float64),
            "modal_theta": np.asarray(modal_theta, dtype=np.float64),
            "modal_mass": np.asarray(modal_mass, dtype=np.float64),
            "capacitance_f": np.asarray([capacitance], dtype=np.float64),
            "mode_nodal_displacements": np.asarray(mode_nodal_displacements, dtype=np.float64),
            "mode_dof_vectors": np.asarray(mode_dof_vectors, dtype=np.float64),
            "raw_points": np.asarray(raw_points, dtype=np.float64),
            "raw_tetra_cells": np.asarray(raw_tetra_cells, dtype=np.int64),
            "raw_tetra_tags": np.asarray(raw_tetra_tags, dtype=np.int32),
            "raw_triangle_cells": np.asarray(raw_triangle_cells, dtype=np.int32),
            "raw_triangle_tags": np.asarray(raw_triangle_tags, dtype=np.int32),
            "plate_dimensions_m": plate_dimensions_m,
            "total_thickness_m": np.asarray([total_thickness_m], dtype=np.float64),
            "substrate_volume_m3": np.asarray([substrate_volume], dtype=np.float64),
            "piezo_volume_m3": np.asarray([piezo_volume], dtype=np.float64),
            "substrate_cell_count": np.asarray([substrate_cell_count], dtype=np.int32),
            "piezo_cell_count": np.asarray([piezo_cell_count], dtype=np.int32),
        }
        payload.update(
            build_solver_provenance_arrays(
                eigensolver_backend=str(actual_eigensolver_backend),
                solver_element_order=int(element_order),
                requested_solver_element_order=int(requested_element_order),
                requested_eigensolver_backend=str(requested_backend),
                used_eigensolver_fallback=bool(used_eigensolver_fallback),
                used_element_order_fallback=bool(used_element_order_fallback),
                solver_parity_valid=bool(solver_parity_valid),
                parity_invalid_reason=str(parity_invalid_reason),
                strict_parity_requested=bool(strict_parity),
                diagnostic_only=bool(diagnostic_only),
            )
        )
        payload.update(modal_diagnostics)
        return payload
    finally:
        _destroy_petsc_object(vi)
        _destroy_petsc_object(vr)
        _destroy_petsc_object(eps_solver)
        _destroy_petsc_object(M)
        _destroy_petsc_object(K)
        garbage_cleanup = getattr(PETSc, "garbage_cleanup", None)
        if callable(garbage_cleanup):
            try:
                garbage_cleanup(comm=comm)
            except TypeError:
                garbage_cleanup()


def _build_peak_search_grid(
    seed_frequency_hz: float,
    lower_factor: float,
    upper_factor: float,
    search_points: int,
) -> np.ndarray:
    f_min = max(1.0e-9, float(lower_factor) * float(seed_frequency_hz))
    f_max = float(upper_factor) * float(seed_frequency_hz)
    f_max = max(f_max, 1.05 * float(seed_frequency_hz))
    if f_max <= f_min:
        f_max = max(1.10 * float(seed_frequency_hz), 1.05 * f_min)
    return np.linspace(f_min, f_max, int(search_points), dtype=np.float64)


def _resolve_peak_search_seed(
    *,
    modal_model: dict[str, np.ndarray],
    peak_search_seed: str,
) -> tuple[str, float]:
    freq_n = np.sort(np.asarray(modal_model["eigenfreq_hz"], dtype=np.float64).reshape(-1))
    if freq_n.size == 0:
        raise ValueError("modal_model does not contain any eigenfrequencies.")
    f1_hz = float(freq_n[0])
    dominant_frequency_hz = float(
        np.asarray(modal_model.get("dominant_drive_coupling_mode_frequency_hz", [np.nan]), dtype=np.float64).reshape(-1)[0]
    )
    suspect_mode_ordering = bool(np.asarray(modal_model.get("suspect_mode_ordering", [False])).reshape(-1)[0])
    normalized_seed = _normalize_peak_search_seed(peak_search_seed)

    if normalized_seed == "f1":
        return ("f1", f1_hz)
    if normalized_seed == "dominant_coupling":
        if np.isfinite(dominant_frequency_hz) and dominant_frequency_hz > 0.0:
            return ("dominant_coupling", dominant_frequency_hz)
        return ("f1", f1_hz)
    if (
        normalized_seed == "auto"
        and np.isfinite(dominant_frequency_hz)
        and dominant_frequency_hz > 0.0
        and suspect_mode_ordering
    ):
        return ("dominant_coupling", dominant_frequency_hz)
    return ("f1", f1_hz)


def _search_peak_with_adaptive_window(
    modal_model: dict[str, np.ndarray],
    damping_ratio: float,
    resistance_ohm: float,
    search_scale: tuple[float, float],
    search_points: int,
    peak_search_seed: str = "f1",
    max_expansions: int = 12,
) -> tuple[np.ndarray, np.ndarray, int, bool, str, float]:
    seed_source, seed_frequency_hz = _resolve_peak_search_seed(
        modal_model=modal_model,
        peak_search_seed=peak_search_seed,
    )
    lower_factor = float(search_scale[0])
    upper_factor = float(search_scale[1])

    search_freq = _build_peak_search_grid(
        seed_frequency_hz=seed_frequency_hz,
        lower_factor=lower_factor,
        upper_factor=upper_factor,
        search_points=search_points,
    )
    search_voltage = _evaluate_voltage_frf(
        frequencies_hz=search_freq,
        modal_model=modal_model,
        damping_ratio=damping_ratio,
        resistance_ohm=resistance_ohm,
    )
    peak_index = int(np.argmax(np.abs(search_voltage)))
    expansions = 0
    while (peak_index == 0 or peak_index == search_freq.shape[0] - 1) and expansions < int(max_expansions):
        if peak_index == 0:
            lower_factor = max(1.0e-6, 0.5 * lower_factor)
        else:
            upper_factor *= 2.0
        search_freq = _build_peak_search_grid(
            seed_frequency_hz=seed_frequency_hz,
            lower_factor=lower_factor,
            upper_factor=upper_factor,
            search_points=search_points,
        )
        search_voltage = _evaluate_voltage_frf(
            frequencies_hz=search_freq,
            modal_model=modal_model,
            damping_ratio=damping_ratio,
            resistance_ohm=resistance_ohm,
        )
        peak_index = int(np.argmax(np.abs(search_voltage)))
        expansions += 1

    boundary_hit = peak_index == 0 or peak_index == search_freq.shape[0] - 1
    return search_freq, search_voltage, peak_index, boundary_hit, seed_source, seed_frequency_hz


def _refine_peak_frequency(
    search_freq_hz: np.ndarray,
    search_voltage: np.ndarray,
    modal_model: dict[str, np.ndarray],
    damping_ratio: float,
    resistance_ohm: float,
    refinement_points: int = 401,
    refinement_passes: int = 2,
) -> float:
    refined_freq = np.asarray(search_freq_hz, dtype=np.float64)
    refined_voltage = np.asarray(search_voltage, dtype=np.complex128)
    peak_index = int(np.argmax(np.abs(refined_voltage)))
    for _ in range(int(refinement_passes)):
        if peak_index == 0 or peak_index == refined_freq.shape[0] - 1:
            break
        lower_hz = float(refined_freq[peak_index - 1])
        upper_hz = float(refined_freq[peak_index + 1])
        if upper_hz <= lower_hz:
            break
        refined_freq = np.linspace(lower_hz, upper_hz, int(refinement_points), dtype=np.float64)
        refined_voltage = _evaluate_voltage_frf(
            frequencies_hz=refined_freq,
            modal_model=modal_model,
            damping_ratio=damping_ratio,
            resistance_ohm=resistance_ohm,
        )
        peak_index = int(np.argmax(np.abs(refined_voltage)))
    return float(refined_freq[peak_index])


def _inject_exact_frequency(freq_hz: np.ndarray, target_hz: float) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=np.float64).copy()
    if freq_hz.ndim != 1 or freq_hz.size == 0:
        raise ValueError("freq_hz must be a non-empty 1-D array.")
    closest_idx = int(np.argmin(np.abs(freq_hz - float(target_hz))))
    freq_hz[closest_idx] = float(target_hz)
    return freq_hz


def _simple_cantilever_frequency_estimate_hz(
    modal_model: dict[str, np.ndarray],
    mechanical: MechanicalConfig,
    piezo: PiezoConfig,
) -> float:
    plate_dimensions = np.asarray(modal_model.get("plate_dimensions_m", []), dtype=np.float64).reshape(-1)
    if plate_dimensions.size < 2:
        return float("nan")
    length_m = float(plate_dimensions[0])
    width_m = float(plate_dimensions[1])
    if length_m <= 0.0 or width_m <= 0.0:
        return float("nan")

    total_thickness_m = float(np.asarray(modal_model.get("total_thickness_m", [np.nan]), dtype=np.float64)[0])
    substrate_thickness_m = total_thickness_m - float(piezo.thickness_m)
    substrate_volume = float(np.asarray(modal_model.get("substrate_volume_m3", [np.nan]), dtype=np.float64)[0])
    planform_area = length_m * width_m
    if planform_area <= 0.0 or piezo.thickness_m <= 0.0 or substrate_thickness_m <= 0.0:
        return float("nan")

    substrate_fill_fraction = substrate_volume / (planform_area * substrate_thickness_m)
    substrate_fill_fraction = float(np.clip(substrate_fill_fraction, 1.0e-6, 1.0))

    piezo_ex = float(np.asarray(piezo.stiffness_cE_pa, dtype=np.float64)[0, 0])
    beta1 = 1.875104068711961
    unit_width = 1.0

    area_sub = substrate_fill_fraction * unit_width * substrate_thickness_m
    area_pz = unit_width * piezo.thickness_m
    if area_sub <= 0.0 or area_pz <= 0.0:
        return float("nan")

    z_sub = 0.5 * substrate_thickness_m
    z_pz = substrate_thickness_m + 0.5 * piezo.thickness_m
    neutral_axis = (
        mechanical.substrate_E_pa * area_sub * z_sub
        + piezo_ex * area_pz * z_pz
    ) / (
        mechanical.substrate_E_pa * area_sub
        + piezo_ex * area_pz
    )

    inertia_sub = substrate_fill_fraction * unit_width * substrate_thickness_m ** 3 / 12.0
    inertia_pz = unit_width * piezo.thickness_m ** 3 / 12.0
    bending_rigidity = (
        mechanical.substrate_E_pa * (inertia_sub + area_sub * (z_sub - neutral_axis) ** 2)
        + piezo_ex * (inertia_pz + area_pz * (z_pz - neutral_axis) ** 2)
    )
    mass_per_length = mechanical.substrate_rho * area_sub + mechanical.piezo_rho * area_pz
    if bending_rigidity <= 0.0 or mass_per_length <= 0.0:
        return float("nan")

    omega_1 = (beta1 ** 2) * math.sqrt(bending_rigidity / (mass_per_length * length_m ** 4))
    return omega_1 / (2.0 * math.pi)


def _log_modal_diagnostics(
    modal_model: dict[str, np.ndarray],
) -> None:
    plate_dimensions = np.asarray(modal_model.get("plate_dimensions_m", []), dtype=np.float64).reshape(-1)
    total_thickness_m = float(np.asarray(modal_model.get("total_thickness_m", [np.nan]), dtype=np.float64)[0])
    substrate_volume = float(np.asarray(modal_model.get("substrate_volume_m3", [np.nan]), dtype=np.float64)[0])
    piezo_volume = float(np.asarray(modal_model.get("piezo_volume_m3", [np.nan]), dtype=np.float64)[0])
    substrate_cell_count = int(np.asarray(modal_model.get("substrate_cell_count", [0]), dtype=np.int32)[0])
    piezo_cell_count = int(np.asarray(modal_model.get("piezo_cell_count", [0]), dtype=np.int32)[0])
    capacitance = float(np.asarray(modal_model.get("capacitance_f", [np.nan]), dtype=np.float64)[0])
    eigenfreq_hz = np.asarray(modal_model.get("eigenfreq_hz", []), dtype=np.float64).reshape(-1)
    modal_theta = np.asarray(modal_model.get("modal_theta", []), dtype=np.float64).reshape(-1)
    modal_force = np.asarray(modal_model.get("modal_force", []), dtype=np.float64).reshape(-1)
    eigensolver_backend = str(np.asarray(modal_model.get("eigensolver_backend", ["unknown"])).reshape(-1)[0])
    drive_coupling_score = np.asarray(modal_model.get("drive_coupling_score", []), dtype=np.float64).reshape(-1)
    dominant_drive_coupling_mode_index = int(
        np.asarray(modal_model.get("dominant_drive_coupling_mode_index", [-1]), dtype=np.int32).reshape(-1)[0]
    )
    dominant_drive_coupling_mode_frequency_hz = float(
        np.asarray(modal_model.get("dominant_drive_coupling_mode_frequency_hz", [np.nan]), dtype=np.float64).reshape(-1)[0]
    )
    suspect_mode_ordering = bool(np.asarray(modal_model.get("suspect_mode_ordering", [False])).reshape(-1)[0])
    print(
        "Modal diagnostics: "
        f"plate=({plate_dimensions[0]:.6g}, {plate_dimensions[1]:.6g}) m, "
        f"thickness={total_thickness_m:.6g} m, "
        f"substrate_volume={substrate_volume:.6g} m^3, "
        f"piezo_volume={piezo_volume:.6g} m^3, "
        f"substrate_cells={substrate_cell_count}, "
        f"piezo_cells={piezo_cell_count}, "
        f"capacitance={capacitance:.6g} F, "
        f"eigensolver_backend={eigensolver_backend}"
    )
    print(
        "Modal diagnostics: "
        f"eigenfreq_hz[:6]={np.array2string(eigenfreq_hz[:6], precision=6, separator=', ')}, "
        f"modal_theta[:6]={np.array2string(modal_theta[:6], precision=6, separator=', ')}, "
        f"modal_force[:6]={np.array2string(modal_force[:6], precision=6, separator=', ')}, "
        f"drive_coupling_score[:6]={np.array2string(drive_coupling_score[:6], precision=6, separator=', ')}"
    )
    print(
        "Modal diagnostics: "
        f"dominant_drive_coupling_mode_index={dominant_drive_coupling_mode_index}, "
        f"dominant_drive_coupling_mode_frequency_hz={dominant_drive_coupling_mode_frequency_hz:.6g}, "
        f"suspect_mode_ordering={suspect_mode_ordering}"
    )


def _warn_if_frequency_scale_is_suspicious(
    modal_model: dict[str, np.ndarray],
    mechanical: MechanicalConfig,
    piezo: PiezoConfig,
) -> None:
    f1_hz = float(np.asarray(modal_model["eigenfreq_hz"], dtype=np.float64).reshape(-1)[0])
    estimate_hz = _simple_cantilever_frequency_estimate_hz(
        modal_model=modal_model,
        mechanical=mechanical,
        piezo=piezo,
    )
    if not np.isfinite(estimate_hz) or estimate_hz <= 0.0:
        return
    ratio = f1_hz / estimate_hz
    if ratio < 0.1 or ratio > 10.0:
        dimensions = np.asarray(modal_model.get("plate_dimensions_m", []), dtype=np.float64).reshape(-1)
        warnings.warn(
            "Fundamental frequency sanity check failed: "
            f"f1={f1_hz:.6g} Hz, simple cantilever estimate={estimate_hz:.6g} Hz, "
            f"ratio={ratio:.3e}, plate=({dimensions[0]:.6g}, {dimensions[1]:.6g}) m.",
            stacklevel=2,
        )


def _warn_if_open_circuit_resonance_is_inverted(
    search_freq_hz: np.ndarray,
    modal_model: dict[str, np.ndarray],
    damping_ratio: float,
) -> None:
    short_circuit_voltage = _evaluate_voltage_frf(
        frequencies_hz=search_freq_hz,
        modal_model=modal_model,
        damping_ratio=damping_ratio,
        resistance_ohm=1.0e-6,
    )
    open_circuit_voltage = _evaluate_voltage_frf(
        frequencies_hz=search_freq_hz,
        modal_model=modal_model,
        damping_ratio=damping_ratio,
        resistance_ohm=float("inf"),
    )
    short_circuit_hz = float(search_freq_hz[int(np.argmax(np.abs(short_circuit_voltage)))])
    open_peak_hz = float(search_freq_hz[int(np.argmax(np.abs(open_circuit_voltage)))])
    if open_peak_hz + 1.0e-12 < short_circuit_hz:
        warnings.warn(
            "Open-circuit resonance check failed: "
            f"open-circuit peak={open_peak_hz:.6g} Hz is below short-circuit f1={short_circuit_hz:.6g} Hz.",
            stacklevel=2,
        )


def _solve_reduced_system(
    omega: float,
    modal_model: dict[str, np.ndarray],
    damping_ratio: float,
    resistance_ohm: float,
) -> tuple[np.ndarray, complex]:
    freq_n = modal_model["eigenfreq_hz"]
    theta = modal_model["modal_theta"]
    force = modal_model["modal_force"]
    capacitance = float(modal_model["capacitance_f"][0])

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


def _evaluate_voltage_frf(
    frequencies_hz: np.ndarray,
    modal_model: dict[str, np.ndarray],
    damping_ratio: float,
    resistance_ohm: float,
) -> np.ndarray:
    voltage = np.zeros_like(frequencies_hz, dtype=np.complex128)
    for idx, f_hz in enumerate(frequencies_hz):
        _, voltage[idx] = _solve_reduced_system(
            omega=2.0 * math.pi * float(f_hz),
            modal_model=modal_model,
            damping_ratio=damping_ratio,
            resistance_ohm=resistance_ohm,
        )
    return voltage


def _build_modal_save_payload(
    *,
    sample_id: int,
    element_order: int,
    mechanical: MechanicalConfig,
    piezo: PiezoConfig,
    modal_model: dict[str, np.ndarray],
    mode1_top_surface_strain_eqv: np.ndarray,
    harmonic_top_surface_strain_eqv: np.ndarray,
    harmonic_field_frequency_hz: float,
    frf_search_seed_source: str,
    frf_search_seed_frequency_hz: float,
) -> dict[str, np.ndarray]:
    mode1_frequency_hz = float(np.asarray(modal_model["eigenfreq_hz"], dtype=np.float64).reshape(-1)[0])
    harmonic_frequency_array = np.asarray(harmonic_field_frequency_hz, dtype=np.float64)
    return {
        "sample_id": np.asarray(sample_id, dtype=np.int32),
        "element_order": np.asarray([int(element_order)], dtype=np.int32),
        "solver_element_order": np.asarray(
            modal_model.get("solver_element_order", np.asarray([int(element_order)], dtype=np.int32))
        ),
        "eigenfreq_hz": modal_model["eigenfreq_hz"],
        "modal_force": modal_model["modal_force"],
        "modal_theta": modal_model["modal_theta"],
        "modal_mass": modal_model["modal_mass"],
        "capacitance_f": modal_model["capacitance_f"],
        "eigensolver_backend": np.asarray(modal_model.get("eigensolver_backend", ["unknown"])),
        "requested_solver_element_order": np.asarray(
            modal_model.get("requested_solver_element_order", np.asarray([int(element_order)], dtype=np.int32))
        ),
        "requested_eigensolver_backend": np.asarray(
            modal_model.get("requested_eigensolver_backend", np.asarray(["unknown"]))
        ),
        "used_eigensolver_fallback": np.asarray(
            modal_model.get("used_eigensolver_fallback", np.asarray([False], dtype=np.bool_))
        ),
        "used_element_order_fallback": np.asarray(
            modal_model.get("used_element_order_fallback", np.asarray([False], dtype=np.bool_))
        ),
        "solver_parity_valid": np.asarray(
            modal_model.get("solver_parity_valid", np.asarray([True], dtype=np.bool_))
        ),
        "parity_invalid_reason": np.asarray(
            modal_model.get("parity_invalid_reason", np.asarray([""]))
        ),
        "strict_parity_requested": np.asarray(
            modal_model.get("strict_parity_requested", np.asarray([False], dtype=np.bool_))
        ),
        "diagnostic_only": np.asarray(
            modal_model.get("diagnostic_only", np.asarray([False], dtype=np.bool_))
        ),
        "low_mode_eigenfreq_hz": np.asarray(
            modal_model.get("low_mode_eigenfreq_hz", np.asarray([], dtype=np.float64)),
            dtype=np.float64,
        ),
        "low_mode_modal_force": np.asarray(
            modal_model.get("low_mode_modal_force", np.asarray([], dtype=np.float64)),
            dtype=np.float64,
        ),
        "low_mode_modal_theta": np.asarray(
            modal_model.get("low_mode_modal_theta", np.asarray([], dtype=np.float64)),
            dtype=np.float64,
        ),
        "drive_coupling_score": np.asarray(
            modal_model.get("drive_coupling_score", np.asarray([], dtype=np.float64)),
            dtype=np.float64,
        ),
        "dominant_drive_coupling_mode_index": np.asarray(
            modal_model.get("dominant_drive_coupling_mode_index", np.asarray([-1], dtype=np.int32))
        ),
        "dominant_drive_coupling_mode_frequency_hz": np.asarray(
            modal_model.get("dominant_drive_coupling_mode_frequency_hz", np.asarray([np.nan], dtype=np.float64))
        ),
        "suspect_mode_ordering": np.asarray(
            modal_model.get("suspect_mode_ordering", np.asarray([False], dtype=np.bool_))
        ),
        "frf_search_seed_source": np.asarray([str(frf_search_seed_source)]),
        "frf_search_seed_frequency_hz": np.asarray([float(frf_search_seed_frequency_hz)], dtype=np.float64),
        "substrate_volume_m3": modal_model["substrate_volume_m3"],
        "piezo_volume_m3": modal_model["piezo_volume_m3"],
        "substrate_cell_count": modal_model["substrate_cell_count"],
        "piezo_cell_count": modal_model["piezo_cell_count"],
        "substrate_rho": np.asarray(mechanical.substrate_rho, dtype=np.float64),
        "piezo_rho": np.asarray(mechanical.piezo_rho, dtype=np.float64),
        "damping_ratio": np.asarray(mechanical.damping_ratio, dtype=np.float64),
        "base_acceleration_m_per_s2": np.asarray(mechanical.base_acceleration_m_per_s2, dtype=np.float64),
        "piezo_thickness_m": np.asarray(piezo.thickness_m, dtype=np.float64),
        "resistance_ohm": np.asarray(piezo.resistance_ohm, dtype=np.float64),
        "mode1_frequency_hz": np.asarray(mode1_frequency_hz, dtype=np.float64),
        "mode1_top_surface_strain_eqv": np.asarray(mode1_top_surface_strain_eqv, dtype=np.float64),
        "harmonic_field_frequency_hz": harmonic_frequency_array,
        "harmonic_top_surface_strain_eqv": np.asarray(harmonic_top_surface_strain_eqv, dtype=np.float64),
        "field_frequency_hz": harmonic_frequency_array,
        "top_surface_strain_eqv": np.asarray(harmonic_top_surface_strain_eqv, dtype=np.float64),
    }


def solve_modal_voltage_frf(
    mesh_path: str | Path,
    response_dir: str | Path,
    num_modes: int = 8,
    search_scale: tuple[float, float] = (0.5, 2.0),
    search_points: int = 301,
    peak_search_seed: str = "f1",
    frf_points: int = 256,
    normalized_range: tuple[float, float] = (0.9, 1.1),
    mechanical: MechanicalConfig | None = None,
    piezo: PiezoConfig | None = None,
    modes_output_dir: str | Path | None = None,
    element_order: int = 2,
    requested_element_order: int | None = None,
    store_mode_shapes: bool = False,
    eigensolver_backend: str = "auto",
    allow_eigensolver_fallback: bool = False,
    strict_parity: bool = False,
) -> Path:
    mechanical = mechanical or MechanicalConfig()
    piezo = piezo or PiezoConfig()
    mesh_path = Path(mesh_path)
    sample_id = _extract_sample_id(mesh_path)
    requested_element_order = int(element_order) if requested_element_order is None else int(requested_element_order)

    try:
        modal_model = _assemble_modal_model(
            mesh_path=mesh_path,
            num_modes=num_modes,
            mechanical=mechanical,
            piezo=piezo,
            element_order=element_order,
            requested_element_order=requested_element_order,
            store_mode_shapes=store_mode_shapes,
            eigensolver_backend=eigensolver_backend,
            allow_eigensolver_fallback=allow_eigensolver_fallback,
            strict_parity=strict_parity,
        )
    except SolverParityError as exc:
        diagnostic_path = _write_solver_diagnostic_payload(
            output_dir=response_dir,
            sample_id=sample_id,
            payload=exc.diagnostic_payload,
        )
        print(f"Saved solver diagnostic payload to {diagnostic_path}", flush=True)
        raise
    _log_modal_diagnostics(modal_model)
    _warn_if_frequency_scale_is_suspicious(
        modal_model=modal_model,
        mechanical=mechanical,
        piezo=piezo,
    )
    search_freq, search_voltage, peak_index, boundary_hit, frf_search_seed_source, frf_search_seed_frequency_hz = _search_peak_with_adaptive_window(
        modal_model=modal_model,
        damping_ratio=mechanical.damping_ratio,
        resistance_ohm=piezo.resistance_ohm,
        search_scale=search_scale,
        search_points=search_points,
        peak_search_seed=peak_search_seed,
    )
    print(
        "FRF peak search seed: "
        f"source={frf_search_seed_source}, frequency_hz={frf_search_seed_frequency_hz:.6g}",
        flush=True,
    )
    _warn_if_open_circuit_resonance_is_inverted(
        search_freq_hz=search_freq,
        modal_model=modal_model,
        damping_ratio=mechanical.damping_ratio,
    )
    if boundary_hit:
        warnings.warn(
            "FRF peak search still landed on the search-window boundary after adaptive expansion.",
            stacklevel=2,
        )
    f_peak_hz = _refine_peak_frequency(
        search_freq_hz=search_freq,
        search_voltage=search_voltage,
        modal_model=modal_model,
        damping_ratio=mechanical.damping_ratio,
        resistance_ohm=piezo.resistance_ohm,
    )

    freq_hz = _inject_exact_frequency(
        np.linspace(
            normalized_range[0] * f_peak_hz,
            normalized_range[1] * f_peak_hz,
            int(frf_points),
            dtype=np.float64,
        ),
        f_peak_hz,
    )
    voltage = _evaluate_voltage_frf(
        frequencies_hz=freq_hz,
        modal_model=modal_model,
        damping_ratio=mechanical.damping_ratio,
        resistance_ohm=piezo.resistance_ohm,
    )
    saved_peak_ratio = float(freq_hz[int(np.argmax(np.abs(voltage)))] / f_peak_hz)
    if abs(saved_peak_ratio - 1.0) >= 5.0e-3:
        raise AssertionError(
            f"Saved FRF peak is not centered on f_peak_hz: saved_peak_ratio={saved_peak_ratio:.6g}."
        )
    mode1_top_surface_strain = np.zeros(0, dtype=np.float64)
    harmonic_top_surface_strain = np.zeros(0, dtype=np.float64)
    mode_dof_vectors = np.asarray(modal_model["mode_dof_vectors"], dtype=np.float64)
    if mode_dof_vectors.size > 0:
        mode1_top_surface_strain = _compute_top_surface_fe_strain_from_mode_dofs(
            raw_points=modal_model["raw_points"],
            raw_tetra_cells=modal_model["raw_tetra_cells"],
            raw_triangle_cells=modal_model["raw_triangle_cells"],
            raw_triangle_tags=modal_model["raw_triangle_tags"],
            mode_dof_vectors=mode_dof_vectors,
            element_order=element_order,
        )
        q_peak, _ = _solve_reduced_system(
            omega=2.0 * math.pi * f_peak_hz,
            modal_model=modal_model,
            damping_ratio=mechanical.damping_ratio,
            resistance_ohm=piezo.resistance_ohm,
        )
        harmonic_top_surface_strain = _compute_top_surface_fe_strain_from_mode_dofs(
            raw_points=modal_model["raw_points"],
            raw_tetra_cells=modal_model["raw_tetra_cells"],
            raw_triangle_cells=modal_model["raw_triangle_cells"],
            raw_triangle_tags=modal_model["raw_triangle_tags"],
            mode_dof_vectors=mode_dof_vectors,
            element_order=element_order,
            modal_coefficients=q_peak,
        )
    response_modal_diagnostics = compute_drive_coupling_diagnostics(
        eigenfreq_hz=np.asarray(modal_model["eigenfreq_hz"], dtype=np.float64),
        modal_force=np.asarray(modal_model["modal_force"], dtype=np.float64),
        modal_theta=np.asarray(modal_model["modal_theta"], dtype=np.float64),
    )
    response_modal_diagnostics.update(
        {
            "frf_search_seed_source": np.asarray([str(frf_search_seed_source)]),
            "frf_search_seed_frequency_hz": np.asarray([float(frf_search_seed_frequency_hz)], dtype=np.float64),
        }
    )
    response_path = save_fem_response(
        sample_id=sample_id,
        f_peak_hz=f_peak_hz,
        freq_hz=freq_hz,
        voltage_mag=np.abs(voltage),
        output_dir=response_dir,
        quality_flag=1,
        solver_provenance=build_solver_provenance_arrays(
            eigensolver_backend=str(np.asarray(modal_model["eigensolver_backend"]).reshape(-1)[0]),
            solver_element_order=int(np.asarray(modal_model["solver_element_order"], dtype=np.int32).reshape(-1)[0]),
            requested_solver_element_order=int(
                np.asarray(modal_model["requested_solver_element_order"], dtype=np.int32).reshape(-1)[0]
            ),
            requested_eigensolver_backend=str(
                np.asarray(modal_model["requested_eigensolver_backend"]).reshape(-1)[0]
            ),
            used_eigensolver_fallback=bool(np.asarray(modal_model["used_eigensolver_fallback"]).reshape(-1)[0]),
            used_element_order_fallback=bool(np.asarray(modal_model["used_element_order_fallback"]).reshape(-1)[0]),
            solver_parity_valid=bool(np.asarray(modal_model["solver_parity_valid"]).reshape(-1)[0]),
            parity_invalid_reason=str(np.asarray(modal_model["parity_invalid_reason"]).reshape(-1)[0]),
            strict_parity_requested=bool(np.asarray(modal_model["strict_parity_requested"]).reshape(-1)[0]),
            diagnostic_only=bool(np.asarray(modal_model["diagnostic_only"]).reshape(-1)[0]),
        ),
        modal_diagnostics=response_modal_diagnostics,
    )

    if modes_output_dir is not None:
        modes_output_dir = Path(modes_output_dir)
        modes_output_dir.mkdir(parents=True, exist_ok=True)
        modal_save = _build_modal_save_payload(
            sample_id=sample_id,
            element_order=element_order,
            mechanical=mechanical,
            piezo=piezo,
            modal_model=modal_model,
            mode1_top_surface_strain_eqv=mode1_top_surface_strain,
            harmonic_top_surface_strain_eqv=harmonic_top_surface_strain,
            harmonic_field_frequency_hz=f_peak_hz,
            frf_search_seed_source=frf_search_seed_source,
            frf_search_seed_frequency_hz=frf_search_seed_frequency_hz,
        )
        np.savez_compressed(
            modes_output_dir / f"sample_{sample_id:04d}_modal.npz",
            **modal_save,
        )
    return response_path


def _response_output_path(response_dir: str | Path, sample_id: int) -> Path:
    return Path(response_dir) / f"sample_{int(sample_id):04d}_response.npz"


def _modal_output_path(modes_output_dir: str | Path | None, sample_id: int) -> Path | None:
    if modes_output_dir is None:
        return None
    return Path(modes_output_dir) / f"sample_{int(sample_id):04d}_modal.npz"


def _modal_output_has_explicit_surface_fields(modal_path: Path) -> bool:
    if not modal_path.exists():
        return False
    try:
        with np.load(modal_path) as modal:
            return has_explicit_surface_strain_fields(modal)
    except Exception:
        return False


def solve_modal_voltage_frf_batch(
    mesh_paths: list[str | Path],
    response_dir: str | Path,
    num_modes: int = 8,
    search_scale: tuple[float, float] = (0.5, 2.0),
    search_points: int = 301,
    peak_search_seed: str = "f1",
    frf_points: int = 256,
    normalized_range: tuple[float, float] = (0.9, 1.1),
    mechanical: MechanicalConfig | None = None,
    piezo: PiezoConfig | None = None,
    modes_output_dir: str | Path | None = None,
    element_order: int = 2,
    requested_element_order: int | None = None,
    store_mode_shapes: bool = False,
    skip_existing: bool = False,
    eigensolver_backend: str = "auto",
    allow_eigensolver_fallback: bool = False,
    strict_parity: bool = False,
) -> list[Path]:
    response_dir = Path(response_dir)
    response_dir.mkdir(parents=True, exist_ok=True)
    if modes_output_dir is not None:
        Path(modes_output_dir).mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    total = len(mesh_paths)
    for idx, mesh_path in enumerate(mesh_paths, start=1):
        mesh_path = Path(mesh_path)
        sample_id = _extract_sample_id(mesh_path)
        response_path = _response_output_path(response_dir, sample_id)
        modal_path = _modal_output_path(modes_output_dir, sample_id)
        modal_ready = modal_path is None or modal_path.exists()
        modal_missing_strain = False
        if modal_path is not None and modal_path.exists() and store_mode_shapes:
            modal_missing_strain = not _modal_output_has_explicit_surface_fields(modal_path)
            modal_ready = modal_ready and not modal_missing_strain
        if skip_existing and response_path.exists() and modal_ready:
            print(f"[{idx}/{total}] Skipping {mesh_path.name} (existing outputs found).")
            saved.append(response_path)
            gc.collect()
            continue
        if modal_missing_strain:
            print(f"[{idx}/{total}] Solving {mesh_path.name} (refreshing missing explicit modal/harmonic strain data)")
        else:
            print(f"[{idx}/{total}] Solving {mesh_path.name}")
        try:
            saved.append(
                solve_modal_voltage_frf(
                    mesh_path=mesh_path,
                    response_dir=response_dir,
                    num_modes=num_modes,
                    search_scale=search_scale,
                    search_points=search_points,
                    peak_search_seed=peak_search_seed,
                    frf_points=frf_points,
                    normalized_range=normalized_range,
                    mechanical=mechanical,
                    piezo=piezo,
                    modes_output_dir=modes_output_dir,
                    element_order=element_order,
                    requested_element_order=requested_element_order,
                    store_mode_shapes=store_mode_shapes,
                    eigensolver_backend=eigensolver_backend,
                    allow_eigensolver_fallback=allow_eigensolver_fallback,
                    strict_parity=strict_parity,
                )
            )
        finally:
            gc.collect()
    return saved


def _resolve_mesh_paths(mesh_args: list[str], mesh_dir_arg: str) -> list[Path]:
    mesh_paths: list[Path] = [Path(value) for value in mesh_args if str(value).strip()]
    if mesh_dir_arg.strip():
        mesh_dir = Path(mesh_dir_arg)
        mesh_paths.extend(sorted(mesh_dir.glob("plate3d_*_fenicsx.npz")))
    # preserve order while removing duplicates
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in mesh_paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the base-excited piezoelectric plate FRF with FEniCSx modal reduction.",
    )
    parser.add_argument(
        "--mesh",
        action="append",
        default=[],
        help="Path to one 3D plate3d_*_fenicsx.npz mesh. Repeat the flag to solve multiple meshes.",
    )
    parser.add_argument(
        "--mesh-dir",
        default="",
        help="Optional directory containing plate3d_*_fenicsx.npz files to solve in one batch.",
    )
    parser.add_argument(
        "--response-dir",
        default="data/fem_responses",
        help="Output directory for sample responses.",
    )
    parser.add_argument(
        "--modes-dir",
        default="data/modal_data",
        help="Optional output directory for modal diagnostics.",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=8,
        help="Number of structural modes to retain.",
    )
    parser.add_argument(
        "--search-points",
        type=int,
        default=301,
        help="Number of coarse points used to locate the fundamental FRF peak before refinement.",
    )
    parser.add_argument(
        "--peak-search-seed",
        default="f1",
        choices=["auto", "f1", "dominant_coupling"],
        help=(
            "Seed for the adaptive FRF peak search. Production runs default to 'f1'; "
            "'dominant_coupling' remains an explicit debug override, and 'auto' only switches "
            "to dominant coupling when suspect mode ordering is detected."
        ),
    )
    parser.add_argument(
        "--frf-points",
        type=int,
        default=256,
        help="Number of samples saved in the final normalized FRF window.",
    )
    parser.add_argument(
        "--element-order",
        type=int,
        default=2,
        help="Lagrange order for the solid displacement field. Use 2 by default for thin-plate bending accuracy.",
    )
    parser.add_argument(
        "--requested-element-order",
        type=int,
        default=None,
        help="Original requested displacement order. Defaults to --element-order.",
    )
    parser.add_argument(
        "--eigensolver-backend",
        default="auto",
        choices=["shift_invert_lu", "iterative_gd", "auto"],
        help="Eigen backend. Strict parity defaults 'auto' to shift_invert_lu and rejects iterative_gd.",
    )
    parser.add_argument(
        "--allow-eigensolver-fallback",
        action="store_true",
        help="Permit automatic fallback from shift_invert_lu to iterative_gd for diagnostic runs only.",
    )
    parser.add_argument(
        "--strict-parity",
        action="store_true",
        help="Require the strict parity solver path: shift_invert_lu only, with no automatic eigensolver fallback.",
    )
    parser.add_argument(
        "--store-mode-shapes",
        action="store_true",
        help="Store per-mode nodal fields for extra diagnostics. Disabled by default to keep screening runs fast.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip meshes whose response (and modal file, if requested) already exist.",
    )
    parser.add_argument(
        "--substrate-rho",
        type=float,
        default=None,
        help="Substrate density in kg/m^3. Defaults to the shared problem spec when available.",
    )
    parser.add_argument(
        "--piezo-rho",
        type=float,
        default=None,
        help="Piezo density in kg/m^3. Defaults to the shared problem spec when available.",
    )
    parser.add_argument(
        "--problem-spec",
        default="",
        help=(
            "Optional shared problem specification YAML. Defaults to configs/peh_inverse_design_spec.yaml "
            "when present."
        ),
    )
    args = parser.parse_args()
    if bool(args.strict_parity) and bool(args.allow_eigensolver_fallback):
        raise ValueError("--allow-eigensolver-fallback cannot be combined with --strict-parity.")

    mesh_paths = _resolve_mesh_paths(args.mesh, args.mesh_dir)
    if not mesh_paths:
        raise ValueError("Provide at least one --mesh path or a --mesh-dir containing plate3d_*_fenicsx.npz files.")

    project_root = Path(__file__).resolve().parents[1]
    if args.problem_spec:
        problem_spec = load_problem_spec(args.problem_spec, project_root=project_root)
    else:
        default_spec_path = default_problem_spec_path(project_root)
        problem_spec = load_problem_spec(default_spec_path, project_root=project_root) if default_spec_path.exists() else None

    if problem_spec is not None:
        mechanical_kwargs = build_mechanical_config_kwargs(problem_spec)
        runtime_defaults = build_runtime_defaults(problem_spec)
        if args.substrate_rho is not None:
            mechanical_kwargs["substrate_rho"] = float(args.substrate_rho)
        else:
            mechanical_kwargs["substrate_rho"] = float(runtime_defaults["substrate_rho"])
        if args.piezo_rho is not None:
            mechanical_kwargs["piezo_rho"] = float(args.piezo_rho)
        else:
            mechanical_kwargs["piezo_rho"] = float(runtime_defaults["piezo_rho"])
        piezo_kwargs = build_piezo_config_kwargs(problem_spec)
        mechanical = MechanicalConfig(**mechanical_kwargs)
        piezo = PiezoConfig(**piezo_kwargs)
    else:
        mechanical = MechanicalConfig(
            substrate_rho=MechanicalConfig.substrate_rho if args.substrate_rho is None else float(args.substrate_rho),
            piezo_rho=MechanicalConfig.piezo_rho if args.piezo_rho is None else float(args.piezo_rho),
        )
        piezo = PiezoConfig()

    saved_paths = solve_modal_voltage_frf_batch(
        mesh_paths=[str(path) for path in mesh_paths],
        response_dir=args.response_dir,
        num_modes=int(args.num_modes),
        search_points=int(args.search_points),
        peak_search_seed=str(args.peak_search_seed),
        frf_points=int(args.frf_points),
        mechanical=mechanical,
        piezo=piezo,
        modes_output_dir=args.modes_dir,
        element_order=int(args.element_order),
        requested_element_order=(
            int(args.element_order) if args.requested_element_order is None else int(args.requested_element_order)
        ),
        store_mode_shapes=bool(args.store_mode_shapes),
        skip_existing=bool(args.skip_existing),
        eigensolver_backend=str(args.eigensolver_backend),
        allow_eigensolver_fallback=bool(args.allow_eigensolver_fallback),
        strict_parity=bool(args.strict_parity),
    )
    print(f"Saved {len(saved_paths)} response file(s) to {Path(args.response_dir)}")



if __name__ == "__main__":
    main()
