from __future__ import annotations

import argparse
import gc
import hashlib
import math
import re
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from peh_inverse_design.core.mesh_tags import FACET_TOP_ELECTRODE_TAG, VOLUME_PIEZO_TAG, VOLUME_SUBSTRATE_TAG
    from peh_inverse_design.geometry.modal_surface_fields import has_explicit_surface_strain_fields
    from peh_inverse_design.core.problem_spec import (
        build_runtime_defaults,
        build_mechanical_config_kwargs,
        build_piezo_config_kwargs,
        default_problem_spec_path,
        load_problem_spec,
    )
    from peh_inverse_design.datasets.response_dataset import normalize_voltage_amplitude_convention
    from peh_inverse_design.datasets.response_dataset import save_fem_response
    from peh_inverse_design.solver.modal_frf import evaluate_voltage_frf, solve_reduced_system
else:
    from ..core.mesh_tags import FACET_TOP_ELECTRODE_TAG, VOLUME_PIEZO_TAG, VOLUME_SUBSTRATE_TAG
    from ..geometry.modal_surface_fields import has_explicit_surface_strain_fields
    from ..core.problem_spec import (
        build_runtime_defaults,
        build_mechanical_config_kwargs,
        build_piezo_config_kwargs,
        default_problem_spec_path,
        load_problem_spec,
    )
    from ..datasets.response_dataset import normalize_voltage_amplitude_convention
    from ..datasets.response_dataset import save_fem_response
    from .modal_frf import evaluate_voltage_frf, solve_reduced_system

# Internal aliases kept for the existing call sites and tests.
_solve_reduced_system = solve_reduced_system
_evaluate_voltage_frf = evaluate_voltage_frf


EIGENSOLVER_BACKENDS = (
    "shift_invert_cholesky",
    "shift_invert_lu",
    "iterative_lobpcg_gamg",
    "iterative_gd_gamg",
    "shift_invert_cholesky_ooc",
)

_DIRECT_EIGENSOLVER_BACKENDS = {
    "shift_invert_cholesky",
    "shift_invert_lu",
    "shift_invert_cholesky_ooc",
}

PIEZO_ELASTIC_MODELS = ("anisotropic", "isotropic_vrh")
PIEZO_VOIGT_ORDER = "xx,yy,zz,yz,xz,xy"
ISOTROPIC_VRH_E_PA = 6.10655e10
ISOTROPIC_VRH_NU = 0.39480


def _normalize_piezo_elastic_model(model: str) -> str:
    normalized = str(model).strip().lower()
    if normalized not in PIEZO_ELASTIC_MODELS:
        choices = "|".join(PIEZO_ELASTIC_MODELS)
        raise ValueError(f"Unsupported piezo elastic model: {model}. Expected one of: {choices}")
    return normalized


def _normalize_piezo_voigt_order(voigt_order: object) -> str:
    if isinstance(voigt_order, (tuple, list)):
        tokens = [str(value).strip().lower() for value in voigt_order if str(value).strip()]
    else:
        tokens = [token.strip().lower() for token in str(voigt_order).replace(";", ",").split(",") if token.strip()]
    if not tokens:
        return PIEZO_VOIGT_ORDER
    return ",".join(tokens)


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
    capacitance_eps33s_f_per_m: float = 1.26934e-08
    elastic_model: str = "anisotropic"
    voigt_order: str = PIEZO_VOIGT_ORDER
    isotropic_E_pa: float = ISOTROPIC_VRH_E_PA
    isotropic_nu: float = ISOTROPIC_VRH_NU
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "elastic_model", _normalize_piezo_elastic_model(self.elastic_model))
        object.__setattr__(self, "voigt_order", _normalize_piezo_voigt_order(self.voigt_order))
        if float(self.isotropic_E_pa) <= 0.0:
            raise ValueError("isotropic_E_pa must be strictly positive.")
        nu = float(self.isotropic_nu)
        if not (-1.0 < nu < 0.5):
            raise ValueError("isotropic_nu must be between -1 and 0.5 for a stable isotropic material.")


def _extract_sample_id(mesh_path: Path) -> int:
    matches = re.findall(r"(\d+)", mesh_path.stem)
    if not matches:
        raise ValueError(f"Could not infer sample id from {mesh_path}.")
    return int(matches[-1])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str, dtype: object | None = None) -> object | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key] if dtype is None else np.asarray(data[key], dtype=dtype)).reshape(-1)
    if arr.size == 0:
        return None
    return arr[0]


def _load_solver_mesh_provenance(mesh_path: Path) -> dict[str, np.ndarray]:
    mesh_path = Path(mesh_path)
    provenance: dict[str, np.ndarray] = {
        "mesh_file_sha256": np.asarray([_file_sha256(mesh_path)]),
    }
    if mesh_path.suffix != ".npz":
        return provenance
    try:
        with np.load(mesh_path, allow_pickle=True) as mesh:
            for key in [
                "mesh_preset",
                "requested_solver_mesh_size_m",
                "solver_mesh_size_m",
                "solver_mesh_coarsening_passes",
                "solver_mesh_coarsening_allowed",
                "max_solver_vector_dofs",
                "estimated_q2_vector_dofs",
                "substrate_layers",
                "piezo_layers",
            ]:
                value = _npz_scalar(mesh, key)
                if value is not None:
                    provenance[key] = np.asarray([value])
            if "points" in mesh.files:
                provenance["mesh_point_count"] = np.asarray([int(np.asarray(mesh["points"]).shape[0])], dtype=np.int64)
            if "tetra_cells" in mesh.files:
                provenance["mesh_tetra_count"] = np.asarray([int(np.asarray(mesh["tetra_cells"]).shape[0])], dtype=np.int64)
    except Exception:
        return provenance
    return provenance


def _scalar_from_arrays(values: dict[str, np.ndarray], key: str) -> object | None:
    if key not in values:
        return None
    arr = np.asarray(values[key]).reshape(-1)
    if arr.size == 0:
        return None
    return arr[0]


def _scalars_match(actual: object, expected: object) -> bool:
    if isinstance(expected, (str, np.str_)):
        return str(actual) == str(expected)
    if isinstance(expected, (bool, np.bool_)):
        return bool(int(actual)) == bool(expected)
    if isinstance(expected, (int, np.integer)):
        return int(actual) == int(expected)
    try:
        return bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=1.0e-15))
    except Exception:
        return str(actual) == str(expected)


def _output_matches_mesh_provenance(
    output_path: Path,
    mesh_provenance: dict[str, np.ndarray],
    *,
    require_layer_metadata: bool,
) -> bool:
    if not output_path.exists():
        return False
    expected_hash = _scalar_from_arrays(mesh_provenance, "mesh_file_sha256")
    if expected_hash is None:
        return False
    try:
        with np.load(output_path, allow_pickle=True) as output:
            values = {key: np.asarray(output[key]) for key in output.files}
    except Exception:
        return False
    actual_hash = _scalar_from_arrays(values, "mesh_file_sha256")
    if actual_hash is None or str(actual_hash) != str(expected_hash):
        return False
    if not require_layer_metadata:
        return True
    for key in ["mesh_preset", "substrate_layers", "piezo_layers"]:
        expected = _scalar_from_arrays(mesh_provenance, key)
        if expected is None:
            continue
        actual = _scalar_from_arrays(values, key)
        if actual is None or not _scalars_match(actual, expected):
            return False
    return True


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


def _isotropic_bulk_modulus(E: float, nu: float) -> float:
    return float(E) / (3.0 * (1.0 - 2.0 * float(nu)))


def _isotropic_shear_modulus(E: float, nu: float) -> float:
    return float(E) / (2.0 * (1.0 + float(nu)))


def _piezo_effective_stiffness_matrix(piezo: PiezoConfig) -> np.ndarray:
    elastic_model = _normalize_piezo_elastic_model(piezo.elastic_model)
    if elastic_model == "anisotropic":
        return np.asarray(piezo.stiffness_cE_pa, dtype=np.float64)
    if elastic_model == "isotropic_vrh":
        return _isotropic_stiffness_matrix(float(piezo.isotropic_E_pa), float(piezo.isotropic_nu))
    raise AssertionError(f"Unhandled piezo elastic model: {piezo.elastic_model}")


def _with_piezo_elastic_model(piezo: PiezoConfig, piezo_elastic_model: str | None) -> PiezoConfig:
    if piezo_elastic_model is None:
        return replace(piezo, elastic_model=_normalize_piezo_elastic_model(piezo.elastic_model))
    return replace(piezo, elastic_model=_normalize_piezo_elastic_model(piezo_elastic_model))


def _piezo_elastic_metadata(piezo: PiezoConfig) -> dict[str, np.ndarray]:
    elastic_model = _normalize_piezo_elastic_model(piezo.elastic_model)
    if elastic_model == "isotropic_vrh":
        isotropic_E_pa = float(piezo.isotropic_E_pa)
        isotropic_nu = float(piezo.isotropic_nu)
        isotropic_bulk_modulus_pa = _isotropic_bulk_modulus(isotropic_E_pa, isotropic_nu)
        isotropic_shear_modulus_pa = _isotropic_shear_modulus(isotropic_E_pa, isotropic_nu)
    else:
        isotropic_E_pa = float("nan")
        isotropic_nu = float("nan")
        isotropic_bulk_modulus_pa = float("nan")
        isotropic_shear_modulus_pa = float("nan")
    return {
        "piezo_elastic_model": np.asarray([elastic_model]),
        "piezo_voigt_order": np.asarray([_normalize_piezo_voigt_order(piezo.voigt_order)]),
        "isotropic_E_pa": np.asarray([isotropic_E_pa], dtype=np.float64),
        "isotropic_nu": np.asarray([isotropic_nu], dtype=np.float64),
        "isotropic_bulk_modulus_pa": np.asarray([isotropic_bulk_modulus_pa], dtype=np.float64),
        "isotropic_shear_modulus_pa": np.asarray([isotropic_shear_modulus_pa], dtype=np.float64),
    }


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
        if (
            "MUMPS error in numerical factorization" in text
            or "INFOG(1)=-13" in text
            or "INFOG(1) = -13" in text
            or "INFOG(1)=-9" in text
            or "INFOG(1) = -9" in text
        ):
            return True
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return False


def _is_petsc_out_of_memory_failure(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        text = str(current).strip().lower()
        if "out of memory" in text:
            return True
        if "memory requested" in text and ("error code 55" in text or "petsc" in text):
            return True
        if "petscmallocalign" in text and "memory" in text:
            return True
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return False


def _normalize_eigensolver_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    legacy_aliases = {
        "iterative_gd": "iterative_gd_gamg",
        "gd": "iterative_gd_gamg",
        "lobpcg": "iterative_lobpcg_gamg",
    }
    normalized = legacy_aliases.get(normalized, normalized)
    if normalized not in EIGENSOLVER_BACKENDS:
        choices = "|".join(EIGENSOLVER_BACKENDS)
        raise ValueError(f"Unsupported eigensolver backend: {backend}. Expected one of: {choices}")
    return normalized


def _set_mumps_ooc_options(*, PETSc, comm) -> dict[str, int]:
    options: dict[str, int] = {
        "mat_mumps_icntl_22": 1,
        "mat_mumps_icntl_14": 200,
    }
    comm_size = int(getattr(comm, "size", 1))
    if comm_size > 1:
        options["mat_mumps_icntl_28"] = 2
        options["mat_mumps_icntl_29"] = 2

    petsc_options = PETSc.Options()
    for key, value in options.items():
        petsc_options[key] = str(int(value))
    rendered = ", ".join(f"{key}={value}" for key, value in options.items())
    print(f"Using MUMPS out-of-core options: {rendered}", flush=True)
    return options


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
    normalized_backend = _normalize_eigensolver_backend(backend)
    eps_solver = SLEPc.EPS().create(comm)
    eps_solver.setOperators(K, M)
    eps_solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps_solver.setDimensions(num_modes)

    if normalized_backend in _DIRECT_EIGENSOLVER_BACKENDS:
        if normalized_backend == "shift_invert_cholesky_ooc":
            _set_mumps_ooc_options(PETSc=PETSc, comm=comm)
        eps_solver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps_solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps_solver.setTarget(0.0)
        st = eps_solver.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        ksp = st.getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        if normalized_backend == "shift_invert_lu":
            pc.setType(PETSc.PC.Type.LU)
        else:
            pc.setType(PETSc.PC.Type.CHOLESKY)
        pc.setFactorSolverType("mumps")
        eps_solver.setFromOptions()
        return eps_solver, normalized_backend

    if normalized_backend in {"iterative_gd_gamg", "iterative_lobpcg_gamg"}:
        if normalized_backend == "iterative_lobpcg_gamg":
            eps_solver.setType(SLEPc.EPS.Type.LOBPCG)
        else:
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
        return eps_solver, normalized_backend

    raise AssertionError(f"Unhandled eigensolver backend: {backend}")


def _assemble_modal_model(
    mesh_path: Path,
    num_modes: int,
    mechanical: MechanicalConfig,
    piezo: PiezoConfig,
    element_order: int = 1,
    store_mode_shapes: bool = False,
    eigensolver_backend: str = "shift_invert_cholesky",
) -> dict[str, np.ndarray]:
    MPI, PETSc, SLEPc, fem, fem_petsc, io, io_gmsh, ufl = _load_fenicsx()
    import basix.ufl  # type: ignore
    import dolfinx.mesh as dmesh  # type: ignore

    comm = MPI.COMM_WORLD
    mesh_path = Path(mesh_path)
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
    C_pz = ufl.as_matrix(_piezo_effective_stiffness_matrix(piezo).tolist())

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
    requested_eigensolver_backend = _normalize_eigensolver_backend(eigensolver_backend)
    actual_eigensolver_backend = requested_eigensolver_backend
    try:
        K = fem_petsc.assemble_matrix(a_form, bcs=bcs, diag=1.0)
        M = fem_petsc.assemble_matrix(m_form, bcs=bcs, diag=0.0)
        K.assemble()
        M.assemble()

        eps_solver, actual_eigensolver_backend = _build_eps_solver(
            comm=comm,
            K=K,
            M=M,
            num_modes=num_modes,
            PETSc=PETSc,
            SLEPc=SLEPc,
            backend=requested_eigensolver_backend,
        )
        try:
            eps_solver.solve()
        except Exception as exc:
            _destroy_petsc_object(eps_solver)
            eps_solver = None
            if (
                actual_eigensolver_backend in _DIRECT_EIGENSOLVER_BACKENDS
                and (_is_mumps_factorization_failure(exc) or _is_petsc_out_of_memory_failure(exc))
            ):
                raise RuntimeError(
                    f"Eigensolver backend '{actual_eigensolver_backend}' failed during sparse direct "
                    "factorization. PETSc/SLEPc objects were destroyed; restart a fresh solver process "
                    "with a different backend or MPI-rank setting."
                ) from exc
            raise

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
        capacitance_eps33s_f_per_m = getattr(piezo, "capacitance_eps33s_f_per_m", piezo.eps33s_f_per_m)
        capacitance = capacitance_eps33s_f_per_m * piezo_volume / (piezo.thickness_m ** 2)
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

        return {
            "element_order": np.asarray([int(element_order)], dtype=np.int32),
            "eigenfreq_hz": np.asarray(eigenfreq_hz, dtype=np.float64),
            "modal_force": np.asarray(modal_force, dtype=np.float64),
            "modal_theta": np.asarray(modal_theta, dtype=np.float64),
            "modal_mass": np.asarray(modal_mass, dtype=np.float64),
            "capacitance_f": np.asarray([capacitance], dtype=np.float64),
            "capacitance_eps33s_f_per_m": np.asarray([capacitance_eps33s_f_per_m], dtype=np.float64),
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
            "eigensolver_backend": np.asarray([str(actual_eigensolver_backend)]),
        }
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
    f1_hz: float,
    lower_factor: float,
    upper_factor: float,
    search_points: int,
) -> np.ndarray:
    f_min = max(1.0e-9, float(lower_factor) * float(f1_hz))
    f_max = float(upper_factor) * float(f1_hz)
    f_max = max(f_max, 1.05 * float(f1_hz))
    if f_max <= f_min:
        f_max = max(1.10 * float(f1_hz), 1.05 * f_min)
    return np.linspace(f_min, f_max, int(search_points), dtype=np.float64)


def _search_peak_with_adaptive_window(
    modal_model: dict[str, np.ndarray],
    damping_ratio: float,
    resistance_ohm: float,
    search_scale: tuple[float, float],
    search_points: int,
    max_expansions: int = 12,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    freq_n = np.sort(np.asarray(modal_model["eigenfreq_hz"], dtype=np.float64).reshape(-1))
    if freq_n.size == 0:
        raise ValueError("modal_model does not contain any eigenfrequencies.")
    f1_hz = float(freq_n[0])
    lower_factor = float(search_scale[0])
    upper_factor = float(search_scale[1])

    search_freq = _build_peak_search_grid(
        f1_hz=f1_hz,
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
            f1_hz=f1_hz,
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
    return search_freq, search_voltage, peak_index, boundary_hit


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

    piezo_ex = float(_piezo_effective_stiffness_matrix(piezo)[0, 0])
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
        f"modal_force[:6]={np.array2string(modal_force[:6], precision=6, separator=', ')}"
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
    mesh_provenance: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    mode1_frequency_hz = float(np.asarray(modal_model["eigenfreq_hz"], dtype=np.float64).reshape(-1)[0])
    harmonic_frequency_array = np.asarray(harmonic_field_frequency_hz, dtype=np.float64)
    payload = {
        "sample_id": np.asarray(sample_id, dtype=np.int32),
        "element_order": np.asarray([int(element_order)], dtype=np.int32),
        "eigenfreq_hz": modal_model["eigenfreq_hz"],
        "modal_force": modal_model["modal_force"],
        "modal_theta": modal_model["modal_theta"],
        "modal_mass": modal_model["modal_mass"],
        "capacitance_f": modal_model["capacitance_f"],
        "capacitance_eps33s_f_per_m": modal_model.get(
            "capacitance_eps33s_f_per_m",
            np.asarray([piezo.capacitance_eps33s_f_per_m], dtype=np.float64),
        ),
        "eigensolver_backend": np.asarray(modal_model.get("eigensolver_backend", ["unknown"])),
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
    payload.update(_piezo_elastic_metadata(piezo))
    if mesh_provenance is not None:
        for key, value in mesh_provenance.items():
            if key in payload:
                raise ValueError(f"Mesh provenance cannot overwrite modal field '{key}'.")
            payload[str(key)] = np.asarray(value)
    return payload


def solve_modal_voltage_frf(
    mesh_path: str | Path,
    response_dir: str | Path,
    num_modes: int = 8,
    search_scale: tuple[float, float] = (0.5, 2.0),
    search_points: int = 301,
    frf_points: int = 256,
    normalized_range: tuple[float, float] = (0.9, 1.1),
    mechanical: MechanicalConfig | None = None,
    piezo: PiezoConfig | None = None,
    modes_output_dir: str | Path | None = None,
    element_order: int = 2,
    store_mode_shapes: bool = False,
    eigensolver_backend: str = "shift_invert_cholesky",
    piezo_elastic_model: str | None = None,
    house_voltage_amplitude_convention: str = "peak",
) -> Path:
    mechanical = mechanical or MechanicalConfig()
    piezo = _with_piezo_elastic_model(piezo or PiezoConfig(), piezo_elastic_model)
    mesh_path = Path(mesh_path)
    sample_id = _extract_sample_id(mesh_path)
    mesh_provenance = _load_solver_mesh_provenance(mesh_path)

    modal_model = _assemble_modal_model(
        mesh_path=mesh_path,
        num_modes=num_modes,
        mechanical=mechanical,
        piezo=piezo,
        element_order=element_order,
        store_mode_shapes=store_mode_shapes,
        eigensolver_backend=eigensolver_backend,
    )
    _log_modal_diagnostics(modal_model)
    _warn_if_frequency_scale_is_suspicious(
        modal_model=modal_model,
        mechanical=mechanical,
        piezo=piezo,
    )
    search_freq, search_voltage, peak_index, boundary_hit = _search_peak_with_adaptive_window(
        modal_model=modal_model,
        damping_ratio=mechanical.damping_ratio,
        resistance_ohm=piezo.resistance_ohm,
        search_scale=search_scale,
        search_points=search_points,
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
    response_path = save_fem_response(
        sample_id=sample_id,
        f_peak_hz=f_peak_hz,
        freq_hz=freq_hz,
        voltage_mag=np.abs(voltage),
        output_dir=response_dir,
        quality_flag=1,
        voltage_amplitude_convention=house_voltage_amplitude_convention,
        metadata=mesh_provenance,
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
            mesh_provenance=mesh_provenance,
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


def _modal_output_matches_element_order(
    modal_path: Path,
    expected_element_order: int,
    mesh_provenance: dict[str, np.ndarray] | None = None,
    expected_eigensolver_backend: str | None = None,
    expected_piezo_elastic_model: str | None = None,
) -> bool:
    if not modal_path.exists():
        return False
    try:
        with np.load(modal_path, allow_pickle=True) as modal:
            if "element_order" not in modal.files:
                return False
            actual = int(np.asarray(modal["element_order"], dtype=np.int32).reshape(-1)[0])
            if expected_eigensolver_backend is not None:
                if "eigensolver_backend" not in modal.files:
                    return False
                actual_backend = _normalize_eigensolver_backend(str(np.asarray(modal["eigensolver_backend"]).reshape(-1)[0]))
            if expected_piezo_elastic_model is not None:
                if "piezo_elastic_model" not in modal.files:
                    return False
                actual_piezo_elastic_model = _normalize_piezo_elastic_model(
                    str(np.asarray(modal["piezo_elastic_model"]).reshape(-1)[0])
                )
    except Exception:
        return False
    if actual != int(expected_element_order):
        return False
    if expected_eigensolver_backend is not None and actual_backend != _normalize_eigensolver_backend(expected_eigensolver_backend):
        return False
    if (
        expected_piezo_elastic_model is not None
        and actual_piezo_elastic_model != _normalize_piezo_elastic_model(expected_piezo_elastic_model)
    ):
        return False
    if mesh_provenance is not None and not _output_matches_mesh_provenance(
        modal_path,
        mesh_provenance,
        require_layer_metadata=True,
    ):
        return False
    return True


def _response_output_is_peak(
    response_path: Path,
    mesh_provenance: dict[str, np.ndarray] | None = None,
) -> bool:
    if not response_path.exists():
        return False
    try:
        with np.load(response_path, allow_pickle=True) as response:
            if "peak_voltage_form" in response.files:
                normalize_voltage_amplitude_convention(str(np.asarray(response["peak_voltage_form"]).reshape(-1)[0]))
    except Exception:
        return False
    if mesh_provenance is not None and not _output_matches_mesh_provenance(
        response_path,
        mesh_provenance,
        require_layer_metadata=True,
    ):
        return False
    return True


def solve_modal_voltage_frf_batch(
    mesh_paths: list[str | Path],
    response_dir: str | Path,
    num_modes: int = 8,
    search_scale: tuple[float, float] = (0.5, 2.0),
    search_points: int = 301,
    frf_points: int = 256,
    normalized_range: tuple[float, float] = (0.9, 1.1),
    mechanical: MechanicalConfig | None = None,
    piezo: PiezoConfig | None = None,
    modes_output_dir: str | Path | None = None,
    element_order: int = 2,
    store_mode_shapes: bool = False,
    eigensolver_backend: str = "shift_invert_cholesky",
    piezo_elastic_model: str | None = None,
    skip_existing: bool = False,
    house_voltage_amplitude_convention: str = "peak",
) -> list[Path]:
    piezo_elastic_model = (
        _normalize_piezo_elastic_model(piezo_elastic_model)
        if piezo_elastic_model is not None
        else _normalize_piezo_elastic_model((piezo or PiezoConfig()).elastic_model)
    )
    response_dir = Path(response_dir)
    response_dir.mkdir(parents=True, exist_ok=True)
    if modes_output_dir is not None:
        Path(modes_output_dir).mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    total = len(mesh_paths)
    for idx, mesh_path in enumerate(mesh_paths, start=1):
        mesh_path = Path(mesh_path)
        sample_id = _extract_sample_id(mesh_path)
        mesh_provenance = _load_solver_mesh_provenance(mesh_path)
        response_path = _response_output_path(response_dir, sample_id)
        response_ready = _response_output_is_peak(response_path, mesh_provenance)
        modal_path = _modal_output_path(modes_output_dir, sample_id)
        modal_ready = modal_path is None or _modal_output_matches_element_order(
            modal_path,
            int(element_order),
            mesh_provenance,
            expected_eigensolver_backend=eigensolver_backend,
            expected_piezo_elastic_model=piezo_elastic_model,
        )
        modal_missing_strain = False
        if modal_path is not None and modal_ready and store_mode_shapes:
            modal_missing_strain = not _modal_output_has_explicit_surface_fields(modal_path)
            modal_ready = modal_ready and not modal_missing_strain
        if skip_existing and response_ready and modal_ready:
            print(f"[{idx}/{total}] Skipping {mesh_path.name} (existing outputs found).")
            saved.append(response_path)
            gc.collect()
            continue
        if modal_missing_strain:
            print(f"[{idx}/{total}] Solving {mesh_path.name} (refreshing missing explicit modal/harmonic strain data)")
        elif modal_path is not None and modal_path.exists() and not modal_ready:
            print(f"[{idx}/{total}] Solving {mesh_path.name} (existing modal output has a different element_order/backend)")
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
                    frf_points=frf_points,
                    normalized_range=normalized_range,
                    mechanical=mechanical,
                    piezo=piezo,
                    modes_output_dir=modes_output_dir,
                    element_order=element_order,
                    store_mode_shapes=store_mode_shapes,
                    eigensolver_backend=eigensolver_backend,
                    piezo_elastic_model=piezo_elastic_model,
                    house_voltage_amplitude_convention=house_voltage_amplitude_convention,
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
        "--frf-points",
        type=int,
        default=256,
        help="Number of samples saved in the final normalized FRF window.",
    )
    parser.add_argument(
        "--freq-ratio-min",
        type=float,
        default=0.9,
        help="Lower bound of the normalized FRF window as a fraction of the peak frequency f_peak.",
    )
    parser.add_argument(
        "--freq-ratio-max",
        type=float,
        default=1.1,
        help="Upper bound of the normalized FRF window as a fraction of the peak frequency f_peak.",
    )
    parser.add_argument(
        "--element-order",
        type=int,
        default=2,
        help="Lagrange order for the solid displacement field. Use 2 by default for thin-plate bending accuracy.",
    )
    parser.add_argument(
        "--store-mode-shapes",
        action="store_true",
        help="Store per-mode nodal fields for extra diagnostics. Disabled by default to keep screening runs fast.",
    )
    parser.add_argument(
        "--eigensolver-backend",
        default="shift_invert_cholesky",
        choices=list(EIGENSOLVER_BACKENDS),
        help="SLEPc/PETSc eigensolver backend. Cholesky/MUMPS shift-invert is the default parity backend.",
    )
    parser.add_argument(
        "--piezo-elastic-model",
        default="anisotropic",
        choices=list(PIEZO_ELASTIC_MODELS),
        help="PZT elastic stiffness model: original anisotropic ANSYS 3D matrix or isotropic VRH diagnostic.",
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
    parser.add_argument(
        "--house-voltage-amplitude-convention",
        default=None,
        choices=["peak"],
        help="Canonical in-house voltage convention. Only peak amplitudes are supported.",
    )
    args = parser.parse_args()

    mesh_paths = _resolve_mesh_paths(args.mesh, args.mesh_dir)
    if not mesh_paths:
        raise ValueError("Provide at least one --mesh path or a --mesh-dir containing plate3d_*_fenicsx.npz files.")

    project_root = Path(__file__).resolve().parents[2]
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
        house_voltage_amplitude_convention = str(
            runtime_defaults["house_voltage_amplitude_convention"]
            if args.house_voltage_amplitude_convention is None
            else args.house_voltage_amplitude_convention
        )
    else:
        mechanical = MechanicalConfig(
            substrate_rho=MechanicalConfig.substrate_rho if args.substrate_rho is None else float(args.substrate_rho),
            piezo_rho=MechanicalConfig.piezo_rho if args.piezo_rho is None else float(args.piezo_rho),
        )
        piezo = PiezoConfig()
        house_voltage_amplitude_convention = (
            "peak" if args.house_voltage_amplitude_convention is None else str(args.house_voltage_amplitude_convention)
        )

    saved_paths = solve_modal_voltage_frf_batch(
        mesh_paths=[str(path) for path in mesh_paths],
        response_dir=args.response_dir,
        num_modes=int(args.num_modes),
        search_points=int(args.search_points),
        frf_points=int(args.frf_points),
        normalized_range=(float(args.freq_ratio_min), float(args.freq_ratio_max)),
        mechanical=mechanical,
        piezo=piezo,
        modes_output_dir=args.modes_dir,
        element_order=int(args.element_order),
        store_mode_shapes=bool(args.store_mode_shapes),
        eigensolver_backend=str(args.eigensolver_backend),
        piezo_elastic_model=str(args.piezo_elastic_model),
        skip_existing=bool(args.skip_existing),
        house_voltage_amplitude_convention=house_voltage_amplitude_convention,
    )
    print(f"Saved {len(saved_paths)} response file(s) to {Path(args.response_dir)}")


def _run_cli() -> None:
    try:
        main()
    except Exception as exc:
        if _is_petsc_out_of_memory_failure(exc) or _is_mumps_factorization_failure(exc):
            print(
                "FEniCSx modal solve ran out of PETSc/SLEPc direct-factorization memory; exiting with "
                "status 137 so the pipeline can retry in a fresh container with different solver settings.",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(137) from exc
        raise


if __name__ == "__main__":
    _run_cli()
