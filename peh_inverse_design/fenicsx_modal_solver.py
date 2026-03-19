from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from peh_inverse_design.mesh_tags import FACET_TOP_ELECTRODE_TAG, VOLUME_PIEZO_TAG, VOLUME_SUBSTRATE_TAG
    from peh_inverse_design.response_dataset import save_fem_response
else:
    from .mesh_tags import FACET_TOP_ELECTRODE_TAG, VOLUME_PIEZO_TAG, VOLUME_SUBSTRATE_TAG
    from .response_dataset import save_fem_response


@dataclass(frozen=True)
class MechanicalConfig:
    substrate_E_pa: float = 1.9305e11
    substrate_nu: float = 0.30
    substrate_rho: float = 7930.0
    piezo_rho: float = 7800.0
    damping_ratio: float = 0.025
    base_acceleration_m_per_s2: float = 2.5


@dataclass(frozen=True)
class PiezoConfig:
    thickness_m: float = 2.667e-4
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


def _extract_sample_id(mesh_path: Path) -> int:
    matches = re.findall(r"(\d+)", mesh_path.stem)
    if not matches:
        raise ValueError(f"Could not infer sample id from {mesh_path}.")
    return int(matches[-1])


def _build_component_point_maps(V, raw_points: np.ndarray):
    raw_points = np.asarray(raw_points, dtype=np.float64)
    point_lookup = {
        tuple(np.round(point, 9)): idx
        for idx, point in enumerate(raw_points)
    }
    component_maps: list[tuple[np.ndarray, np.ndarray]] = []
    for comp in range(3):
        Vc, parent_dofs = V.sub(comp).collapse()
        coords = np.asarray(Vc.tabulate_dof_coordinates(), dtype=np.float64).reshape(-1, 3)
        point_ids = np.empty(coords.shape[0], dtype=np.int64)
        for idx, coord in enumerate(coords):
            key = tuple(np.round(coord, 9))
            raw_idx = point_lookup.get(key)
            if raw_idx is None:
                dist2 = np.sum((raw_points - coord) ** 2, axis=1)
                raw_idx = int(np.argmin(dist2))
                if dist2[raw_idx] > 1.0e-16:
                    raise KeyError(f"Could not match dof coordinate {coord} to raw mesh points.")
            point_ids[idx] = raw_idx
        component_maps.append((np.asarray(parent_dofs, dtype=np.int64), point_ids))
    return component_maps


def _mode_to_nodal_displacement(
    mode_values: np.ndarray,
    component_maps: list[tuple[np.ndarray, np.ndarray]],
    n_points: int,
) -> np.ndarray:
    nodal = np.zeros((n_points, 3), dtype=np.float64)
    for comp, (parent_dofs, point_ids) in enumerate(component_maps):
        nodal[point_ids, comp] = mode_values[parent_dofs]
    return nodal


def _equivalent_surface_strain_from_displacement(
    points: np.ndarray,
    triangle: np.ndarray,
    displacement: np.ndarray,
) -> float:
    xy = np.asarray(points[triangle, :2], dtype=np.float64)
    uv = np.asarray(displacement[triangle, :2], dtype=np.complex128)
    A = np.column_stack([np.ones(3), xy])
    invA = np.linalg.inv(A)
    grads = invA[1:, :].T

    grad_u = uv[:, 0] @ grads
    grad_v = uv[:, 1] @ grads
    eps_xx = grad_u[0]
    eps_yy = grad_v[1]
    eps_xy = 0.5 * (grad_u[1] + grad_v[0])

    eq = np.sqrt(
        np.real(
            eps_xx * np.conj(eps_xx)
            - eps_xx * np.conj(eps_yy)
            + eps_yy * np.conj(eps_yy)
            + 3.0 * eps_xy * np.conj(eps_xy)
        )
    )
    return float(eq)


def _compute_top_surface_strain(
    points: np.ndarray,
    triangle_cells: np.ndarray,
    triangle_tags: np.ndarray,
    nodal_displacement: np.ndarray,
) -> np.ndarray:
    top_triangles = np.asarray(triangle_cells[triangle_tags == FACET_TOP_ELECTRODE_TAG], dtype=np.int64)
    if top_triangles.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(
        [
            _equivalent_surface_strain_from_displacement(points, tri, nodal_displacement)
            for tri in top_triangles
        ],
        dtype=np.float64,
    )


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


def _assemble_modal_model(
    mesh_path: Path,
    num_modes: int,
    mechanical: MechanicalConfig,
    piezo: PiezoConfig,
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
    if mesh_path.suffix == ".npz":
        raw = np.load(mesh_path)
        raw_points = np.asarray(raw["points"], dtype=np.float64)
        raw_tetra_cells = np.asarray(raw["tetra_cells"], dtype=np.int64)
        raw_tetra_tags = np.asarray(raw["tetra_tags"], dtype=np.int32)
        raw_triangle_cells = np.asarray(raw["triangle_cells"], dtype=np.int32)
        raw_triangle_tags = np.asarray(raw["triangle_tags"], dtype=np.int32)

        domain = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,), dtype=np.float64))
        mesh = dmesh.create_mesh(comm, raw_tetra_cells, domain, raw_points)
        cell_entities = np.arange(raw_tetra_cells.shape[0], dtype=np.int32)
        cell_tags = dmesh.meshtags(mesh, mesh.topology.dim, cell_entities, raw_tetra_tags)
    elif mesh_path.suffix == ".xdmf":
        with io.XDMFFile(comm, str(mesh_path), "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            cell_tags = xdmf.read_meshtags(mesh, name="Grid")
    else:
        mesh, cell_tags, _ = io_gmsh.read_from_msh(str(mesh_path), comm, 0, gdim=3)
    gdim = mesh.geometry.dim
    V = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    component_maps = _build_component_point_maps(V, raw_points) if raw_points.size else []

    fdim = mesh.topology.dim - 1
    x_coords = np.asarray(mesh.geometry.x, dtype=np.float64)
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

    K = fem_petsc.assemble_matrix(a_form, bcs=bcs, diag=1.0)
    M = fem_petsc.assemble_matrix(m_form, bcs=bcs, diag=0.0)
    K.assemble()
    M.assemble()

    eps_solver = SLEPc.EPS().create(comm)
    eps_solver.setOperators(K, M)
    eps_solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps_solver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps_solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps_solver.setTarget(0.0)
    eps_solver.setDimensions(num_modes)
    st = eps_solver.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    eps_solver.setFromOptions()
    eps_solver.solve()

    nconv = eps_solver.getConverged()
    if nconv <= 0:
        raise RuntimeError("SLEPc did not converge any eigenpairs.")

    e_col3 = np.asarray(piezo.e_matrix_c_per_m2, dtype=np.float64)[:, 2]
    e_col3_constant = ufl.as_vector(np.asarray(e_col3, dtype=np.float64).tolist())
    one = PETSc.ScalarType(1.0)

    piezo_volume_local = fem.assemble_scalar(fem.form(one * dx(VOLUME_PIEZO_TAG)))
    piezo_volume = comm.allreduce(piezo_volume_local, op=MPI.SUM)
    capacitance = piezo.eps33s_f_per_m * piezo_volume / (piezo.thickness_m ** 2)

    eigenfreq_hz: list[float] = []
    modal_force: list[float] = []
    modal_theta: list[float] = []
    modal_mass: list[float] = []
    mode_nodal_displacements: list[np.ndarray] = []

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
        if component_maps:
            mode_nodal_displacements.append(
                _mode_to_nodal_displacement(mode.x.array, component_maps, raw_points.shape[0])
            )

    if not eigenfreq_hz:
        raise RuntimeError("No positive eigenfrequencies were extracted from the mesh.")

    return {
        "eigenfreq_hz": np.asarray(eigenfreq_hz, dtype=np.float64),
        "modal_force": np.asarray(modal_force, dtype=np.float64),
        "modal_theta": np.asarray(modal_theta, dtype=np.float64),
        "modal_mass": np.asarray(modal_mass, dtype=np.float64),
        "capacitance_f": np.asarray([capacitance], dtype=np.float64),
        "mode_nodal_displacements": np.asarray(mode_nodal_displacements, dtype=np.float64),
        "raw_points": np.asarray(raw_points, dtype=np.float64),
        "raw_tetra_cells": np.asarray(raw_tetra_cells, dtype=np.int64),
        "raw_tetra_tags": np.asarray(raw_tetra_tags, dtype=np.int32),
        "raw_triangle_cells": np.asarray(raw_triangle_cells, dtype=np.int32),
        "raw_triangle_tags": np.asarray(raw_triangle_tags, dtype=np.int32),
    }


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
    A[-1, -1] = (1.0 / resistance_ohm) - 1j * omega * capacitance
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


def solve_modal_voltage_frf(
    mesh_path: str | Path,
    response_dir: str | Path,
    num_modes: int = 12,
    search_scale: tuple[float, float] = (0.8, 1.2),
    search_points: int = 801,
    frf_points: int = 256,
    normalized_range: tuple[float, float] = (0.9, 1.1),
    mechanical: MechanicalConfig | None = None,
    piezo: PiezoConfig | None = None,
    modes_output_dir: str | Path | None = None,
) -> Path:
    mechanical = mechanical or MechanicalConfig()
    piezo = piezo or PiezoConfig()
    mesh_path = Path(mesh_path)
    sample_id = _extract_sample_id(mesh_path)

    modal_model = _assemble_modal_model(
        mesh_path=mesh_path,
        num_modes=num_modes,
        mechanical=mechanical,
        piezo=piezo,
    )
    f1 = float(modal_model["eigenfreq_hz"][0])
    search_freq = np.linspace(search_scale[0] * f1, search_scale[1] * f1, int(search_points), dtype=np.float64)
    search_voltage = _evaluate_voltage_frf(
        frequencies_hz=search_freq,
        modal_model=modal_model,
        damping_ratio=mechanical.damping_ratio,
        resistance_ohm=piezo.resistance_ohm,
    )
    search_voltage_mag = np.abs(search_voltage)
    peak_index = int(np.argmax(search_voltage_mag))
    f_peak_hz = float(search_freq[peak_index])

    freq_hz = np.linspace(
        normalized_range[0] * f_peak_hz,
        normalized_range[1] * f_peak_hz,
        int(frf_points),
        dtype=np.float64,
    )
    voltage = _evaluate_voltage_frf(
        frequencies_hz=freq_hz,
        modal_model=modal_model,
        damping_ratio=mechanical.damping_ratio,
        resistance_ohm=piezo.resistance_ohm,
    )
    top_surface_strain = np.zeros(0, dtype=np.float64)
    mode_nodal_displacements = np.asarray(modal_model["mode_nodal_displacements"], dtype=np.float64)
    if mode_nodal_displacements.size > 0:
        q_peak, _ = _solve_reduced_system(
            omega=2.0 * math.pi * f_peak_hz,
            modal_model=modal_model,
            damping_ratio=mechanical.damping_ratio,
            resistance_ohm=piezo.resistance_ohm,
        )
        nodal_peak = np.tensordot(q_peak, mode_nodal_displacements, axes=(0, 0))
        top_surface_strain = _compute_top_surface_strain(
            points=modal_model["raw_points"],
            triangle_cells=modal_model["raw_triangle_cells"],
            triangle_tags=modal_model["raw_triangle_tags"],
            nodal_displacement=nodal_peak,
        )
    response_path = save_fem_response(
        sample_id=sample_id,
        f_peak_hz=f_peak_hz,
        freq_hz=freq_hz,
        voltage_mag=np.abs(voltage),
        output_dir=response_dir,
        quality_flag=1,
    )

    if modes_output_dir is not None:
        modes_output_dir = Path(modes_output_dir)
        modes_output_dir.mkdir(parents=True, exist_ok=True)
        modal_save = {
            "sample_id": np.asarray(sample_id, dtype=np.int32),
            "eigenfreq_hz": modal_model["eigenfreq_hz"],
            "modal_force": modal_model["modal_force"],
            "modal_theta": modal_model["modal_theta"],
            "modal_mass": modal_model["modal_mass"],
            "capacitance_f": modal_model["capacitance_f"],
            "field_frequency_hz": np.asarray(f_peak_hz, dtype=np.float64),
            "top_surface_strain_eqv": top_surface_strain,
        }
        np.savez_compressed(
            modes_output_dir / f"sample_{sample_id:04d}_modal.npz",
            **modal_save,
        )
    return response_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the base-excited piezoelectric plate FRF with FEniCSx modal reduction.",
    )
    parser.add_argument("--mesh", required=True, help="Path to one 3D .msh file.")
    parser.add_argument("--response-dir", default="data/fem_responses", help="Output directory for sample responses.")
    parser.add_argument("--modes-dir", default="data/modal_data", help="Optional output directory for modal diagnostics.")
    parser.add_argument("--num-modes", type=int, default=12, help="Number of structural modes to retain.")
    parser.add_argument("--search-points", type=int, default=801, help="Number of points used to locate the fundamental FRF peak.")
    parser.add_argument("--frf-points", type=int, default=256, help="Number of samples saved in the final normalized FRF window.")
    parser.add_argument("--substrate-rho", type=float, default=7930.0, help="Substrate density in kg/m^3.")
    parser.add_argument("--piezo-rho", type=float, default=7800.0, help="Piezo density in kg/m^3.")
    args = parser.parse_args()

    response_path = solve_modal_voltage_frf(
        mesh_path=args.mesh,
        response_dir=args.response_dir,
        num_modes=args.num_modes,
        search_points=args.search_points,
        frf_points=args.frf_points,
        mechanical=MechanicalConfig(
            substrate_rho=float(args.substrate_rho),
            piezo_rho=float(args.piezo_rho),
        ),
        modes_output_dir=args.modes_dir,
    )
    print(f"Saved response to {response_path}")


if __name__ == "__main__":
    main()
