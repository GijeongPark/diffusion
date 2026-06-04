from __future__ import annotations

import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from peh_inverse_design.solver.fenicsx_modal_solver import (
    EIGENSOLVER_BACKENDS,
    ISOTROPIC_VRH_E_PA,
    ISOTROPIC_VRH_NU,
    MechanicalConfig,
    PiezoConfig,
    _build_eps_solver,
    _build_modal_save_payload,
    _compute_top_surface_cellwise_strain,
    _isotropic_bulk_modulus,
    _isotropic_shear_modulus,
    _is_mumps_factorization_failure,
    _is_petsc_out_of_memory_failure,
    _piezo_effective_stiffness_matrix,
    _normalize_eigensolver_backend,
    _normalize_piezo_elastic_model,
    _remap_raw_cell_tags_to_created_mesh,
    _run_cli,
    solve_modal_voltage_frf_batch,
)
from peh_inverse_design.datasets.response_dataset import save_fem_response


def _mesh_provenance(mesh_path: Path) -> dict[str, np.ndarray]:
    return {"mesh_file_sha256": np.asarray([hashlib.sha256(mesh_path.read_bytes()).hexdigest()])}


class FenicsxModalSolverTests(unittest.TestCase):
    class _FakePC:
        def __init__(self) -> None:
            self.pc_type = None
            self.factor_solver_type = None

        def setType(self, pc_type) -> None:
            self.pc_type = pc_type

        def setFactorSolverType(self, factor_solver_type: str) -> None:
            self.factor_solver_type = factor_solver_type

    class _FakeKSP:
        def __init__(self) -> None:
            self.ksp_type = None
            self.pc = FenicsxModalSolverTests._FakePC()

        def setType(self, ksp_type) -> None:
            self.ksp_type = ksp_type

        def getPC(self):
            return self.pc

    class _FakeST:
        def __init__(self) -> None:
            self.st_type = None
            self.ksp = FenicsxModalSolverTests._FakeKSP()

        def setType(self, st_type) -> None:
            self.st_type = st_type

        def getKSP(self):
            return self.ksp

    class _FakeEPSImpl:
        def __init__(self) -> None:
            self.problem_type = None
            self.dimensions = None
            self.eps_type = None
            self.which = None
            self.target = None
            self.st = FenicsxModalSolverTests._FakeST()

        def setOperators(self, _K, _M) -> None:
            pass

        def setProblemType(self, problem_type) -> None:
            self.problem_type = problem_type

        def setDimensions(self, dimensions: int) -> None:
            self.dimensions = dimensions

        def setType(self, eps_type) -> None:
            self.eps_type = eps_type

        def setWhichEigenpairs(self, which) -> None:
            self.which = which

        def setTarget(self, target: float) -> None:
            self.target = target

        def getST(self):
            return self.st

        def setTolerances(self, **_kwargs) -> None:
            pass

        def setFromOptions(self) -> None:
            pass

    class _FakeSLEPc:
        class EPS:
            last = None

            class ProblemType:
                GHEP = "ghep"

            class Type:
                KRYLOVSCHUR = "krylovschur"
                GD = "gd"
                LOBPCG = "lobpcg"

            class Which:
                TARGET_MAGNITUDE = "target_magnitude"
                SMALLEST_REAL = "smallest_real"

            def __init__(self) -> None:
                self.impl = FenicsxModalSolverTests._FakeEPSImpl()

            def create(self, _comm):
                FenicsxModalSolverTests._FakeSLEPc.EPS.last = self.impl
                return self.impl

        class ST:
            class Type:
                SINVERT = "sinvert"
                PRECOND = "precond"

    class _FakePETSc:
        class KSP:
            class Type:
                PREONLY = "preonly"

        class PC:
            class Type:
                LU = "lu"
                CHOLESKY = "cholesky"
                GAMG = "gamg"

        class Options(dict):
            pass

    def test_supported_eigensolver_backend_names_are_normalized(self) -> None:
        self.assertIn("shift_invert_cholesky", EIGENSOLVER_BACKENDS)
        self.assertEqual(_normalize_eigensolver_backend("iterative_gd"), "iterative_gd_gamg")

    def test_piezo_elastic_model_isotropic_vrh_builds_expected_stiffness(self) -> None:
        self.assertEqual(_normalize_piezo_elastic_model("isotropic_vrh"), "isotropic_vrh")
        stiffness = _piezo_effective_stiffness_matrix(PiezoConfig(elastic_model="isotropic_vrh"))

        self.assertAlmostEqual(stiffness[0, 0], 1.2593230791500423e11)
        self.assertAlmostEqual(stiffness[0, 1], 8.215147912234578e10)
        self.assertAlmostEqual(stiffness[3, 3], 2.1890414396329224e10)
        self.assertAlmostEqual(_isotropic_bulk_modulus(ISOTROPIC_VRH_E_PA, ISOTROPIC_VRH_NU), 9.674508871989859e10)
        self.assertAlmostEqual(_isotropic_shear_modulus(ISOTROPIC_VRH_E_PA, ISOTROPIC_VRH_NU), 2.1890414396329224e10)

    def test_shift_invert_cholesky_uses_cholesky_mumps(self) -> None:
        eps, backend = _build_eps_solver(
            comm=object(),
            K=object(),
            M=object(),
            num_modes=3,
            PETSc=self._FakePETSc,
            SLEPc=self._FakeSLEPc,
            backend="shift_invert_cholesky",
        )

        self.assertIs(eps, self._FakeSLEPc.EPS.last)
        self.assertEqual(backend, "shift_invert_cholesky")
        self.assertEqual(eps.eps_type, "krylovschur")
        self.assertEqual(eps.st.st_type, "sinvert")
        self.assertEqual(eps.st.ksp.pc.pc_type, "cholesky")
        self.assertEqual(eps.st.ksp.pc.factor_solver_type, "mumps")

    def test_shift_invert_lu_uses_lu_mumps(self) -> None:
        eps, backend = _build_eps_solver(
            comm=object(),
            K=object(),
            M=object(),
            num_modes=3,
            PETSc=self._FakePETSc,
            SLEPc=self._FakeSLEPc,
            backend="shift_invert_lu",
        )

        self.assertEqual(backend, "shift_invert_lu")
        self.assertEqual(eps.st.ksp.pc.pc_type, "lu")
        self.assertEqual(eps.st.ksp.pc.factor_solver_type, "mumps")

    def test_iterative_backends_use_gamg_preconditioner(self) -> None:
        for backend, expected_eps_type in [
            ("iterative_lobpcg_gamg", "lobpcg"),
            ("iterative_gd_gamg", "gd"),
        ]:
            with self.subTest(backend=backend):
                eps, actual_backend = _build_eps_solver(
                    comm=object(),
                    K=object(),
                    M=object(),
                    num_modes=3,
                    PETSc=self._FakePETSc,
                    SLEPc=self._FakeSLEPc,
                    backend=backend,
                )

                self.assertEqual(actual_backend, backend)
                self.assertEqual(eps.eps_type, expected_eps_type)
                self.assertEqual(eps.st.st_type, "precond")
                self.assertEqual(eps.st.ksp.pc.pc_type, "gamg")

    def test_is_petsc_out_of_memory_failure_follows_exception_context(self) -> None:
        inner = RuntimeError("petsc4py.PETSc.Error: error code 55\nOut of memory.\nMemory requested 1234")
        outer = SystemError("EPS.solve returned a result with an exception set")
        outer.__context__ = inner

        self.assertTrue(_is_petsc_out_of_memory_failure(outer))

    def test_is_mumps_factorization_failure_follows_exception_context(self) -> None:
        inner = RuntimeError("MUMPS error in numerical factorization: INFOG(1)=-13")
        outer = SystemError("EPS.solve returned a result with an exception set")
        outer.__context__ = inner

        self.assertTrue(_is_mumps_factorization_failure(outer))

    def test_run_cli_exits_137_for_known_petsc_oom(self) -> None:
        inner = RuntimeError("petsc4py.PETSc.Error: error code 55\nOut of memory.\nMemory requested 1234")
        outer = SystemError("EPS.solve returned a result with an exception set")
        outer.__context__ = inner

        with mock.patch("peh_inverse_design.solver.fenicsx_modal_solver.main", side_effect=outer):
            stderr_buffer = io.StringIO()
            with contextlib.redirect_stderr(stderr_buffer):
                with self.assertRaises(SystemExit) as excinfo:
                    _run_cli()

        self.assertEqual(excinfo.exception.code, 137)
        self.assertIn("exiting with status 137", stderr_buffer.getvalue())

    def test_run_cli_exits_137_for_mumps_factorization_memory_failure(self) -> None:
        inner = RuntimeError("MUMPS error in numerical factorization: INFOG(1)=-13")
        outer = RuntimeError("backend failed")
        outer.__context__ = inner

        with mock.patch("peh_inverse_design.solver.fenicsx_modal_solver.main", side_effect=outer):
            stderr_buffer = io.StringIO()
            with contextlib.redirect_stderr(stderr_buffer):
                with self.assertRaises(SystemExit) as excinfo:
                    _run_cli()

        self.assertEqual(excinfo.exception.code, 137)
        self.assertIn("direct-factorization memory", stderr_buffer.getvalue())

    def test_skip_existing_reuses_peak_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_path = root / "plate3d_0000_fenicsx.npz"
            mesh_path.write_bytes(b"stub")
            response_dir = root / "responses"
            save_fem_response(
                sample_id=0,
                f_peak_hz=1.0,
                freq_hz=np.asarray([0.9, 1.0, 1.1], dtype=np.float64),
                voltage_mag=np.asarray([1.0, 2.0, 1.5], dtype=np.float64),
                output_dir=response_dir,
                metadata=_mesh_provenance(mesh_path),
            )

            with mock.patch("peh_inverse_design.solver.fenicsx_modal_solver.solve_modal_voltage_frf") as solve_mock:
                saved = solve_modal_voltage_frf_batch(
                    mesh_paths=[mesh_path],
                    response_dir=response_dir,
                    skip_existing=True,
                    house_voltage_amplitude_convention="peak",
                )

            solve_mock.assert_not_called()
            self.assertEqual(saved, [response_dir / "sample_0000_response.npz"])

    def test_skip_existing_recomputes_response_when_existing_file_is_tagged_rms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_path = root / "plate3d_0000_fenicsx.npz"
            mesh_path.write_bytes(b"stub")
            response_dir = root / "responses"
            response_path = save_fem_response(
                sample_id=0,
                f_peak_hz=1.0,
                freq_hz=np.asarray([0.9, 1.0, 1.1], dtype=np.float64),
                voltage_mag=np.asarray([1.0, 2.0, 1.5], dtype=np.float64),
                output_dir=response_dir,
                metadata=_mesh_provenance(mesh_path),
            )
            with np.load(response_path, allow_pickle=True) as response:
                payload = {key: np.asarray(response[key]) for key in response.files}
            payload["peak_voltage_form"] = np.asarray("rms")
            np.savez_compressed(response_path, **payload)

            with mock.patch(
                "peh_inverse_design.solver.fenicsx_modal_solver.solve_modal_voltage_frf",
                return_value=response_path,
            ) as solve_mock:
                saved = solve_modal_voltage_frf_batch(
                    mesh_paths=[mesh_path],
                    response_dir=response_dir,
                    skip_existing=True,
                    house_voltage_amplitude_convention="peak",
                )

            solve_mock.assert_called_once()
            self.assertEqual(saved, [response_path])

    def test_skip_existing_recomputes_when_modal_element_order_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mesh_path = root / "plate3d_0000_fenicsx.npz"
            mesh_path.write_bytes(b"stub")
            response_dir = root / "responses"
            modal_dir = root / "modal"
            modal_dir.mkdir(parents=True, exist_ok=True)
            response_path = save_fem_response(
                sample_id=0,
                f_peak_hz=1.0,
                freq_hz=np.asarray([0.9, 1.0, 1.1], dtype=np.float64),
                voltage_mag=np.asarray([1.0, 2.0, 1.5], dtype=np.float64),
                output_dir=response_dir,
                metadata=_mesh_provenance(mesh_path),
            )
            np.savez_compressed(
                modal_dir / "sample_0000_modal.npz",
                element_order=np.asarray([1], dtype=np.int32),
                **_mesh_provenance(mesh_path),
            )

            with mock.patch(
                "peh_inverse_design.solver.fenicsx_modal_solver.solve_modal_voltage_frf",
                return_value=response_path,
            ) as solve_mock:
                saved = solve_modal_voltage_frf_batch(
                    mesh_paths=[mesh_path],
                    response_dir=response_dir,
                    modes_output_dir=modal_dir,
                    element_order=2,
                    skip_existing=True,
                    house_voltage_amplitude_convention="peak",
                )

            solve_mock.assert_called_once()
            self.assertEqual(saved, [response_path])

    def test_remap_raw_cell_tags_matches_created_mesh_after_point_reordering(self) -> None:
        raw_tetra_cells = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        raw_tetra_tags = np.asarray([12], dtype=np.int32)
        input_global_indices = np.asarray([3, 1, 0, 2], dtype=np.int64)
        created_cell_geometry_dofs = np.asarray([[2, 1, 3, 0]], dtype=np.int64)

        remapped = _remap_raw_cell_tags_to_created_mesh(
            raw_tetra_cells=raw_tetra_cells,
            raw_tetra_tags=raw_tetra_tags,
            created_cell_geometry_dofs=created_cell_geometry_dofs,
            input_global_indices=input_global_indices,
        )

        np.testing.assert_array_equal(remapped, np.asarray([12], dtype=np.int32))

    def test_remap_raw_cell_tags_matches_created_mesh_after_cell_reordering(self) -> None:
        raw_tetra_cells = np.asarray(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
            ],
            dtype=np.int64,
        )
        raw_tetra_tags = np.asarray([11, 12], dtype=np.int32)
        input_global_indices = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
        created_cell_geometry_dofs = np.asarray(
            [
                [1, 2, 3, 4],
                [0, 1, 2, 3],
            ],
            dtype=np.int64,
        )

        remapped = _remap_raw_cell_tags_to_created_mesh(
            raw_tetra_cells=raw_tetra_cells,
            raw_tetra_tags=raw_tetra_tags,
            created_cell_geometry_dofs=created_cell_geometry_dofs,
            input_global_indices=input_global_indices,
        )

        np.testing.assert_array_equal(remapped, np.asarray([12, 11], dtype=np.int32))

    def test_remap_raw_cell_tags_matches_created_mesh_after_cell_and_point_reordering(self) -> None:
        raw_tetra_cells = np.asarray(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
            ],
            dtype=np.int64,
        )
        raw_tetra_tags = np.asarray([11, 12], dtype=np.int32)

        # The created mesh reorders geometry points relative to the raw NPZ input.
        input_global_indices = np.asarray([2, 0, 3, 1, 4], dtype=np.int64)
        # These created cells use geometry-dof indices, not the original raw-point ids.
        created_cell_geometry_dofs = np.asarray(
            [
                [1, 3, 0, 2],  # maps back to raw vertices [0, 1, 2, 3]
                [3, 0, 2, 4],  # maps back to raw vertices [1, 2, 3, 4]
            ],
            dtype=np.int64,
        )

        remapped = _remap_raw_cell_tags_to_created_mesh(
            raw_tetra_cells=raw_tetra_cells,
            raw_tetra_tags=raw_tetra_tags,
            created_cell_geometry_dofs=created_cell_geometry_dofs,
            input_global_indices=input_global_indices,
        )

        np.testing.assert_array_equal(remapped, np.asarray([11, 12], dtype=np.int32))

    def test_build_modal_save_payload_writes_explicit_modal_and_harmonic_fields(self) -> None:
        modal_model = {
            "eigenfreq_hz": np.asarray([1.25, 3.5], dtype=np.float64),
            "modal_force": np.asarray([0.1, 0.2], dtype=np.float64),
            "modal_theta": np.asarray([0.01, 0.02], dtype=np.float64),
            "modal_mass": np.asarray([1.0, 1.0], dtype=np.float64),
            "capacitance_f": np.asarray([5.0e-7], dtype=np.float64),
            "capacitance_eps33s_f_per_m": np.asarray([1.729e-8], dtype=np.float64),
            "substrate_volume_m3": np.asarray([1.0e-4], dtype=np.float64),
            "piezo_volume_m3": np.asarray([2.0e-5], dtype=np.float64),
            "substrate_cell_count": np.asarray([12], dtype=np.int32),
            "piezo_cell_count": np.asarray([8], dtype=np.int32),
        }

        payload = _build_modal_save_payload(
            sample_id=7,
            element_order=2,
            mechanical=MechanicalConfig(),
            piezo=PiezoConfig(),
            modal_model=modal_model,
            mode1_top_surface_strain_eqv=np.asarray([10.0, 20.0], dtype=np.float64),
            harmonic_top_surface_strain_eqv=np.asarray([30.0, 40.0], dtype=np.float64),
            harmonic_field_frequency_hz=1.2,
        )

        self.assertIn("mode1_frequency_hz", payload)
        self.assertIn("mode1_top_surface_strain_eqv", payload)
        self.assertIn("harmonic_field_frequency_hz", payload)
        self.assertIn("harmonic_top_surface_strain_eqv", payload)
        self.assertIn("capacitance_eps33s_f_per_m", payload)
        self.assertIn("piezo_elastic_model", payload)
        self.assertIn("piezo_voigt_order", payload)
        self.assertIn("isotropic_E_pa", payload)
        np.testing.assert_array_equal(payload["top_surface_strain_eqv"], np.asarray([30.0, 40.0], dtype=np.float64))
        self.assertEqual(float(payload["mode1_frequency_hz"]), 1.25)
        self.assertEqual(float(payload["harmonic_field_frequency_hz"]), 1.2)
        self.assertEqual(float(np.asarray(payload["capacitance_eps33s_f_per_m"]).reshape(-1)[0]), 1.729e-8)
        self.assertEqual(str(np.asarray(payload["piezo_elastic_model"]).reshape(-1)[0]), "anisotropic")
        self.assertTrue(np.isnan(float(np.asarray(payload["isotropic_E_pa"]).reshape(-1)[0])))

    def test_build_modal_save_payload_writes_isotropic_vrh_metadata(self) -> None:
        modal_model = {
            "eigenfreq_hz": np.asarray([1.25], dtype=np.float64),
            "modal_force": np.asarray([0.1], dtype=np.float64),
            "modal_theta": np.asarray([0.01], dtype=np.float64),
            "modal_mass": np.asarray([1.0], dtype=np.float64),
            "capacitance_f": np.asarray([5.0e-7], dtype=np.float64),
            "substrate_volume_m3": np.asarray([1.0e-4], dtype=np.float64),
            "piezo_volume_m3": np.asarray([2.0e-5], dtype=np.float64),
            "substrate_cell_count": np.asarray([12], dtype=np.int32),
            "piezo_cell_count": np.asarray([8], dtype=np.int32),
        }

        payload = _build_modal_save_payload(
            sample_id=7,
            element_order=2,
            mechanical=MechanicalConfig(),
            piezo=PiezoConfig(elastic_model="isotropic_vrh"),
            modal_model=modal_model,
            mode1_top_surface_strain_eqv=np.asarray([], dtype=np.float64),
            harmonic_top_surface_strain_eqv=np.asarray([], dtype=np.float64),
            harmonic_field_frequency_hz=1.2,
        )

        self.assertEqual(str(np.asarray(payload["piezo_elastic_model"]).reshape(-1)[0]), "isotropic_vrh")
        self.assertEqual(str(np.asarray(payload["piezo_voigt_order"]).reshape(-1)[0]), "xx,yy,zz,yz,xz,xy")
        self.assertEqual(float(np.asarray(payload["isotropic_E_pa"]).reshape(-1)[0]), ISOTROPIC_VRH_E_PA)
        self.assertEqual(float(np.asarray(payload["isotropic_nu"]).reshape(-1)[0]), ISOTROPIC_VRH_NU)

    def test_top_surface_cellwise_strain_is_root_dominant_for_cantilever_like_mode(self) -> None:
        x_nodes = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        y_nodes = np.asarray([0.0, 1.0], dtype=np.float64)
        z_nodes = np.asarray([0.0, 1.0], dtype=np.float64)

        def node_id(ix: int, iy: int, iz: int) -> int:
            return iz * (len(x_nodes) * len(y_nodes)) + iy * len(x_nodes) + ix

        points = []
        for z in z_nodes:
            for y in y_nodes:
                for x in x_nodes:
                    points.append([x, y, z])
        points = np.asarray(points, dtype=np.float64)

        tetra_cells: list[list[int]] = []
        triangle_cells: list[list[int]] = []
        for ix in range(len(x_nodes) - 1):
            v000 = node_id(ix, 0, 0)
            v100 = node_id(ix + 1, 0, 0)
            v010 = node_id(ix, 1, 0)
            v110 = node_id(ix + 1, 1, 0)
            v001 = node_id(ix, 0, 1)
            v101 = node_id(ix + 1, 0, 1)
            v011 = node_id(ix, 1, 1)
            v111 = node_id(ix + 1, 1, 1)
            tetra_cells.extend(
                [
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                ]
            )
            triangle_cells.extend(
                [
                    [v001, v101, v111],
                    [v001, v111, v011],
                ]
            )

        tetra_cells_array = np.asarray(tetra_cells, dtype=np.int64)
        triangle_cells_array = np.asarray(triangle_cells, dtype=np.int32)
        triangle_tags = np.full(triangle_cells_array.shape[0], 105, dtype=np.int32)

        def w(x: np.ndarray) -> np.ndarray:
            return x ** 2 * (3.0 - x)

        def dw_dx(x: np.ndarray) -> np.ndarray:
            return 6.0 * x - 3.0 * x ** 2

        nodal_displacement = np.zeros((points.shape[0], 3), dtype=np.float64)
        x = points[:, 0]
        z_centered = points[:, 2] - 0.5
        nodal_displacement[:, 0] = -z_centered * dw_dx(x)
        nodal_displacement[:, 2] = w(x)

        strain = _compute_top_surface_cellwise_strain(
            points=points,
            tetra_cells=tetra_cells_array,
            triangle_cells=triangle_cells_array,
            triangle_tags=triangle_tags,
            nodal_displacement=nodal_displacement,
        )
        centroids = np.mean(points[triangle_cells_array], axis=1)
        root_mean = float(np.mean(strain[centroids[:, 0] <= 0.25]))
        tip_mean = float(np.mean(strain[centroids[:, 0] >= 0.75]))
        top_decile = strain >= np.percentile(strain, 90.0)
        top_decile_x = float(np.mean(centroids[top_decile, 0]))

        self.assertGreater(root_mean, tip_mean)
        self.assertLess(top_decile_x, 0.35)


if __name__ == "__main__":
    unittest.main()
