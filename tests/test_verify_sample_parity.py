from __future__ import annotations

import unittest

from peh_inverse_design.verify_sample_parity import (
    _apply_strict_ansys_parity_overrides,
    _parity_match_score,
    _parity_rank_key,
)


class VerifySampleParityTests(unittest.TestCase):
    def test_parity_match_score_includes_selected_voltage_error(self) -> None:
        record = {
            "mode1_vs_ansys_modal_error_percent": 3.0,
            "f_peak_vs_ansys_frf_peak_error_percent": -4.0,
            "selected_voltage_error_percent": 12.5,
        }

        self.assertEqual(_parity_match_score(record), 12.5)

    def test_parity_rank_penalizes_diagnostic_and_failed_runs(self) -> None:
        ok_record = {
            "case_name": "ok",
            "status": "ok",
            "solver_parity_valid": True,
            "diagnostic_only": False,
            "mode1_vs_ansys_modal_error_percent": 1.0,
            "f_peak_vs_ansys_frf_peak_error_percent": 2.0,
            "selected_voltage_error_percent": 3.0,
        }
        diagnostic_record = {
            **ok_record,
            "case_name": "diagnostic",
            "diagnostic_only": True,
        }
        failed_record = {
            **ok_record,
            "case_name": "failed",
            "status": "solve_failed",
        }

        self.assertLess(_parity_rank_key(ok_record), _parity_rank_key(diagnostic_record))
        self.assertLess(_parity_rank_key(diagnostic_record), _parity_rank_key(failed_record))

    def test_strict_ansys_parity_override_respects_explicit_voltage_form(self) -> None:
        class Args:
            strict_ansys_parity = True
            mesh_preset = "default"
            allow_diagnostic_fallbacks = True
            ansys_voltage_form = "rms"

        updated = _apply_strict_ansys_parity_overrides(
            Args(),
            explicit_voltage_form=True,
        )

        self.assertEqual(updated.mesh_preset, "ansys_parity")
        self.assertFalse(updated.allow_diagnostic_fallbacks)
        self.assertEqual(updated.ansys_voltage_form, "rms")


if __name__ == "__main__":
    unittest.main()
