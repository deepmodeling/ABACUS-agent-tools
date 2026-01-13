"""
Unit tests for SCF parameter validator.

Tests cover:
- Range validations for all parameters
- Dependency validations (mixing, smearing, k-points, spin)
- Cross-parameter compatibility checks
- Error vs warning classification
- Validation result formatting
"""

import pytest

from src.abacusagent.modules.submodules.scf.schema import (
    SCFParameters,
    SmearingMethod,
    MixingType,
)
from src.abacusagent.modules.submodules.scf.validator import (
    SCFParameterValidator,
    ValidationResult,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_create_error_result(self):
        """Test creating error validation result."""
        result = ValidationResult(
            is_valid=False,
            parameter="ecutwfc",
            message="ecutwfc must be > 0",
            severity="error"
        )

        assert result.is_valid is False
        assert result.parameter == "ecutwfc"
        assert result.message == "ecutwfc must be > 0"
        assert result.severity == "error"

    def test_create_warning_result(self):
        """Test creating warning validation result."""
        result = ValidationResult(
            is_valid=True,
            parameter="ecutwfc",
            message="ecutwfc is low",
            severity="warning"
        )

        assert result.is_valid is True
        assert result.severity == "warning"

    def test_create_info_result(self):
        """Test creating info validation result."""
        result = ValidationResult(
            is_valid=True,
            parameter="mixing_ndim",
            message="Using default",
            severity="info"
        )

        assert result.is_valid is True
        assert result.severity == "info"

    def test_result_to_dict(self):
        """Test converting validation result to dictionary."""
        result = ValidationResult(
            is_valid=False,
            parameter="test",
            message="Test message",
            severity="error"
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["is_valid"] is False
        assert result_dict["parameter"] == "test"
        assert result_dict["message"] == "Test message"
        assert result_dict["severity"] == "error"


class TestRangeValidations:
    """Test range validations for individual parameters."""

    def test_ecutwfc_valid(self):
        """Test valid ecutwfc values."""
        validator = SCFParameterValidator()
        params = SCFParameters(ecutwfc=100.0)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.errors) == 0

    def test_ecutwfc_negative(self):
        """Test that negative ecutwfc raises error."""
        validator = SCFParameterValidator()
        params = SCFParameters(ecutwfc=-50.0)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert len(validator.errors) == 1
        assert "ecutwfc must be > 0" in validator.errors[0]

    def test_ecutwfc_zero(self):
        """Test that zero ecutwfc raises error."""
        validator = SCFParameterValidator()
        params = SCFParameters(ecutwfc=0.0)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert "ecutwfc must be > 0" in validator.errors[0]

    def test_ecutwfc_too_low_warning(self):
        """Test that very low ecutwfc generates warning."""
        validator = SCFParameterValidator()
        params = SCFParameters(ecutwfc=15.0)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True  # Warning, not error
        assert len(validator.warnings) >= 1
        assert any("very low" in w for w in validator.warnings)

    def test_ecutwfc_too_high_warning(self):
        """Test that very high ecutwfc generates warning."""
        validator = SCFParameterValidator()
        params = SCFParameters(ecutwfc=250.0)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.warnings) >= 1
        assert any("very high" in w for w in validator.warnings)

    def test_scf_thr_valid(self):
        """Test valid scf_thr values."""
        validator = SCFParameterValidator()
        params = SCFParameters(scf_thr=1e-6)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.errors) == 0

    def test_scf_thr_negative(self):
        """Test that negative scf_thr raises error."""
        validator = SCFParameterValidator()
        params = SCFParameters(scf_thr=-1e-6)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert "scf_thr must be > 0" in validator.errors[0]

    def test_scf_thr_too_loose_warning(self):
        """Test that loose scf_thr generates warning."""
        validator = SCFParameterValidator()
        params = SCFParameters(scf_thr=1e-2)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.warnings) >= 1
        assert any("loose" in w for w in validator.warnings)

    def test_scf_thr_too_tight_warning(self):
        """Test that very tight scf_thr generates warning."""
        validator = SCFParameterValidator()
        params = SCFParameters(scf_thr=1e-13)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.warnings) >= 1
        assert any("very tight" in w or "hard to converge" in w for w in validator.warnings)

    def test_scf_nmax_valid(self):
        """Test valid scf_nmax values."""
        validator = SCFParameterValidator()
        params = SCFParameters(scf_nmax=100)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.errors) == 0

    def test_scf_nmax_negative(self):
        """Test that negative scf_nmax raises error."""
        validator = SCFParameterValidator()
        params = SCFParameters(scf_nmax=-10)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert "scf_nmax must be > 0" in validator.errors[0]

    def test_scf_nmax_too_low_warning(self):
        """Test that low scf_nmax generates warning."""
        validator = SCFParameterValidator()
        params = SCFParameters(scf_nmax=10)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.warnings) >= 1
        assert any("low" in w for w in validator.warnings)

    def test_mixing_beta_valid(self):
        """Test valid mixing_beta values."""
        validator = SCFParameterValidator()
        params = SCFParameters(mixing_beta=0.4)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.errors) == 0

    def test_mixing_beta_too_high(self):
        """Test that mixing_beta > 1 raises error."""
        validator = SCFParameterValidator()
        params = SCFParameters(mixing_beta=1.5)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert "mixing_beta must be in (0, 1]" in validator.errors[0]

    def test_mixing_beta_zero(self):
        """Test that mixing_beta = 0 raises error."""
        validator = SCFParameterValidator()
        params = SCFParameters(mixing_beta=0.0)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert "mixing_beta must be in (0, 1]" in validator.errors[0]

    def test_mixing_beta_high_warning(self):
        """Test that high mixing_beta generates warning."""
        validator = SCFParameterValidator()
        params = SCFParameters(mixing_beta=0.9)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.warnings) >= 1
        assert any("high" in w or "instability" in w for w in validator.warnings)

    def test_smearing_sigma_valid(self):
        """Test valid smearing_sigma values."""
        validator = SCFParameterValidator()
        params = SCFParameters(smearing_sigma=0.015)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.errors) == 0

    def test_smearing_sigma_negative(self):
        """Test that negative smearing_sigma raises error."""
        validator = SCFParameterValidator()
        params = SCFParameters(smearing_sigma=-0.01)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert "smearing_sigma must be > 0" in validator.errors[0]

    def test_kspacing_valid(self):
        """Test valid kspacing values."""
        validator = SCFParameterValidator()
        params = SCFParameters(kspacing=0.3)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.errors) == 0

    def test_kspacing_negative(self):
        """Test that negative kspacing raises error."""
        validator = SCFParameterValidator()
        params = SCFParameters(kspacing=-0.3)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert "kspacing must be > 0" in validator.errors[0]

    def test_out_chg_valid(self):
        """Test valid out_chg values."""
        validator = SCFParameterValidator()
        for value in [-1, 0, 1]:
            params = SCFParameters(out_chg=value)
            context = {"basis_type": "lcao", "soc": False, "nspin": 1}

            is_valid, results = validator.validate_all(params, context)
            assert is_valid is True

    def test_out_chg_invalid(self):
        """Test that invalid out_chg raises error."""
        validator = SCFParameterValidator()
        params = SCFParameters(out_chg=5)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert "out_chg must be -1, 0, or 1" in validator.errors[0]


class TestDependencyValidations:
    """Test dependency validations between parameters."""

    def test_mixing_ndim_with_pulay(self):
        """Test mixing_ndim is appropriate for pulay mixing."""
        validator = SCFParameterValidator()
        params = SCFParameters(
            mixing_type=MixingType.PULAY,
            mixing_ndim=8
        )
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        # Should not have warnings about mixing_ndim being ignored

    def test_mixing_ndim_with_plain_warning(self):
        """Test mixing_ndim with plain mixing generates warning."""
        validator = SCFParameterValidator()
        params = SCFParameters(
            mixing_type=MixingType.PLAIN,
            mixing_ndim=8
        )
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.warnings) >= 1
        assert any("mixing_ndim is ignored" in w for w in validator.warnings)

    def test_mixing_gg0_with_kerker(self):
        """Test mixing_gg0 is appropriate for kerker mixing."""
        validator = SCFParameterValidator()
        params = SCFParameters(
            mixing_type=MixingType.KERKER,
            mixing_gg0=1.5
        )
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True

    def test_mixing_gg0_with_pulay_warning(self):
        """Test mixing_gg0 with pulay mixing generates warning."""
        validator = SCFParameterValidator()
        params = SCFParameters(
            mixing_type=MixingType.PULAY,
            mixing_gg0=1.5
        )
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.warnings) >= 1
        assert any("mixing_gg0 is ignored" in w for w in validator.warnings)

    def test_gamma_only_and_kspacing_conflict(self):
        """Test that gamma_only and kspacing are mutually exclusive."""
        validator = SCFParameterValidator()
        params = SCFParameters(
            gamma_only=True,
            kspacing=0.3
        )
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert len(validator.errors) == 1
        assert "mutually exclusive" in validator.errors[0]

    def test_gamma_only_without_kspacing(self):
        """Test gamma_only without kspacing is valid."""
        validator = SCFParameterValidator()
        params = SCFParameters(gamma_only=True)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True

    def test_kspacing_without_gamma_only(self):
        """Test kspacing without gamma_only is valid."""
        validator = SCFParameterValidator()
        params = SCFParameters(kspacing=0.3)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True

    def test_soc_requires_nspin_4(self):
        """Test that soc=True requires nspin=4."""
        validator = SCFParameterValidator()
        params = SCFParameters(nspin=2)
        context = {"basis_type": "lcao", "soc": True, "nspin": 2}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert len(validator.errors) == 1
        assert "nspin" in validator.errors[0]
        assert "soc" in validator.errors[0].lower()

    def test_soc_with_nspin_4_valid(self):
        """Test that soc=True with nspin=4 is valid."""
        validator = SCFParameterValidator()
        params = SCFParameters(nspin=4)
        context = {"basis_type": "lcao", "soc": True, "nspin": 4}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True


class TestCrossParameterValidations:
    """Test cross-parameter compatibility checks."""

    def test_out_mul_with_lcao(self):
        """Test out_mul with LCAO basis is valid."""
        validator = SCFParameterValidator()
        params = SCFParameters(out_mul=True)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True

    def test_out_mul_with_pw_warning(self):
        """Test out_mul with PW basis generates warning."""
        validator = SCFParameterValidator()
        params = SCFParameters(out_mul=True)
        context = {"basis_type": "pw", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is True
        assert len(validator.warnings) >= 1
        assert any("Mulliken" in w and "LCAO" in w for w in validator.warnings)


class TestMultipleErrors:
    """Test handling of multiple validation errors."""

    def test_multiple_errors(self):
        """Test that multiple errors are all reported."""
        validator = SCFParameterValidator()
        params = SCFParameters(
            ecutwfc=-50.0,      # Error: negative
            mixing_beta=1.5,    # Error: > 1
            scf_nmax=-10        # Error: negative
        )
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False
        assert len(validator.errors) == 3

    def test_errors_and_warnings(self):
        """Test that errors and warnings can coexist."""
        validator = SCFParameterValidator()
        params = SCFParameters(
            ecutwfc=-50.0,      # Error: negative
            scf_thr=1e-2        # Warning: loose
        )
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert is_valid is False  # Errors block validation
        assert len(validator.errors) >= 1
        assert len(validator.warnings) >= 1


class TestValidatorState:
    """Test validator state management."""

    def test_validator_resets_state(self):
        """Test that validator resets state between validations."""
        validator = SCFParameterValidator()

        # First validation with error
        params1 = SCFParameters(ecutwfc=-50.0)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}
        is_valid1, results1 = validator.validate_all(params1, context)

        assert is_valid1 is False
        assert len(validator.errors) == 1

        # Second validation without error
        params2 = SCFParameters(ecutwfc=100.0)
        is_valid2, results2 = validator.validate_all(params2, context)

        assert is_valid2 is True
        assert len(validator.errors) == 0  # Should be reset


class TestValidationResults:
    """Test validation result collection."""

    def test_validation_results_structure(self):
        """Test that validation results have correct structure."""
        validator = SCFParameterValidator()
        params = SCFParameters(ecutwfc=-50.0)
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert isinstance(result, ValidationResult)
            assert hasattr(result, 'is_valid')
            assert hasattr(result, 'parameter')
            assert hasattr(result, 'message')
            assert hasattr(result, 'severity')

    def test_severity_classification(self):
        """Test that severity is correctly classified."""
        validator = SCFParameterValidator()
        params = SCFParameters(
            ecutwfc=-50.0,      # Error
            scf_thr=1e-2        # Warning
        )
        context = {"basis_type": "lcao", "soc": False, "nspin": 1}

        is_valid, results = validator.validate_all(params, context)

        errors = [r for r in results if r.severity == "error"]
        warnings = [r for r in results if r.severity == "warning"]

        assert len(errors) >= 1
        assert len(warnings) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
