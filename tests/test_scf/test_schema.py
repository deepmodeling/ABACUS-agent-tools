"""
Unit tests for SCF parameter schema.

Tests cover:
- SCFParameters dataclass creation and validation
- Enum value validation (SmearingMethod, MixingType, BasisType)
- ParameterProvenance and SCFAuditTrail structures
- Serialization to dictionaries
"""

import pytest
from datetime import datetime

from src.abacusagent.modules.submodules.scf.schema import (
    SCFParameters,
    ParameterProvenance,
    SCFAuditTrail,
    SmearingMethod,
    MixingType,
    BasisType,
)


class TestEnums:
    """Test enum definitions and values."""

    def test_smearing_method_values(self):
        """Test SmearingMethod enum has all expected values."""
        assert SmearingMethod.GAUSSIAN.value == "gaussian"
        assert SmearingMethod.FERMI_DIRAC.value == "fd"
        assert SmearingMethod.FIXED.value == "fixed"
        assert SmearingMethod.METHFESSEL_PAXTON.value == "mp"
        assert SmearingMethod.MARZARI_VANDERBILT.value == "mv"
        assert SmearingMethod.COLD.value == "cold"

    def test_mixing_type_values(self):
        """Test MixingType enum has all expected values."""
        assert MixingType.PLAIN.value == "plain"
        assert MixingType.KERKER.value == "kerker"
        assert MixingType.PULAY.value == "pulay"
        assert MixingType.PULAY_KERKER.value == "pulay-kerker"
        assert MixingType.BROYDEN.value == "broyden"

    def test_basis_type_values(self):
        """Test BasisType enum has all expected values."""
        assert BasisType.PW.value == "pw"
        assert BasisType.LCAO.value == "lcao"
        assert BasisType.LCAO_IN_PW.value == "lcao_in_pw"

    def test_enum_from_string(self):
        """Test creating enums from string values."""
        assert SmearingMethod("gaussian") == SmearingMethod.GAUSSIAN
        assert MixingType("pulay") == MixingType.PULAY
        assert BasisType("lcao") == BasisType.LCAO

    def test_enum_invalid_value(self):
        """Test that invalid enum values raise ValueError."""
        with pytest.raises(ValueError):
            SmearingMethod("invalid")
        with pytest.raises(ValueError):
            MixingType("invalid")
        with pytest.raises(ValueError):
            BasisType("invalid")


class TestSCFParameters:
    """Test SCFParameters dataclass."""

    def test_create_empty_parameters(self):
        """Test creating SCFParameters with all defaults (None)."""
        params = SCFParameters()

        assert params.ecutwfc is None
        assert params.scf_thr is None
        assert params.scf_nmax is None
        assert params.smearing_method is None
        assert params.smearing_sigma is None
        assert params.mixing_type is None
        assert params.mixing_beta is None
        assert params.mixing_ndim is None
        assert params.mixing_gg0 is None
        assert params.kspacing is None
        assert params.gamma_only is None
        assert params.symmetry is None
        assert params.out_chg is None
        assert params.out_mul is None
        assert params.chg_extrap is None
        assert params.ks_solver is None
        assert params.nspin is None

    def test_create_with_convergence_params(self):
        """Test creating SCFParameters with convergence parameters."""
        params = SCFParameters(
            ecutwfc=100.0,
            scf_thr=1e-6,
            scf_nmax=100
        )

        assert params.ecutwfc == 100.0
        assert params.scf_thr == 1e-6
        assert params.scf_nmax == 100

    def test_create_with_smearing_params(self):
        """Test creating SCFParameters with smearing parameters."""
        params = SCFParameters(
            smearing_method=SmearingMethod.GAUSSIAN,
            smearing_sigma=0.015
        )

        assert params.smearing_method == SmearingMethod.GAUSSIAN
        assert params.smearing_sigma == 0.015

    def test_create_with_mixing_params(self):
        """Test creating SCFParameters with mixing parameters."""
        params = SCFParameters(
            mixing_type=MixingType.PULAY,
            mixing_beta=0.4,
            mixing_ndim=8,
            mixing_gg0=0.0
        )

        assert params.mixing_type == MixingType.PULAY
        assert params.mixing_beta == 0.4
        assert params.mixing_ndim == 8
        assert params.mixing_gg0 == 0.0

    def test_create_with_kpoint_params(self):
        """Test creating SCFParameters with k-point parameters."""
        params = SCFParameters(
            kspacing=0.3,
            gamma_only=False
        )

        assert params.kspacing == 0.3
        assert params.gamma_only is False

    def test_create_with_all_params(self):
        """Test creating SCFParameters with all parameters set."""
        params = SCFParameters(
            ecutwfc=120.0,
            scf_thr=1e-7,
            scf_nmax=200,
            smearing_method=SmearingMethod.METHFESSEL_PAXTON,
            smearing_sigma=0.02,
            mixing_type=MixingType.PULAY_KERKER,
            mixing_beta=0.4,
            mixing_ndim=10,
            mixing_gg0=1.5,
            kspacing=0.25,
            gamma_only=False,
            symmetry=True,
            out_chg=1,
            out_mul=True,
            chg_extrap="first-order",
            ks_solver="genelpa",
            nspin=2
        )

        assert params.ecutwfc == 120.0
        assert params.scf_thr == 1e-7
        assert params.scf_nmax == 200
        assert params.smearing_method == SmearingMethod.METHFESSEL_PAXTON
        assert params.smearing_sigma == 0.02
        assert params.mixing_type == MixingType.PULAY_KERKER
        assert params.mixing_beta == 0.4
        assert params.mixing_ndim == 10
        assert params.mixing_gg0 == 1.5
        assert params.kspacing == 0.25
        assert params.gamma_only is False
        assert params.symmetry is True
        assert params.out_chg == 1
        assert params.out_mul is True
        assert params.chg_extrap == "first-order"
        assert params.ks_solver == "genelpa"
        assert params.nspin == 2

    def test_enum_string_conversion(self):
        """Test that enum values can be accessed as strings."""
        params = SCFParameters(
            smearing_method=SmearingMethod.GAUSSIAN,
            mixing_type=MixingType.PULAY
        )

        assert params.smearing_method.value == "gaussian"
        assert params.mixing_type.value == "pulay"


class TestParameterProvenance:
    """Test ParameterProvenance dataclass."""

    def test_create_user_input_provenance(self):
        """Test creating provenance for user input."""
        prov = ParameterProvenance(
            parameter_name="ecutwfc",
            value=100.0,
            source="user_input",
            reasoning="Explicitly provided by user"
        )

        assert prov.parameter_name == "ecutwfc"
        assert prov.value == 100.0
        assert prov.source == "user_input"
        assert prov.reasoning == "Explicitly provided by user"
        assert prov.depends_on is None
        assert prov.inference_rule is None
        assert isinstance(prov.timestamp, str)

    def test_create_default_provenance(self):
        """Test creating provenance for default value."""
        prov = ParameterProvenance(
            parameter_name="scf_thr",
            value=1e-6,
            source="default",
            reasoning="Standard convergence threshold"
        )

        assert prov.parameter_name == "scf_thr"
        assert prov.value == 1e-6
        assert prov.source == "default"
        assert prov.reasoning == "Standard convergence threshold"

    def test_create_inferred_provenance(self):
        """Test creating provenance for inferred value."""
        prov = ParameterProvenance(
            parameter_name="mixing_beta",
            value=0.4,
            source="inferred",
            reasoning="Default for pulay mixing",
            depends_on=["mixing_type"],
            inference_rule="pulay_mixing_beta"
        )

        assert prov.parameter_name == "mixing_beta"
        assert prov.value == 0.4
        assert prov.source == "inferred"
        assert prov.reasoning == "Default for pulay mixing"
        assert prov.depends_on == ["mixing_type"]
        assert prov.inference_rule == "pulay_mixing_beta"

    def test_create_dependency_provenance(self):
        """Test creating provenance for dependency-based value."""
        prov = ParameterProvenance(
            parameter_name="nspin",
            value=4,
            source="dependency",
            reasoning="Required by spin-orbit coupling",
            depends_on=["soc"]
        )

        assert prov.parameter_name == "nspin"
        assert prov.value == 4
        assert prov.source == "dependency"
        assert prov.depends_on == ["soc"]

    def test_provenance_to_dict(self):
        """Test converting provenance to dictionary."""
        prov = ParameterProvenance(
            parameter_name="ecutwfc",
            value=100.0,
            source="user_input",
            reasoning="Test reasoning"
        )

        prov_dict = prov.to_dict()

        assert isinstance(prov_dict, dict)
        assert prov_dict["parameter_name"] == "ecutwfc"
        assert prov_dict["value"] == 100.0
        assert prov_dict["source"] == "user_input"
        assert prov_dict["reasoning"] == "Test reasoning"
        assert "timestamp" in prov_dict

    def test_provenance_timestamp_format(self):
        """Test that timestamp is in ISO format."""
        prov = ParameterProvenance(
            parameter_name="test",
            value=1.0,
            source="default",
            reasoning="Test"
        )

        # Should be parseable as ISO datetime
        timestamp = datetime.fromisoformat(prov.timestamp)
        assert isinstance(timestamp, datetime)


class TestSCFAuditTrail:
    """Test SCFAuditTrail dataclass."""

    def test_create_empty_audit_trail(self):
        """Test creating empty audit trail."""
        trail = SCFAuditTrail(
            calculation_id="test123",
            parameters={},
            validation_results=[],
            warnings=[],
            errors=[]
        )

        assert trail.calculation_id == "test123"
        assert len(trail.parameters) == 0
        assert len(trail.validation_results) == 0
        assert len(trail.warnings) == 0
        assert len(trail.errors) == 0

    def test_create_audit_trail_with_parameters(self):
        """Test creating audit trail with parameters."""
        prov1 = ParameterProvenance(
            parameter_name="ecutwfc",
            value=100.0,
            source="user_input",
            reasoning="User provided"
        )
        prov2 = ParameterProvenance(
            parameter_name="scf_thr",
            value=1e-6,
            source="default",
            reasoning="Standard default"
        )

        trail = SCFAuditTrail(
            calculation_id="test123",
            parameters={"ecutwfc": prov1, "scf_thr": prov2},
            validation_results=[],
            warnings=[],
            errors=[]
        )

        assert len(trail.parameters) == 2
        assert "ecutwfc" in trail.parameters
        assert "scf_thr" in trail.parameters
        assert trail.parameters["ecutwfc"].value == 100.0
        assert trail.parameters["scf_thr"].value == 1e-6

    def test_audit_trail_with_warnings_and_errors(self):
        """Test audit trail with warnings and errors."""
        trail = SCFAuditTrail(
            calculation_id="test123",
            parameters={},
            validation_results=[],
            warnings=["Warning 1", "Warning 2"],
            errors=["Error 1"]
        )

        assert len(trail.warnings) == 2
        assert len(trail.errors) == 1
        assert trail.warnings[0] == "Warning 1"
        assert trail.errors[0] == "Error 1"

    def test_audit_trail_to_dict(self):
        """Test converting audit trail to dictionary."""
        prov = ParameterProvenance(
            parameter_name="ecutwfc",
            value=100.0,
            source="user_input",
            reasoning="Test"
        )

        trail = SCFAuditTrail(
            calculation_id="test123",
            parameters={"ecutwfc": prov},
            validation_results=[{"test": "result"}],
            warnings=["Warning"],
            errors=[]
        )

        trail_dict = trail.to_dict()

        assert isinstance(trail_dict, dict)
        assert trail_dict["calculation_id"] == "test123"
        assert "parameters" in trail_dict
        assert "ecutwfc" in trail_dict["parameters"]
        assert trail_dict["validation_results"] == [{"test": "result"}]
        assert trail_dict["warnings"] == ["Warning"]
        assert trail_dict["errors"] == []

    def test_audit_trail_nested_serialization(self):
        """Test that nested provenance objects are properly serialized."""
        prov = ParameterProvenance(
            parameter_name="mixing_beta",
            value=0.4,
            source="inferred",
            reasoning="Inferred from mixing_type",
            depends_on=["mixing_type"],
            inference_rule="pulay_beta"
        )

        trail = SCFAuditTrail(
            calculation_id="test123",
            parameters={"mixing_beta": prov},
            validation_results=[],
            warnings=[],
            errors=[]
        )

        trail_dict = trail.to_dict()

        # Check nested provenance is properly converted
        assert isinstance(trail_dict["parameters"]["mixing_beta"], dict)
        assert trail_dict["parameters"]["mixing_beta"]["value"] == 0.4
        assert trail_dict["parameters"]["mixing_beta"]["source"] == "inferred"
        assert trail_dict["parameters"]["mixing_beta"]["depends_on"] == ["mixing_type"]


class TestSchemaIntegration:
    """Integration tests for schema components."""

    def test_complete_workflow(self):
        """Test complete workflow: create params, provenance, and audit trail."""
        # Create parameters
        params = SCFParameters(
            ecutwfc=100.0,
            scf_thr=1e-6,
            mixing_type=MixingType.PULAY
        )

        # Create provenances
        prov1 = ParameterProvenance(
            parameter_name="ecutwfc",
            value=params.ecutwfc,
            source="user_input",
            reasoning="User provided"
        )
        prov2 = ParameterProvenance(
            parameter_name="scf_thr",
            value=params.scf_thr,
            source="user_input",
            reasoning="User provided"
        )
        prov3 = ParameterProvenance(
            parameter_name="mixing_type",
            value=params.mixing_type.value,
            source="user_input",
            reasoning="User provided"
        )

        # Create audit trail
        trail = SCFAuditTrail(
            calculation_id="integration_test",
            parameters={
                "ecutwfc": prov1,
                "scf_thr": prov2,
                "mixing_type": prov3
            },
            validation_results=[],
            warnings=[],
            errors=[]
        )

        # Verify
        assert len(trail.parameters) == 3
        assert trail.parameters["ecutwfc"].value == 100.0
        assert trail.parameters["mixing_type"].value == "pulay"

        # Test serialization
        trail_dict = trail.to_dict()
        assert isinstance(trail_dict, dict)
        assert len(trail_dict["parameters"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
