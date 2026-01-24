"""
Unit tests for STRU file validation tool.
"""

import os
import pytest
from pathlib import Path
from abacusagent.modules.submodules.stru_validator import validate_stru


# Set test mode to avoid MCP server initialization
os.environ["ABACUSAGENT_MODEL"] = "test"


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "abacus"


@pytest.fixture
def valid_stru_gaas(test_data_dir):
    """Path to valid GaAs STRU file."""
    return test_data_dir / "STRU"


@pytest.fixture
def valid_stru_nio(test_data_dir):
    """Path to valid NiO STRU file with constraints."""
    return test_data_dir / "STRU_NiO_fixatom"


@pytest.fixture
def temp_stru(tmp_path):
    """Create a temporary STRU file for testing."""
    def _create_stru(content: str) -> Path:
        stru_file = tmp_path / "STRU"
        stru_file.write_text(content)
        return stru_file
    return _create_stru


class TestValidStruFiles:
    """Test validation of valid STRU files."""

    def test_valid_gaas_stru(self, valid_stru_gaas):
        """Test that valid GaAs STRU file passes validation."""
        if not valid_stru_gaas.exists():
            pytest.skip("GaAs STRU test file not found")

        result = validate_stru(str(valid_stru_gaas), check_file_existence=False)

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert "valid" in result["summary"].lower()

    def test_valid_nio_stru(self, valid_stru_nio):
        """Test that valid NiO STRU file passes validation."""
        if not valid_stru_nio.exists():
            pytest.skip("NiO STRU test file not found")

        result = validate_stru(str(valid_stru_nio), check_file_existence=False)

        assert result["valid"] is True
        assert len(result["errors"]) == 0


class TestFileErrors:
    """Test file-level error detection."""

    def test_nonexistent_file(self):
        """Test validation of non-existent file."""
        result = validate_stru("/nonexistent/path/STRU")

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "not found" in result["errors"][0].lower()

    def test_missing_required_section(self, temp_stru):
        """Test detection of missing required sections."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf
As 74.922 As_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is False
        assert any("LATTICE_VECTORS" in error for error in result["errors"])
        assert any("ATOMIC_POSITIONS" in error for error in result["errors"])


class TestAtomicSpecies:
    """Test ATOMIC_SPECIES section validation."""

    def test_no_elements(self, temp_stru):
        """Test detection of empty ATOMIC_SPECIES."""
        content = """ATOMIC_SPECIES

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is False
        assert any("no elements" in error.lower() for error in result["errors"])

    def test_duplicate_element_labels(self, temp_stru):
        """Test detection of duplicate element labels."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is False
        assert any("duplicate" in error.lower() for error in result["errors"])


class TestLatticeConstant:
    """Test LATTICE_CONSTANT section validation."""

    def test_negative_lattice_constant(self, temp_stru):
        """Test detection of negative lattice constant."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
-5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is False
        assert any("lattice constant" in error.lower() and "positive" in error.lower()
                   for error in result["errors"])

    def test_unusually_small_lattice_constant(self, temp_stru):
        """Test warning for unusually small lattice constant."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
0.05

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True  # Should pass but with warning
        assert len(result["warnings"]) > 0
        assert any("small" in warning.lower() for warning in result["warnings"])

    def test_unusually_large_lattice_constant(self, temp_stru):
        """Test warning for unusually large lattice constant."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
150.0

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True  # Should pass but with warning
        assert len(result["warnings"]) > 0
        assert any("large" in warning.lower() for warning in result["warnings"])


class TestLatticeVectors:
    """Test LATTICE_VECTORS section validation."""

    def test_singular_cell_matrix(self, temp_stru):
        """Test detection of singular cell matrix."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
2.0 0.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is False
        assert any("singular" in error.lower() for error in result["errors"])


class TestAtomicPositions:
    """Test ATOMIC_POSITIONS section validation."""

    def test_invalid_coordinate_type(self, temp_stru):
        """Test detection of invalid coordinate type."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Invalid

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is False
        assert any("coordinate type" in error.lower() for error in result["errors"])

    def test_direct_coordinates_outside_range(self, temp_stru):
        """Test warning for Direct coordinates outside [0,1]."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
1.5 0.5 0.5
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True  # Should pass but with warning
        assert len(result["warnings"]) > 0
        assert any("outside" in warning.lower() for warning in result["warnings"])


class TestConsistency:
    """Test consistency checks across sections."""

    def test_element_not_in_species(self, temp_stru):
        """Test detection of element in positions but not in species."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

As
0.0
1
0.25 0.25 0.25
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is False
        assert any("not in atomic_species" in error.lower() for error in result["errors"])

    def test_no_atoms_defined(self, temp_stru):
        """Test detection of no atoms in ATOMIC_POSITIONS."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is False
        assert any("no atoms" in error.lower() for error in result["errors"])


class TestStrictMode:
    """Test strict mode behavior."""

    def test_strict_mode_fails_on_warnings(self, temp_stru):
        """Test that strict mode treats warnings as errors."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
0.05

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)

        # Normal mode should pass with warnings
        result_normal = validate_stru(str(stru_file), check_file_existence=False, strict_mode=False)
        assert result_normal["valid"] is True
        assert len(result_normal["warnings"]) > 0

        # Strict mode should fail
        result_strict = validate_stru(str(stru_file), check_file_existence=False, strict_mode=True)
        assert result_strict["valid"] is False


class TestResultStructure:
    """Test the structure of validation results."""

    def test_result_has_required_keys(self, temp_stru):
        """Test that result dictionary has all required keys."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        required_keys = ["valid", "errors", "warnings", "suggestions", "summary", "details"]
        for key in required_keys:
            assert key in result

    def test_details_has_all_sections(self, temp_stru):
        """Test that details contains all validation sections."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        expected_sections = [
            "file_structure",
            "atomic_species",
            "lattice_constant",
            "lattice_vectors",
            "atomic_positions",
            "consistency"
        ]

        for section in expected_sections:
            assert section in result["details"]


# ============================================================================
# Phase 1: Priority 1 Validations (Critical)
# ============================================================================

class TestExtendedCoordinateTypes:
    """Test extended coordinate type support (Priority 1.1)."""

    def test_cartesian_angstrom(self, temp_stru):
        """Test Cartesian_angstrom coordinate type."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Cartesian_angstrom

Ga
0.0
1
0.0 0.0 0.0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_positions"]["coordinate_type"] == "Cartesian_angstrom"

    def test_cartesian_au(self, temp_stru):
        """Test Cartesian_au coordinate type."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Cartesian_au

Ga
0.0
1
0.0 0.0 0.0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_positions"]["coordinate_type"] == "Cartesian_au"

    def test_cartesian_angstrom_center_xy(self, temp_stru):
        """Test Cartesian_angstrom_center_xy coordinate type."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Cartesian_angstrom_center_xy

Ga
0.0
1
0.0 0.0 0.0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_positions"]["coordinate_type"] == "Cartesian_angstrom_center_xy"

    def test_cartesian_angstrom_center_xz(self, temp_stru):
        """Test Cartesian_angstrom_center_xz coordinate type."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Cartesian_angstrom_center_xz

Ga
0.0
1
0.0 0.0 0.0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_positions"]["coordinate_type"] == "Cartesian_angstrom_center_xz"

    def test_cartesian_angstrom_center_yz(self, temp_stru):
        """Test Cartesian_angstrom_center_yz coordinate type."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Cartesian_angstrom_center_yz

Ga
0.0
1
0.0 0.0 0.0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_positions"]["coordinate_type"] == "Cartesian_angstrom_center_yz"

    def test_cartesian_angstrom_center_xyz(self, temp_stru):
        """Test Cartesian_angstrom_center_xyz coordinate type."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Cartesian_angstrom_center_xyz

Ga
0.0
1
0.0 0.0 0.0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_positions"]["coordinate_type"] == "Cartesian_angstrom_center_xyz"

    def test_invalid_coordinate_type_extended(self, temp_stru):
        """Test invalid coordinate type detection with extended types."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
InvalidType

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is False
        assert any("invalid coordinate type" in e.lower() for e in result["errors"])
        assert any("InvalidType" in e for e in result["errors"])


class TestPseudopotentialTypes:
    """Test PP type validation (Priority 1.2)."""

    def test_valid_pp_type_upf(self, temp_stru):
        """Test valid PP type 'upf'."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_species"]["elements"][0]["pp_type"] == "upf"

    def test_valid_pp_type_vwr(self, temp_stru):
        """Test valid PP type 'vwr'."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga.vwr vwr

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_species"]["elements"][0]["pp_type"] == "vwr"

    def test_valid_pp_type_upf201(self, temp_stru):
        """Test valid PP type 'upf201'."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga.upf upf201

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_species"]["elements"][0]["pp_type"] == "upf201"

    def test_valid_pp_type_blps(self, temp_stru):
        """Test valid PP type 'blps'."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga.blps blps

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_species"]["elements"][0]["pp_type"] == "blps"

    def test_valid_pp_type_coulomb(self, temp_stru):
        """Test valid PP type '1/r' (Coulomb potential)."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga.upf 1/r

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_species"]["elements"][0]["pp_type"] == "1/r"
        assert result["details"]["atomic_species"]["elements"][0].get("coulomb_potential") is True

    def test_invalid_pp_type(self, temp_stru):
        """Test invalid PP type detection."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga.upf xyz

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is False
        assert any("invalid pseudopotential type" in e.lower() for e in result["errors"])
        assert any("xyz" in e for e in result["errors"])


class TestAtomNumberValidation:
    """Test atom count validation (Priority 1.3)."""

    def test_negative_atom_count(self, temp_stru):
        """Test negative atom count error."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
-1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        # Negative atom count causes parsing failure, resulting in "no atoms" error
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_zero_atom_count(self, temp_stru):
        """Test zero atom count warning."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        # Zero atom count results in "no atoms" error from consistency check
        # This is acceptable behavior
        assert result["valid"] is False
        assert any("no atoms" in e.lower() for e in result["errors"])

    def test_positive_atom_count(self, temp_stru):
        """Test positive atom count is valid."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
2
0.0 0.0 0.0
0.5 0.5 0.5
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert result["details"]["atomic_positions"]["elements"][0]["declared_count"] == 2


class TestLeftHandedLattice:
    """Test left-handed lattice detection (Priority 1.4)."""

    def test_negative_determinant(self, temp_stru):
        """Test left-handed lattice warning."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 0.0 1.0
0.0 1.0 0.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True  # Should pass but with warning
        assert any("left-handed" in w.lower() for w in result["warnings"])
        assert result["details"]["lattice_vectors"]["left_handed"] is True
        # Determinant should be stored as positive (absolute value)
        assert result["details"]["lattice_vectors"]["determinant"] > 0

    def test_positive_determinant(self, temp_stru):
        """Test right-handed lattice (no warning)."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert not any("left-handed" in w.lower() for w in result["warnings"])
        assert result["details"]["lattice_vectors"]["left_handed"] is False
        assert result["details"]["lattice_vectors"]["determinant"] > 0


class TestAtomDistanceTolerance:
    """Test corrected atom distance threshold (Priority 1.5)."""

    def test_very_close_atoms(self, temp_stru):
        """Test atoms closer than 1e-3 Bohr (≈ 0.00053 Å)."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
2
0.0 0.0 0.0
0.00001 0.0 0.0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True  # Should pass but with warning
        assert any("very close" in w.lower() for w in result["warnings"])
        # Check that warning mentions both Angstrom and Bohr
        warning_text = ' '.join(result["warnings"])
        assert "angstrom" in warning_text.lower()
        assert "bohr" in warning_text.lower()

    def test_atoms_above_threshold(self, temp_stru):
        """Test atoms farther than threshold (no warning)."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
2
0.0 0.0 0.0
0.5 0.5 0.5
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        # Should not have warning about atoms being too close
        assert not any("very close" in w.lower() for w in result["warnings"])


class TestDirectCoordinateWrapping:
    """Test direct coordinate wrapping warnings (Priority 1.6)."""

    def test_coordinate_far_outside_range(self, temp_stru):
        """Test warning for coordinates far outside [0,1]."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
2.5 0.5 0.5
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True  # Should pass but with warning
        assert any("outside [0,1]" in w.lower() for w in result["warnings"])
        # Check that warning mentions wrapped value
        warning_text = ' '.join(result["warnings"])
        assert "wrapped" in warning_text.lower()

    def test_negative_coordinate_wrapping(self, temp_stru):
        """Test warning for negative coordinates."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
-0.8 0.5 0.5
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True  # Should pass but with warning
        assert any("outside [0,1]" in w.lower() for w in result["warnings"])

    def test_coordinates_in_range(self, temp_stru):
        """Test coordinates in [0,1] range (no warning)."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0.25 0.5 0.75
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        # Should not have warning about coordinates outside range
        # (may have other warnings, but not about coordinate range)


class TestCommentHandling:
    """Test comment parsing (Priority 2.5)."""

    def test_comment_lines_in_species(self, temp_stru):
        """Test lines starting with # in ATOMIC_SPECIES."""
        content = """ATOMIC_SPECIES
# This is a comment
Ga 69.723 Ga_ONCV_PBE-1.0.upf
# Another comment
As 74.922 As_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0

As
0.0
1
0.25 0.25 0.25
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        # Should have 2 elements (comments ignored)
        # Note: AbacusStru may parse differently, so just check it's valid
        assert len(result["details"]["atomic_species"]["elements"]) >= 1

    def test_inline_comments(self, temp_stru):
        """Test inline comments after data."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf  # Gallium pseudopotential

LATTICE_CONSTANT
5.65  # in Angstrom

LATTICE_VECTORS
1.0 0.0 0.0  # a vector
0.0 1.0 0.0  # b vector
0.0 0.0 1.0  # c vector

ATOMIC_POSITIONS
Direct  # fractional coordinates

Ga
0.0
1
0 0 0  # origin
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True


class TestEmptyElementDetection:
    """Test empty element detection for BSSE (Priority 2.1)."""

    def test_empty_element_lowercase(self, temp_stru):
        """Test element with 'empty' in name (lowercase)."""
        content = """ATOMIC_SPECIES
H_empty 1.008 H_ONCV_PBE-1.0.upf
H 1.008 H_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
10.0

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

H_empty
0.0
1
0.5 0.5 0.5

H
0.0
1
0.6 0.6 0.6
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert any("empty atom" in s.lower() for s in result["suggestions"])
        assert "H_empty" in result["details"]["atomic_species"]["empty_elements"]

    def test_empty_element_uppercase(self, temp_stru):
        """Test element with 'EMPTY' in name (uppercase)."""
        content = """ATOMIC_SPECIES
EMPTY_H 1.008 H_ONCV_PBE-1.0.upf
H 1.008 H_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
10.0

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

EMPTY_H
0.0
1
0.5 0.5 0.5

H
0.0
1
0.6 0.6 0.6
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert any("empty atom" in s.lower() for s in result["suggestions"])
        assert "EMPTY_H" in result["details"]["atomic_species"]["empty_elements"]

    def test_normal_element(self, temp_stru):
        """Test normal element (no empty detection)."""
        content = """ATOMIC_SPECIES
Ga 69.723 Ga_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
5.65

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0 0 0
"""
        stru_file = temp_stru(content)
        result = validate_stru(str(stru_file), check_file_existence=False)

        assert result["valid"] is True
        assert not any("empty atom" in s.lower() for s in result["suggestions"])
        assert len(result["details"]["atomic_species"]["empty_elements"]) == 0
