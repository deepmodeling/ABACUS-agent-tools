"""
MCP tool for validating ABACUS STRU files.

This module provides a comprehensive validation tool for ABACUS STRU files,
checking file structure, format correctness, physical validity, and providing
detailed error messages with actionable suggestions for fixing issues.
"""

from pathlib import Path
from typing import Dict, Any
from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.stru_validator import validate_stru as _validate_stru


@mcp.tool()
def validate_stru(
    stru_file: str,
    check_file_existence: bool = True,
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    Validate an ABACUS STRU file for correctness and physical validity.

    This tool performs comprehensive validation of ABACUS STRU files, checking:
    - File structure and required sections
    - ATOMIC_SPECIES format and validity
    - NUMERICAL_ORBITAL section (if present)
    - LATTICE_CONSTANT and LATTICE_VECTORS
    - ATOMIC_POSITIONS format and coordinates
    - Consistency across sections
    - Physical plausibility (atom distances, cell volume, etc.)

    Args:
        stru_file: Path to the STRU file to validate (relative or absolute)
        check_file_existence: Whether to check if referenced pseudopotential
            and orbital files exist (default: True). Set to False if files
            are in a different location or will be provided later.
        strict_mode: If True, treat warnings as errors and fail validation
            (default: False). Use this for strict validation before production runs.

    Returns:
        Dictionary containing:
        - valid (bool): Overall validation status (True if no errors)
        - errors (list): Critical issues that must be fixed
        - warnings (list): Potential issues that should be reviewed
        - suggestions (list): Improvement recommendations
        - summary (str): Human-readable summary of validation results
        - details (dict): Detailed results organized by category:
            - file_structure: Sections found and missing
            - atomic_species: Element definitions and validity
            - numerical_orbital: Orbital files (if present)
            - lattice_constant: Lattice constant value and validity
            - lattice_vectors: Cell matrix, determinant, and volume
            - atomic_positions: Coordinate type and atom positions
            - consistency: Cross-section consistency checks
            - physical_validity: Physical plausibility checks

    Example:
        >>> result = validate_stru("STRU")
        >>> if result['valid']:
        ...     print("STRU file is valid")
        ... else:
        ...     print("Validation failed:")
        ...     for error in result['errors']:
        ...         print(f"  {error}")

        >>> # Strict mode - warnings cause failure
        >>> result = validate_stru("STRU", strict_mode=True)

        >>> # Skip file existence checks
        >>> result = validate_stru("STRU", check_file_existence=False)
    """
    return _validate_stru(stru_file, check_file_existence, strict_mode)
