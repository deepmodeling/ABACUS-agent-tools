"""
STRU file validation implementation.

This module provides comprehensive validation for ABACUS STRU files,
checking file structure, format correctness, physical validity, and
providing detailed error messages with actionable suggestions.
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from abacustest.lib_prepare.abacus import AbacusStru

# Constants matching C++ implementation
BOHR_TO_ANGSTROM = 0.529177249
MIN_DISTANCE_BOHR = 1.0e-3
MIN_DISTANCE_ANGSTROM = MIN_DISTANCE_BOHR * BOHR_TO_ANGSTROM  # ≈ 0.00053 Å

# Valid coordinate types (from read_atoms.cpp:35-55)
VALID_COORD_TYPES = [
    "Direct",
    "Cartesian",
    "Cartesian_angstrom",
    "Cartesian_au",
    "Cartesian_angstrom_center_xy",
    "Cartesian_angstrom_center_xz",
    "Cartesian_angstrom_center_yz",
    "Cartesian_angstrom_center_xyz"
]

# Valid pseudopotential types (from read_atom_species.cpp:52-66)
VALID_PP_TYPES = ["auto", "upf", "vwr", "upf201", "blps", "1/r"]


class ValidationResult:
    """Helper class to accumulate validation results."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.suggestions: List[str] = []
        self.details: Dict[str, Any] = {}

    def add_error(self, message: str, section: str = "general"):
        """Add a critical error."""
        self.errors.append(message)

    def add_warning(self, message: str, section: str = "general"):
        """Add a warning."""
        self.warnings.append(message)

    def add_suggestion(self, message: str):
        """Add a suggestion."""
        self.suggestions.append(message)

    def is_valid(self, strict_mode: bool = False) -> bool:
        """Check if validation passed."""
        if self.errors:
            return False
        if strict_mode and self.warnings:
            return False
        return True

    def to_dict(self, strict_mode: bool = False) -> Dict[str, Any]:
        """Convert to dictionary format."""
        valid = self.is_valid(strict_mode)

        # Generate summary
        if valid:
            summary = "✓ STRU file is valid"
            if self.warnings:
                summary += f" ({len(self.warnings)} warning(s))"
        else:
            summary = f"✗ Validation failed: {len(self.errors)} error(s)"
            if self.warnings:
                summary += f", {len(self.warnings)} warning(s)"

        return {
            "valid": valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "summary": summary,
            "details": self.details
        }


def validate_stru(
    stru_file: str,
    check_file_existence: bool = True,
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    Validate an ABACUS STRU file.

    Args:
        stru_file: Path to STRU file to validate
        check_file_existence: Whether to check if referenced PP/orbital files exist
        strict_mode: Treat warnings as errors

    Returns:
        Dictionary with validation results including:
        - valid: Overall validation status
        - errors: List of critical issues
        - warnings: List of potential issues
        - suggestions: List of improvement recommendations
        - summary: Human-readable summary
        - details: Detailed results by category
    """
    result = ValidationResult()
    stru_path = Path(stru_file)

    # Check file exists
    if not stru_path.exists():
        result.add_error(
            f"ERROR: [File] STRU file not found\n"
            f"  Location: {stru_file}\n"
            f"  Fix: Check the file path is correct"
        )
        return result.to_dict(strict_mode)

    # Try to read the file
    try:
        with open(stru_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        result.add_error(
            f"ERROR: [File] Cannot read STRU file\n"
            f"  Location: {stru_file}\n"
            f"  Error: {str(e)}\n"
            f"  Fix: Check file permissions and encoding"
        )
        return result.to_dict(strict_mode)

    # Validate file structure first
    _validate_file_structure(lines, result)

    # Check for duplicate elements in raw file (before AbacusStru deduplicates)
    _check_duplicate_elements(lines, result)

    # If critical structure errors, don't continue parsing
    if result.errors:
        return result.to_dict(strict_mode)

    # Try to parse using AbacusStru for valid files
    stru = None
    try:
        # Temporarily redirect stdout/stderr to suppress AbacusStru warnings
        import sys
        import io
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        try:
            stru = AbacusStru.ReadStru(str(stru_path))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    except SystemExit:
        # AbacusStru calls sys.exit() on errors, catch it
        pass
    except Exception:
        # Other parsing errors
        pass

    # If AbacusStru parsing failed, do manual parsing
    if stru is None:
        stru = _manual_parse_stru(lines, result)
        if stru is None:
            return result.to_dict(strict_mode)

    # Validate each section
    _validate_atomic_species(stru, lines, result, stru_path.parent if check_file_existence else None)
    _validate_numerical_orbital(stru, lines, result, stru_path.parent if check_file_existence else None)
    _validate_lattice_constant(stru, lines, result)
    _validate_lattice_vectors(stru, lines, result)
    _validate_atomic_positions(stru, lines, result)
    _validate_consistency(stru, result)
    _validate_physical(stru, result)

    return result.to_dict(strict_mode)


def strip_comments(line: str) -> str:
    """Remove comments from line (text after #)."""
    comment_pos = line.find('#')
    if comment_pos >= 0:
        return line[:comment_pos].strip()
    return line.strip()


def _manual_parse_stru(lines: List[str], result: ValidationResult):
    """
    Manually parse STRU file when AbacusStru.ReadStru() fails.
    Returns a minimal structure object for validation.
    """
    class ManualStru:
        def __init__(self):
            self.elements = []
            self.masses = {}
            self.pp_files = {}
            self.pp_types = {}
            self.orb_files = {}
            self.lat0 = None
            self.cells = None
            self.coords_type = None
            self.coords = {}
            self.magmoms = {}
            self.empty_elements = []

    stru = ManualStru()
    content = '\n'.join(lines)

    # Parse ATOMIC_SPECIES
    if "ATOMIC_SPECIES" in content:
        try:
            start_idx = next(i for i, line in enumerate(lines) if "ATOMIC_SPECIES" in line)
            i = start_idx + 1
            while i < len(lines):
                raw_line = lines[i]
                line = strip_comments(raw_line).strip()
                if not line:
                    i += 1
                    continue
                if any(section in line for section in ["NUMERICAL_ORBITAL", "LATTICE_CONSTANT", "LATTICE_VECTORS", "ATOMIC_POSITIONS"]):
                    break
                parts = line.split()
                if len(parts) >= 3:
                    label = parts[0]
                    # Always append to preserve duplicates for validation
                    stru.elements.append(label)
                    # Store in dict (will overwrite if duplicate, but we keep list for detection)
                    stru.masses[label] = float(parts[1])
                    stru.pp_files[label] = parts[2]
                    # Check for PP type (4th column, optional)
                    if len(parts) >= 4:
                        stru.pp_types[label] = parts[3]
                    # Check for empty element (BSSE)
                    if "empty" in label.lower():
                        stru.empty_elements.append(label)
                i += 1
        except Exception:
            pass

    # Parse LATTICE_CONSTANT
    if "LATTICE_CONSTANT" in content:
        try:
            start_idx = next(i for i, line in enumerate(lines) if "LATTICE_CONSTANT" in line)
            i = start_idx + 1
            while i < len(lines):
                raw_line = lines[i]
                line = strip_comments(raw_line).strip()
                if line:
                    stru.lat0 = float(line.split()[0])
                    break
                i += 1
        except Exception:
            pass

    # Parse LATTICE_VECTORS
    if "LATTICE_VECTORS" in content:
        try:
            start_idx = next(i for i, line in enumerate(lines) if "LATTICE_VECTORS" in line)
            vectors = []
            i = start_idx + 1
            while i < len(lines) and len(vectors) < 3:
                raw_line = lines[i]
                line = strip_comments(raw_line).strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        vectors.append([float(parts[0]), float(parts[1]), float(parts[2])])
                i += 1
            if len(vectors) == 3:
                stru.cells = vectors
        except Exception:
            pass

    # Parse ATOMIC_POSITIONS
    if "ATOMIC_POSITIONS" in content:
        try:
            start_idx = next(i for i, line in enumerate(lines) if "ATOMIC_POSITIONS" in line)
            coord_type = strip_comments(lines[start_idx + 1]).split()[0]
            stru.coords_type = coord_type

            i = start_idx + 2
            while i < len(lines):
                raw_line = lines[i]
                line = strip_comments(raw_line).strip()
                if not line:
                    i += 1
                    continue

                # Check if this is an element label (single word on a line)
                parts = line.split()
                if len(parts) == 1:
                    elem = parts[0]
                    i += 1
                    # Read magnetism
                    if i < len(lines):
                        mag_line = strip_comments(lines[i]).strip()
                        try:
                            mag = float(mag_line.split()[0])
                            stru.magmoms[elem] = []
                        except:
                            pass
                        i += 1
                    # Read atom count
                    if i < len(lines):
                        count_line = strip_comments(lines[i]).strip()
                        try:
                            count = int(count_line.split()[0])
                            stru.coords[elem] = []
                            i += 1
                            # Read coordinates
                            for _ in range(count):
                                if i < len(lines):
                                    coord_line = strip_comments(lines[i]).strip()
                                    if coord_line:
                                        coord_parts = coord_line.split()
                                        if len(coord_parts) >= 3:
                                            coords = [float(coord_parts[0]), float(coord_parts[1]), float(coord_parts[2])]
                                            stru.coords[elem].append(coords)
                                    i += 1
                        except:
                            i += 1
                else:
                    i += 1
        except Exception:
            pass

    return stru


def _check_duplicate_elements(lines: List[str], result: ValidationResult):
    """Check for duplicate element labels in ATOMIC_SPECIES section."""
    content = '\n'.join(lines)
    if "ATOMIC_SPECIES" not in content:
        return

    try:
        start_idx = next(i for i, line in enumerate(lines) if "ATOMIC_SPECIES" in line)
        i = start_idx + 1
        seen_labels = set()

        while i < len(lines):
            raw_line = lines[i]
            line = strip_comments(raw_line).strip()
            if not line:
                i += 1
                continue
            if any(section in line for section in ["NUMERICAL_ORBITAL", "LATTICE_CONSTANT", "LATTICE_VECTORS", "ATOMIC_POSITIONS"]):
                break

            parts = line.split()
            if len(parts) >= 3:
                label = parts[0]
                if label in seen_labels:
                    result.add_error(
                        f"ERROR: [ATOMIC_SPECIES] Duplicate element label\n"
                        f"  Label: {label}\n"
                        f"  Fix: Each element label must be unique"
                    )
                seen_labels.add(label)
            i += 1
    except Exception:
        pass


def _validate_file_structure(lines: List[str], result: ValidationResult):
    """Validate overall file structure."""
    details = {"sections_found": [], "sections_missing": []}

    required_sections = [
        "ATOMIC_SPECIES",
        "LATTICE_CONSTANT",
        "LATTICE_VECTORS",
        "ATOMIC_POSITIONS"
    ]

    content = '\n'.join(lines)

    for section in required_sections:
        if section in content:
            details["sections_found"].append(section)
        else:
            details["sections_missing"].append(section)
            result.add_error(
                f"ERROR: [File Structure] Required section missing\n"
                f"  Section: {section}\n"
                f"  Fix: Add the {section} section to the STRU file"
            )

    result.details["file_structure"] = details


def _validate_atomic_species(
    stru: AbacusStru,
    lines: List[str],
    result: ValidationResult,
    base_path: Optional[Path] = None
):
    """Validate ATOMIC_SPECIES section."""
    details = {"elements": [], "issues": [], "empty_elements": []}

    # Get elements from AbacusStru object
    elements = getattr(stru, '_element', None) or getattr(stru, 'elements', None)
    if not elements:
        result.add_error(
            f"ERROR: [ATOMIC_SPECIES] No elements defined\n"
            f"  Fix: Add at least one element to ATOMIC_SPECIES section"
        )
        result.details["atomic_species"] = details
        return

    seen_labels = set()
    masses = getattr(stru, '_mass', None) or getattr(stru, 'masses', None)
    pp_files = getattr(stru, '_pp', None) or getattr(stru, 'pp_files', None)

    # Get labels (which may differ from elements, e.g., "H_empty" -> "H")
    labels = getattr(stru, '_label', None)

    # Parse PP types from raw file (not available in AbacusStru)
    pp_types = {}
    content = '\n'.join(lines)
    if "ATOMIC_SPECIES" in content:
        try:
            start_idx = next(i for i, line in enumerate(lines) if "ATOMIC_SPECIES" in line)
            i = start_idx + 1
            while i < len(lines):
                raw_line = lines[i]
                line = strip_comments(raw_line).strip()
                if not line:
                    i += 1
                    continue
                if any(section in line for section in ["NUMERICAL_ORBITAL", "LATTICE_CONSTANT", "LATTICE_VECTORS", "ATOMIC_POSITIONS"]):
                    break
                parts = line.split()
                if len(parts) >= 4:
                    label = parts[0]
                    pp_types[label] = parts[3]
                i += 1
        except Exception:
            pass

    # Iterate over elements (or labels if available)
    elem_list = labels if labels else elements
    for i, elem in enumerate(elem_list):
        elem_info = {"label": elem, "valid": True}

        # Check for duplicate labels
        if elem in seen_labels:
            result.add_error(
                f"ERROR: [ATOMIC_SPECIES] Duplicate element label\n"
                f"  Label: {elem}\n"
                f"  Fix: Each element label must be unique"
            )
            elem_info["valid"] = False
        seen_labels.add(elem)

        # Priority 2.1: Check for empty element (BSSE calculations)
        if "empty" in elem.lower():
            result.add_suggestion(
                f"SUGGESTION: Element '{elem}' detected as empty atom\n"
                f"  Purpose: For BSSE (Basis Set Superposition Error) calculations\n"
                f"  Note: Empty atoms use ghost basis functions"
            )
            details["empty_elements"].append(elem)

        # Check mass
        if masses:
            # Handle both dict and list formats
            if isinstance(masses, dict):
                mass = masses.get(elem)
            elif isinstance(masses, list) and i < len(masses):
                mass = masses[i]
            else:
                mass = None

            if mass is not None:
                if mass <= 0:
                    result.add_error(
                        f"ERROR: [ATOMIC_SPECIES] Invalid mass\n"
                        f"  Element: {elem}\n"
                        f"  Mass: {mass}\n"
                        f"  Expected: Positive float\n"
                        f"  Fix: Set mass to a positive value"
                    )
                    elem_info["valid"] = False
                elem_info["mass"] = mass

        # Check pseudopotential file
        if pp_files:
            # Handle both dict and list formats
            if isinstance(pp_files, dict):
                pp_file = pp_files.get(elem)
            elif isinstance(pp_files, list) and i < len(pp_files):
                pp_file = pp_files[i]
            else:
                pp_file = None

            if pp_file:
                elem_info["pp_file"] = pp_file

            if base_path and pp_file:
                pp_path = base_path / pp_file
                if not pp_path.exists():
                    result.add_warning(
                        f"WARNING: [ATOMIC_SPECIES] Pseudopotential file not found\n"
                        f"  Element: {elem}\n"
                        f"  File: {pp_file}\n"
                        f"  Suggestion: Check the file path or set ABACUS_PP_PATH"
                    )

        # Priority 1.2: Validate pseudopotential type
        if elem in pp_types:
            pp_type = pp_types[elem]
            elem_info["pp_type"] = pp_type

            if pp_type not in VALID_PP_TYPES:
                result.add_error(
                    f"ERROR: [ATOMIC_SPECIES] Invalid pseudopotential type\n"
                    f"  Element: {elem}\n"
                    f"  Type: {pp_type}\n"
                    f"  Valid types: {', '.join(VALID_PP_TYPES)}\n"
                    f"  Fix: Use a valid PP type or omit for 'auto'"
                )
                elem_info["valid"] = False
            elif pp_type == "1/r":
                elem_info["coulomb_potential"] = True

        details["elements"].append(elem_info)

    result.details["atomic_species"] = details


def _validate_numerical_orbital(
    stru: AbacusStru,
    lines: List[str],
    result: ValidationResult,
    base_path: Optional[Path] = None
):
    """Validate NUMERICAL_ORBITAL section if present."""
    details = {"present": False, "orbital_files": []}

    # Check if NUMERICAL_ORBITAL section exists
    content = '\n'.join(lines)
    if "NUMERICAL_ORBITAL" not in content:
        result.details["numerical_orbital"] = details
        return

    details["present"] = True

    orb_files = getattr(stru, '_orb', None) or getattr(stru, 'orb_files', None)
    elements = getattr(stru, '_element', None) or getattr(stru, 'elements', None)

    if orb_files:
        for i, orb_file in enumerate(orb_files):
            elem = elements[i] if elements and i < len(elements) else f"Element_{i}"
            orb_info = {"element": elem, "file": orb_file, "exists": None}

            if base_path and orb_file:
                orb_path = base_path / orb_file
                orb_info["exists"] = orb_path.exists()
                if not orb_path.exists():
                    result.add_warning(
                        f"WARNING: [NUMERICAL_ORBITAL] Orbital file not found\n"
                        f"  Element: {elem}\n"
                        f"  File: {orb_file}\n"
                        f"  Suggestion: Check the file path or set ABACUS_ORB_PATH"
                    )

            details["orbital_files"].append(orb_info)

        # Check if number of orbital files matches number of elements
        if elements and len(orb_files) != len(elements):
            result.add_warning(
                f"WARNING: [NUMERICAL_ORBITAL] Orbital file count mismatch\n"
                f"  Elements: {len(elements)}\n"
                f"  Orbital files: {len(orb_files)}\n"
                f"  Suggestion: Provide orbital files for all elements"
            )

    result.details["numerical_orbital"] = details


def _validate_lattice_constant(stru: AbacusStru, lines: List[str], result: ValidationResult):
    """Validate LATTICE_CONSTANT section."""
    details = {"value": None, "valid": True}

    lat0 = getattr(stru, '_lattice_constant', None) or getattr(stru, 'lat0', None)
    if lat0 is None:
        result.add_error(
            f"ERROR: [LATTICE_CONSTANT] Lattice constant not defined\n"
            f"  Fix: Add LATTICE_CONSTANT section with a positive value"
        )
        details["valid"] = False
        result.details["lattice_constant"] = details
        return

    details["value"] = lat0

    if lat0 <= 0:
        result.add_error(
            f"ERROR: [LATTICE_CONSTANT] Invalid lattice constant\n"
            f"  Value: {lat0}\n"
            f"  Expected: Positive float\n"
            f"  Fix: Set lattice constant to a positive value"
        )
        details["valid"] = False
    elif lat0 < 0.1:
        result.add_warning(
            f"WARNING: [LATTICE_CONSTANT] Unusually small lattice constant\n"
            f"  Value: {lat0} Angstrom\n"
            f"  Suggestion: Verify this is the intended value"
        )
    elif lat0 > 100:
        result.add_warning(
            f"WARNING: [LATTICE_CONSTANT] Unusually large lattice constant\n"
            f"  Value: {lat0} Angstrom\n"
            f"  Suggestion: Verify this is the intended value"
        )

    result.details["lattice_constant"] = details


def _validate_lattice_vectors(stru: AbacusStru, lines: List[str], result: ValidationResult):
    """Validate LATTICE_VECTORS section."""
    details = {"vectors": None, "determinant": None, "volume": None, "valid": True, "left_handed": False}

    cells = getattr(stru, '_cell', None) or getattr(stru, 'cells', None)
    if cells is None:
        result.add_error(
            f"ERROR: [LATTICE_VECTORS] Lattice vectors not defined\n"
            f"  Fix: Add LATTICE_VECTORS section with 3 vectors"
        )
        details["valid"] = False
        result.details["lattice_vectors"] = details
        return

    cells = np.array(cells)
    details["vectors"] = cells.tolist()

    # Check shape
    if cells.shape != (3, 3):
        result.add_error(
            f"ERROR: [LATTICE_VECTORS] Invalid lattice vectors shape\n"
            f"  Shape: {cells.shape}\n"
            f"  Expected: (3, 3)\n"
            f"  Fix: Provide exactly 3 vectors with 3 components each"
        )
        details["valid"] = False
        result.details["lattice_vectors"] = details
        return

    # Check determinant (non-singular)
    det = np.linalg.det(cells)

    # Priority 1.4: Left-handed lattice detection
    if det < 0:
        result.add_warning(
            f"WARNING: [LATTICE_VECTORS] Left-handed lattice detected\n"
            f"  Determinant: {det:.6e}\n"
            f"  Note: Using absolute value for volume calculation\n"
            f"  Suggestion: Consider using right-handed coordinate system"
        )
        details["left_handed"] = True
        det = abs(det)

    details["determinant"] = float(det)

    if abs(det) < 1e-10:
        result.add_error(
            f"ERROR: [LATTICE_VECTORS] Singular cell matrix\n"
            f"  Determinant: {det}\n"
            f"  Fix: Lattice vectors must be linearly independent"
        )
        details["valid"] = False

    # Calculate volume
    lat0 = getattr(stru, '_lattice_constant', None) or getattr(stru, 'lat0', None)
    if lat0:
        volume = abs(det) * (lat0 ** 3)
        details["volume"] = float(volume)

        if volume < 1.0:
            result.add_warning(
                f"WARNING: [LATTICE_VECTORS] Unusually small cell volume\n"
                f"  Volume: {volume:.2f} Angstrom^3\n"
                f"  Suggestion: Verify lattice vectors and constant are correct"
            )
        elif volume > 100000:
            result.add_warning(
                f"WARNING: [LATTICE_VECTORS] Unusually large cell volume\n"
                f"  Volume: {volume:.2f} Angstrom^3\n"
                f"  Suggestion: Verify lattice vectors and constant are correct"
            )

    result.details["lattice_vectors"] = details


def _validate_atomic_positions(stru: AbacusStru, lines: List[str], result: ValidationResult):
    """Validate ATOMIC_POSITIONS section."""
    details = {"coordinate_type": None, "elements": [], "valid": True}

    # Priority 1.1: Parse coordinate type from raw file (AbacusStru normalizes it)
    content = '\n'.join(lines)
    coord_type = None
    if "ATOMIC_POSITIONS" in content:
        start_idx = next((i for i, line in enumerate(lines) if "ATOMIC_POSITIONS" in line), None)
        if start_idx is not None and start_idx + 1 < len(lines):
            coord_type = strip_comments(lines[start_idx + 1]).split()[0]
            details["coordinate_type"] = coord_type

            # Extended coordinate type support
            if coord_type not in VALID_COORD_TYPES:
                result.add_error(
                    f"ERROR: [ATOMIC_POSITIONS] Invalid coordinate type\n"
                    f"  Type: {coord_type}\n"
                    f"  Expected: One of {', '.join(VALID_COORD_TYPES)}\n"
                    f"  Fix: Use a valid coordinate type"
                )
                details["valid"] = False

    # Fallback to AbacusStru if we couldn't parse from lines
    if coord_type is None:
        cartesian = getattr(stru, '_cartesian', None)
        if cartesian is not None:
            coord_type = "Cartesian" if cartesian else "Direct"
            details["coordinate_type"] = coord_type

    # Get elements and coordinates
    elements = getattr(stru, '_element', None) or getattr(stru, 'elements', None)
    labels = getattr(stru, '_label', None)
    atom_numbers = getattr(stru, '_atom_number', None)
    coords = getattr(stru, '_coord', None)

    # For validation, use labels from ATOMIC_SPECIES (which may differ from elements)
    species_labels = labels if labels else elements

    if not elements:
        result.details["atomic_positions"] = details
        return

    # Build coords dict by element
    if coords and labels and atom_numbers:
        coord_idx = 0
        for i, (label, count) in enumerate(zip(labels, atom_numbers)):
            elem_info = {
                "label": label,
                "declared_count": count,
                "actual_count": count,
                "valid": True
            }

            # Priority 1.3: Atom number validation
            if count < 0:
                result.add_error(
                    f"ERROR: [ATOMIC_POSITIONS] Negative atom count\n"
                    f"  Element: {label}\n"
                    f"  Count: {count}\n"
                    f"  Fix: Atom count must be non-negative"
                )
                elem_info["valid"] = False
            elif count == 0:
                result.add_warning(
                    f"WARNING: [ATOMIC_POSITIONS] Zero atoms for element\n"
                    f"  Element: {label}\n"
                    f"  Note: If this is intentional (e.g., for DP model), ignore this warning\n"
                    f"  Suggestion: Verify this is not a mistake"
                )

            # Check if element exists in ATOMIC_SPECIES
            # Use species_labels for comparison (which includes labels like "H_empty")
            if species_labels and label not in species_labels:
                result.add_error(
                    f"ERROR: [ATOMIC_POSITIONS] Unknown element\n"
                    f"  Element: {label}\n"
                    f"  Fix: Add {label} to ATOMIC_SPECIES section"
                )
                elem_info["valid"] = False

            # Validate coordinates for this element (only if count > 0)
            if count > 0:
                elem_coords = coords[coord_idx:coord_idx + count]
                coord_idx += count

                # Priority 1.6: Direct coordinate wrapping validation
                if details["coordinate_type"] and details["coordinate_type"] == "Direct":
                    for j, coord in enumerate(elem_coords):
                        for k, val in enumerate(coord[:3]):
                            # Check if significantly outside [0, 1]
                            if val < -0.5 or val > 1.5:
                                # Match C++ wrapping: fmod(x + 10000, 1.0)
                                wrapped = (val + 10000) % 1.0
                                result.add_warning(
                                    f"WARNING: [ATOMIC_POSITIONS] Direct coordinate outside [0,1]\n"
                                    f"  Element: {label}, Atom: {j+1}, Component: {['x','y','z'][k]}\n"
                                    f"  Value: {val}\n"
                                    f"  Wrapped value: {wrapped:.6f}\n"
                                    f"  Note: Coordinates will be wrapped for periodic boundaries\n"
                                    f"  Suggestion: Consider using coordinates in [0,1] range"
                                )
                            elif val < -0.1 or val > 1.1:
                                # Keep existing warning for moderate deviations
                                result.add_warning(
                                    f"WARNING: [ATOMIC_POSITIONS] Direct coordinate outside [0,1]\n"
                                    f"  Element: {label}, Atom: {j+1}, Component: {['x','y','z'][k]}\n"
                                    f"  Value: {val}\n"
                                    f"  Suggestion: Direct coordinates are typically in [0,1]"
                                )

            details["elements"].append(elem_info)

    result.details["atomic_positions"] = details


def _validate_consistency(stru: AbacusStru, result: ValidationResult):
    """Validate consistency across sections."""
    details = {"checks": []}

    # Get elements and labels - handle both AbacusStru and ManualStru formats
    elements = getattr(stru, '_element', None) or getattr(stru, 'elements', None)
    labels = getattr(stru, '_label', None)
    coords = getattr(stru, '_coord', None) or getattr(stru, 'coords', None)

    # For consistency checking, use labels from ATOMIC_SPECIES
    # AbacusStru may have both _label (from ATOMIC_POSITIONS) and elements (from ATOMIC_SPECIES)
    # We need to get the actual labels from ATOMIC_SPECIES for comparison
    species_labels = labels if labels else elements

    # Check all elements in ATOMIC_POSITIONS exist in ATOMIC_SPECIES
    # Handle both list (from AbacusStru) and dict (from manual parser) formats
    if labels and species_labels:
        # AbacusStru format - labels is a list
        # Note: AbacusStru may normalize element names (e.g., "H_empty" -> "H")
        # but keeps original labels, so we check if labels are in species_labels
        for label in labels:
            # Check if label exists in species_labels (exact match)
            if label not in species_labels:
                result.add_error(
                    f"ERROR: [Consistency] Element in ATOMIC_POSITIONS not in ATOMIC_SPECIES\n"
                    f"  Element: {label}\n"
                    f"  Fix: Add {label} to ATOMIC_SPECIES section"
                )
                details["checks"].append({
                    "check": "element_consistency",
                    "passed": False,
                    "element": label
                })
            else:
                details["checks"].append({
                    "check": "element_consistency",
                    "passed": True,
                    "element": label
                })
    elif coords and isinstance(coords, dict) and elements:
        # Manual parser format - coords is a dict with element keys
        for label in coords.keys():
            if label not in elements:
                result.add_error(
                    f"ERROR: [Consistency] Element in ATOMIC_POSITIONS not in ATOMIC_SPECIES\n"
                    f"  Element: {label}\n"
                    f"  Fix: Add {label} to ATOMIC_SPECIES section"
                )
                details["checks"].append({
                    "check": "element_consistency",
                    "passed": False,
                    "element": label
                })
            else:
                details["checks"].append({
                    "check": "element_consistency",
                    "passed": True,
                    "element": label
                })

    # Calculate total atom count
    total_atoms = 0
    if coords:
        if isinstance(coords, list):
            total_atoms = len(coords)
        elif isinstance(coords, dict):
            total_atoms = sum(len(c) for c in coords.values())
        details["total_atoms"] = total_atoms

    if total_atoms == 0:
        result.add_error(
            f"ERROR: [Consistency] No atoms defined\n"
            f"  Fix: Add atomic positions to ATOMIC_POSITIONS section"
        )

    result.details["consistency"] = details


def _validate_physical(stru: AbacusStru, result: ValidationResult):
    """Validate physical plausibility."""
    details = {"checks": []}

    # Get necessary attributes
    coords = getattr(stru, '_coord', None)
    cells = getattr(stru, '_cell', None)
    lat0 = getattr(stru, '_lattice_constant', None) or getattr(stru, 'lat0', None)
    cartesian = getattr(stru, '_cartesian', False)

    # Check for atoms too close together
    if coords and cells and lat0:
        try:
            # Convert all coordinates to Cartesian
            cells_array = np.array(cells) * lat0
            all_positions = []

            for coord in coords:
                if not cartesian:
                    # Convert Direct to Cartesian
                    pos = np.dot(coord[:3], cells_array)
                else:
                    pos = np.array(coord[:3]) * lat0
                all_positions.append(pos)

            # Check pairwise distances
            min_distance = float('inf')
            for i in range(len(all_positions)):
                for j in range(i + 1, len(all_positions)):
                    dist = np.linalg.norm(all_positions[i] - all_positions[j])
                    min_distance = min(min_distance, dist)

                    # Priority 1.5: Corrected atom distance tolerance (1e-3 Bohr ≈ 0.00053 Å)
                    if dist < MIN_DISTANCE_ANGSTROM:
                        result.add_warning(
                            f"WARNING: [Physical] Atoms very close together\n"
                            f"  Distance: {dist:.6f} Angstrom ({dist/BOHR_TO_ANGSTROM:.6f} Bohr)\n"
                            f"  Threshold: {MIN_DISTANCE_ANGSTROM:.6f} Angstrom ({MIN_DISTANCE_BOHR} Bohr)\n"
                            f"  Atoms: {i+1} and {j+1}\n"
                            f"  Suggestion: Verify atomic positions are correct"
                        )

            details["min_distance"] = float(min_distance) if min_distance != float('inf') else None
        except Exception as e:
            details["checks"].append({
                "check": "atom_distances",
                "passed": False,
                "error": str(e)
            })

    # Check magnetic moments if present
    magmoms = getattr(stru, '_magmom', None)
    if magmoms:
        for i, mag in enumerate(magmoms):
            if mag is not None and abs(mag) > 10:
                result.add_warning(
                    f"WARNING: [Physical] Unusually large magnetic moment\n"
                    f"  Element index: {i+1}\n"
                    f"  Magnetic moment: {mag}\n"
                    f"  Suggestion: Verify this is the intended value"
                )

    result.details["physical_validity"] = details
