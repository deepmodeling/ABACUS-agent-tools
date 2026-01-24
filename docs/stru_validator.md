# STRU File Validation Tool

## Overview

The `validate_stru` tool provides comprehensive validation for ABACUS STRU files, checking file structure, format correctness, physical validity, and providing detailed error messages with actionable suggestions for fixing issues.

## Features

- **File Structure Validation**: Checks for required sections (ATOMIC_SPECIES, LATTICE_CONSTANT, LATTICE_VECTORS, ATOMIC_POSITIONS)
- **Format Validation**: Validates data types, ranges, and formats for all sections
- **Consistency Checks**: Ensures consistency across sections (e.g., elements in ATOMIC_POSITIONS exist in ATOMIC_SPECIES)
- **Physical Validity**: Checks for physically plausible values (atom distances, cell volume, magnetic moments)
- **Detailed Error Messages**: Provides specific error locations and actionable fix suggestions
- **Strict Mode**: Optional mode that treats warnings as errors
- **File Reference Checking**: Optionally verifies that pseudopotential and orbital files exist

## Usage

### Basic Usage

```python
from abacusagent.modules.submodules.stru_validator import validate_stru

# Validate a STRU file
result = validate_stru("STRU")

if result['valid']:
    print("✓ STRU file is valid")
else:
    print("✗ Validation failed:")
    for error in result['errors']:
        print(f"  {error}")
```

### MCP Tool Usage

When using through the MCP server:

```python
# The tool is automatically registered as an MCP tool
# and can be called by LLMs through the MCP protocol

result = validate_stru(
    stru_file="path/to/STRU",
    check_file_existence=True,
    strict_mode=False
)
```

### Parameters

- **stru_file** (str): Path to the STRU file to validate (relative or absolute)
- **check_file_existence** (bool, default=True): Whether to check if referenced pseudopotential and orbital files exist
- **strict_mode** (bool, default=False): If True, treat warnings as errors and fail validation

### Return Value

Returns a dictionary with the following structure:

```python
{
    "valid": bool,              # Overall validation status
    "errors": List[str],        # Critical issues (must fix)
    "warnings": List[str],      # Potential issues (should review)
    "suggestions": List[str],   # Improvement recommendations
    "summary": str,             # Human-readable summary
    "details": {                # Detailed results by category
        "file_structure": {...},
        "atomic_species": {...},
        "numerical_orbital": {...},
        "lattice_constant": {...},
        "lattice_vectors": {...},
        "atomic_positions": {...},
        "consistency": {...},
        "physical_validity": {...}
    }
}
```

## Validation Categories

### 1. File Structure
- Checks for presence of all required sections
- Validates section ordering (warnings only)

### 2. ATOMIC_SPECIES
- At least one element defined
- Valid element labels (no duplicates)
- Positive masses
- Pseudopotential file references

### 3. NUMERICAL_ORBITAL (if present)
- Number of orbital files matches number of elements
- Orbital file references

### 4. LATTICE_CONSTANT
- Single positive float value
- Reasonable range (warns if < 0.1 or > 100 Angstrom)

### 5. LATTICE_VECTORS
- Exactly 3 vectors with 3 components each
- Non-singular cell matrix (determinant ≠ 0)
- Reasonable cell volume

### 6. ATOMIC_POSITIONS
- Valid coordinate type (Direct, Cartesian, Cartesian_angstrom, Cartesian_au, Cartesian_angstrom_center_xy/xz/yz/xyz)
- At least 3 coordinates (x, y, z) per atom
- Direct coordinates typically in [0, 1] (warning if outside)
- Optional atom attributes (see Atom Attributes section below)

### 7. Consistency
- All elements in ATOMIC_POSITIONS exist in ATOMIC_SPECIES
- No duplicate element blocks
- Total atom count > 0

### 8. Physical Validity
- Atoms not too close together (< 0.00053 Angstrom / 1e-3 Bohr)
- Reasonable magnetic moments (|mag| < 10)

## Atom Attributes

The validator supports parsing and validation of optional atom attributes that can appear after atomic coordinates in the ATOMIC_POSITIONS section. These attributes match the ABACUS C++ implementation (read_atoms.cpp:206-316).

### Supported Attributes

#### 1. Movement Constraints
Controls which directions an atom can move during relaxation/MD.

**New format (recommended):**
```
0.0 0.0 0.0 m 1 1 0
```
- `m`: keyword
- Three values: 0 (frozen) or 1 (movable) for x, y, z directions

**Old format (deprecated):**
```
0.0 0.0 0.0 0 0 1
```
- Three numeric values immediately after coordinates
- Still supported but triggers deprecation warning

**Validation:**
- Values must be 0 or 1
- Deprecation warning for old format

#### 2. Velocities
Initial velocities for molecular dynamics.

**Format:**
```
0.0 0.0 0.0 v 1.0 2.0 3.0
```
or
```
0.0 0.0 0.0 vel 1.0 2.0 3.0
0.0 0.0 0.0 velocity 1.0 2.0 3.0
```
- Keywords: `v`, `vel`, or `velocity`
- Three float values for vx, vy, vz

**Validation:**
- Must have exactly 3 numeric values

#### 3. Magnetic Moments
Initial magnetic moments for spin-polarized calculations.

**Scalar format (z-component only):**
```
0.0 0.0 0.0 mag 2.0
```

**Vector format (x, y, z components):**
```
0.0 0.0 0.0 mag 1.0 2.0 3.0
```
- Keywords: `mag` or `magmom`
- 1 value (scalar) or 3 values (vector)

**Validation:**
- Cannot use both vector magnetic moment and angles on same atom (ERROR)

#### 4. Angles
Alternative way to specify magnetic moment direction using spherical coordinates.

**Format:**
```
0.0 0.0 0.0 angle1 45.0 angle2 90.0
```
- `angle1`: polar angle (degrees)
- `angle2`: azimuthal angle (degrees)

**Validation:**
- Warning if outside [-360, 360] degrees
- Cannot use with vector magnetic moment (ERROR)

#### 5. Lambda Parameters (DFT+U)
Hubbard U parameters for DFT+U calculations.

**Scalar format (z-component only):**
```
0.0 0.0 0.0 lambda 0.5
```

**Vector format (x, y, z components):**
```
0.0 0.0 0.0 lambda 0.1 0.2 0.3
```
- Keyword: `lambda`
- 1 value (scalar) or 3 values (vector)

#### 6. Spin Constraints
Constrain spin direction during calculations.

**Scalar format (z-component only):**
```
0.0 0.0 0.0 sc 1.0
```

**Vector format (x, y, z components):**
```
0.0 0.0 0.0 sc 0.1 0.2 0.3
```
- Keyword: `sc`
- 1 value (scalar) or 3 values (vector)

### Multiple Attributes

Multiple attributes can be specified on the same line:

```
0.0 0.0 0.0 m 1 1 0 v 0.1 0.2 0.3 mag 2.0
```

### Comments

Attributes support inline comments:

```
0.0 0.0 0.0 m 1 1 0 mag 2.0  # frozen in xy, mag moment 2.0
```

### Validation Results

Attribute validation results are included in the `details["atomic_positions"]["attributes"]` section:

```python
{
    "total_atoms_with_attributes": int,
    "movement_constraints": {
        "count": int,
        "old_format_count": int,
        "new_format_count": int
    },
    "velocities": {
        "count": int
    },
    "magnetic_moments": {
        "scalar_count": int,
        "vector_count": int,
        "angle_count": int,
        "conflicts": []  # Atoms with both vector mag and angles
    },
    "lambda_parameters": {
        "scalar_count": int,
        "vector_count": int
    },
    "spin_constraints": {
        "scalar_count": int,
        "vector_count": int
    },
    "issues": []
}
```

### Attribute Validation Errors

#### Invalid Movement Values
```
ERROR: [ATOMIC_POSITIONS] Invalid movement constraint values
  Element: H, Atom: 1
  Values: 1 2 0
  Expected: Each value must be 0 (frozen) or 1 (movable)
  Fix: Use 0 to freeze or 1 to allow movement in each direction
```

#### Conflicting Magnetic Specifications
```
ERROR: [ATOMIC_POSITIONS] Conflicting magnetic moment specifications
  Element: H, Atom: 1
  Found: Vector magnetic moment AND angles
  Fix: Use either vector magnetic moment (mag x y z) OR angles (angle1/angle2), not both
```

#### Angle Out of Range
```
WARNING: [ATOMIC_POSITIONS] Angle outside reasonable range
  Element: H, Atom: 1
  angle1: 500.0 degrees
  Reasonable range: [-360.0, 360.0]
  Suggestion: Verify this is the intended value
```

#### Deprecated Format
```
WARNING: [ATOMIC_POSITIONS] Deprecated movement constraint format
  Format: Numeric values after coordinates (e.g., '0 0 1')
  Suggestion: Use new keyword format: 'm 0 0 1'
  Note: Old format still works but may be removed in future versions
```

### Example: Complete STRU with Attributes

```
ATOMIC_SPECIES
Ni 58.693 Ni_ONCV_PBE-1.0.upf
O 15.999 O_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
1.889726

LATTICE_VECTORS
4.17 2.085 2.085
2.085 4.17 2.085
2.085 2.085 4.17

ATOMIC_POSITIONS
Direct

Ni
0.0
2
0.0 0.0 0.0 m 0 0 1 mag 2.0
0.5 0.5 0.5 m 1 1 1 mag -2.0

O
0.0
2
0.25 0.25 0.25 m 0 0 0 mag 0.0
0.75 0.75 0.75 m 1 0 1 mag 0.0
```

This file will validate successfully with:
- 4 movement constraints (new format)
- 4 scalar magnetic moments
- 1 deprecation warning if old format is used



## Examples

### Example 1: Basic Validation

```python
result = validate_stru("STRU")

if result['valid']:
    print(f"✓ {result['summary']}")
    if result['warnings']:
        print(f"\nWarnings: {len(result['warnings'])}")
        for warning in result['warnings']:
            print(f"  {warning}")
```

### Example 2: Strict Mode

```python
# Strict mode treats warnings as errors
result = validate_stru("STRU", strict_mode=True)

if not result['valid']:
    print("Validation failed in strict mode")
    print(f"Errors: {len(result['errors'])}")
    print(f"Warnings: {len(result['warnings'])}")
```

### Example 3: Skip File Existence Checks

```python
# Useful when pseudopotential/orbital files are in a different location
result = validate_stru("STRU", check_file_existence=False)
```

### Example 4: Detailed Information

```python
result = validate_stru("STRU")

# Access detailed validation results
print("Lattice constant:", result['details']['lattice_constant']['value'])
print("Cell volume:", result['details']['lattice_vectors']['volume'])
print("Total atoms:", result['details']['consistency']['total_atoms'])

# Check specific sections
for elem in result['details']['atomic_species']['elements']:
    print(f"Element {elem['label']}: mass={elem['mass']}")
```

## Error Message Format

### Errors (Critical Issues)

```
ERROR: [Section] Description
  Location: Line X or Section Y
  Found: <actual value>
  Expected: <expected format>
  Fix: <specific suggestion>
```

### Warnings (Potential Issues)

```
WARNING: [Section] Description
  Location: Line X or Section Y
  Details: <explanation>
  Suggestion: <recommendation>
```

### Suggestions (Improvements)

```
SUGGESTION: <improvement>
  Reason: <why this would be better>
```

## Common Validation Errors

### Missing Required Section

```
ERROR: [File Structure] Required section missing
  Section: LATTICE_VECTORS
  Fix: Add the LATTICE_VECTORS section to the STRU file
```

### Duplicate Element Labels

```
ERROR: [ATOMIC_SPECIES] Duplicate element label
  Label: Ga
  Fix: Each element label must be unique
```

### Singular Cell Matrix

```
ERROR: [LATTICE_VECTORS] Singular cell matrix
  Determinant: 0.0
  Fix: Lattice vectors must be linearly independent
```

### Element Mismatch

```
ERROR: [Consistency] Element in ATOMIC_POSITIONS not in ATOMIC_SPECIES
  Element: As
  Fix: Add As to ATOMIC_SPECIES section
```

## Integration with ABACUS Workflows

The validation tool is designed to be used before running ABACUS calculations:

```python
# Validate STRU file before preparing calculation
result = validate_stru("STRU")

if result['valid']:
    # Proceed with ABACUS preparation
    abacus_prepare(...)
else:
    # Report errors to user
    print("Please fix the following errors:")
    for error in result['errors']:
        print(error)
```

## Testing

Run the unit tests:

```bash
pytest tests/test_stru_validator.py -v
```

Run the example script:

```bash
python examples/validate_stru_example.py
```

## Implementation Details

- **Parser**: Uses `AbacusStru.ReadStru()` for valid files, falls back to manual parsing for invalid files
- **Error Handling**: Gracefully handles `sys.exit()` calls from `AbacusStru` to provide better error messages
- **Dual Format Support**: Handles both `AbacusStru` object format and manual parser format
- **Comprehensive Coverage**: Validates all major sections and common error cases

## Limitations

- File existence checks are relative to the STRU file directory
- Physical validity checks are heuristic-based (e.g., minimum atom distance threshold)
- Attribute parsing focuses on validation; runtime operations (unit conversions, default values) are handled by ABACUS

## Future Enhancements

Potential improvements for future versions:

- Validation of advanced ABACUS features (DFT+U, vdW corrections, etc.)
- Integration with pseudopotential/orbital databases
- Automatic fixing of common issues
- Performance optimization for large STRU files
- Support for STRU file generation from validation results
