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
- Valid coordinate type (Direct or Cartesian)
- At least 3 coordinates (x, y, z) per atom
- Direct coordinates typically in [0, 1] (warning if outside)
- Valid movement flags and magnetic moments (if present)

### 7. Consistency
- All elements in ATOMIC_POSITIONS exist in ATOMIC_SPECIES
- No duplicate element blocks
- Total atom count > 0

### 8. Physical Validity
- Atoms not too close together (< 0.5 Angstrom)
- Reasonable magnetic moments (|mag| < 10)

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

- Does not validate advanced features like DFT+U parameters or external fields
- File existence checks are relative to the STRU file directory
- Physical validity checks are heuristic-based (e.g., minimum atom distance threshold)

## Future Enhancements

Potential improvements for future versions:

- Validation of advanced ABACUS features (DFT+U, vdW corrections, etc.)
- Integration with pseudopotential/orbital databases
- Automatic fixing of common issues
- Performance optimization for large STRU files
- Support for STRU file generation from validation results
