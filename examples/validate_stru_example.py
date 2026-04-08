#!/usr/bin/env python3
"""
Example script demonstrating the STRU file validation tool.

This script shows how to use the validate_stru tool to check ABACUS STRU files
for correctness and physical validity.
"""

import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from abacusagent.modules.submodules.stru_validator import validate_stru


def print_validation_result(result, filename):
    """Pretty print validation results."""
    print(f"\n{'='*70}")
    print(f"Validation Results for: {filename}")
    print(f"{'='*70}")

    print(f"\n{result['summary']}")

    if result['errors']:
        print(f"\n❌ ERRORS ({len(result['errors'])}):")
        print("-" * 70)
        for error in result['errors']:
            print(error)
            print()

    if result['warnings']:
        print(f"\n⚠️  WARNINGS ({len(result['warnings'])}):")
        print("-" * 70)
        for warning in result['warnings']:
            print(warning)
            print()

    if result['suggestions']:
        print(f"\n💡 SUGGESTIONS ({len(result['suggestions'])}):")
        print("-" * 70)
        for suggestion in result['suggestions']:
            print(f"  • {suggestion}")

    print(f"\n{'='*70}\n")


def main():
    """Run validation examples."""

    # Example 1: Validate a valid STRU file
    print("\n" + "="*70)
    print("Example 1: Validating a correct STRU file")
    print("="*70)

    test_stru = os.path.join(os.path.dirname(__file__), '..', 'tests', 'abacus', 'STRU')
    if os.path.exists(test_stru):
        result = validate_stru(test_stru, check_file_existence=False)
        print_validation_result(result, test_stru)
    else:
        print(f"Test file not found: {test_stru}")

    # Example 2: Validate with strict mode
    print("\n" + "="*70)
    print("Example 2: Strict mode validation")
    print("="*70)
    print("Strict mode treats warnings as errors.\n")

    if os.path.exists(test_stru):
        result = validate_stru(test_stru, check_file_existence=False, strict_mode=True)
        print(f"Valid in normal mode: {validate_stru(test_stru, check_file_existence=False)['valid']}")
        print(f"Valid in strict mode: {result['valid']}")

    # Example 3: Show detailed validation information
    print("\n" + "="*70)
    print("Example 3: Detailed validation information")
    print("="*70)

    if os.path.exists(test_stru):
        result = validate_stru(test_stru, check_file_existence=False)

        print("\nFile Structure:")
        print(f"  Sections found: {', '.join(result['details']['file_structure']['sections_found'])}")

        print("\nAtomic Species:")
        for elem in result['details']['atomic_species']['elements']:
            print(f"  {elem['label']}: mass={elem.get('mass', 'N/A')}")

        print("\nLattice:")
        lat_const = result['details']['lattice_constant']
        print(f"  Constant: {lat_const['value']} Angstrom")

        lat_vec = result['details']['lattice_vectors']
        if lat_vec.get('volume'):
            print(f"  Volume: {lat_vec['volume']:.2f} Angstrom^3")

        print("\nAtomic Positions:")
        print(f"  Coordinate type: {result['details']['atomic_positions']['coordinate_type']}")
        for elem in result['details']['atomic_positions']['elements']:
            print(f"  {elem['label']}: {elem['actual_count']} atoms")

        if 'total_atoms' in result['details']['consistency']:
            print(f"\nTotal atoms: {result['details']['consistency']['total_atoms']}")


if __name__ == '__main__':
    main()
