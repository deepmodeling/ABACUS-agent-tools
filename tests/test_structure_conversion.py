import unittest
import os
import tempfile
import glob
from pathlib import Path

os.environ["ABACUSAGENT_MODEL"] = "test"

from abacusagent.modules.structure_editor import (
    convert_to_primitive,
    convert_to_conventional,
)


class TestStructureConversion(unittest.TestCase):
    """Tests for structure conversion functions with proper cleanup."""

    def setUp(self):
        """Create test structure files and set up temporary directory."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)
        self.original_cwd = os.getcwd()

        # Change to test directory so output files are created there
        os.chdir(self.test_path)

        # Create test structure files
        from ase.build import bulk
        from ase.io import write

        # FCC conventional cell (4 atoms)
        fcc_conventional = bulk("Al", "fcc", a=4.05, cubic=True)
        self.fcc_conventional_file = self.test_path / "Al_fcc_conventional.cif"
        write(self.fcc_conventional_file, fcc_conventional, format="cif")

        # FCC primitive cell (1 atom)
        fcc_primitive = bulk("Al", "fcc", a=4.05, cubic=False)
        self.fcc_primitive_file = self.test_path / "Al_fcc_primitive.cif"
        write(self.fcc_primitive_file, fcc_primitive, format="cif")

        # BCC conventional cell (2 atoms)
        bcc_conventional = bulk("Fe", "bcc", a=2.87, cubic=True)
        self.bcc_conventional_file = self.test_path / "Fe_bcc_conventional.cif"
        write(self.bcc_conventional_file, bcc_conventional, format="cif")

        # BCC primitive cell (1 atom)
        bcc_primitive = bulk("Fe", "bcc", a=2.87, cubic=False)
        self.bcc_primitive_file = self.test_path / "Fe_bcc_primitive.cif"
        write(self.bcc_primitive_file, bcc_primitive, format="cif")

    def tearDown(self):
        """Clean up and return to original directory."""
        # Clean up any remaining generated files
        for pattern in ["*_primitive*", "*_conventional*"]:
            for filepath in glob.glob(pattern):
                try:
                    os.remove(filepath)
                except (OSError, PermissionError):
                    pass

        # Return to original directory
        os.chdir(self.original_cwd)

        # Clean up temporary directory
        self.test_dir.cleanup()

    def test_primitive_conversion(self):
        """Test converting conventional cells to primitive cells."""
        test_cases = [
            ("FCC", self.fcc_conventional_file, 1, "Fm-3m"),
            ("BCC", self.bcc_conventional_file, 1, "Im-3m"),
        ]

        for name, input_file, expected_atoms, expected_spacegroup in test_cases:
            with self.subTest(structure=name):
                result = convert_to_primitive(
                    input_file,
                    stru_type="cif",
                    output_format="cif",
                    tolerance=0.25,
                )

                self.assertIn("output_file", result)
                self.assertIn("num_atoms", result)
                self.assertIn("cell", result)
                self.assertIn("spacegroup", result)

                self.assertEqual(result["num_atoms"], expected_atoms)
                self.assertTrue(os.path.exists(result["output_file"]))
                self.assertEqual(result["spacegroup"], expected_spacegroup)

                # Check cell format
                cell = result["cell"]
                self.assertIsInstance(cell, list)
                self.assertEqual(len(cell), 3)
                for row in cell:
                    self.assertIsInstance(row, list)
                    self.assertEqual(len(row), 3)

    def test_conventional_conversion(self):
        """Test converting primitive cells to conventional cells."""
        test_cases = [
            ("FCC", self.fcc_primitive_file, 4, "Fm-3m"),
            ("BCC", self.bcc_primitive_file, 2, "Im-3m"),
        ]

        for name, input_file, expected_atoms, expected_spacegroup in test_cases:
            with self.subTest(structure=name):
                result = convert_to_conventional(
                    input_file,
                    stru_type="cif",
                    output_format="cif",
                    tolerance=0.01,
                )

                self.assertIn("output_file", result)
                self.assertIn("num_atoms", result)
                self.assertIn("cell", result)
                self.assertIn("spacegroup", result)

                self.assertEqual(result["num_atoms"], expected_atoms)
                self.assertTrue(os.path.exists(result["output_file"]))
                self.assertEqual(result["spacegroup"], expected_spacegroup)

    def test_round_trip_conversion(self):
        """Test round-trip conversion: conventional -> primitive -> conventional."""
        test_cases = [
            ("FCC", self.fcc_conventional_file, 4),
            ("BCC", self.bcc_conventional_file, 2),
        ]

        for name, input_file, expected_atoms in test_cases:
            with self.subTest(structure=name):
                # Convert conventional to primitive
                primitive_result = convert_to_primitive(
                    input_file,
                    stru_type="cif",
                    output_format="cif",
                    tolerance=0.25,
                )

                # Convert primitive back to conventional
                conventional_result = convert_to_conventional(
                    primitive_result["output_file"],
                    stru_type="cif",
                    output_format="cif",
                    tolerance=0.01,
                )

                # Should get back to original number of atoms
                self.assertEqual(conventional_result["num_atoms"], expected_atoms)

                # Space group should be consistent
                self.assertEqual(
                    primitive_result["spacegroup"], conventional_result["spacegroup"]
                )

    def test_output_formats(self):
        """Test conversion with different output formats."""
        output_formats = [
            ("cif", ".cif"),
            ("poscar", ".vasp"),
            ("abacus/stru", ".stru"),
        ]

        for format_name, file_extension in output_formats:
            with self.subTest(format=format_name):
                # Test primitive conversion
                result = convert_to_primitive(
                    self.fcc_conventional_file,
                    stru_type="cif",
                    output_format=format_name,
                    tolerance=0.25,
                )

                self.assertIn("output_file", result)
                self.assertTrue(os.path.exists(result["output_file"]))
                self.assertTrue(str(result["output_file"]).endswith(file_extension))

                # Test conventional conversion
                result = convert_to_conventional(
                    self.fcc_primitive_file,
                    stru_type="cif",
                    output_format=format_name,
                    tolerance=0.01,
                )

                self.assertIn("output_file", result)
                self.assertTrue(os.path.exists(result["output_file"]))
                self.assertTrue(str(result["output_file"]).endswith(file_extension))

    def test_error_handling(self):
        """Test error handling for non-existent files."""
        # Need to be in test directory for this test
        non_existent = Path("non_existent_file.cif")

        # Test primitive conversion
        result = convert_to_primitive(
            non_existent,
            stru_type="cif",
            output_format="cif",
            tolerance=0.25,
        )
        self.assertIn("message", result)
        self.assertIn("failed", result["message"].lower())

        # Test conventional conversion
        result = convert_to_conventional(
            non_existent,
            stru_type="cif",
            output_format="cif",
            tolerance=0.01,
        )
        self.assertIn("message", result)
        self.assertIn("failed", result["message"].lower())

    def test_tolerance_parameter(self):
        """Test that tolerance parameter affects symmetry detection."""
        # Test with different tolerance values
        tolerances = [0.001, 0.25, 1.0]

        for tolerance in tolerances:
            with self.subTest(tolerance=tolerance):
                result = convert_to_primitive(
                    self.fcc_conventional_file,
                    stru_type="cif",
                    output_format="cif",
                    tolerance=tolerance,
                )

                self.assertIn("output_file", result)
                self.assertIn("num_atoms", result)
                self.assertIn("spacegroup", result)

                # Should succeed and give correct number of atoms
                self.assertEqual(result["num_atoms"], 1)
                self.assertEqual(result["spacegroup"], "Fm-3m")

    def test_same_input_output_format(self):
        """Test conversion when input and output formats are the same."""
        # Test with explicit same format
        result = convert_to_primitive(
            self.fcc_conventional_file,
            stru_type="cif",
            output_format="cif",
            tolerance=0.25,
        )

        self.assertIn("output_file", result)
        self.assertTrue(os.path.exists(result["output_file"]))
        self.assertTrue(str(result["output_file"]).endswith(".cif"))


if __name__ == "__main__":
    unittest.main()
