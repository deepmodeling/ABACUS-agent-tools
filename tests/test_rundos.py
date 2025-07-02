import unittest
import os, sys
from pathlib import Path
import tempfile
import shutil

#sys.path.append(str(Path(__file__).parent.parent / "src" / "abacusagent" / "modules"))
from abacusagent.modules.dos import run_dos, pygrep
from abacusagent.modules.dos import plot_dos_pdos as mkplots 

class TestRunDos(unittest.TestCase):
    def setUp(self):
        # Set up the environment for the test
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        
        self.data_dir = Path(__file__).parent / "plot_dos"

        ### copy "INPUT", "STRU", *.upf, and *.orb in data_dir to test_path
        for item in self.data_dir.iterdir():
            if item.is_file() and (item.name.endswith('.upf') or item.name.endswith('.orb') or item.name in ['INPUT', 'STRU']):
                target_path = self.test_path / item.name
                target_path.write_bytes(item.read_bytes())
            elif item.is_dir() and item.name == "OUT.diamond":
        # Copy directory recursively, overwrite if it exists
                target_dir = self.test_path / item.name
                try:
                    shutil.copytree(item, target_dir)
                except FileExistsError:
                    shutil.rmtree(target_dir)
                    shutil.copytree(item, target_dir)

        # Set the environment variable for the test

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

        print(f"Test directory set up at: {self.test_path}")

    def tearDown(self):
        # Clean up the environment after the test
        os.chdir(self.original_cwd)
    
    def test_run_dos(self):
        """
        Test the run_dos function with a valid input.
        """

        # ignore the screen output
        sys.stdout = open(os.devnull, 'w')

        # Call the run_dos function
        dosfilepaths = run_dos('.', test_mode=True)

        for i, plot_file in enumerate(dosfilepaths):
            self.assertTrue(os.path.exists(plot_file), f"File {plot_file} does not exist.")
            self.assertTrue(os.path.isfile(plot_file), f"{plot_file} is not a file.")
            self.assertTrue(os.path.getsize(plot_file) > 0, f"{plot_file} is empty.")


class TestPygrep(unittest.TestCase):
    def setUp(self):
        # Set up the environment for the test
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)

        # Set the environment variable for the test
        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)
        print(f"Test directory set up at: {self.test_path}")

    def tearDown(self):
        # Clean up the environment after the test
        os.chdir(self.original_cwd)

    def test_pygrep(self):
        test_file = self.test_path / "test.txt"
        test_file.write_text("Hello world!\nThis is a test.")

        result = pygrep("Hello", str(test_file))
        self.assertEqual(result, "Hello world!\n")
        result = pygrep("hello", str(test_file))
        self.assertEqual(result, "")
        



