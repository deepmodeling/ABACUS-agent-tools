import os
import shutil
from pathlib import Path
import unittest
import tempfile
import pytest
import inspect
from abacusagent.modules.cube import (
    abacus_cal_elf,
    abacus_cal_charge_density_difference,
)
from utils import initilize_test_env, load_test_ref_result

initilize_test_env()


class TestCubeCalculation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_si_prim = (
            Path(__file__).parent / "abacus_inputs_dirs/Si-prim/"
        )
        self.abacus_inputs_dir_h2 = Path(__file__).parent / "abacus_inputs_dirs/H2/"
        self.stru_scf = self.abacus_inputs_dir_si_prim / "STRU_scf"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)

    @pytest.mark.smoke
    def test_abacus_cal_elf_si_prim(self):
        """
        Test the abacus_cal_elf function with Si primitive cell.
        """
        test_func_name = inspect.currentframe().f_code.co_name

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_si_prim, test_work_dir)
        shutil.copy2(self.abacus_inputs_dir_si_prim / "STRU_scf", test_work_dir / "STRU")

        outputs = abacus_cal_elf(test_work_dir)

        elf_file = outputs["elf_file"]
        self.assertIsInstance(elf_file, Path)
        self.assertTrue(elf_file.exists())
        self.assertTrue(outputs["elf_work_path"].exists())

        # Test that the ELF calculation ran successfully
        self.assertTrue(outputs["elf_work_path"].joinpath("OUT.ABACUS", "ELF.cube").exists())

    def test_abacus_cal_charge_density_difference_si_prim(self):
        """
        Test the abacus_cal_charge_density_difference function with Si primitive cell.
        """
        test_func_name = inspect.currentframe().f_code.co_name

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_si_prim, test_work_dir)
        shutil.copy2(self.abacus_inputs_dir_si_prim / "STRU_scf", test_work_dir / "STRU")

        outputs = abacus_cal_charge_density_difference(test_work_dir, subsys1_atom_index=[1])
        print(outputs)

        charge_density_diff_file = outputs["charge_density_difference_cube_file"]
        self.assertIsInstance(charge_density_diff_file, Path)
        self.assertTrue(charge_density_diff_file.exists())
        self.assertTrue(outputs["charge_density_diff_work_path"].exists())

        # Test that the charge density difference calculation ran successfully
        self.assertTrue(outputs["charge_density_diff_work_path"].joinpath("chg_density_diff.cube").exists())

    def test_abacus_cal_charge_density_difference_h2(self):
        """
        Test the abacus_cal_charge_density_difference function with H2 molecule.
        """
        test_func_name = inspect.currentframe().f_code.co_name

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_h2, test_work_dir)
        shutil.copy2(self.abacus_inputs_dir_h2 / "STRU_relaxed", test_work_dir / "STRU")

        outputs = abacus_cal_charge_density_difference(test_work_dir, subsys1_atom_index=[1])
        print(outputs)

        charge_density_diff_file = outputs["charge_density_difference_cube_file"]
        self.assertIsInstance(charge_density_diff_file, Path)
        self.assertTrue(charge_density_diff_file.exists())
        self.assertTrue(outputs["charge_density_diff_work_path"].exists())

        # Test that the charge density difference calculation ran successfully
        self.assertTrue(outputs["charge_density_diff_work_path"].joinpath("chg_density_diff.cube").exists())
