import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
import pytest
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.work_function import abacus_cal_work_function

initilize_test_env()

class TestAbacusWorkFunction(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_al110 = Path(__file__).parent / 'abacus_inputs_dirs/Al110/'
        self.stru_al110 = self.abacus_inputs_dir_al110 / 'STRU'
        self.input_al110 = self.abacus_inputs_dir_al110 / 'INPUT_work_function'
        self.abacus_inputs_dir_zno0001 = Path(__file__).parent / 'abacus_inputs_dirs/ZnO0001/'
        self.stru_zno0001 = self.abacus_inputs_dir_zno0001 / 'STRU'
        self.input_zno0001 =  self.abacus_inputs_dir_zno0001 / 'INPUT_work_function'

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)
    
    @pytest.mark.smoke
    def test_abacus_cal_work_function_al110(self):
        """
        Calculate the work function of Al(100) surface.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        
        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_al110, test_work_dir)
        shutil.copy2(self.stru_al110, test_work_dir / 'STRU')
        shutil.copy2(self.input_al110, test_work_dir / 'INPUT')

        outputs = abacus_cal_work_function(test_work_dir, vacuum_direction='b')

        print(outputs)

        self.assertIsInstance(outputs['averaged_elecstat_pot_plot'], get_path_type())
        self.assertIsInstance(outputs['averaged_elecstat_pot_dat_file'], get_path_type())
        self.assertEqual(len(outputs['work_function_results']), len(ref_results['work_function_results']))
        for i in range(len(outputs['work_function_results'])):
            self.assertAlmostEqual(outputs['work_function_results'][i]['work_function'], ref_results['work_function_results'][i]['work_function'], places=2)

    def test_abacus_cal_work_function_zno0001_dipole_corr(self):
        """
        Calculate the work function of ZnO(0001) surface with dipole correction.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        
        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_zno0001, test_work_dir)
        shutil.copy2(self.stru_zno0001, test_work_dir / "STRU")
        shutil.copy2(self.input_zno0001, test_work_dir / 'INPUT')

        outputs = abacus_cal_work_function(test_work_dir, vacuum_direction='c', dipole_correction=True)

        print(outputs)

        self.assertIsInstance(outputs['averaged_elecstat_pot_plot'], get_path_type())
        self.assertIsInstance(outputs['averaged_elecstat_pot_dat_file'], get_path_type())
        self.assertEqual(len(outputs['work_function_results']), len(ref_results['work_function_results']))
        for i in range(len(outputs['work_function_results'])):
            self.assertAlmostEqual(outputs['work_function_results'][i]['work_function'], ref_results['work_function_results'][i]['work_function'], places=2)
