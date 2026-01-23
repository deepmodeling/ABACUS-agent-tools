import os
import shutil
from pathlib import Path
import unittest
import tempfile
import pytest
import inspect
from utils import initilize_test_env, load_test_ref_result, get_path_type
from abacusagent.modules.dos import abacus_dos_run

initilize_test_env()

class TestAbacusDosRun(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_nacl_prim = Path(__file__).parent / 'abacus_inputs_dirs/NaCl-prim/'
        self.stru_dos_nacl_prim = self.abacus_inputs_dir_nacl_prim / "STRU_dos"
        self.input_dos_nacl_prim = self.abacus_inputs_dir_nacl_prim / "INPUT_nspin1"
        self.input_dos_pw_nacl_prim = self.abacus_inputs_dir_nacl_prim / "INPUT_pw_nspin1"
        self.abacus_inputs_dir_fe_bcc_prim = Path(__file__).parent / 'abacus_inputs_dirs/Fe-BCC-prim/'
        self.stru_dos_fe_bcc_prim = self.abacus_inputs_dir_fe_bcc_prim / "STRU_dos"
        self.input_dos_pw_fe_bcc_prim = self.abacus_inputs_dir_fe_bcc_prim / "INPUT_pw_nspin2_gammaonly"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)

    def tearDown(self):
        os.chdir(self.original_cwd)
    
    @pytest.mark.smoke
    def test_abacus_dos_run_species(self):
        """
        Test the abacus_dos_run function with PDOS plotting mode set to different species.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_nacl_prim, test_work_dir)
        shutil.copy2(self.stru_dos_nacl_prim, test_work_dir / "STRU")
        shutil.copy2(self.input_dos_nacl_prim, test_work_dir / "INPUT")

        outputs = abacus_dos_run(test_work_dir,
                                 pdos_mode='species',
                                 dos_edelta_ev = 0.01,
                                 dos_sigma = 0.07,
                                 dos_emin_ev=-20,
                                 dos_emax_ev=20)
        print(outputs)
        
        dos_fig_path = outputs['dos_fig_path']
        dos_data_path = outputs['dos_data_path']
        pdos_fig_path = outputs['pdos_fig_path']
        pdos_data_path = outputs['pdos_data_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(dos_data_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertIsInstance(pdos_data_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])

    def test_abacus_dos_run_species_shell(self):
        """
        Test the abacus_dos_run function with PDOS plotting mode set to different species and shell.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_nacl_prim, test_work_dir)
        shutil.copy2(self.stru_dos_nacl_prim, test_work_dir / "STRU")
        shutil.copy2(self.input_dos_nacl_prim, test_work_dir / "INPUT")

        outputs = abacus_dos_run(test_work_dir,
                                 pdos_mode='species+shell')
        
        dos_fig_path = outputs['dos_fig_path']
        dos_data_path = outputs['dos_data_path']
        pdos_fig_path = outputs['pdos_fig_path']
        pdos_data_path = outputs['pdos_data_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(dos_data_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertIsInstance(pdos_data_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])

    def test_abacus_dos_run_species_orbital(self):
        """
        Test the abacus_dos_run function with PDOS plotting mode set to different species and orbitals.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_nacl_prim, test_work_dir)
        shutil.copy2(self.stru_dos_nacl_prim, test_work_dir / "STRU")
        shutil.copy2(self.input_dos_nacl_prim, test_work_dir / "INPUT")

        outputs = abacus_dos_run(test_work_dir,
                                 pdos_mode='species+orbital')
        
        dos_fig_path = outputs['dos_fig_path']
        dos_data_path = outputs['dos_data_path']
        pdos_fig_path = outputs['pdos_fig_path']
        pdos_data_path = outputs['pdos_data_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(dos_data_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertIsInstance(pdos_data_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])

    def test_abacus_dos_run_atoms(self):
        """
        Test the abacus_dos_run function with PDOS plotting mode set to different species and shell.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_nacl_prim, test_work_dir)
        shutil.copy2(self.stru_dos_nacl_prim, test_work_dir / "STRU")
        shutil.copy2(self.input_dos_nacl_prim, test_work_dir / "INPUT")

        outputs = abacus_dos_run(test_work_dir,
                                 pdos_mode='atoms',
                                 pdos_atom_indices = [1, 2])
        
        dos_fig_path = outputs['dos_fig_path']
        dos_data_path = outputs['dos_data_path']
        pdos_fig_path = outputs['pdos_fig_path']
        pdos_data_path = outputs['pdos_data_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(dos_data_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertIsInstance(pdos_data_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])


    def test_abacus_dos_run_species_nspin2(self):
        """
        Test the abacus_dos_run function with nspin=2 case and PDOS plotting mode set to different species.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_fe_bcc_prim, test_work_dir)
        shutil.copy2(self.stru_dos_fe_bcc_prim, test_work_dir / "STRU")

        outputs = abacus_dos_run(test_work_dir,
                                 pdos_mode='species',
                                 dos_edelta_ev = 0.01,
                                 dos_sigma = 0.07,
                                 dos_emin_ev = -20,
                                 dos_emax_ev =  20)
        
        dos_fig_path = outputs['dos_fig_path']
        dos_data_path = outputs['dos_data_path']
        pdos_fig_path = outputs['pdos_fig_path']
        pdos_data_path = outputs['pdos_data_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(dos_data_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertIsInstance(pdos_data_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])

    def test_abacus_dos_run_species_shell_nspin2(self):
        """
        Test the abacus_dos_run function with nspin=2 case and PDOS plotting mode set to different species and shell.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_fe_bcc_prim, test_work_dir)
        shutil.copy2(self.stru_dos_fe_bcc_prim, test_work_dir / "STRU")

        outputs = abacus_dos_run(test_work_dir,
                                 pdos_mode='species+shell')
        
        dos_fig_path = outputs['dos_fig_path']
        dos_data_path = outputs['dos_data_path']
        pdos_fig_path = outputs['pdos_fig_path']
        pdos_data_path = outputs['pdos_data_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(dos_data_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertIsInstance(pdos_data_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])

    def test_abacus_dos_run_species_orbital_nspin2(self):
        """
        Test the abacus_dos_run function with nspin=2 case and PDOS plotting mode set to different species and orbitals.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_fe_bcc_prim, test_work_dir)
        shutil.copy2(self.stru_dos_fe_bcc_prim, test_work_dir / "STRU")

        outputs = abacus_dos_run(test_work_dir,
                                 pdos_mode='species+orbital')
        
        dos_fig_path = outputs['dos_fig_path']
        dos_data_path = outputs['dos_data_path']
        pdos_fig_path = outputs['pdos_fig_path']
        pdos_data_path = outputs['pdos_data_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(dos_data_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertIsInstance(pdos_data_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])

    def test_abacus_dos_run_atoms_nspin2(self):
        """
        Test the abacus_dos_run function with nspin=2 case and PDOS plotting mode set to different species.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_fe_bcc_prim, test_work_dir)
        shutil.copy2(self.stru_dos_fe_bcc_prim, test_work_dir / "STRU")

        outputs = abacus_dos_run(test_work_dir,
                                 pdos_mode='atoms',
                                 pdos_atom_indices = [1],
                                 dos_edelta_ev = 0.01,
                                 dos_sigma = 0.07,
                                 dos_emin_ev = -20,
                                 dos_emax_ev =  20)
        
        dos_fig_path = outputs['dos_fig_path']
        dos_data_path = outputs['dos_data_path']
        pdos_fig_path = outputs['pdos_fig_path']
        pdos_data_path = outputs['pdos_data_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(dos_data_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        self.assertIsInstance(pdos_data_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])
    
    @pytest.mark.smoke
    def test_abacus_dos_run_pw_nspin1(self):
        """
        Test the abacus_dos_run function with nspin=1 case with pw basis
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_nacl_prim, test_work_dir)
        shutil.copy2(self.stru_dos_nacl_prim, test_work_dir / "STRU")
        shutil.copy2(self.input_dos_pw_nacl_prim, test_work_dir / "INPUT")

        outputs = abacus_dos_run(test_work_dir)
        print(outputs)

        dos_fig_path = outputs['dos_fig_path']
        dos_data_path = outputs['dos_data_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(dos_data_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'], places=6)

    def test_abacus_dos_run_pw_nspin2(self):
        """
        Test the abacus_dos_run function with nspin=2 case with pw basis
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)

        test_work_dir = self.test_path / test_func_name
        shutil.copytree(self.abacus_inputs_dir_fe_bcc_prim, test_work_dir)
        shutil.copy2(self.stru_dos_fe_bcc_prim, test_work_dir / "STRU")
        shutil.copy2(self.input_dos_pw_fe_bcc_prim, test_work_dir / "INPUT")

        outputs = abacus_dos_run(test_work_dir)
        print(outputs)

        dos_fig_path = outputs['dos_fig_path']
        dos_data_path = outputs['dos_data_path']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(dos_data_path, get_path_type())
        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'], places=6)
