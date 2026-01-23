import os
import shutil
from pathlib import Path
import unittest
import tempfile
import inspect
import pytest
from abacusagent.modules.tool_wrapper import *
from utils import initilize_test_env, load_test_ref_result, get_path_type

initilize_test_env()

@pytest.mark.tool_wrapper
class TestToolWrapper(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        self.test_path = Path(self.test_dir.name)
        self.abacus_inputs_dir_si_prim = Path(__file__).parent / 'abacus_inputs_dirs/Si-prim/'
        self.stru_scf = self.abacus_inputs_dir_si_prim / "STRU_scf"
        self.abacus_inputs_dir_si_prim_elastic = Path(__file__).parent / 'abacus_inputs_dirs/Si-prim-elastic/'
        self.stru_elastic = self.abacus_inputs_dir_si_prim_elastic / "STRU_cell_relaxed"
        self.abacus_inputs_dir_fe_bcc_prim = Path(__file__).parent / 'abacus_inputs_dirs/Fe-BCC-prim/'
        self.stru_fe_bcc_prim = self.abacus_inputs_dir_fe_bcc_prim / "STRU_cell_relaxed"
        self.abacus_inputs_dir_al110 = Path(__file__).parent / 'abacus_inputs_dirs/Al110/'
        self.stru_al110 = self.abacus_inputs_dir_al110 / "STRU"
        self.abacus_inputs_dir_tial = Path(__file__).parent / 'abacus_inputs_dirs/gamma-TiAl-P4mmm/'
        self.stru_tial = self.abacus_inputs_dir_tial / "STRU"
        self.abacus_inputs_dir_nacl_prim = Path(__file__).parent / 'abacus_inputs_dirs/NaCl-prim/'
        self.stru_nacl_eos_cubic = self.abacus_inputs_dir_nacl_prim / "STRU_cubic_eos"
        self.abacus_inputs_dir_zno = Path(__file__).parent / 'abacus_inputs_dirs/ZnO/'
        self.stru_zno = self.abacus_inputs_dir_zno / "STRU"

        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)


    def tearDown(self):
        os.chdir(self.original_cwd)
    
    @pytest.mark.smoke
    def test_run_abacus_calculation_scf(self):
        """
        Test the wrapper function of doing SCF calculation.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")

        outputs = abacus_calculation_scf(test_work_dir / "STRU",
                                         stru_type='abacus/stru',
                                         lcao=True,
                                         nspin=1,
                                         dft_functional='PBE',
                                         dftu=False,
                                         dftu_param=None,
                                         init_mag=None)
        print(outputs)

        self.assertTrue(outputs['normal_end'])
        self.assertTrue(outputs['converge'])
        self.assertAlmostEqual(outputs['energy'], ref_results['energy'], delta=1e-6)
    
    def test_run_abacus_calculation_dftu_initmag(self):
        """
        Test the wrapper function of doing SCF calculation with DFT+U and initial magnetic moment.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_zno, test_work_dir / "STRU")

        dftu_param = {
            'element': ['Zn', 'O'],
            'orbital': ['d', 'p'],
            'U_value': [4.0, 1.0]
        }
        init_mag = {
            'element': ['Zn', 'O'],
            'mag': [1.0, 0.5]
        }

        outputs = abacus_calculation_scf(test_work_dir / "STRU",
                                         stru_type='abacus/stru',
                                         lcao=True,
                                         nspin=2,
                                         dft_functional='PBE',
                                         dftu=True,
                                         dftu_param=dftu_param,
                                         init_mag=init_mag)
        print(outputs)

        self.assertTrue(outputs['normal_end'])
        self.assertTrue(outputs['converge'])
        self.assertAlmostEqual(outputs['energy'], ref_results['energy'], delta=1e-6)

    def test_run_abacus_calculation_elf(self):
        """
        Test the wrapper function of doing SCF calculation.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")

        outputs = abacus_cal_elf(test_work_dir / "STRU",
                                 stru_type='abacus/stru',
                                 lcao=True,
                                 nspin=1,
                                 dft_functional='PBE',
                                 dftu=False,
                                 dftu_param=None,
                                 init_mag=None)
        print(outputs)

        self.assertIsInstance(outputs['elf_file'], get_path_type())
        self.assertTrue(os.path.exists(outputs['elf_file']))
    
    def test_run_abacus_calculation_relax(self):
        """
        Test the wrapper function of doing relax calculation.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")

        relax_precision='medium'
        outputs = abacus_do_relax(test_work_dir / "STRU",
                                  stru_type='abacus/stru',
                                  lcao=True,
                                  nspin=1,
                                  dft_functional='PBE',
                                  dftu=False,
                                  dftu_param=None,
                                  init_mag=None,
                                  max_steps=100,
                                  relax_cell=True,
                                  relax_precision=relax_precision,
                                  relax_method='cg',
                                  fixed_axes=None)
        print(outputs)

        self.assertTrue(outputs['final_stru'].exists())
        self.assertTrue(outputs['relax_converge'])

    def test_run_abacus_calculation_dos(self):
        """
        Test the wrapper function of calculating DOS.
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = abacus_dos_run(test_work_dir / "STRU",
                                 stru_type='abacus/stru',
                                 lcao=True,
                                 nspin=1,
                                 dft_functional='PBE',
                                 dftu=False,
                                 dftu_param=None,
                                 init_mag=None,
                                 max_steps=100,
                                 relax=True,
                                 relax_cell=True,
                                 relax_method='cg',
                                 relax_precision='medium',
                                 fixed_axes=None,
                                 pdos_mode='species+shell',
                                 dos_edelta_ev=0.01,
                                 dos_sigma=0.07)
        print(outputs)

        dos_fig_path = outputs['dos_fig_path']
        pdos_fig_path = outputs['pdos_fig_path']
        dos_data_path = outputs['dos_data_path']
        pdos_data_paths = outputs['pdos_data_paths']

        self.assertIsInstance(dos_fig_path, get_path_type())
        self.assertIsInstance(dos_data_path, get_path_type())
        self.assertIsInstance(pdos_fig_path, get_path_type())
        for pdos_data_path in pdos_data_paths:
            self.assertIsInstance(pdos_data_path, get_path_type())

        self.assertTrue(outputs['scf_normal_end'])
        self.assertTrue(outputs['scf_converge'])
        self.assertTrue(outputs['nscf_normal_end'])
        self.assertAlmostEqual(outputs['scf_energy'], ref_results['scf_energy'])
    
    def test_run_abacus_calculation_bader_charge(self):
        """
        Test the abacus_calculation_scf function to calculate Bader charge
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = abacus_badercharge_run(stru_file=self.stru_scf,
                                         stru_type='abacus/stru',
                                         lcao=True,
                                         nspin=1,
                                         dft_functional="PBE",
                                         dftu=False,
                                         dftu_param=None,
                                         init_mag=None,
                                         relax_cell=True,
                                         relax_precision='medium',
                                         relax_method='cg',
                                         fixed_axes=None)
        print(outputs)

        self.assertIsInstance(outputs['bader_result_csv'], get_path_type())
    
    def test_run_abacus_calculation_band(self):
        """
        Test the abacus_calculation_scf function to calculate band
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = abacus_cal_band(stru_file = self.stru_scf,
                                  stru_type='abacus/stru',
                                  lcao=True,
                                  nspin=1,
                                  dft_functional="PBE",
                                  dftu=False,
                                  dftu_param=None,
                                  init_mag=None,
                                  relax_cell=True,
                                  relax_precision='medium',
                                  relax_method='cg',
                                  fixed_axes=None,
                                  mode='auto',
                                  energy_min=-10,
                                  energy_max=10,
                                  insert_point_nums=30)
        print(outputs)

        band_picture = outputs['band_picture']
        self.assertIsInstance(band_picture, get_path_type())
        self.assertAlmostEqual(outputs['band_gap'], ref_results['band_gap'], delta=1e-4)
    
    @pytest.mark.long
    def test_run_abacus_calculation_elastic_properties(self):
        """
        Test the abacus_calculation_scf function to calculate elastic properties
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_elastic, test_work_dir / "STRU")
        
        outputs = abacus_cal_elastic(stru_file = self.stru_scf,
                                     stru_type='abacus/stru',
                                     lcao=True,
                                     nspin=1,
                                     dft_functional="PBE",
                                     dftu=False,
                                     dftu_param=None,
                                     init_mag=None,
                                     relax_cell=True,
                                     relax_precision='medium',
                                     relax_method='cg',
                                     fixed_axes=None,)
        print(outputs)

        # Compare calculated and reference elastic tensor
        self.assertEqual(len(outputs['elastic_tensor']), len(ref_results['elastic_tensor']))
        for elastic_tensor_output_row, elastic_tensor_ref_row in zip(outputs['elastic_tensor'], ref_results['elastic_tensor']):
            self.assertEqual(len(elastic_tensor_output_row), len(elastic_tensor_ref_row))
            for element_output, element_ref in zip(elastic_tensor_output_row, elastic_tensor_ref_row):
                self.assertAlmostEqual(element_output, element_ref, delta=0.1)

        self.assertAlmostEqual(outputs['bulk_modulus'], ref_results['bulk_modulus'], delta=0.1)
        self.assertAlmostEqual(outputs['shear_modulus'], ref_results['shear_modulus'], delta=0.1)
        self.assertAlmostEqual(outputs['young_modulus'], ref_results['young_modulus'], delta=0.1)
        self.assertAlmostEqual(outputs['poisson_ratio'], ref_results['poisson_ratio'], delta=0.01)
    
    @pytest.mark.long
    def test_run_abacus_calculation_phonon_dispersion(self):
        """
        Test the abacus_calculation_scf function to calculate phonon dispersion
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        outputs = abacus_phonon_dispersion(stru_file = self.stru_scf,
                                           stru_type='abacus/stru',
                                           lcao=True,
                                           nspin=1,
                                           dft_functional="PBE",
                                           dftu=False,
                                           dftu_param=None,
                                           init_mag=None,
                                           relax_cell=True,
                                           relax_precision='medium',
                                           relax_method='cg',
                                           fixed_axes=None,)
        print(outputs)
        
        self.assertIsInstance(outputs['band_dos_plot'], get_path_type())

        self.assertAlmostEqual(outputs['entropy'], ref_results['entropy'], delta=1e-2)
        self.assertAlmostEqual(outputs['free_energy'], ref_results['free_energy'], delta=1e-2)
        self.assertAlmostEqual(outputs['max_frequency_THz'], ref_results['max_frequency_THz'], delta=1e-2)
        self.assertAlmostEqual(outputs['max_frequency_K'], ref_results['max_frequency_K'], delta=1e-2)
    
    def test_run_abacus_calculation_md(self):
        """
        Test the abacus_calculation_scf function to do AIMD calculation
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_scf, test_work_dir / "STRU")
        
        md_nstep = 5
        outputs = abacus_run_md(stru_file = self.stru_tial,
                                stru_type='abacus/stru',
                                lcao=True,
                                nspin=1,
                                dft_functional="PBE",
                                dftu=False,
                                dftu_param=None,
                                init_mag=None,
                                relax_cell=True,
                                relax_precision='medium',
                                relax_method='cg',
                                fixed_axes=None,
                                md_type='nve',
                                md_nstep=md_nstep,
                                md_dt=1.0,
                                md_tfirst=300)
        print(outputs)
        
        self.assertTrue(outputs['normal_end'])
        self.assertEqual(outputs['traj_frame_nums'], md_nstep+1)
        self.assertIsInstance(outputs['md_traj_file'], get_path_type())

    def test_run_abacus_calculation_vacancy_formation_energy(self):
        """
        Test the abacus_calculation_scf function to calculate vacancy formation energy
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_tial, test_work_dir / "STRU")
        
        outputs = abacus_vacancy_formation_energy(stru_file = self.stru_tial,
                                                  stru_type='abacus/stru',
                                                  lcao=True,
                                                  nspin=1,
                                                  dft_functional="PBE",
                                                  dftu=False,
                                                  dftu_param=None,
                                                  init_mag=None,
                                                  relax_cell=True,
                                                  relax_precision='medium',
                                                  relax_method='cg',
                                                  fixed_axes=None,
                                                  supercell=[1, 1, 1],
                                                  vacancy_index = 1)
        
        print(outputs)

        self.assertTrue(outputs['original_stru_job_relax_converge'])
        self.assertTrue(outputs['original_stru_job_relax_converge'])
        self.assertAlmostEqual(outputs['vacancy_formation_energy'], ref_results['vacancy_formation_energy'], delta=2)
   
    def test_run_abacus_calculation_work_function(self):
        """
        Test the abacus_calculation_scf function to calculate work function
        """
        test_func_name = inspect.currentframe().f_code.co_name
        ref_results = load_test_ref_result(test_func_name)
        
        test_work_dir = self.test_path / test_func_name
        os.makedirs(test_work_dir, exist_ok=True)
        shutil.copy2(self.stru_al110, test_work_dir / "STRU")

        outputs = abacus_cal_work_function(stru_file=self.stru_al110,
                                           stru_type='abacus/stru',
                                           lcao=True,
                                           nspin=1,
                                           dft_functional="PBE",
                                           dftu=False,
                                           dftu_param=None,
                                           init_mag=None,
                                           relax=False,
                                           relax_cell=True,
                                           relax_precision='medium',
                                           relax_method='cg',
                                           fixed_axes=None,
                                           vacuum_direction='y',
                                           dipole_correction=False,)
        print(outputs)

        self.assertIsInstance(outputs['averaged_elecstat_pot_plot'], get_path_type())
        self.assertEqual(len(outputs['work_function_results']), len(ref_results['work_function_results']))
        for i in range(len(outputs['work_function_results'])):
            self.assertAlmostEqual(outputs['work_function_results'][i]['work_function'], ref_results['work_function_results'][i]['work_function'], places=2)
        
