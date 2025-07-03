import unittest
import os, sys
import shutil
import math
import json
from pathlib import Path
import tempfile
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput

os.environ["ABACUSAGENT_MODEL"] = "test"  # Set the model to test
from abacusagent.modules.abacus import abacus_prepare, abacus_modify_input, abacus_modify_stru, abacus_collect_data

class TestAbacusPrepare(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)  
        self.test_path = Path(self.test_dir.name)
        
        self.data_dir = Path(__file__).parent / "abacus"
        self.pp_path = (self.data_dir / "pp").resolve()
        self.orb_path = (self.data_dir / "orb").resolve()
        self.stru_file1 = (self.data_dir / "STRU").resolve()
        self.stru_file2 = (self.data_dir / "POSCAR").resolve()
        
        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)
        
        print(f"Test directory: {self.test_path}")
            
    def tearDown(self):
        os.chdir(self.original_cwd) 
        
    def test_abacus_prepare(self):
        """
        Test the abacus_prepare function.
        """
        
        # catch the screen output
        sys.stdout = open(os.devnull, 'w')
        
        outputs = abacus_prepare(str(self.stru_file1.absolute()),
            stru_type = "abacus/stru",
            pp_path= self.pp_path,
            orb_path = self.orb_path,
            job_type= "scf",
            lcao= True
        )
        self.assertTrue(os.path.exists(outputs["job_path"]))
        self.assertTrue(os.path.exists(outputs["job_path"] / "INPUT"))
        self.assertTrue(os.path.exists(outputs["job_path"] / "STRU"))
        self.assertTrue(os.path.exists(outputs["job_path"] / "As_ONCV_PBE-1.0.upf"))
        self.assertTrue(os.path.exists(outputs["job_path"] / "As_gga_8au_100Ry_2s2p1d.orb"))
        self.assertTrue(os.path.exists(outputs["job_path"] / "Ga_ONCV_PBE-1.0.upf"))
        self.assertTrue(os.path.exists(outputs["job_path"] / "Ga_gga_9au_100Ry_2s2p2d.orb"))
        
        input_files = ["INPUT", "STRU", "As_ONCV_PBE-1.0.upf", "As_gga_8au_100Ry_2s2p1d.orb",
                       "Ga_ONCV_PBE-1.0.upf", "Ga_gga_9au_100Ry_2s2p2d.orb","struinfo.txt"]
        self.assertCountEqual(outputs["input_files"], input_files)
        self.assertTrue("input_content" in outputs)
        


class TestAbacusModifyInput(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)  
        self.test_path = Path(self.test_dir.name)
        
        self.data_dir = Path(__file__).parent / "abacus"
        
        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)
        
        print(f"Test directory: {self.test_path}")
            
    def tearDown(self):
        os.chdir(self.original_cwd) 

    def test_abacus_modify_input_basic(self):
        """
        Test modify INPUT parameters without modifying DFT+U parameters
        """
        extra_input = {'vdw_method': 'd3_bj', 'nspin': 2}
        remove_input = {'gamma_only'}
        # copy INPUT and STRU file to current directory
        shutil.copy2(self.data_dir / "INPUT_ref", "INPUT")
        shutil.copy2(self.data_dir / "STRU", "STRU")

        original_input_param = ReadInput("INPUT")
        outputs = abacus_modify_input(".",
                                      extra_input=extra_input,
                                      remove_input=remove_input)
        
        modified_input_param = ReadInput("INPUT")
        self.assertEqual(modified_input_param['vdw_method'], extra_input['vdw_method'])
        self.assertEqual(modified_input_param['nspin'], extra_input['nspin'])
        self.assertEqual(modified_input_param['ecutwfc'], original_input_param['ecutwfc'])

    def test_abacus_modify_input_dftu(self):
        """
        Test modify INPUT parameters about DFT+U settings
        """

        dft_plus_u_settings = {'Fe': ['d', 3.0],
                               'O':  0.5}
        extra_input = {'vdw_method': 'd3_bj', 'nspin': 2}
        # copy INPUT and STRU file to current directory
        shutil.copy2(self.data_dir / "INPUT_LiFePO4_old", "INPUT")
        shutil.copy2(self.data_dir / "STRU_LiFePO4", "STRU")

        original_input_param = ReadInput("INPUT")

        outputs = abacus_modify_input(".",
                                      dft_plus_u_settings=dft_plus_u_settings,
                                      extra_input=extra_input)

        modified_input_param = ReadInput("INPUT")
        self.assertEqual(modified_input_param['vdw_method'], extra_input['vdw_method'])
        self.assertEqual(modified_input_param['nspin'], extra_input['nspin'])
        self.assertEqual(modified_input_param['ecutwfc'], original_input_param['ecutwfc'])

        self.assertEqual(modified_input_param['dft_plus_u'], 1)
        orbital_corr_modified = modified_input_param['orbital_corr'].split()
        orbital_corr_ref = ['2', '-1', '-1', '1']
        self.assertEqual(len(orbital_corr_modified), len(orbital_corr_ref))
        for i in range(len(orbital_corr_modified)):
            self.assertEqual(orbital_corr_modified[i], orbital_corr_ref[i])
        
        hubbard_u_modified = modified_input_param['hubbard_u'].split()
        for i in range(len(hubbard_u_modified)):
            hubbard_u_modified[i] = float(hubbard_u_modified[i])
        hubbard_u_ref = [3.0, 0.0, 0.0, 0.5]
        self.assertEqual(len(hubbard_u_modified), len(hubbard_u_ref))
        for i in range(len(hubbard_u_modified)):
            self.assertAlmostEqual(hubbard_u_modified[i], hubbard_u_ref[i])


class TestAbacusModifyStru(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)  
        self.test_path = Path(self.test_dir.name)
        
        self.data_dir = Path(__file__).parent / "abacus"
        
        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)
        
        print(f"Test directory: {self.test_path}")
            
    def tearDown(self):
        os.chdir(self.original_cwd) 

    def test_abacus_modify_stru_pp_orb(self):
        """
        Test modify pseudopotential and orbital in STRU file
        """
        pp = {'Ni': 'Ni_ONCV_PBE-1.2.upf', 'O': 'O_ONCV_PBE-1.2.upf'}
        orb = {'Ni': 'Ni_gga_10au_6s3p3d2f.orb', 'O': 'O_gga_10au_3s3p2d.orb'}
        shutil.copy2(self.data_dir / "STRU_NiO_ref", "STRU")
        shutil.copy2(self.data_dir / "INPUT_ref", "INPUT")

        outputs = abacus_modify_stru(".", 
                                    pp=pp,
                                    orb=orb)
        

        modified_stru = AbacusStru.ReadStru("STRU")

        for idx, element in enumerate(modified_stru.get_element(number=False,total=False)):
            self.assertEqual(modified_stru.get_pp()[idx],  pp[element])
            self.assertEqual(modified_stru.get_orb()[idx], orb[element])


    def test_abacus_modify_stru_fixed_atoms(self):
        """
        Test modify atom fixation in STRU file
        """
        fix_atoms_idx = [0, 2, 3]
        movable_coors = [[0, 0, 1],
                         [0, 0, 0],
                         [1, 0, 1]]
        shutil.copy2(self.data_dir / "STRU_NiO_ref", "STRU")
        shutil.copy2(self.data_dir / "INPUT_ref", "INPUT")

        outputs = abacus_modify_stru(".", 
                                    fix_atoms_idx=fix_atoms_idx,
                                    movable_coords=movable_coors)
        

        modified_stru = AbacusStru.ReadStru("STRU")
        modified_stru_move = modified_stru.get_move()
        
        for fix_idx, fix_atom_idx in enumerate(fix_atoms_idx):
            self.assertEqual(len(movable_coors[fix_idx]), len(modified_stru_move[fix_atom_idx]))
            for coord_idx in range(len(movable_coors[fix_idx])):
                self.assertEqual(movable_coors[fix_idx][coord_idx],
                                 modified_stru_move[fix_atom_idx][coord_idx])


    def test_abacus_modify_initial_magmoms_nspin2(self):
        """
        Test modify magnetic moment for every atom in STRU file in the nspin=2 case
        """
        initial_magmoms = [2.0, 2.0, 0.0, 0.0]
        shutil.copy2(self.data_dir / "STRU_NiO_ref", "STRU")
        shutil.copy2(self.data_dir / "INPUT_ref", "INPUT")

        outputs = abacus_modify_stru(".", 
                                    initial_magmoms=initial_magmoms)
        

        modified_stru = AbacusStru.ReadStru("STRU")
        modified_stru_initial_magmoms = modified_stru.get_atommag()

        for idx, value in enumerate(initial_magmoms):
            self.assertIsInstance(modified_stru_initial_magmoms[idx], float)
            self.assertAlmostEqual(value, modified_stru_initial_magmoms[idx])


    def test_abacus_modify_initial_magmoms_nspin4(self):
        """
        Test modify magnetic moment for every atom in STRU file in the nspin=4 case
        """
        initial_magmoms = [[2.0, 0.0, 0.0],
                           [2.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0]]
        shutil.copy2(self.data_dir / "STRU_NiO_ref", "STRU")
        shutil.copy2(self.data_dir / "INPUT_ref", "INPUT")

        outputs = abacus_modify_stru(".", 
                                    initial_magmoms=initial_magmoms)

        modified_stru = AbacusStru.ReadStru("STRU")
        modified_stru_initial_magmoms = modified_stru.get_atommag()

        for idx in range(len(initial_magmoms)):
            self.assertIsInstance(modified_stru_initial_magmoms[idx], list)
            self.assertEqual(len(initial_magmoms[idx]), len(modified_stru_initial_magmoms[idx]))
            for idx2 in range(len(initial_magmoms[idx])):
                self.assertAlmostEqual(initial_magmoms[idx][idx2],
                                       modified_stru_initial_magmoms[idx][idx2])


    def test_abacus_modify_initial_magmoms_nspin4_angle(self):
        """
        Test modify magnetic moment for every atom in STRU file in the nspin=4 case with angle
        """
        initial_magmoms = [2.0, 2.0, 0.0, 0.0]
        angle1 = [5.0, 10.0, 15.0, 20.0]
        angle2 = [20.0, 15.0, 10.0, 0.0]
        shutil.copy2(self.data_dir / "INPUT_ref", "INPUT")
        shutil.copy2(self.data_dir / "STRU_NiO_ref", "STRU")

        outputs = abacus_modify_stru(".", 
                                    initial_magmoms=initial_magmoms,
                                    angle1=angle1,
                                    angle2=angle2)
        

        modified_stru = AbacusStru.ReadStru("STRU")
        modified_stru_initial_magmoms = modified_stru.get_atommag()
        modified_stru_angle1 = modified_stru.get_angle1()
        modified_stru_angle2 = modified_stru.get_angle2()

        self.assertEqual(len(initial_magmoms), len(modified_stru_initial_magmoms))
        self.assertEqual(len(initial_magmoms), len(modified_stru_angle1))
        self.assertEqual(len(initial_magmoms), len(modified_stru_angle2))
        for idx in range(len(initial_magmoms)):
            magx, magy, magz = modified_stru_initial_magmoms[idx]
            mag = math.sqrt(magx**2 + magy**2 + magz**2)
            self.assertAlmostEqual(initial_magmoms[idx], mag)
            self.assertAlmostEqual(angle1[idx], modified_stru_angle1[idx])
            self.assertAlmostEqual(angle2[idx], modified_stru_angle2[idx])


class TestAbacusCollectData(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)  
        self.test_path = Path(self.test_dir.name)
        
        self.data_dir = Path(__file__).parent / "abacus/Si-sp"
        self.metrics_ref_json = self.data_dir / "metrics-ref.json"
        
        self.original_cwd = os.getcwd()
        os.chdir(self.test_path)
        
        print(f"Test directory: {self.test_path}")
            
    def tearDown(self):
        os.chdir(self.original_cwd) 
    
    def test_abacus_collect_data(self):
        """
        Test collect data from directory of abacus jobs
        """

        metrics = ["normal_end", "natom", "ibzk", "nelec", "nbands", "scf_steps", "energy_per_atom"]
        with open(self.metrics_ref_json, "r") as fin:
            data_ref = json.load(fin)
        
        outputs = abacus_collect_data(self.data_dir,
                                     metrics)
        collected_metrics = outputs['collected_metrics']
        
        for metric in metrics:
            if metric in ['nelec', 'energy_per_atom']:
                self.assertAlmostEqual(data_ref[metric], collected_metrics[metric])
            else:
                self.assertEqual(data_ref[metric], collected_metrics[metric])
        
        if os.path.exists(self.data_dir / "metrics.json"):
            os.remove(self.data_dir / "metrics.json")

