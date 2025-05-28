import unittest
import os, sys
from pathlib import Path
import tempfile

os.environ["ABACUSAGENT_MODEL"] = "test"  # Set the model to test
from abacusagent.modules.abacus import abacus_prepare

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
        
        outputs = abacus_prepare(self.stru_file1,
            stru_type = "abacus/stru",
            pp_path= self.pp_path,
            orb_path = self.orb_path,
            job_type= "scf",
            lcao= True
        )
        self.assertTrue(os.path.exists(outputs["job_path"]))
        self.assertEqual(outputs["job_path"], Path("000000").absolute())
        self.assertTrue(os.path.exists("000000/INPUT"))
        self.assertTrue(os.path.exists("000000/STRU"))
        self.assertTrue(os.path.exists("000000/As_ONCV_PBE-1.0.upf"))
        self.assertTrue(os.path.exists("000000/As_gga_8au_100Ry_2s2p1d.orb"))
        self.assertTrue(os.path.exists("000000/Ga_ONCV_PBE-1.0.upf"))
        self.assertTrue(os.path.exists("000000/Ga_gga_9au_100Ry_2s2p2d.orb"))

