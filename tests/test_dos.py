import unittest
import os, sys, glob
from pathlib import Path
os.environ["ABACUSAGENT_MODEL"] = "test"

from abacusagent.modules.dos import plot_dos_pdos as mkplots 

class TestPlotDos(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).parent / "plot_dos"


    def tearDown(self):
        for pngfile in glob.glob(os.path.join(self.data_dir, "*.png")):
            os.remove(pngfile)
    
    def test_run_dos(self):
        """
        Test the run_dos function with a valid input.
        """

        # ignore the screen output
        sys.stdout = open(os.devnull, 'w')

        # Call the run_dos function
        results_test = mkplots(self.data_dir,self.data_dir, dpi=20)
        results_ref = [Path(self.data_dir) / "DOS.png",
                       Path(self.data_dir) / "Si1_pdos.png",
                       Path(self.data_dir) / "Si2_pdos.png"]
    
        self.assertListEqual([Path(p) for p in results_test], results_ref)
        
        

        

        



