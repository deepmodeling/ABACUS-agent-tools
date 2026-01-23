import unittest
import os, sys, glob, shutil
from pathlib import Path

os.environ["ABACUSAGENT_MODEL"] = "test"

from abacusagent.modules.submodules.dos import plot_write_dos_pdos as mkplots


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
        sys.stdout = open(os.devnull, "w")

        # Call the run_dos function
        results_figs, results_datas = mkplots(self.data_dir, self.data_dir, "species", dos_emin_ev=-1, dos_emax_ev=1)
        
        output_dir = Path(glob.glob("*plot_write_dos_pdos*")[0]).absolute()
        results_figs_ref = [
            output_dir / "DOS.png",
            output_dir / "PDOS.png",
        ]
        results_datas_ref = [
            output_dir / "DOS.dat",
            output_dir / "PDOS.dat",
        ]

        self.assertListEqual([Path(p) for p in results_figs], results_figs_ref)
        self.assertListEqual([Path(p) for p in results_datas], results_datas_ref)

        if os.path.exists(self.data_dir / "metrics.json"):
            os.remove(self.data_dir / "metrics.json")
        
        for dir in glob.glob("*plot_write_dos_pdos*"):
            shutil.rmtree(dir)
