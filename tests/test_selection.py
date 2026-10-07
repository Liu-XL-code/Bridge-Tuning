import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from bridge_tuning.selection import assess, coverage


class SelectionTests(unittest.TestCase):
    def test_directional_nearest_support_and_sample_coverage(self):
        c=np.array([[0.,0.],[2.,0.]])
        b=np.array([[0.,0.],[1.,0.],[10.,0.]])
        self.assertAlmostEqual(assess(c,b)["D_target_to_bridge"],.5)
        self.assertAlmostEqual(coverage(c),2.)
        self.assertNotEqual(assess(c,b)["D_target_to_bridge"],assess(b,c)["D_target_to_bridge"])
        self.assertAlmostEqual(assess(c,b)["delta_W"],coverage(b)-coverage(c))
        with self.assertRaises(ValueError):coverage(np.ones((1,2)))
