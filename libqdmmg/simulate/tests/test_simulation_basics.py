'''

author: Linus Bjarne Dittmer

This class contains unittests to test the file

    libqdmmg/general/gaussian.py

'''

import unittest
import numpy
import libqdmmg.general as gen
import libqdmmg.simulate as sim

class TestGaussian(unittest.TestCase):
  
    def setUp(self):
        self.s = sim.Simulation(1, 1.0, dim=2)

    def tearDown(self):
        del self.s

    def test_methods(self):
        self.assertTrue(hasattr(self.s, 'bind_potential'))

    def test_variables(self):
        self.assertTrue(hasattr(self.s, 'dim'))

        self.assertTrue(hasattr(self.s, 'tsteps'))
        self.assertTrue(hasattr(self.s, 'tstep_val'))
        
        self.assertTrue(hasattr(self.s, 'logger'))
        self.assertTrue(hasattr(self.s, 'verbose'))
        
        self.assertTrue(hasattr(self.s, 'potential'))
        self.assertTrue(hasattr(self.s, 'active_gaussian'))
        self.assertTrue(hasattr(self.s, 'previous_wavefunction'))
        self.assertTrue(hasattr(self.s, 'generations'))

if __name__ == '__main__':
    unittest.main()

