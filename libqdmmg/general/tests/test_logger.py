'''

author: Linus Bjarne Dittmer

This class contains unittests to test the file

    libqdmmg/general/gaussian.py

'''

import unittest
import numpy
import libqdmmg.general as gen
import libqdmmg.simulate as sim

class TestLogger(unittest.TestCase):
   
    def setUp(self):
        self.l = self.genLogger()

    def tearDown(self):
        del self.l

    def genLogger(self, dim=2):
        s = sim.Simulation(1, 1.0, dim=dim)
        return s.logger

    def test_methods(self):
        self.assertTrue(hasattr(self.l, 'error'))
        self.assertTrue(hasattr(self.l, 'warn'))
        self.assertTrue(hasattr(self.l, 'info'))
        self.assertTrue(hasattr(self.l, 'debug1'))
        self.assertTrue(hasattr(self.l, 'debug2'))
        self.assertTrue(hasattr(self.l, 'debug3'))
        

if __name__ == '__main__':
    unittest.main()

