'''

author: Linus Bjarne Dittmer

This class contains unittests to test the file

    libqdmmg/general/wavepacket.py

'''

import unittest
import numpy
import libqdmmg.general as gen
import libqdmmg.simulate as sim
import libqdmmg.integrate as intor

class TestWavepacket(unittest.TestCase):
   
    def setUp(self):
        self.w = self.genWavepacket()

    def tearDown(self):
        del self.w
        del self.s

    def genWavepacket(self, dim=2):
        self.s = sim.Simulation(2, 1.0, dim=dim)
        s = self.s
        g1 = gen.Gaussian(s)
        g2 = gen.Gaussian(s)
        w = gen.Wavepacket(s)
        w.bind_gaussian(g1, numpy.ones(2))
        w.bind_gaussian(g2, 0.5*numpy.ones(2))
        return w

    def test_getCoeffs(self):
        coeffs = self.w.get_coeffs(0)
        norm2 = intor.int_request(self.s, 'int_ovlp_ww', self.w, self.w, 0)
        self.assertAlmostEqual(norm2, 1.0, delta=10**-7)

    def test_evaluate(self):
        v1 = self.w.evaluate(numpy.array([0.0, 0.0]), 0)
        v2 = 0.7978845607507745
        self.assertAlmostEqual(v1, v2, delta=10**-7)

if __name__ == '__main__':
    unittest.main()

