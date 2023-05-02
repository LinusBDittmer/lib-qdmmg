'''

author: Linus Bjarne Dittmer

This class contains unittests to test the file

    libqdmmg/general/wavepacket.py

'''

import unittest
import numpy
import libqdmmg.general as gen
import libqdmmg.simulate as sim

class TestWavepacket(unittest.TestCase):
   
    def setUp(self):
        self.w = self.genWavepacket()

    def genWavepacket(self, dim=2):
        s = sim.Simulation(1, 1.0, dim=dim)
        g1 = gen.Gaussian(s)
        g2 = gen.Gaussian(s, centre=numpy.array([0.0, 0.5]))
        w = gen.Wavepacket(s)
        w.bindGaussian(g1, numpy.array([1.0]))
        w.bindGaussian(g2, numpy.array([1.0]))
        return w

    def test_getCoeffs(self):
        coeffs = self.w.getCoeffs(0)
        norm2 = numpy.linalg.norm(coeffs)**2
        self.assertAlmostEqual(norm2, 1.0, delta=10**-7)

    def test_evaluate(self):
        v1 = self.w.evaluate(numpy.array([0.0, 0.0]), 0)
        v2 = 2**(-0.5) * (1 + numpy.exp(-0.5**2))
        self.assertAlmostEqual(v1, v2, delta=10**-7)

if __name__ == '__main__':
    unittest.main()

