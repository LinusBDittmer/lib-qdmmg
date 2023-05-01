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
   
    def genGaussian(self, dim=2):
        s = sim.Simulation(2, 1.0, dim=dim)
        g = gen.Gaussian(s)
        return g

    def test_evaluate(self):
        g = self.genGaussian()
        self.assertAlmostEqual(g.evaluate(numpy.zeros(2), 0), 1.0, delta=10**-9)

    def test_evaluate_d(self):
        g = self.genGaussian()
        self.assertAlmostEqual(g.evaluateD(numpy.zeros(2), 0), numpy.zeros(2), delta=10**-9)

    def test_u_amplitude(self):
        g = self.genGaussian()
        u_amp = 2 * 2**0.5 / numpy.pi
        self.assertAlmostEqual(g.u_amplitude(0), u_amp, delta=10**-7)

    def test_v_amplitude(self):
        g = self.genGaussian()
        v_amp = 2.0 / numpy.pi
        self.assertAlmostEqual(g.v_amplitude(0, 0), v_amp, delta=10**-7)

    def test_evaluate_u(self):
        g = self.genGaussian()
        u_amp = 2 * 2**0.5 / numpy.pi
        self.assertAlmostEqual(g.evaluateU(numpy.zeros(2), 0), u_amp, delta=10**-7)

    def test_evaluate_v(self):
        g = self.genGaussian()
        v_amp = 2.0 / numpy.pi
        self.assertAlmostEqual(g.evaluateV(numpy.zeros(2), 0, 0), v_amp, delta=10**-7)

if __name__ == '__main__':
    unittest.main()

