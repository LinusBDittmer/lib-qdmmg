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
        self.g = self.genGaussian()

    def tearDown(self):
        del self.g

    def genGaussian(self, dim=2):
        s = sim.Simulation(2, 1.0, dim=dim)
        g = gen.Gaussian(s)
        return g

    def test_evaluate(self):
        self.assertAlmostEqual(self.g.evaluate(numpy.zeros(2), 0), 1.0, delta=10**-9)

    def test_evaluate_d(self):
        self.assertTrue(numpy.allclose(self.g.evaluateD(numpy.zeros(2), 0), numpy.zeros(2), atol=10**-7))

    def test_u_amplitude(self):
        u_amp = 2 * 2**0.5 / numpy.pi
        self.assertAlmostEqual(self.g.u_amplitude(0), u_amp, delta=10**-7)

    def test_v_amplitude(self):
        v_amp = 2.0 / numpy.pi
        self.assertAlmostEqual(self.g.v_amplitude(0, 0), v_amp, delta=10**-7)

    def test_evaluate_u(self):
        u_amp = 2 * 2**0.5 / numpy.pi
        self.assertAlmostEqual(float(self.g.evaluateU(numpy.zeros(2), 0)), u_amp, delta=10**-7)

    def test_evaluate_v(self):
        v_amp = 2.0 / numpy.pi
        self.assertTrue(numpy.allclose(self.g.evaluateV(numpy.zeros(2), 0), v_amp, atol=10**-7))

if __name__ == '__main__':
    unittest.main()

