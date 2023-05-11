'''

author: Linus Bjarne Dittmer

This class contains unittests to test the file

    libqdmmg/general/gaussian.py

'''

import unittest
import numpy
import scipy.integrate as si
import libqdmmg.general as gen
import libqdmmg.simulate as sim

class TestGaussian(unittest.TestCase):
  
    def setUp(self):
        self.g = self.genGaussian()

    def tearDown(self):
        del self.g

    def genGaussian(self, dim=2):
        s = sim.Simulation(2, 1.0, dim=dim)
        g = gen.Gaussian(s, centre=numpy.array((0.0, 0.0)))
        return g

    def test_evaluate(self):
        self.assertAlmostEqual(self.g.evaluate(numpy.zeros(2), 0), 1.0, delta=10**-9)

    def test_evaluate_d(self):
        self.assertTrue(numpy.allclose(self.g.evaluateD(numpy.zeros(2), 0), numpy.zeros(2), atol=10**-7))

    def test_evaluate_u(self):
        kernel = lambda x0, x1 : self.g.evaluateU(numpy.array((x0, x1)), 0) * (x0)**2 * self.g.evaluate(numpy.array((x0, x1)), 0)
        int1 = si.nquad(kernel, [[-4, 4], [-4, 4]])[0]
        self.assertAlmostEqual(int1, 1.0, delta=10**-7)

    def test_evaluate_v(self):
        kernel = lambda x0, x1 : self.g.evaluateV(numpy.array((x0, x1)), 0) * x0**2 * self.g.evaluate(numpy.array((x0, x1)), 0)
        int1 = si.nquad(kernel, [[-4, 4], [-4, 4]])[0]
        self.assertAlmostEqual(int1, 0.5, delta=10**-7)

if __name__ == '__main__':
    unittest.main()

