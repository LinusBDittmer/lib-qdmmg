'''

author: Linus Bjarne Dittmer

This class contains unittests to test the file

    libqdmmg/potential/potential.py

'''

import unittest
import numpy
import scipy.integrate as si
import libqdmmg.general as gen
import libqdmmg.simulate as sim
import libqdmmg.integrate as intor
import libqdmmg.potential as pot

class TestPotential(unittest.TestCase):
  
    def setUp(self):
        self.dim = 3
        self.sim = sim.Simulation(2, 1.0, dim=self.dim)
        self.pot = pot.HarmonicOscillator(self.sim, numpy.ones(self.dim))

    def tearDown(self):
        del self.sim
        del self.pot
        del self.dim

    def test_evaluate(self):
        p = self.pot.evaluate(numpy.ones(self.dim))
        self.assertAlmostEqual(p, 0.5 * self.dim, delta=10**-7)

    def test_gradient(self):
        p = self.pot.gradient(numpy.ones(self.dim))
        self.assertTrue(numpy.allclose(p, numpy.ones(self.dim)))

    def test_hessian(self):
        p = self.pot.hessian(numpy.ones(self.dim))
        self.assertTrue(numpy.allclose(p, numpy.diag(numpy.ones(self.dim))))


class TestPotentialIntegrator(unittest.TestCase):

    def setUp(self):
        self.dim = 3
        self.sim = sim.Simulation(2, 1.0, dim=self.dim)
        self.pot = pot.TrialPotential(self.sim, numpy.array((1, 1, 1)), numpy.array((1, 1, 1)))
        
        self.pot_intor = self.pot.gen_potential_integrator()
        self.g1 = gen.Gaussian(self.sim, centre=numpy.array((-0.5, 0.0, 1.0)))
        self.g2 = gen.Gaussian(self.sim, centre=numpy.array((-0.5, 0.0, 1.0))) #, momentum=numpy.array((1.0, 0.1, -0.1)))

    def tearDown(self):
        del self.dim
        del self.sim
        del self.pot
        del self.pot_intor
        del self.g1
        del self.g2

    def scipyIntegral(self, f):
        func = lambda x0, x1, x2 : self.g2.evaluate(numpy.array((x0, x1, x2)), 0) * self.pot.evaluate(numpy.array((x0, x1, x2))) * f(x0, x1, x2)
        bounds = [-4.0, 4.0]
        return si.nquad(func, [bounds, bounds, bounds])[0]

    def test_gVg(self):
        f = lambda x0, x1, x2 : self.g1.evaluate(numpy.array((x0, x1, x2)), 0)
        int1 = self.pot_intor._int_gVg(self.g1, self.g2, 0)
        int2 = self.scipyIntegral(f)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-4)

    def test_uVg(self):
        f = lambda x0, x1, x2 : self.g1.evaluateU(numpy.array((x0, x1, x2)), 0)
        int1 = self.pot_intor._int_uVg(self.g1, self.g2, 0)
        int2 = self.scipyIntegral(f)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-4)
    
    def test_uVxg(self):
        # index 1
        f = lambda x0, x1, x2 : self.g1.evaluateU(numpy.array((x0, x1, x2)), 0) * x1
        int1 = self.pot_intor._int_uVxg(self.g1, self.g2, 0, 1)
        int2 = self.scipyIntegral(f)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-4)

    def test_vVg(self):
        # vindex 0
        f = lambda x0, x1, x2 : self.g1.evaluateV(numpy.array((x0, x1, x2)), 0)
        int1 = self.pot_intor._int_vVg(self.g1, self.g2, 0, 0)
        int2 = self.scipyIntegral(f)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-4)

    def test_vVxg(self):
        # vindex 0, index 1
        f = lambda x0, x1, x2 : self.g1.evaluateV(numpy.array((x0, x1, x2)), 0) * x1
        int1 = self.pot_intor._int_vVxg(self.g1, self.g2, 0, 0, 1)
        int2 = self.scipyIntegral(f)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-4)

if __name__ == '__main__':
    unittest.main()

