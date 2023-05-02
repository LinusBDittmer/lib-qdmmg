'''

author: Linus Bjarne Dittmer

This class contains unittests to test parts of the file

    libqdmmg/integrate/anal_integrator.py

For the sake of structure and partial testing, the integral tests are split into their leading type (g-, u- or v-), each being handled by the respective class TestGIntegrals, TestUIntegrals and TestVIntegrals.

'''

import unittest
import numpy
import scipy.integrate as si
import libqdmmg.general as gen
import libqdmmg.simulate as sim
import libqdmmg.integrate.anal_integrator as intor

class TestGIntegrals(unittest.TestCase):
  
    def setUp(self):
        self.s = sim.Simulation(2, 1.0, dim=3)
        self.g1 = self.genGaussian(centre=[0.0, 0.5, 0.0])
        self.g2 = self.genGaussian(centre=[0.5, 0.5, 0.1])

    def tearDown(self):
        del self.g1
        del self.g2
        del self.s

    def genGaussian(self, centre=None):
        g = gen.Gaussian(self.s, centre=centre)
        return g

    def scipyIntegral(self, f):
        func = lambda x0, x1, x2 : f(x0, x1, x2) * self.g1.evaluate(numpy.array([x0,x1,x2]), 0) * self.g2.evaluate(numpy.array([x0,x1,x2]), 0)
        bounds = [-4.0, 4.0]
        return si.nquad(func, [bounds, bounds, bounds])[0]

    def test_gg(self):
        int2 = intor.int_gg(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0])
        int1 = self.scipyIntegral(lambda x0, x1, x2 : 1)
        self.assertAlmostEqual(int1, int2, delta=10**-3)

    def test_gxg(self):
        # index 0
        int2 = intor.int_gxg(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x0)
        self.assertAlmostEqual(int1, int2, delta=10**-3)
    
    def test_gx2g_m(self):
        # indices 1, 2
        int2 = intor.int_gx2g_mixed(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 1, 2)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x1 * x2)
        self.assertAlmostEqual(int1, int2, delta=10**-3)
    
    def test_gx2g_d(self):
        # index 0
        int2 = intor.int_gx2g_diag(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x0 * x0)
        self.assertAlmostEqual(int1, int2, delta=10**-3)

    def test_gx3g_m(self):
        # indinces 0, 1
        int2 = intor.int_gx3g_mixed(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0, 1)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x0 * x1 * x1)
        self.assertAlmostEqual(int1, int2, delta=10**-3)

    def test_gx3g_d(self):
        # index 1
        int2 = intor.int_gx3g_diag(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 1)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x1 * x1 * x1)
        self.assertAlmostEqual(int1, int2, delta=10**-3)

class TestUIntegrals(unittest.TestCase):

    def setUp(self):
        self.s = sim.Simulation(2, 1.0, dim=3)
        self.g1 = self.genGaussian(centre=[0.0, 0.5, 0.0])
        self.g2 = self.genGaussian(centre=[0.5, 0.5, 0.1])

    def tearDown(self):
        del self.g1
        del self.g2
        del self.s

    def genGaussian(self, centre=None):
        g = gen.Gaussian(self.s, centre=centre)
        return g

    def scipyIntegral(self, f):
        func = lambda x0, x1, x2 : f(x0, x1, x2) * self.g1.evaluateU(numpy.array([x0,x1,x2]), 0) * self.g2.evaluate(numpy.array([x0,x1,x2]), 0)
        bounds = [-10.0, 10.0]
        return si.nquad(func, [bounds, bounds, bounds])[0]

    def test_ug(self):
        int2 = intor.int_ug(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0])
        int1 = self.scipyIntegral(lambda x0, x1, x2 : 1)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

    def test_uxg(self):
        # index 1
        int2 = intor.int_uxg(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 1)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x1)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

    def test_ux2g_m(self):
        # indices 0, 2
        int2 = intor.int_ux2g_mixed(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0, 2)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x0 * x2)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

    def test_ux2g_d(self):
        # index 0
        int2 = intor.int_ux2g_diag(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x0 * x0)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

    def test_ux3g_m(self):
        # indinces 0, 2
        int2 = intor.int_ux3g_mixed(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0, 2)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x0 * x2 * x2)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

    def test_ux3g_d(self):
        # index 1
        int2 = intor.int_ux3g_diag(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 1)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x1 * x1 * x1)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

class TestVIntegrals(unittest.TestCase):

    def setUp(self):
        self.s = sim.Simulation(2, 1.0, dim=3)
        self.g1 = self.genGaussian(centre=[0.0, 0.3, 0.0])
        self.g2 = self.genGaussian(centre=[0.5, 0.5, 0.1], momentum=[1.0, 0.0, 0.0])

    def tearDown(self):
        del self.g1
        del self.g2
        del self.s

    def genGaussian(self, centre=None, momentum=None):
        g = gen.Gaussian(self.s, centre=centre, momentum=momentum)
        return g

    def scipyIntegral(self, f):
        func = lambda x0, x1, x2 : f(x0, x1, x2) * self.g1.evaluateV(numpy.array([x0,x1,x2]), 0) * self.g2.evaluate(numpy.array([x0,x1,x2]), 0)
        bounds = [-10.0, 10.0]
        return si.nquad(func, [bounds, bounds, bounds])[0]

    def test_vg(self):
        int2 = intor.int_vg(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : 1)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

    def test_vxg(self):
        # index 0
        int2 = intor.int_vxg(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0, 0)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x0)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

    def test_vx2g_m(self):
        # indices 0, 1
        int2 = intor.int_vx2g_mixed(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0, 0, 1)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x0 * x1)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

    def test_vx2g_d(self):
        # index 1
        int2 = intor.int_vx2g_diag(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0, 1)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x1 * x1)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

    def test_vx3g_m(self):
        # indinces 0, 1
        int2 = intor.int_vx3g_mixed(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0, 0, 1)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x0 * x1 * x1)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

    def test_vx3g_d(self):
        # index 1
        int2 = intor.int_vx3g_diag(self.g1.width, self.g2.width, self.g1.centre[0], self.g2.centre[0], self.g1.momentum[0], self.g2.momentum[0], self.g1.phase[0], self.g2.phase[0], 0, 1)
        int1 = self.scipyIntegral(lambda x0, x1, x2 : x1 * x1 * x1)
        self.assertAlmostEqual(int1.real, int2.real, delta=10**-3)

if __name__ == '__main__':
    unittest.main()

