'''

author: Linus Bjarne Dittmer

This class contains unittests to test the file

    libqdmmg/general/gaussian.py

'''

import unittest
import numpy
import libqdmmg.general as gen
import libqdmmg.simulate as sim
import libqdmmg.potential as pot

class TestEOMInit(unittest.TestCase):
  
    def setUp(self):
        self.simulation = sim.Simulation(2, 1.0, dim=1)
        self.potential = pot.HarmonicOscillator(self.simulation, numpy.ones(1))
        self.simulation.bind_potential(self.potential)
        self.simulation.active_gaussian = gen.Gaussian(self.simulation, width=numpy.array(0.5))

    def tearDown(self):
        del self.simulation
        del self.potential

    def test_simulation(self):
        self.simulation.gen_wavefunction()
        self.assertAlmostEqual(self.simulation.active_gaussian.centre[1], 0.0, delta=10**-9)

if __name__ == '__main__':
    unittest.main()

