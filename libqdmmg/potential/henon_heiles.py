'''

@author Linus Bjarne Dittmer


'''

import numpy
from libqdmmg.potential.potential import Potential, PotentialIntegrator
from functools import reduce

class HenonHeiles(Potential):

    def __init__(self, sim, omega, l):
        super().__init__(sim)
        self.reduced_mass = numpy.ones(sim.dim) / omega
        self.l = l

    def evaluate(self, x):
        x = numpy.array(x)
        harmonic_terms = 0.5 * reduce(numpy.dot, (1/self.reduced_mass, x*x))
        anharmonic_terms = 0.0
        for i in range(len(x)-1):
            anharmonic_terms += x[i]*x[i]*x[i+1] - x[i+1]*x[i+1]*x[i+1] / 3.0
        anharmonic_terms *= self.l
        return harmonic_terms + anharmonic_terms

    def gen_potential_integrator(self):
        return HenonHeilesIntegrator(self)

class HenonHeilesIntegrator(PotentialIntegrator):
    
    def __init__(self, hh):
        super().__init__(hh)
        assert isinstance(hh, HenonHeiles), f"Only Henon-Heiles potential permitted, received {type(hh)}"


        
