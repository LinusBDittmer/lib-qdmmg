'''

@author Linus Bjarne Dittmer


'''

import numpy
from libqdmmg.potential.potential import Potential, PotentialIntegrator
from functools import reduce

class HarmonicOscillator(Potential):

    def __init__(self, sim, forces):
        super().__init__(sim)
        assert len(forces) == sim.dim, f"Number of received force constants incorrect, received {len(forces)}, expected {sim.dim}"
        self.forces = forces

    def evaluate(self, x):
        return 0.5 * reduce(numpy.dot, (self.forces, x*x))

    def gradient(self, x):
        return self.forces * x

    def hessian(self, x):
        return numpy.diag(self.forces)

    def gen_potential_integrator(self):
        return HarmonicOscillatorIntegrator(self)

class HarmonicOscillatorIntegrator(PotentialIntegrator):
    
    def __init__(self, ho):
        super().__init__(ho)
        assert isinstance(ho, HarmonicOscillator), f"Only Harmonic Oscillator potential permitted, received {type(ho)}"


        
