'''

@author Linus Bjarne Dittmer


'''

import numpy
from libqdmmg.potential.potential import Potential, PotentialIntegrator
from functools import reduce

class CoupledHarmonicOscillator(Potential):

    def __init__(self, sim, forces, coupling):
        super().__init__(sim)
        assert len(forces) == sim.dim, f"Number of received force constants incorrect, received {len(forces)}, expected {sim.dim}"
        self.forces = forces
        self.coupling = coupling

    def evaluate(self, x):
        x = numpy.array(x)
        return 0.5 * (reduce(numpy.dot, (self.forces, x*x)) + self.coupling * reduce(numpy.dot, (x[:self.sim.dim-1], numpy.roll(x, 1)[1:])))

    def gen_potential_integrator(self):
        return CoupledHarmonicOscillatorIntegrator(self)

class CoupledHarmonicOscillatorIntegrator(PotentialIntegrator):
    
    def __init__(self, cho):
        super().__init__(cho)
        assert isinstance(cho, CoupledHarmonicOscillator), f"Only Coupled Harmonic Oscillator potential permitted, received {type(ho)}"


        
