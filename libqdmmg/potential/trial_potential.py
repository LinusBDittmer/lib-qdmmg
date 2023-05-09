'''

@author Linus Bjarne Dittmer


'''

import numpy
from libqdmmg.potential.potential import Potential, PotentialIntegrator
from functools import reduce

class TrialPotential(Potential):

    def __init__(self, sim, grad, forces):
        super().__init__(sim)
        assert len(forces) == sim.dim, f"Number of received force constants incorrect, received {len(forces)}, expected {sim.dim}"
        self.forces = forces
        self.grad = grad

    def evaluate(self, x):
        return 0.5 * reduce(numpy.dot, (self.forces, x*x)) + reduce(numpy.dot, (self.grad, x))

    def gen_potential_integrator(self):
        return TrialPotentialIntegrator(self)

    def gradient(self, x):
        return self.forces * x + self.grad

    def hessian(self, x):
        return numpy.diag(self.forces)

class TrialPotentialIntegrator(PotentialIntegrator):
    
    def __init__(self, tp):
        super().__init__(tp)
        assert isinstance(tp, TrialPotential), f"Inadmissible potential of type {type(ho)}"


        
