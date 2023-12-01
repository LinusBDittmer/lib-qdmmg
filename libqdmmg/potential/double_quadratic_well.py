'''

@author Linus Bjarne Dittmer


'''

import numpy
from libqdmmg.potential.potential import Potential, PotentialIntegrator
from functools import reduce

class DoubleQuadraticWell(Potential):

    def __init__(self, sim, quartic=1.0, quadratic=0.0, shift=None, coupling=0.0):
        super().__init__(sim)
        if coupling is None:
            coupling = numpy.zeros(sim.dim-1)
        if shift is None:
            shift = numpy.zeros(sim.dim)
        self.quartic = quartic
        self.quadratic = quadratic
        self.coupling = coupling
        self.shift = shift

    def evaluate(self, x):
        x = numpy.array(x) - self.shift
        val = self.quartic * numpy.sum(x*x*x*x) - self.quadratic * numpy.sum(x*x)
        shiftx = numpy.roll(x, -1)
        shiftx[-1] = 0
        val += self.coupling * numpy.dot(x, shiftx)
        return val

    def gradient(self, x):
        x = numpy.array(x) - self.shift
        grad = 4 * self.quartic * x*x*x - 2 * self.quadratic * x
        xsp = numpy.roll(x, -1)
        xsm = numpy.roll(x, 1)
        xsp[-1] = 0
        xsm[0] = 0
        grad += self.coupling * (xsp + xsm)
        return grad

    def hessian(self, x):
        x = numpy.array(x) - self.shift
        dh = numpy.diag(12 * self.quartic * x*x - 2 * self.quadratic)
        dh += numpy.diag(self.coupling * numpy.ones(self.sim.dim-1), 1)
        dh += numpy.diag(self.coupling * numpy.ones(self.sim.dim-1), -1)
        return dh


    def gen_potential_integrator(self):
        return DoubleQuadraticWellIntegrator(self)

class DoubleQuadraticWellIntegrator(PotentialIntegrator):
    
    def __init__(self, dqw):
        super().__init__(dqw)
        assert isinstance(dqw, DoubleQuadraticWell), f"Only DoubleQuadraticWellIntegrator potential permitted, received {type(dqw)}"


        
