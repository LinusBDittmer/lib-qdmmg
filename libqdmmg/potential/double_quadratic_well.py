'''

@author Linus Bjarne Dittmer


'''

import numpy
from libqdmmg.potential.potential import Potential, PotentialIntegrator
from functools import reduce

class DoubleQuadraticWell(Potential):

    def __init__(self, sim, quartic=None, cubic=None, quadratic=None, linear=None, constant=None, shift=None):
        super().__init__(sim)
        if quartic is None and cubic is None and quadratic is None and linear is None and constant is None and shift is None:
            quartic = numpy.ones(sim.dim)
            cubic = numpy.zeros(sim.dim)
            quadratic = -numpy.ones(sim.dim)
            linear = 0.2 * numpy.ones(sim.dim)
            constant = 0.5 * numpy.ones(sim.dim)
            shift = 0.7 * numpy.ones(sim.dim)
        if quartic is None:
            quartic = numpy.zeros(sim.dim)
        if cubic is None:
            cubic = numpy.zeros(sim.dim)
        if quadratic is None:
            quadratic = numpy.zeros(sim.dim)
        if linear is None:
            linear = numpy.zeros(sim.dim)
        if constant is None:
            constant = numpy.zeros(sim.dim)
        if shift is None:
            shift = numpy.zeros(sim.dim)
        self.quartic = quartic
        self.cubic = cubic
        self.quadratic = quadratic
        self.linear = linear
        self.constant = constant
        self.shift = shift

    def evaluate(self, x):
        x = numpy.array(x) - self.shift
        return reduce(numpy.dot, (self.quartic, x*x*x*x)) + reduce(numpy.dot, (self.cubic, x*x*x)) + reduce(numpy.dot, (self.quadratic, x*x)) + reduce(numpy.dot, (self.linear, x)) + numpy.sum(self.constant)

    def gradient(self, x):
        x = numpy.array(x) - self.shift
        return self.num_gradient(x)
        return 4 * self.quartic * x*x*x + 3 * self.cubic * x*x + 2 * self.quadratic * x + self.linear

    def hessian(self, x):
        x = numpy.array(x) - self.shift
        return self.num_hessian(x)
        h = 12 * self.quartic * x*x + 6 * self.cubic * x + 2 * self.quadratic
        return numpy.diag(h)

    def gen_potential_integrator(self):
        return DoubleQuadraticWellIntegrator(self)

class DoubleQuadraticWellIntegrator(PotentialIntegrator):
    
    def __init__(self, dqw):
        super().__init__(dqw)
        assert isinstance(dqw, DoubleQuadraticWell), f"Only DoubleQuadraticWellIntegrator potential permitted, received {type(dqw)}"


        
