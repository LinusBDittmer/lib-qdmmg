'''

Container for General Gaussian used for wavepacket propagation

'''

import numpy
from functools import reduce

class Gaussian:

    def __init__(self, sim, centre=None, width=None, momentum=None, phase=0.0):
        tsteps = sim.tsteps
        self.centre = numpy.zeros((tsteps, sim.dim))
        self.momentum = numpy.zeros((tsteps, sim.dim))
        self.phase = numpy.zeros(tsteps, dtype=numpy.complex128)
        self.width = width
        self.phase[0] = phase
        self.logger = sim.logger

        if type(self.centre) is numpy.ndarray:
            self.centre[0] = centre
        if type(self.width) is not numpy.ndarray:
            self.width = numpy.ones(sim.dim, dtype=numpy.complex128)
        if type(self.momentum) is numpy.ndarray:
            self.momentum[0] = momentum


    def evaluate(self, x, t, is_index=True):
        # y = exp(-a(x-c)**2 + px + g)
        if not is_index:
            t = int(t * sim.tstep_val)
        xs = x - self.centre[t]
        ep = - reduce(numpy.dot, (self.width, xs*xs)) + reduce(numpy.dot, (self.momentum, x)) + self.phase
        return numpy.exp(ep)

    def evaluateD(self, x, t, is_index=True):
        # y' = y * (-2a(x-c) + p)
        y = self.evaluate(x, t, is_index)
        if not is_index:
            t = int(t* sim.tstep_val)
        return y * reduce(numpy.add, (-2*reduce(numpy.multiply, (self.width, x-self.centre)), self.momentum))

