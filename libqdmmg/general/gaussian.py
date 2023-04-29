'''

Container for General Gaussian used for wavepacket propagation

'''

import numpy
import copy
from functools import reduce

class Gaussian:

    def __init__(self, sim, centre=None, width=None, momentum=None, phase=0.0):
        tsteps = sim.tsteps
        self.sim = sim
        self.centre = numpy.zeros((tsteps, sim.dim))
        self.d_centre = numpy.zeros(self.centre.shape)
        self.momentum = numpy.zeros((tsteps, sim.dim))
        self.d_momentum = numpy.zeros(self.momentum.shape)
        self.phase = numpy.zeros(tsteps)
        self.d_phase = numpy.zeros(self.phase.shape)
        self.width = numpy.ones(sim.dim)
        self.phase[0] = phase
        self.logger = sim.logger
        self.v_amp = -numpy.ones(tsteps)

        if type(centre) is numpy.ndarray:
            self.centre[0] = centre
        if type(width) is numpy.ndarray:
            self.width = width
        if type(momentum) is numpy.ndarray:
            self.momentum[0] = momentum


    def evaluate(self, x, t, is_index=True):
        # y = exp(-a(x-c)**2 + px + g)
        if not is_index:
            t = int(t * sim.tstep_val)
        xs = x - self.centre[t]
        ep = - reduce(numpy.dot, (self.width, xs*xs)) + 1j*(reduce(numpy.dot, (self.momentum[t], x)) + self.phase[t])
        return numpy.exp(ep)

    def evaluateD(self, x, t):
        # y' = y * (-2a(x-c) + p)
        y = self.evaluate(x, t, is_index)
        return y * reduce(numpy.add, (-2*reduce(numpy.multiply, (self.width, x-self.centre[t])), 1j*self.momentum[t]))

    def u_amplitude(self, t):
        return 2 * numpy.pi**(-self.sim.dim * 0.5) * numpy.linalg.norm(self.width)**(self.sim.dim * 0.5)

    def v_amplitde(self, t, index):
        if self.v_amp[t] < 0:
            w_prod = reduce(numpy.prod, self.width)
            ep = reduce(numpy.dot, (self.width, self.centre[t]*self.centre[t]))
            self.v_amp[t] = 2 * self.width[0] * numpy.pi(-self.sim.dim*0.5) * w_prod**0.5 * numpy.exp(2 * ep)
        return self.width[index] / self.width[0] * self.v_amp[t]


    def step_forward(self):
        if self.sim.t == 0:
            self.centre[1] = self.centre[0] + self.sim.tstep_val * self.d_centre[0]
            self.momentum[1] = self.momentum[0] + self.sim.tstep_val * self.d_momentum[0]
            self.phase[1] = self.phase[0] + self.sim.tstep_val * self.d_phase[0]
        else:
            self.centre[t+1] = self.centre[t-1] + 2 * sim.tstep_val * self.d_centre[t]
            self.momentum[t+1] = self.momentum[t-1] + 2 * sim.tstep_val * self.d_momentum[t]
            self.phase[t+1] = self.phase[t-1] + 2 * sim.tstep_val * self.d_phase[t]

    def copy(self):
        g = Gaussian(self.sim)
        g.centre = numpy.copy(self.centre)
        g.d_centre = numpy.copy(self.d_centre)
        g.momentum = numpy.copy(self.momentum)
        g.d_momentum = numpy.copy(self.d_momentum)
        g.phase = numpy.copy(self.phase)
        g.d_phase = numpy.copy(self.d_phase)
        g.v_amp = numpy.copy(self.v_amp)
        g.width = numpy.copy(self.width)
        return g
