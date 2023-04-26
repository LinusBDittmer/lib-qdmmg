'''

Class for a general Wavepacket. Used to describe prior wavefunctions

'''

import numpy

class Wavepacket:

    def __init__(self, sim, g1):
        self.sim = sim
        self.gaussians = []
        self.gauss_coeff = numpy.array([])
        self.logger = sim.logger

    def bindGaussian(self, g1, coeff):
        self.gaussians.append(g1)
        if abs(coeff.real - coeff) > 10**-3:
            coeff = abs(coeff)
            self.logger.warn("Significant complex phase in coefficient detected. Phase should be exported to basis and coeffs kept real. Phase is discarded.")
        self.gauss_coeff = numpy.append(self.gauss_coeff, coeff.real)
        self.gauss_coeff /= numpy.linalg.norm(self.gauss_coeff)



