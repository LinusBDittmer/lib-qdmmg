from libqdmmg import simulate as sim
from libqdmmg import general as g
from libqdmmg import integrate as intor
import numpy

s = sim.Simulation(1, 1, verbose=5)
s.dim = 1
g1 = g.Gaussian(s, centre=numpy.array((1)))
g2 = g.Gaussian(s, centre=numpy.array((0.5)))
wp = g.Wavepacket(s, g1)
wp.bindGaussian(g2, 0.3+0.1j)
s.logger.info(wp.gauss_coeff)
s.logger.info(numpy.linalg.norm(wp.gauss_coeff))


