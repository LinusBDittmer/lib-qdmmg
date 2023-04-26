from libqdmmg import simulate as sim
from libqdmmg import general as g
from libqdmmg import integrate as intor
import numpy

s = sim.Simulation(1, 1, verbose=5)
s.dim = 1
g1 = g.Gaussian(s, centre=numpy.array((1)))
g2 = g.Gaussian(s, centre=numpy.array((0.5)))
gg = intor.int_request(sim, 'int_gg', g1, g2, 0, 0)
gxg = intor.int_request(sim, 'int_gxg', g1, g2, 0, 0, useM=False)
s.logger.info(gg)
s.logger.info(gxg)


