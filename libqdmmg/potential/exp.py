import numpy
import scipy
import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot

s = sim.Simulation(2, 1.0, dim=2)
g1 = gen.Gaussian(s, centre=numpy.array((0.1, 0)))
ho = pot.HarmonicOscillator(s, forces=numpy.ones(2))
ho_intor = ho.gen_potential_integrator()
print("Analytical")
print(ho_intor._int_uVxg(g1, g1, 0, 0))
p0 = numpy.array((0.0, 0))
p = lambda x0, x1 : ho.evaluate(p0) + numpy.dot(numpy.array((x0, x1)), ho.gradient(p0)) + 0.5 * numpy.einsum('i,ij,j->', numpy.array((x0, x1)), ho.hessian(p0), numpy.array((x0, x1)))
l = lambda x0, x1 : g1.evaluateU(numpy.array((x0, x1)), 0) * g1.evaluate(numpy.array((x0, x1)), 0) * p(x0-p0[0], x1-p0[0])
print("Real:")
#print(scipy.integrate.nquad(l, [[-4, 4], [-4, 4]]))
print(scipy.integrate.nquad(lambda x0, x1 : x0 * g1.evaluateU(numpy.array((x0, x1)), 0) * ho.evaluate(numpy.array((x0, x1))) * g1.evaluate(numpy.array((x0, x1)), 0), [[-4, 4], [-4, 4]]))
