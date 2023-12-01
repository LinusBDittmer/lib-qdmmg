import libqdmmg.potential as pot
import libqdmmg.simulate as sim
import matplotlib.pyplot as plt
import numpy

s = sim.Simulation(2, 10.0, dim=2)
dqw = pot.DoubleQuadraticWell(s, quartic=1.2484*10**-4, quadratic=8.9191*10**-4, shift=numpy.array([1.78, 1.78]))

m1, m2 = -5.0, 5.0
res = 150
xspace = numpy.linspace(m1, m2, num=res)
potval = numpy.zeros((res, res))

for i in range(res):
    for j in range(res):
        #potval[i,j] = dqw.evaluate((xspace[i], xspace[j]))
        #potval[i,j] = numpy.linalg.norm(dqw.gradient((xspace[i], xspace[j])))
        potval[i,j] = numpy.trace(dqw.hessian((xspace[i], xspace[j])))

potval -= numpy.min(potval)
potval /= numpy.max(potval)
potval = numpy.power(potval, 0.3)

plt.figure(figsize=(12, 12))
plt.imshow(potval, extent=(m1, m2, m1, m2))
plt.savefig("potential_render_hess.png", bbox_inches='tight', dpi=200)

