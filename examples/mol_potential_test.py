import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot
import libqdmmg.properties as prop

s = sim.Simulation(1000, 0.5, dim=3, verbose=9, generations=3)
geometry, charge, mult = pot.from_xyz('./benchmarking/HOF/geometry/base.xyz')
p = pot.MolecularPotential(s, geometry, charge=charge, multiplicity=mult, rounding=5, basis='6-31g', theory='rhf')
s.bind_potential(p)

num=100
index=1
space = numpy.linspace(-0.05, 0.05, num=num)
plt.figure()
fig, ax = plt.subplots(figsize=(16, 9))
ppoints = numpy.zeros(num)
gradients = numpy.zeros(num)
hessians = numpy.zeros(num)
for i in range(num):
    point = numpy.zeros(3)
    point[index] = space[i]
    ppoints[i] = p.evaluate(point)
    grad = p.gradient(point)
    hess = p.hessian(point)
    gradients[i] = grad[index]
    hessians[i] = hess[index,index]
    print(hess)


ax.plot(space, ppoints)
for i in range(0, num, 20):
    ax.plot(space, 0.5*hessians[i]*(space-space[i])*(space-space[i])+gradients[i]*(space-space[i])+ppoints[i], color='green')
ax.set_ylim(-0.05, 0.5)
plt.savefig("molpottest_" + str(index) + ".png", bbox_inches='tight', dpi=200)
plt.close()
