'''

@author Linus Bjarne Dittmer

'''

import numpy

class Grid:

    def __init__(self, sim, resolution):
        self.sim = sim
        self.resolution = resolution

    def define_by_gaussian(self, g1, t):
        centre = g1.centre[t]
        halfwidth = 3.0 * g1.width
        self.gridaxes = numpy.array([numpy.linspace(centre[i]-halfwidth[i], centre[i]+halfwidth[i], num=self.resolution) for i in range(self.sim.dim)])

    def define_by_two_gaussians(self, g1, g2, t):
        halfwidth = 3.0 * (g1.width + g2.width)
        centre = (g1.width * g1.centre + g2.width * g2.centre) / (g1.width + g2.width)
        self.gridaxes = numpy.array([numpy.linspace(centre[i]-halfwidth[i], centre[i]+halfwidth[i], num=self.resolution) for i in range(self.sim.dim)])

    def define_by_ug(self, g1, g2, t):
        self.define_by_two_gaussians(g1, g2, t)

    def define_by_vg(self, g1, g2, t):
        vg = g1.copy()
        vg.centre *= -1
        self.define_by_two_gaussians(vg, g2, t)

    def gridpoint(self, indices):
        return numpy.array([self.gridaxes[j,indices[j]] for j in range(self.sim.dim)])

    def pointweight(self, indices, t):
        return abs(numpy.prod(self.gridaxes[:,1]-self.gridaxes[:,0]))



if __name__ == '__main__':
    import libqdmmg.simulate as sim
    import libqdmmg.general as gen


    s = sim.Simulation(10, 0.1, dim=3)
    grid = Grid(s, 10)
    g = gen.Gaussian(s)

    grid.define_by_gaussian(g, 0)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                print(grid.gridpoint((i,j,k)))
