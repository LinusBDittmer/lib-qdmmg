'''

@author Linus Bjarne Dittmer

'''


import numpy


class NumericalIntegrator:

    def __init__(self, sim):
        self.sim = sim
        self.grid = None
        self.function = self.one
        self.function2 = self.one
        self.function3 = self.one

    def bind_grid(self, grid):
        self.grid = grid

    def bind_function(self, func):
        self.function = func

    def bind_function2(self, func):
        self.function2 = func

    def bind_function3(self, func):
        self.function3 = func

    def bind_potential(self, potential):
        self.potential = potential

    def one(self, x, t):
        return 1.0

    def grid_eval(self, index, t):
        p = self.grid.gridpoint(index)
        return self.function(p, t) * self.function2(p, t) * self.function3(p, t) * self.potential.evaluate(p)

    def integrate(self, t):
        assert self.grid is not None

        index = numpy.zeros(self.sim.dim, dtype=numpy.int32)
        int_val = 0.0
        for indexnumber in range(self.grid.resolution**self.sim.dim):
            int_val += self.grid_eval(index, t) * self.grid.pointweight(index, t)

            index[0] += 1
            for i in range(self.sim.dim-1):
                if index[i] >= self.grid.resolution:
                    index[i] = 0
                    index[i+1] += 1
                else:
                    break

        return int_val


if __name__ == '__main__':
    import libqdmmg.simulate as sim
    import libqdmmg.integrate as intor
    import libqdmmg.general as gen
    s = sim.Simulation(1, 1, dim=3)
    g = intor.Grid(s, 10)
    gauss = gen.Gaussian(s)
    g.define_by_gaussian(gauss, 0)
    ni = NumericalIntegrator(s)
    ni.bind_grid(g)
    ni.bind_function(gauss.evaluate)
    print(ni.integrate(0))
    print(numpy.pi**(1.5))

