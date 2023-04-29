'''

@author Linus Bjarne Dittmer

'''


import numpy


class NumericalIntegrator:

    def __init__(self, sim):
        self.sim = sim
        self.grid = None
        self.function = None
        self.function2 = None

    def bindGrid(self, grid):
        self.grid = grid

    def bindFunction(self, func):
        self.function = func

    def bindFunction2(self, func):
        self.function2 = func

    def silenceFunction2(self):
        self.function2 = self.one

    def one(self, x, t):
        return 1.0

    def gridEval(self, index, t):
        # Note: index must have shape (1, dim)
        p = self.grid.gridpoint(index)
        return self.function(p, t) * self.function2(p, t)

    def integrate(self, t):
        assert self.grid is not None
        #assert self.function is not None
        if self.function2 is None:
            self.silenceFunction2()

        index = numpy.zeros(self.sim.dim, dtype=numpy.int32)
        int_val = 0.0
        for indexnumber in range(self.grid.resolution**self.sim.dim):
            int_val += self.gridEval(index, t) * self.grid.pointweight(index, t)

            index[0] += 1
            for i in range(self.sim.dim-1):
                if index[i] >= self.grid.resolution:
                    index[i] = 0
                    index[i+1] += 1
                else:
                    break

        return int_val

class NumericalIntegrationCallable:

    def __init__(self, sim):
        self.sim = sim

    def call(self, x, t):
        return 1


if __name__ == '__main__':
    import libqdmmg.simulate as sim
    import libqdmmg.integrate as intor
    import libqdmmg.general as gen
    s = sim.Simulation(1, 1, dim=3)
    g = intor.Grid(s, 10)
    gauss = gen.Gaussian(s)
    g.define_by_gaussian(gauss, 0)
    ni = NumericalIntegrator(s)
    ni.bindGrid(g)
    ni.bindFunction(gauss.evaluate)
    print(ni.integrate(0))
    print(numpy.pi**(1.5))

