'''

File for Gaussian Strategy. This file determines how new Gaussians are placed

'''

import numpy
import libqdmmg.general as gen
import libqdmmg.integrate as intor

class GaussianStrategy:

    def __init__(self, sim):
        self.sim = sim
        self.logger = sim.logger

    def new_position(self, generation):
        return numpy.zeros(self.sim.dim)

    def new_width(self, generation):
        return numpy.zeros(self.sim.dim)

    def random_array(self, lower, upper, symmetric=False, normal=False):
        r = (numpy.random.rand(self.sim.dim) * (upper-lower)) + lower
        if normal:
            r = max(numpy.random.normal(lower, upper), 0.0)
        if not symmetric:
            return r
        return numpy.random.choice(numpy.array([-1, 1]), self.sim.dim) * r


class GaussianGridStrategy(GaussianStrategy):

    def __init__(self, sim, gridsize, gridres, gridshift, prio_size=10):
        super().__init__(sim)
        self.gridsize = numpy.array(gridsize, dtype=float)
        self.gridshift = numpy.array(gridshift, dtype=float)
        self.gridres = numpy.array(gridres, dtype=int)
        self.prio_size = sim.generations+1
        self.cellsize = 2 * gridsize / gridres
        self.widths = numpy.zeros((self.sim.generations, self.sim.dim))
        self.centres = numpy.zeros((self.sim.generations, self.sim.dim))
        self.generated = [False] * self.sim.generations
        self.logger.debug2("Initialising Grid")
        self.init_grid()
        self.logger.debug2("Finished Initialising Grid")

    def init_grid(self):
        energies = numpy.zeros(numpy.prod(self.gridres))
        for i in range(len(energies)):
            p = self.gridpoint(i)
            if numpy.allclose(p, numpy.zeros(p.shape)):
                energies[i] = 10**15
            else:
                energies[i] = self.sim.potential.evaluate(p)
        if self.prio_size > 0:
            self.prio_points = numpy.argsort(energies)[:self.prio_size]
        else:
            self.prio_points = numpy.array([])

    def generate(self, generation):
        if generation < self.prio_size:
            self.centres[generation] = self.gridpoint(self.prio_points[generation])
            hess = self.sim.potential.hessian(self.centres[generation])
            widthsquare = 0.25 * numpy.diag(hess) * self.sim.potential.reduced_mass
            for w in range(len(widthsquare)):
                if widthsquare[w] < 0.01:
                    widthsquare[w] = 0.1
            self.widths[generation] = numpy.sqrt(widthsquare)
        else:
            #TODO Boltzmann selection
            pass
 

    def new_width(self, generation):
        if not self.generated[generation]:
            self.generate(generation)
        return self.widths[generation]

    def new_position(self, generation):
        if not self.generated[generation]:
            self.generate(generation)
        return self.centres[generation]

    def gridpoint(self, i):
        index = []
        for j in range(self.sim.dim-1, 0, -1):
            axis_size = self.gridres[j]
            axis_index = i % axis_size
            index.insert(0, axis_index)
            i = i // axis_size
        index.insert(0, i)
        index = numpy.array(index)

        #for j in range(self.sim.dim):
        #    index[j] = int((i % self.gridres**(j+1)) / self.gridres**j)
        return index * self.cellsize - 0.5 * self.gridsize + self.gridshift


class GaussianSingleRandomStrategy(GaussianStrategy):

    def __init__(self, sim, centre_mu=1.0, centre_sigma=0.3, width_mu=1.0, width_sigma=0.3, init_full=True, init_scale=2.5):
        super().__init__(sim)
        self.centre_mu = centre_mu
        self.centre_sigma = centre_sigma
        self.width_mu = width_mu
        self.width_sigma = width_sigma
        self.centres = numpy.zeros((sim.generations, sim.dim))
        #self.generated = [False] * sim.generations
        if init_full:
            self.generate_gaussians(init_scale)

    def generate_gaussians(self, init_scale=2.5):
        hess = self.sim.potential.hessian(numpy.zeros(self.sim.dim))
        init_width = numpy.sqrt(0.25 * numpy.diag(hess) * self.sim.potential.reduced_mass)
        init_sigma = 1 / numpy.sqrt(0.5 * init_width)
        init_pos = numpy.random.normal(loc=0.0, scale=init_scale*init_sigma, size=(self.sim.generations, self.sim.dim))
        self.centres = self.relax_init_positions(init_pos)
        self.generated = [True] * self.sim.generations

    def relax_init_positions(self, init_pos, optimum=1.5):
        self.logger.info("Finding suitable starting positions...")
        def penalty(x):
            distp = 0.0
            if len(x) == 1:
                return (numpy.linalg.norm(x[0]) - optimum)**2
            for t1 in range(len(x)-1):
                for t2 in range(1, len(x)):
                    distp += (numpy.linalg.norm(x[t1]-x[t2]) - optimum)**2
            return distp
        
        def penaltygrad(x, epsilon=10**-5):
            grad = numpy.zeros(x.shape)
            for t1 in range(len(x)):
                for t2 in range(len(x[0])):
                    xp, xm = numpy.copy(x), numpy.copy(x)
                    xp[t1,t2] += epsilon
                    xm[t1,t2] -= epsilon
                    grad[t1,t2] = 0.5 * (penalty(xp) - penalty(xm)) / epsilon
            return grad

        for i in range(500):
            grad = penaltygrad(init_pos)
            init_pos -= 0.01 * grad
            if numpy.linalg.norm(grad) < 0.1:
                break
        self.logger.info("Found suitable starting positions.")
        return init_pos

    def new_position(self, generation):
        if self.generated[generation]:
            return self.centres[generation]
        centre = self.random_array(self.centre_mu, self.centre_sigma, symmetric=True, normal=True) / numpy.sqrt(self.sim.previous_wavefunction.gaussians[0].width)
        self.generated[generation] = True
        for i in range(10000):
            ghyp = gen.Gaussian(self.sim, centre=centre, width=self.new_width(generation))
            works = True
            for gaussian in self.sim.previous_wavefunction.gaussians:
                if abs(intor.int_request(self.sim, 'int_ovlp_gg', ghyp, gaussian, 0)) > 0.4**2:
                    centre = self.random_array(self.centre_mu, self.centre_sigma, symmetric=True, normal=True) / numpy.sqrt(self.sim.previous_wavefunction.gaussians[0].width)
                    works = False
                    break
            if works:
                break

        return centre

    def new_width(self, generation):
        if self.generated[generation]:
            hess = self.sim.potential.hessian(numpy.zeros(self.sim.dim))
            widthsquare = 0.25 * numpy.diag(hess) * self.sim.potential.reduced_mass
            width = numpy.sqrt(widthsquare)
            return width
        wfactor = max(0.1, self.random_array(self.width_mu, self.width_sigma, normal=True))
        width = numpy.sqrt(self.sim.previous_wavefunction.gaussians[-1].width) * wfactor
        return width
 



