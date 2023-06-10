'''

author : Linus Bjarne Dittmer

'''

import numpy
import scipy.integrate
import libqdmmg.potential as pot
import libqdmmg.integrate as intor
import libqdmmg.general as gen

class Property:

    def __init__(self, sim, descriptor, shape=0, dtype=numpy.complex128):
        self.sim = sim
        self.logger = sim.logger
        self.descriptor = descriptor
        self.shape = shape
        if shape == 0:
            self.values = numpy.zeros(sim.tsteps, dtype=dtype)
        else:
            self.values = numpy.zeros(tuple([sim.tsteps]) + shape, dtype=dtype)

    def kernel(self, obj=None):
        if obj is None:
            obj = self.sim.previous_wavefunction
        for t in range(self.sim.tsteps):
            self.logger.info("Computing Property " + self.descriptor + " at timestep " + str(t))
            self.values[t] = self.compute(obj, t)

    def compute(self, obj, t):
        '''
        === TO BE OVERRIDDEN ===
        '''
        return 0

    def get(self):
        return self.values


    def wavepacketify(self, obj):
        if isinstance(obj, gen.gaussian.Gaussian):
            g = obj.copy()
            obj = gen.Wavepacket(self.sim)
            obj.bind_gaussian(g, numpy.ones(self.sim.tsteps))

        assert isinstance(obj, gen.wavepacket.Wavepacket), f"Only Gaussians and Wavepackets can be processed for kinetic energy. Received {type(obj)}"
        return obj

class PotentialEnergy(Property):

    def __init__(self, sim):
        super().__init__(sim, "Potential Energy", dtype=float)
        if sim.dim == 1:
            self.compute = self.compute1D
        else:
            self.compute = self.computeND

    def integrand1D(self, x, obj, t):
        wp_abs = abs(obj.evaluate(numpy.array((x)), t))
        return (wp_abs*wp_abs*self.sim.potential.evaluate(numpy.array((x)))).real

    def integrandND(self, *x):
        wp_abs = abs(self._obj.evaluate(x, self._t))
        return (wp_abs*wp_abs*self.sim.potential.evaluate(x)).real


    def compute1D(self, obj, t):
        obj = self.wavepacketify(obj)
        return obj.energy_pot(t)
        bounds = [-100, 100]
        return scipy.integrate.quad(self.integrand1D, bounds[0], bounds[1], args=(obj, t))[0]

    def computeND(self, obj, t):
        obj = self.wavepacketify(obj)
        return obj.energy_pot(t)
        bounds = [[-100, 100]] * self.sim.dim
        self._obj = obj
        self._t = t
        return scipy.integrate.nquad(self.integrandND, bounds)[0]


class KineticEnergy(Property):

    def __init__(self, sim):
        super().__init__(sim, "Kinetic Energy", dtype=float)

    def compute(self, obj, t):
        obj = self.wavepacketify(obj)
        return intor.int_request(self.sim, 'int_kinetic_ww', obj, obj, t).real

class TotalEnergy(Property):

    def __init__(self, sim, kinetic_energy, potential_energy):
        super().__init__(sim, "Total Energy", dtype=float)
        self.kinetic_energy = kinetic_energy
        self.potential_energy = potential_energy

    def compute(self, obj, t):
        return self.kinetic_energy.values[t] + self.potential_energy.values[t]

class AverageDisplacement(Property):

    def __init__(self, sim):
        super().__init__(sim, "Average Displacement", dtype=float, shape=(sim.dim,))

    def compute(self, obj, t):
        obj = self.wavepacketify(obj)
        coeffs = obj.get_coeffs(t)
        c = numpy.zeros(obj.gaussians[0].centre[t].shape)
        for i, co in enumerate(coeffs):
            c += co*co * obj.gaussians[i].centre[t]
        return numpy.array(c)

if __name__ == '__main__':
    import libqdmmg.simulate as sim
    import libqdmmg.export as exp
    s = sim.Simulation(20, 0.05, dim=1, verbose=3, generations=1)
    p = pot.HarmonicOscillator(s, numpy.ones(1))
    s.bind_potential(p)
    s.active_gaussian = gen.Gaussian(s, width=0.5*numpy.ones(1), centre=1.0*numpy.ones(1))
    s.gen_wavefunction()
    kin_energy = KineticEnergy(s)
    pot_energy = PotentialEnergy(s)
    tot_energy = TotalEnergy(s)
    displacement = AverageDisplacement(s)
    kin_energy.kernel()
    pot_energy.kernel()
    tot_energy.kernel()
    displacement.kernel()
    s.logger.info("Kinetic Energy:")
    s.logger.info(kin_energy.get())
    s.logger.info("Potential Energy:")
    s.logger.info(pot_energy.get())
    s.logger.info("Total Energy:")
    s.logger.info(tot_energy.get())
