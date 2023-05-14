'''

Central class for Lib-QDMMG


'''

import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.simulate as sim

import numpy

class Simulation:

    def __init__(self, tsteps, tstep_val, verbose=0, dim=1, generations=0):
        self.verbose = verbose
        self.dim = dim
        self.tsteps = tsteps
        self.tstep_val = tstep_val
        self.logger = gen.new_logger(self)

        self.previous_wavefunction = None
        self.active_coeffs = None
        self.d_active_coeffs = None
        self.active_gaussian = None
        self.potential = None
        self.eom_master = sim.EOM_Master(self)
        self.generations = generations
        self.logger.debug3("Initialised Simulation object at " + str(self))

    def bind_potential(self, potential):
        self.logger.debug2("Binding Potential...")
        assert isinstance(potential, pot.potential.Potential), f"Expected Object of type or inheriting libqdmmg.potential.potential.Potential, received {type(potential)}"
        self.potential = potential
        self.logger.info("Bound potential to simulation, type: " + str(type(potential)) + ", content: " + str(potential))


    def step_forward(self, t, isInitial=False):
        if isInitial:
            self.step_forward_initial(t)
        else:
            self.active_gaussian.step_forward(t)
            if t == 0:
                self.active_coeffs[1] = self.active_coeffs[0] + self.tstep_val * self.d_active_coeffs[0]
            else:
                cnew = self.active_coeffs[t-1] + 2 * self.tstep_val * self.d_active_coeffs[t]
                # Smoothmax clampling to avoid blowout
                b = 50.0
                a = 0.999
                cnew = numpy.log(numpy.exp(b*cnew) + numpy.exp(-b*a)) / b
                cnew = -numpy.log(numpy.exp(-b*cnew) + numpy.exp(-b*a)) / b
                self.active_coeffs[t+1] = cnew

    def step_forward_initial(self, t):
        self.active_gaussian.step_forward(t)

    def run_timesteps(self, generation=0, isInitial=False):
        self.logger.info("Beginning timestepping sequence.\n")
        self.logger.info("Total number of timesteps:     " + str(self.tsteps))
        self.logger.info("Timestep interval:             " + str(self.tstep_val) + " (" + str(round(self.tstep_val * 0.02418843265857, 5)) + " fs)\n\n")
        # Main timestepping loop for adding gaussians
        for t in range(self.tsteps):
            self.logger.info("Iteration " + str(t) + " | Generation " + str(generation))
            self.logger.info("t = " + str(round(t * self.tstep_val * 0.02418843265857, 5)) + " fs")
            if t < self.tsteps-1:
                self.eom_master.prepare_next_step(t, isInitial)
                self.step_forward(t, isInitial)
                if not isInitial:
                    self.logger.info("")
                    self.logger.info("Influence coefficient:")
                    self.logger.info(f"          {self.active_coeffs[t]}  -->  {self.active_coeffs[t+1]}")
            self.logger.info("\n\n")

    def random_array(self, lower, upper):
        return (numpy.random.rand(self.dim) * (upper-lower)) + lower

    def gen_wavefunction(self):
        # Main executable function for generation of the final wavefunction

        # Generate first gaussian
        if not isinstance(self.active_gaussian, gen.gaussian.Gaussian):
            self.active_gaussian = gen.Gaussian(self, centre=self.random_array(-1.0, 1.0), width=self.random_array(0.1, 1.0))
        # Timestepping
        self.run_timesteps(isInitial=True)
        # Binding first gaussian to wavepackt
        self.previous_wavefunction = gen.Wavepacket(self)
        self.previous_wavefunction.bind_gaussian(self.active_gaussian.copy(), numpy.ones(self.tsteps))

        for generation in range(self.generations):
            # Generate next gaussian
            self.active_gaussian = gen.Gaussian(self, centre=self.random_array(-1.0, 1.0), width=self.random_array(0.1, 1.0))
            self.active_coeffs = 0.5*numpy.ones(self.tsteps)
            self.d_active_coeffs = numpy.zeros(self.tsteps)
            # Timestepping
            self.run_timesteps(generation=generation+1)
            # Binding new gaussian
            self.previous_wavefunction.bind_gaussian(self.active_gaussian.copy(), numpy.copy(self.active_coeffs))
            # Check convergence

    def get_wavefunction(self):
        if self.previous_wavefunction is None:
            raise gen.SNRException(self)
        return self.previous_wavefunction

if __name__ == '__main__':
    import numpy

    s = sim.Simulation(100, 0.01, dim=1, verbose=5, generations=2)
    p = pot.HarmonicOscillator(s, 10*numpy.ones(1))
    s.bind_potential(p)
    s.gen_wavefunction()
    print(s.active_coeffs)

