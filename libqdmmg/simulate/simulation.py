'''

Central class for Lib-QDMMG


'''

import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.simulate as sim
import libqdmmg.integrate as intor

import numpy

class Simulation:

    def __init__(self, tsteps, tstep_val, verbose=0, dim=1, generations=0, qcutoff=0.1, micro_steps=4):
        self.verbose = verbose
        self.dim = dim
        self.tsteps = tsteps
        self.tstep_val = tstep_val
        self.generations = generations
        self.logger = gen.new_logger(self)
        self.bounding_box = 1000

        self.previous_wavefunction = None
        self.active_coeffs = None
        self.d_active_coeffs = None
        self.active_gaussian = None
        self.potential = None
        self.eom_master = sim.EOM_Master(self, qcutoff=qcutoff)
        self.eom_intor = sim.EOM_Matexp(self, order=1, auxorder=3, micro_steps=micro_steps)
        self.logger.debug3("Initialised Simulation object at " + str(self))

    def bind_potential(self, potential, adapt_strategy=True, strategy_kwargs=None):
        self.logger.debug2("Binding Potential...")
        assert isinstance(potential, pot.potential.Potential), f"Expected Object of type or inheriting libqdmmg.potential.potential.Potential, received {type(potential)}"
        self.potential = potential
        if isinstance(self.potential, pot.mol_potential.MolecularPotential) or not adapt_strategy:
            # TODO add molecular strategy
            self.gaussian_strategy = sim.GaussianSingleRandomStrategy(self)
        else:
            if strategy_kwargs is None:
                strategy_kwargs = {}
            strategy_kwargs.setdefault("extent", numpy.ones(self.dim) * 10.0)
            strategy_kwargs.setdefault("res", numpy.ones(self.dim))
            strategy_kwargs.setdefault("shift", numpy.zeros(self.dim))
            self.gaussian_strategy = sim.GaussianGridStrategy(self, strategy_kwargs["extent"], strategy_kwargs["res"], strategy_kwargs["shift"])
            self.logger.debug2("Changed Gaussian Strategy to Gaussian Grid Strategy")
        self.logger.info("Bound potential to simulation, type: " + str(type(potential)) + ", content: " + str(potential))

    def step_forward(self, t, isInitial=False):
        if isInitial:
            self.step_forward_initial(t)
        else:
            self.active_gaussian.step_forward(t)
            cnew = self.eom_intor.next_coefficient(self.active_coeffs, self.d_active_coeffs, t)
            # Smoothmax clampling to avoid blowout
            #b = 50.0
            # Get overlap for maximum influence coeff
            a = 1.0 / numpy.sqrt(intor.int_request(self, 'int_gg', self.active_gaussian, self.active_gaussian, 0))
            # Hardmax clamping
            cnew = max(-a, min(cnew, a))
            #cnew = numpy.log(numpy.exp(b*cnew) + numpy.exp(-b*a)) / b
            #cnew = -numpy.log(numpy.exp(-b*cnew) + numpy.exp(-b*a)) / b
            self.active_coeffs[t+1] = cnew
            if (abs(self.active_gaussian.centre[t]) > self.bounding_box).any():
                self.logger.warn("Gaussian left relevant Bounding box and is discarded")
                return True
        return False

    def step_forward_initial(self, t):
        self.active_gaussian.step_forward(t)

    def run_timesteps(self, generation=0, isInitial=False):
        self.logger.info("Beginning timestepping sequence.\n")
        self.logger.info("Total number of timesteps:     " + str(self.tsteps))
        self.logger.info("Timestep interval:             " + str(self.tstep_val) + " (" + str(round(self.tstep_val * 0.02418843265857, 5)) + " fs)\n\n")
        # Main timestepping loop for adding gaussians
        for t in range(self.tsteps):
            self.logger.important("Iteration " + str(t) + " | Generation " + str(generation))
            self.logger.info("t = " + str(round(t * self.tstep_val * 0.02418843265857, 5)) + " fs")
            if t < self.tsteps-1:
                self.eom_master.prepare_next_step(t, isInitial)
                abortion = self.step_forward(t, isInitial)
                if abortion:
                    return False
                if not isInitial:
                    self.logger.info("")
                    self.logger.info("Influence coefficient:")
                    self.logger.info(f"          {self.active_coeffs[t]}  -->  {self.active_coeffs[t+1]}")
                    '''
                    if abs(self.active_coeffs[t+1]) < 10**-7:
                        for t1 in range(t, self.tsteps-1):
                            self.active_coeffs[t1] = 0
                            self.active_gaussian.step_forward(t1)
                            self.logger.info("Aborting Gaussian propagation because it provides no value.")
                            self.logger.info("\n\n")
                        break
                    '''
            self.logger.info("\n\n")
        return True

    def random_array(self, lower, upper, symmetric=False, normal=False):
        r = (numpy.random.rand(self.dim) * (upper-lower)) + lower
        if normal:
            r = max(numpy.random.normal(lower, upper), 0.0)
        if not symmetric:
            return r
        return numpy.random.choice(numpy.array([-1, 1]), self.dim) * r

    def generate_new_gaussian(self, generation):
        if generation == -1:
            widthsquare = 0.25 * numpy.diag(self.potential.hessian(numpy.zeros(self.dim))) * self.potential.reduced_mass
            for w in range(len(widthsquare)):
                if widthsquare[w] < 0.01:
                    widthsquare[w] = 0.1
            width = numpy.sqrt(widthsquare)
            self.logger.debug3(f"Width of initial Gaussian :{width}")
            return gen.Gaussian(self, centre=numpy.zeros(self.dim), width=width)
        
        centre = self.gaussian_strategy.new_position(generation)
        width = self.gaussian_strategy.new_width(generation)
        gaussian = gen.Gaussian(self, centre=centre, width=width)
        #coeff = 1.0 / (len(self.previous_wavefunction.gaussians)+1)
        #coeff = min(max(numpy.exp(-gaussian.energy_tot(0)*100), 0.01), 0.99)
        coeff = 10**-9
        return gaussian, coeff

    def gen_wavefunction(self):
        # Main executable function for generation of the final wavefunction

        # Generate first gaussian
        if not isinstance(self.active_gaussian, gen.gaussian.Gaussian):
            #self.active_gaussian = gen.Gaussian(self, centre=self.random_array(-1.0, 1.0), width=self.random_array(0.1, 1.0))
            self.active_gaussian = self.generate_new_gaussian(-1)
        # Timestepping
        self.run_timesteps(isInitial=True)
        # Binding first gaussian to wavepackt
        self.previous_wavefunction = gen.Wavepacket(self)
        self.previous_wavefunction.bind_gaussian(self.active_gaussian.copy(), numpy.ones(self.tsteps))
        generation = 0
        while generation < self.generations:
            # Generate next gaussian
            self.active_coeffs = numpy.zeros(self.tsteps)
            self.active_gaussian, self.active_coeffs[0] = self.generate_new_gaussian(generation) 
            self.d_active_coeffs = numpy.zeros(self.tsteps)
            # Timestepping
            accept_gaussian = self.run_timesteps(generation=generation+1)
            # Binding new gaussian
            if accept_gaussian:
                ac = numpy.copy(self.active_coeffs)
                self.previous_wavefunction.bind_gaussian(self.active_gaussian.copy(), ac)
                generation += 1
            # Check convergence
            # quality = self.previous_wavefunction.propagation_quality()
            # self.logger.info(f"Propagation Fitness: \n\n{-numpy.log(quality)}")
        
        self.redo_coefficients()

    def redo_coefficients(self):
        self.previous_wavefunction.reset_coeffs(dtype=numpy.complex128)
        for t in range(self.tsteps-1):
            ncoeffs = self.eom_intor.renew_coefficients(self.previous_wavefunction.gaussians, self.previous_wavefunction.get_coeffs(t), t)
            self.previous_wavefunction.gauss_coeff[:,t+1] = ncoeffs
            norm = intor.int_request(self, 'int_ovlp_ww', self.previous_wavefunction, self.previous_wavefunction, t+1)
            self.logger.info(f"Wavepacket Norm before Normalisation: {norm}")
            self.previous_wavefunction.gauss_coeff[:,t+1] /= numpy.sqrt(norm)
            norm = abs(intor.int_request(self, 'int_ovlp_ww', self.previous_wavefunction, self.previous_wavefunction, t+1))
            self.logger.info(f"Wavepacket Norm after Normalisation: {norm}")
            self.logger.info(f"Updated Coefficients: {self.previous_wavefunction.get_coeffs(t)}")

    def get_wavefunction(self):
        if self.previous_wavefunction is None:
            raise gen.SNRException(self)
        return self.previous_wavefunction

if __name__ == '__main__':
    import numpy
    import libqdmmg.plotting as plot
    import libqdmmg.export as exp

    s = sim.Simulation(500, 0.005, dim=1, verbose=5, generations=15)
    p = pot.HarmonicOscillator(s, numpy.ones(1))
    s.bind_potential(p)
    s.gen_wavefunction()
    exp.export_to_json(s.get_wavefunction(), 'trial.json')


