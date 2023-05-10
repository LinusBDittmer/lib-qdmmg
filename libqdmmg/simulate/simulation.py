'''

Central class for Lib-QDMMG


'''

import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.simulate as sim

class Simulation:

    def __init__(self, tsteps, tstep_val, verbose=0, dim=1):
        self.verbose = verbose
        self.dim = dim
        self.tsteps = tsteps
        self.tstep_val = tstep_val
        self.logger = gen.new_logger(self)
        #self.t = 0

        self.previous_wavefunction = None
        self.active_gaussian = None
        self.potential = None
        self.eom_master = sim.EOM_Master(self)
        self.generations = 1

    def bind_potential(self, potential):
        assert isinstance(potential, pot.potential.Potential), f"Expected Object of type or inheriting libqdmmg.potential.potential.Potential, received {type(potential)}"
        self.potential = potential


    def step_forward(self, t, isInitial=False):
        if isInitial:
            self.step_forward_initial(t)
        else:
            self.active_gaussian.step_forward()
            #self.previous_wavefunction.step_forward()
        #self.t += 1

    def step_forward_initial(self, t):
        self.active_gaussian.step_forward(t)
        #self.t += 1
        

    def run_timesteps(self, isInitial=False):
        # Main timestepping loop for adding gaussians
        for t in range(self.tsteps):
            self.eom_master.prepare_next_step(t, isInitial)
            if t < self.tsteps-1:
                self.step_forward(t, isInitial)


    def gen_wavefunction(self):
        # Main executable function for generation of the final wavefunction

        # Generate first gaussian
        if not isinstance(self.active_gaussian, gen.gaussian.Gaussian):
            self.active_gaussian = gen.Gaussian(self)
        # Timestepping
        self.run_timesteps(isInitial=True)
        '''
        # Binding first gaussian to wavepackt
        self.previous_wavefunction = gen.Wavepacket(self)
        self.previous_wavefunction.bindGaussian(self.active_gaussian.copy())

        for generation in range(self.generations):
            # Generate next gaussian
            self.active_gaussian = gen.Gaussian(self)
            # Timestepping
            self.run_timesteps()
            # Binding new gaussian
            self.previous_wavefunction.bindGaussian(self.active_gaussian.copy())
            # Check convergence
        '''
