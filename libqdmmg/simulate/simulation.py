'''

Central class for Lib-QDMMG


'''

import libqdmmg.general as gen


class Simulation:

    def __init__(self, tsteps, tstep_val, verbose=0, dim=1):
        self.verbose = verbose
        self.dim = dim
        self.tsteps = tsteps
        self.tstep_val = tstep_val
        self.logger = gen.new_logger(self)
        self.t = 0

        self.previous_wavefunction = None
        self.active_gaussian = None

    def step_forward(self):
        if not isinstance(self.previous_wavefunction, gen.wavepacket.Wavepacket):
            self.step_forward_initial()
            return
        self.active_gaussian.step_forward()
        self.t += 1

    def step_forward_initial(self):
        self.active_gaussian.step_forward()
        self.t += 1


    def run_timesteps(self, isInitial=False):
        # Main timestepping loop for adding gaussians
        pass

    def gen_wavefunction(self):
        # Main executable function for generation of the final wavefunction

        # Generate first gaussian
        self.active_gaussian = gen.Gaussian(self)
        # Timestepping
        self.run_timesteps(isInitial=True)
        # Binding first gaussian to wavepackt
        self.previous_wavefunction = gen.Wavepacket(self)
        self.previous_wavefunction.bindGaussian(self.active_gaussian.copy())

        for generation in range(generations):
            # Generate next gaussian
            self.active_gaussian = gen.Gaussian(self)
            # Timestepping
            self.run_timesteps()
            # Binding new gaussian
            self.previous_wavefunction.bindGaussian(self.active_gaussian.copy())
            # Check convergence
        
