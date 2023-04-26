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

