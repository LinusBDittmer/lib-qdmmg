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
