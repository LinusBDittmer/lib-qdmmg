'''

author: Linus Bjarne Dittmer

'''

import libqdmmg.simulate.eom as eom
import numpy

class EOM_Master:

    def __init__(self, sim):
        self.sim = sim
        self.logger = sim.logger

    def prepare_next_step(self, t, isInitial=False):
        if isInitial:
            self.sim.active_gaussian.d_centre[t] = eom.eom_init_centre(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            self.sim.active_gaussian.d_momentum[t] = eom.eom_init_momentum(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            self.sim.active_gaussian.d_phase[t] = eom.eom_init_phase(self.sim, self.sim.potential, self.sim.active_gaussian, t)
        else:
            self.sim.active_gaussian.d_centre[t] = eom.eom_centre(self.sim.potential, self.sim.active_gaussian, t)
            self.sim.active_gaussian.d_momentum[t] = eom.eom_momentum(self.sim.potential, self.sim.active_gaussian, t)
            self.sim.active_gaussian.d_phase[t] = eom.eom_phase(self.sim.potential, self.sim.active_gaussian, t)

