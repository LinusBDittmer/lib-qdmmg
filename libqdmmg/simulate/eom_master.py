'''

author: Linus Bjarne Dittmer

'''

import libqdmmg.simulate.eom as eom
import numpy

class EOM_Master:

    def __init__(self, sim):
        self.sim = sim
        self.logger = sim.logger
        self.logger.debug3("Initialised EOM_Master object at " + str(self))

    def prepare_next_step(self, t, isInitial=False):
        if isInitial:
            self.logger.info("Calculating EOM by initial set")
            self.logger.debug1("Calculating Centre")
            self.sim.active_gaussian.d_centre[t] = eom.eom_init_centre(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            self.logger.debug1("Calculating Momentum")
            self.sim.active_gaussian.d_momentum[t] = eom.eom_init_momentum(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            self.logger.debug1("Calculating Phase")
            self.sim.active_gaussian.d_phase[t] = eom.eom_init_phase(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            self.logger.info("Finished calculating EOM by initial set")
            #print(self.sim.active_gaussian.d_centre)
            #print(self.sim.active_gaussian.d_momentum)
            #print(self.sim.active_gaussian.d_phase)
        else:
            self.sim.active_gaussian.d_centre[t] = eom.eom_centre(self.sim.potential, self.sim.active_gaussian, t)
            self.sim.active_gaussian.d_momentum[t] = eom.eom_momentum(self.sim.potential, self.sim.active_gaussian, t)
            self.sim.active_gaussian.d_phase[t] = eom.eom_phase(self.sim.potential, self.sim.active_gaussian, t)


if __name__ == '__main__':
    import libqdmmg.simulate as sim
    import libqdmmg.potential as pot
    import libqdmmg.general as gen

    s = sim.Simulation(2, 1, verbose=4, dim=1)
    p = pot.HarmonicOscillator(s, numpy.ones(1))
    g = gen.Gaussian(s)
    s.bind_potential(p)
    s.active_gaussian = g
    s.eom_master.prepare_next_step(0, isInitial=True)
