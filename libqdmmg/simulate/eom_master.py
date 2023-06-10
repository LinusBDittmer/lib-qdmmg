'''

author: Linus Bjarne Dittmer

'''

import libqdmmg.simulate.eom as eom
import numpy.linalg

class EOM_Master:

    def __init__(self, sim, stepping=-1, invert_stepping=False, qcutoff=0.1):
        self.sim = sim
        self.logger = sim.logger
        self.stepping = stepping
        self.invert_stepping = invert_stepping
        self.qcutoff = qcutoff
        self.logger.debug3("Initialised EOM_Master object at " + str(self))
        

    def prepare_next_step(self, t, isInitial=False):
        useInitial = isInitial
        if self.stepping > -1 and not isInitial:
            useInitial = ((t % self.stepping == 0) and not self.invert_stepping) or ((t % self.stepping != 0) and self.invert_stepping)
        if useInitial:
            self.logger.info("Calculating EOM by initial set")
            self.logger.debug1("Calculating Centre")
            self.sim.active_gaussian.d_centre[t] = eom.eom_init_centre(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            self.logger.debug1("Calculating Momentum")
            self.sim.active_gaussian.d_momentum[t] = eom.eom_init_momentum(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            self.logger.debug1("Calculating Phase")
            self.sim.active_gaussian.d_phase[t] = eom.eom_init_phase(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            self.logger.info("Finished calculating EOM by initial set")
        else:
            d_centre1, d_centre2 = eom.eom_centre(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            d_momentum1, d_momentum2 = eom.eom_momentum(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            d_phase1, d_phase2 = eom.eom_phase(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            
            dc1norm = numpy.linalg.norm(d_centre1, 1)
            dc2norm = numpy.linalg.norm(d_centre2, 1)
            dp1norm = numpy.linalg.norm(d_momentum1, 1)
            dp2norm = numpy.linalg.norm(d_momentum2, 1)
            
            d_centre = d_centre1 + d_centre2 * (dc2norm / dc1norm <= self.qcutoff)
            d_momentum = d_momentum1 + d_momentum1 * (dp2norm / dp1norm <= self.qcutoff)
            d_phase = d_phase1 * (abs(d_phase2) / abs(d_phase1) <= self.qcutoff)

            self.logger.debug3(f"Actual centre correction : {d_centre - d_centre1}")
            self.logger.debug3(f"Actual momentum correction : {d_momentum - d_momentum1}")
            self.logger.debug3(f"Actual phase correction : {d_phase - d_phase1}")

            self.sim.active_gaussian.d_centre[t] = d_centre
            self.sim.active_gaussian.d_momentum[t] = d_momentum
            self.sim.active_gaussian.d_phase[t] = d_phase

            d_coeff = eom.eom_coefficient(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            self.sim.d_active_coeffs[t] = d_coeff


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
