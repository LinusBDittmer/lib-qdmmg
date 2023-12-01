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
       
    def _clean(self, x):
        if type(x) is numpy.ndarray:
            if numpy.isnan(x).any():
                return numpy.zeros(x.shape)
            if numpy.isinf(x).any():
                return numpy.zeros(x.shape)
        else:
            if numpy.isnan(x):
                return 0.0
            if numpy.isinf(x):
                return 0.0
        return x

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
            self.logger.debug1(f"Centre Derivative : {self.sim.active_gaussian.d_centre[t]}")
            self.logger.debug1(f"Momentum Derivative : {self.sim.active_gaussian.d_momentum[t]}")
            self.logger.debug1(f"Phase Derivative : {self.sim.active_gaussian.d_phase[t]}")
        else:
            d_centre1, d_centre2 = eom.eom_centre(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            d_momentum1, d_momentum2 = eom.eom_momentum(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            d_phase1, d_phase2 = eom.eom_phase(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            
            dc1norm = numpy.linalg.norm(d_centre1, 1) + 10**-9
            dc2norm = numpy.linalg.norm(d_centre2, 1)
            dp1norm = numpy.linalg.norm(d_momentum1, 1) + 10**-9
            dp2norm = numpy.linalg.norm(d_momentum2, 1)
            
            d_centre2 = d_centre2 * (dc2norm / dc1norm <= self.qcutoff)
            d_momentum2 = d_momentum2 * (dp2norm / dp1norm <= self.qcutoff)
            d_phase2 = d_phase2 * (abs(d_phase2) / abs(d_phase1+10**-9) <= self.qcutoff)

            self.logger.debug3(f"Classical Centre Derivative : {d_centre1}")
            self.logger.debug3(f"Classical Momentum Derivative : {d_momentum1}")
            self.logger.debug3(f"Classical Phase Derivative : {d_phase1}")
            self.logger.debug3(f"Actual centre correction : {d_centre2}")
            self.logger.debug3(f"Actual momentum correction : {d_momentum2}")
            self.logger.debug3(f"Actual phase correction : {d_phase2}")

            self.sim.active_gaussian.d_centre[t] = self._clean(d_centre1)
            self.sim.active_gaussian.d_momentum[t] = self._clean(d_momentum1)
            self.sim.active_gaussian.d_phase[t] = self._clean(d_phase1)
            self.sim.active_gaussian.d_centre_v[t] = self._clean(d_centre2)
            self.sim.active_gaussian.d_momentum_v[t] = self._clean(d_momentum2)
            self.sim.active_gaussian.d_phase_v[t] = self._clean(d_phase2)

            d_coeff = eom.eom_coefficient(self.sim, self.sim.potential, self.sim.active_gaussian, t)
            self.sim.d_active_coeffs[t] = self._clean(d_coeff)


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
