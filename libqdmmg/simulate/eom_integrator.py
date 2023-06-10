'''

Integration Handler for EOM

'''

import numpy
import scipy
import libqdmmg.integrate as intor

class EOM_Integrator:

    def __init__(self, sim):
        self.sim = sim
        self.stepsize = sim.tstep_val

    def next_centre(self, centre, d_centre, t):
        return None

    def next_momentum(self, momentum, d_momentum, t):
        return None

    def next_phase(self, phase, d_phase, t):
        return 0

    def next_coefficient(self, coeff, d_coeff, t):
        return 0


class EOM_EulerIntegrator(EOM_Integrator):

    def __init__(self, sim):
        super().__init__(sim)

    def next_centre(self, centre, d_centre, t):
        return centre[t] + self.stepsize * d_centre[t]

    def next_momentum(self, momentum, d_momentum, t):
        return momentum[t] + self.stepsize * d_momentum[t]

    def next_phase(self, phase, d_phase, t):
        return phase[t] + self.stepsize * d_phase[t]

    def next_coefficient(self, coeff, d_coeff, t):
        return coeff[t] + self.stepsize * d_coeff[t]


class EOM_AdamsBashforth(EOM_Integrator):

    def __init__(self, sim, order=10):
        super().__init__(sim)
        self.order = order
        self.coefficients = [None]*order
        self.calculate_coeffs()

    def calculate_coeffs(self):
        def abpol(x, j, s):
            val = 1.0
            for i in range(s+1):
                if i == j: continue
                val *= x+i
            return val

        for s in range(self.order):
            self.coefficients[s] = numpy.zeros(s+1)
            for j in range(s+1):
                self.coefficients[s][j] = scipy.integrate.quad(abpol, 0, 1, args=(j,s))[0]
                self.coefficients[s][j] *= (-1)**j / (scipy.special.factorial(j) * scipy.special.factorial(s-j))

    def get_effective_coeffs(self, t):
        return self.coefficients[min(self.order-1, t)]

    def next_centre(self, centre, d_centre, t):
        eff_coeff = self.get_effective_coeffs(t)
        relevant_centres = d_centre[max(0, t-self.order+1):t+1]
        unadapted_shift = numpy.einsum('j,jn->n', eff_coeff, relevant_centres)
        return centre[t] + self.stepsize * unadapted_shift

    def next_momentum(self, momentum, d_momentum, t):
        eff_coeff = self.get_effective_coeffs(t)
        relevant_momenta = d_momentum[max(0, t-self.order+1):t+1]
        unadapted_shift = numpy.einsum('j,jn->n', eff_coeff, relevant_momenta)
        return momentum[t] + self.stepsize * unadapted_shift

    def next_phase(self, phase, d_phase, t):
        eff_coeff = self.get_effective_coeffs(t)
        relevant_phases = d_phase[max(0, t-self.order+1):t+1]
        unadapted_shift = numpy.dot(eff_coeff, relevant_phases)
        return phase[t] + self.stepsize * unadapted_shift

    def next_coefficient(self, coeff, d_coeff, t):
        eff_coeff = self.get_effective_coeffs(t)
        relevant_coeffs = d_coeff[max(0, t-self.order+1):t+1]
        unadapted_shift = self.stepsize * numpy.dot(eff_coeff, relevant_coeffs)
        adapted_shift = self.get_energy_conservation_factor(coeff[t], unadapted_shift, t) * unadapted_shift
        return coeff[t] + adapted_shift

    def get_energy_conservation_factor(self, coeff, shift, t, scale=10000):
        return 1.0
        if abs(shift / self.sim.tstep_val) < 10**-5:
            return 1.0
        g1 = self.sim.active_gaussian
        wp = self.sim.previous_wavefunction
        pot_intor = self.sim.potential.gen_potential_integrator()
        eg0 = g1.energy_tot(t)
        eg1 = g1.energy_tot(t+1)
        ewp0 = wp.energy_tot(t)
        ewp1 = wp.energy_tot(t+1)
        w0 = intor.int_request(self.sim, 'int_kinetic_gw', g1, wp, t)
        w1 = intor.int_request(self.sim, 'int_kinetic_gw', g1, wp, t+1)
        s0 = intor.int_request(self.sim, 'int_ovlp_gg', g1, g1, t)
        s1 = intor.int_request(self.sim, 'int_ovlp_gg', g1, g1, t+1)
        st0 = intor.int_request(self.sim, 'int_ovlp_gw', g1, wp, t)
        st1 = intor.int_request(self.sim, 'int_ovlp_gw', g1, wp, t+1)
        
        def obj(x):
            a0 = coeff
            a1 = coeff + shift*(x+1)
            b0 = numpy.sqrt(a0*a0*(st0*st0-s0)+1) - a0*st0
            b1 = numpy.sqrt(a1*a1*(st1*st1-s1)+1) - a1*st1
            return scale*(a0*a0*eg0 + 2*a0*b0*w0 + b0*b0*ewp0 - a1*a1*eg1 - 2*a1*b1*w1 - b1*b1*ewp1) / self.sim.tstep_val

        def jacobj(x):
            a1 = coeff + shift*(x+1)
            b1 = numpy.sqrt(a1*a1*(st1*st1-s1)+1) - a1*st1
            da1 = shift
            db1 = shift * (st1*st1-s1) * a1 / numpy.sqrt(a1*a1*(st1*st1-s1)) - shift*st1
            return -2*scale*(a1*da1*eg1 + w1*(da1*b1+db1*a1) + ewp1*b1*db1) / self.sim.tstep_val

        root_obj = scipy.optimize.root_scalar(obj, x0=0.0, fprime=jacobj, method='newton', rtol=10**-7)
        self.sim.logger.debug3(f"Energy Conservation remainder : {obj(root_obj.root)}")
        if root_obj.converged:
            if root_obj.root > -2 and root_obj.root < 2 and not numpy.isnan(root_obj.root):
                return root_obj.root.real+1
        return 1.0

class EOM_Pade(EOM_Integrator):

    def __init__(self, sim, m_order=4, n_order=2):
        super().__init__(sim)
        self.m_order = m_order
        self.n_order = n_order
        self.eom_early = EOM_AdamsBashforth(sim, order=m_order+n_order)

    def gen_taylor_coeffs(self, coeff0, d_array):
        tcoeffs = numpy.zeros(self.m_order + self.n_order)
        tcoeffs[0] = coeff0
        tcoeffs[1] = d_array[-1]
        #tcoeffs[2] = 0.5 * (d_array[-2] - d_array[-1]) / h
        #tcoeffs[3] = 1 / 3! * (d_array[-3] - 2 * d_array[-2] + d_array[-1]) / h**2
        for i in range(2, len(tcoeffs)):
            for j in range(i):
                tcoeffs[i] += (-1)**j * scipy.special.binom(i-1,j) * d_array[-(j+1)]
            tcoeffs[j] /= (self.sim.tstep_val**i * scipy.special.factorial(i))
        return tcoeffs

    def gen_pade_prediction(self, tcoeffs):
        p, q = scipy.interpolate.pade(tcoeffs, self.m_order)
        return p(self.sim.tstep_val) / q(self.sim.tstep_val)

    def next_centre(self, centre, d_centre, t):
        to = self.m_order + self.n_order
        if t <= to:
            return self.eom_early.next_centre(centre, d_centre, t)
        nc = numpy.zeros(centre[t].shape)
        for i in range(len(nc)):
            tcoeffs = self.gen_taylor_coeffs(centre[t,i], d_centre[t-to:t+1,i])
            nc[i] = self.gen_pade_prediction(tcoeffs)
        return nc

    def next_momentum(self, momentum, d_momentum, t):
        to = self.m_order + self.n_order
        if t <= to:
            return self.eom_early.next_momentum(momentum, d_momentum, t)
        np = numpy.zeros(momentum[t].shape)
        for i in range(len(np)):
            tcoeffs = self.gen_taylor_coeffs(momentum[t,i], d_momentum[t-to:t+1,i])
            np[i] = self.gen_pade_prediction(tcoeffs)
        return np

    def next_phase(self, phase, d_phase, t):
        to = self.m_order + self.n_order
        if t <= to:
            return self.eom_early.next_phase(phase, d_phase, t)
        tcoeffs = self.gen_taylor_coeffs(phase[t], d_phase[t-to:t+1])
        return self.gen_pade_prediction(tcoeffs)

    def next_coefficient(self, coeff, d_coeff, t):
        to = self.m_order + self.n_order
        if t <= to:
            return self.eom_early.next_coefficient(coeff, d_coeff, t)
        tcoeffs = self.gen_taylor_coeffs(coeff[t], d_coeff[t-to:t+1])
        return self.gen_pade_prediction(tcoeffs)

