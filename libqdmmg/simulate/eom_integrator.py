'''

Integration Handler for EOM

'''

import numpy
import scipy

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
        unadapted_shift = numpy.dot(eff_coeff, relevant_coeffs)
        return coeff[t] + self.stepsize * unadapted_shift



