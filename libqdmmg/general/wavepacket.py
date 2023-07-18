'''

Class for a general Wavepacket. Used to describe prior wavefunctions

'''

import numpy
import libqdmmg.integrate as intor

class Wavepacket:
    '''
    Class representing a wavepacket, i. e. a sum of Gaussians. The exact formula is given as

    .. math::

        \Psi(\mathbf{x}, t) = \sum_i \exp\left(-\boldsymbol \alpha_i \cdot (\mathbf{x} - \Bar{\mathbf{x}}_i)^2 + \mathbf{p}_i \cdot \mathbf{x} + \gamma_i \right)

    Wavepackets are used to represent previously calculated Gaussians.

    Attributes
    ----------
    sim : libqdmmg.simulate.simulation.Simulation
        Main Simulation instance. This is masterclass that holds all relevant information. 
    gaussians : list
        List of Gaussians that are contained within the wavepacket. Each element is an instance of libqdmmg.general.gaussian.Gaussian.
    gauss_coeff : 2D ndarray
        Array of shape (number of gaussians, timesteps). This array holds the weighting coefficient for each gaussian at each timestep.
    logger : libqdmmg.general.logger.Logger
        Logger instance for output to the standard out.

    Examples:

    '''

    def __init__(self, sim):
        '''
        Constructor for the Wavepacket class.

        Parameters
        ----------
        sim : libqdmmg.simulate.simulation.Simulation
            Main Simulation instance. This is a masterclass that holds all relevant information.

        '''
        self.sim = sim
        self.gaussians = []
        self.gauss_coeff = numpy.array([[]])
        self.logger = sim.logger

    def bind_gaussian(self, g1, coeff):
        '''
        Function that adds (binds) another Gaussian to the wavepacket. The individual gaussian coefficients are updated to reflect normalisation.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is to be bound to the wavepacket. It is expected that this Gaussian is already completely propagated
        coeff : 1D ndarray
            Array of weighting coefficients relative to the old weighting of the wavepacket with dimension (timesteps,).

        '''

        self.gaussians.append(g1)
        for i, c in enumerate(coeff):
            if abs(c.real - c) > 10**-3:
                coeff[i] = abs(c)
                self.logger.warn("Significant complex phase in coefficient detected. Phase should be exported to basis and coeffs kept real. Phase is discarded.")
        if self.gauss_coeff.size == 0:
            self.gauss_coeff = numpy.ones((1, self.sim.tsteps))
            for t in range(self.sim.tsteps):
                ovlp_gg = intor.int_request(self.sim, 'int_ovlp_gg', g1, g1, t)
                self.gauss_coeff[:,t] /= numpy.sqrt(ovlp_gg.real)
        else:
            for t in range(self.sim.tsteps):
                ovlp_gg = intor.int_request(self.sim, 'int_ovlp_gg', g1, g1, t)
                ovlp_gw = intor.int_request(self.sim, 'int_ovlp_gw', g1, self, t)
                b_coeff = -coeff[t] * ovlp_gw.real + numpy.sqrt(coeff[t]*coeff[t]*(ovlp_gw.real*ovlp_gw.real - ovlp_gg) + 1)
                self.gauss_coeff[:,t] *= b_coeff.real
            self.gauss_coeff = numpy.append(self.gauss_coeff, numpy.array([coeff.real]), axis=0)

    def get_coeffs(self, t):
        '''
        Getter for the array of gaussian coefficients at a specific timestep t.

        Parameters
        ----------
        t : int
            Timestep at which the gaussian coefficients should be returned.

        Returns
        -------
        gauss_coeff : 1D ndarray
            Array of gaussian coefficients at timestep t with shape (number of gaussians,).

        '''
        return self.gauss_coeff[:,t].T

    def get_d_coeffs(self, t):
        '''
        Getter for the differential of coefficients at a specific timestep t. If t is the last timestep, zero is returned

        Parameters
        ----------
        t : int
            Timestep at which the gaussian coefficient differential should be returned.

        Returns
        -------
        gauss_d_coeff : 1D ndarray
            Array of gaussian coefficient differentials at timestep t with shape (number of gaussians,)
        '''
        if t == self.sim.tsteps-1:
            return numpy.zeros(len(self.gaussians))
        elif t == 0:
            return (self.get_coeffs(1) - self.get_coeffs(0)) / self.sim.tstep_val
        return 0.5 * (self.get_coeffs(t+1) - self.get_coeffs(t-1)) / self.sim.tstep_val 

    def evaluate(self, x, t):
        '''
        Function for evaluating the wavepacket.

        Parameters
        ----------
        x : 1D ndarray
            Array with shape (dimensions,) holding the cartesian coordinates of the point.
        t : int
            Timestep index

        Returns
        -------
        w_val : complex128
            Value of the Wavepacket
        '''
        val = 0.0
        x = numpy.array(x)
        for i, gauss in enumerate(self.gaussians):
            val += self.gauss_coeff[i,t] * gauss.evaluate(x, t)
        return val

    def energy_kin(self, t):
        return intor.int_request(self.sim, 'int_kinetic_ww', self, self, t).real        

    def energy_pot(self, t):
        pot_intor = self.sim.potential.gen_potential_integrator()
        return pot_intor.int_request('int_wVw', self, self, t).real

    def energy_tot(self, t):
        return self.energy_kin(t) + self.energy_pot(t)

    def copy(self):
        c = Wavepacket(self.sim)
        c.gaussians = [None] * len(self.gaussians)
        c.gauss_coeff = numpy.copy(self.gauss_coeff)
        for i in range(len(self.gaussians)):
            c.gaussians[i] = self.gaussians[i].copy()
        return c

    def zerotime_wp(self):
        c = Wavepacket(self.sim)
        c.gaussians = [None] * len(self.gaussians)
        c.gauss_coeff = numpy.copy(self.gauss_coeff)
        for i in range(len(self.gaussians)):
            c.gauss_coeff[i] = numpy.array([self.gauss_coeff[i,0]] * self.sim.tsteps)
            c.gaussians[i] = self.gaussians[i].copy()
            c.gaussians[i].centre = numpy.array([self.gaussians[i].centre[0]] * self.sim.tsteps)
            c.gaussians[i].momentum = numpy.array([self.gaussians[i].momentum[0]] * self.sim.tsteps)
            c.gaussians[i].phase = numpy.array([self.gaussians[i].phase[0]] * self.sim.tsteps)
        return c

    def propagation_quality(self):
        quality = numpy.zeros(self.sim.tsteps)
        pot_intor = self.sim.potential.gen_potential_integrator()
        for t in range(self.sim.tsteps):
            tprop = intor.int_request(self.sim, 'int_dovlp_ww', self, self, t)
            eprop = intor.int_request(self.sim, 'int_kinetic_ww', self, self, t)
            eprop += pot_intor.int_request('int_wVw', self, self, t)
            quality[t] = abs(1j * tprop - eprop)
        return quality

    def reset_coeffs(self):
        self.gauss_coeff = numpy.zeros(self.gauss_coeff.shape, dtype=float)
        self.gauss_coeff[0,0] = 1 / numpy.sqrt(intor.int_request(self.sim, 'int_ovlp_gg', self.gaussians[0], self.gaussians[0], 0))

if __name__ == '__main__':
    import libqdmmg.simulate as sim
    import libqdmmg.general as gen
    import libqdmmg.potential as pot
    s = sim.Simulation(2, 1.0, dim=2, verbose=4)
    s.bind_potential(pot.HarmonicOscillator(s, numpy.ones(2)))
    g1 = gen.Gaussian(s)
    g2 = gen.Gaussian(s, centre=numpy.array([0.0, 0.5]))
    w = gen.Wavepacket(s)
    w.bind_gaussian(g1, numpy.array([1.0, 1.0]))
    w.bind_gaussian(g2, numpy.array([0.5, 0.5]))
    s.logger.info(w.get_coeffs(0))
    s.logger.info(intor.int_request(s, 'int_ovlp_ww', w, w, 0))
