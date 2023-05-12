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

    def bindGaussian(self, g1, coeff):
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
        else:
            self.gauss_coeff = numpy.append(self.gauss_coeff, numpy.array([coeff.real]), axis=0)
        for t in range(self.sim.tsteps):
            self.gauss_coeff[:,t] /= abs(numpy.sqrt(intor.int_request(self.sim, 'int_ovlp_ww', self, self, t)))

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


if __name__ == '__main__':
    import libqdmmg.simulate as sim
    import libqdmmg.general as gen
    s = sim.Simulation(2, 1.0, dim=2, verbose=4)
    g1 = gen.Gaussian(s)
    g2 = gen.Gaussian(s, centre=numpy.array([0.0, 0.5]))
    w = gen.Wavepacket(s)
    w.bindGaussian(g1, numpy.array([1.0, 1.0]))
    w.bindGaussian(g2, numpy.array([1.0, 0.5]))
    s.logger.info(w.get_coeffs(0))
    s.logger.info(intor.int_request(s, 'int_ovlp_ww', w, w, 0))
