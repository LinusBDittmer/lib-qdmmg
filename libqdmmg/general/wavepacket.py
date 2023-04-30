'''

Class for a general Wavepacket. Used to describe prior wavefunctions

'''

import numpy

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
        self.gauss_coeff = numpy.array([])
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
        if abs(coeff.real - coeff) > 10**-3:
            coeff = abs(coeff)
            self.logger.warn("Significant complex phase in coefficient detected. Phase should be exported to basis and coeffs kept real. Phase is discarded.")
        self.gauss_coeff = numpy.append(self.gauss_coeff, coeff.real)
        for t, coeff in enumerate(self.gauss_coeff.T):
            self.gauss_coeff[:,t] /= numpy.linalg.norm(coeff)


    def getCoeffs(self, t):
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


