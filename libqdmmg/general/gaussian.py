'''

Container for General Gaussian used for wavepacket propagation

'''

import numpy
import copy
from functools import reduce

class Gaussian:
    '''
    This class represents a Gaussian basis function used to build the wavepacket. Each gaussian is defined as

    .. math::
        
        g(\mathbf{x}, t) = \exp \left(-\boldsymbol \alpha \cdot (\mathbf{x} - \Bar{\mathbf{x}})^2 + i\mathbf{p} \cdot \mathbf{x} + i\gamma \right)

    The individual variables are referred to as the width, centre, momentum and phase of the gaussian. Furthermore, each Gaussian has two types of dual functions defined, which are the u and v functions. These are employed to guarantee specific integral cancellation relations. It is notable that for each Gaussian, only one u is uniquely defined, while there are as many v functions as dimensions.

    Attributes
    ----------
    sim : libqdmmg.simulate.simulation.Simulation
        Main Simulation instance. This is a masterclass holding all noteworthy information and which is directy controlled by user code.
    centre : 2D ndarray
        Array with shape (timesteps, dimensions). This contains the centrepoints at each timestep.
    d_centre : 2D ndarray
        Array with shape (timesteps, dimensions). This array is populated by the EOM generator and is extracted by the time-integration scheme to generate the next value in centre
    momentum : 2D ndarray
        Array with shape (timesteps, dimensions). This array contains the equivalent momenta at each timestep. Note that the actual momentum is given by 1j*momentum
    d_momentum : 2D ndarray
        Array with shape (timesteps, dimensions). This array is populated by the EOM generator and is extracted by the time-integration scheme to generate the next value in momentum.
    phase : 1D ndarray
        Array with shape (dimensions,). This array contains the equivalent phase at each timestep. Note that the actual phase is given by 1j*phase.
    d_phase : 1D ndarray
        Array with shape (dimensions,). This array is populated by the EOM generator and is extracted by the time-integration scheme to generate the next value in phase.
    logger : libqdmmg.general.logger.Logger
        Logger object for output.
    v_amp : 1D ndarray
        Array with shape (dimensions,). This array contains the v amplitude in direction 0 in order to ease calculation of each v dual function.

    Examples:

    '''

    def __init__(self, sim, centre=None, width=None, momentum=None, phase=0.0):
        '''
        Constructor for the Gaussian class.

        Parameters
        ----------
        sim : libqdmmg.simulate.simulation.Simulation
            Main Simulation instance. This is a masterclass holding all relevant information.
        centre : 1D ndarray, optional
            Initial centre of the Gaussian function with shape (dimensions,). Default is a zero origin (centred at the coordinate origin).
        width : 1D ndarray, optional
            Width array of the Gaussian function with shape (dimensions,). Default is an array of form (1, ..., 1) to create an isotropic unit Gaussian.
        momentum : 1D ndarray, optional
            Initial equivalent momentum of the Gaussian function shape (dimensions,) and dtype float32, not complex128. The actual initial momentum is given by 1j*momentum. Default is a zero array.
        phase : float, optional
            Initial equivalent phase. The actual initial phase is given by 1j*phase. Default is 0.
        '''
        
        self.sim = sim
        self.centre = numpy.zeros((tsteps, sim.dim))
        self.d_centre = numpy.zeros(self.centre.shape)
        self.momentum = numpy.zeros((tsteps, sim.dim))
        self.d_momentum = numpy.zeros(self.momentum.shape)
        self.phase = numpy.zeros(tsteps)
        self.d_phase = numpy.zeros(self.phase.shape)
        self.width = numpy.ones(sim.dim)
        self.phase[0] = phase
        self.logger = sim.logger
        self.v_amp = -numpy.ones(tsteps)

        if type(centre) is numpy.ndarray:
            self.centre[0] = centre
        if type(width) is numpy.ndarray:
            self.width = width
        if type(momentum) is numpy.ndarray:
            self.momentum[0] = momentum


    def evaluate(self, x, t):
        '''
        Evaluates the Gaussian function at a given point x and timestep t

        Parameters
        ----------
        x : 1D ndarray
            Point at which the function is to be evaluated
        t : int
            Index of the timestep at which the function is to be evaluated

        Returns
        -------
        val : complex128
            Function value at point x and timestep t

        '''
        # y = exp(-a(x-c)**2 + px + g)
        xs = x - self.centre[t]
        ep = - reduce(numpy.dot, (self.width, xs*xs)) + 1j*(reduce(numpy.dot, (self.momentum[t], x)) + self.phase[t])
        return numpy.exp(ep)

    def evaluateD(self, x, t):
        '''
        Evaluates the gradient of the Gaussian function at a given point x and timestep t

        Parameters
        ----------
        x : 1D ndarray
            Point at which the gradient is to be evaluated
        t : int
            Index of timestep at which the function is to be evaluated

        Returns
        -------
        grad : 1D ndarray
            Gradient of the Gaussian function with dtype complex128.

        '''
        # y' = y * (-2a(x-c) + p)
        y = self.evaluate(x, t, is_index)
        return y * reduce(numpy.add, (-2*reduce(numpy.multiply, (self.width, x-self.centre[t])), 1j*self.momentum[t]))

    def evaluateU(self, x, t):
        '''
        Evaluates the u dual function at a given point x and timestep t

        Parameters
        ----------
        x : 1D ndarray
            Point at which the u dual function is to be evaluated
        t : int
            Index of the timestep at which the u dual function is to be evaluated

        Returns
        -------
        val : complex128
            Value of the u dual function at point x and timestep t

        '''
        xs = x - self.centre[t]
        ep = - reduce(numpy.dot, (self.width, xs*xs))
        return self.u_amplitude(t) * numpy.exp(ep)

    def evaluateV(self, x, t):
        ''' 
        Evaluates the v dual function in direction 0 at a given point x and timestep t. To evaluate the v dual function in any direction k, multiply the result by width[k] / width[0].

        Parameters
        ----------
        x : 1D ndarray
            Point and which the v dual function is to be evaluated
        t : int
            Index of the timestep at which the v dual function is to be evaluated

        Returns
        -------
        val : complex128
            Value of the v dual function in direction 0 at point x and timestep t

        '''
        xs = x + self.centre[t]
        ep = - reduce(numpy.dot, (self.width, xs*xs))
        return self.v_amplitude(t, 0) * numpy.exp(ep)

    def u_amplitude(self, t):
        '''
        Calculates the amplitude of the u dual function.

        Parameters
        ----------
        t : int
            Index of the timestep at which the amplitude is to be calculated

        Returns
        -------
        amp : float32
            Amplitude of the u dual function

        '''

        return 2 * numpy.pi**(-self.sim.dim * 0.5) * numpy.linalg.norm(self.width)**(self.sim.dim * 0.5)

    def v_amplitude(self, t, index):
        '''
        Calculates the amplitude of the v dual function in a specific direction.

        Parameters
        ----------
        t : int
            Index of the timestep at which the amplitude is to be calculated
        index : int
            Directional index in which the apmlitude is to be calculated

        Returns
        -------
        amp : float32
            Ampliutde of the v dual function

        '''
        if self.v_amp[t] < 0:
            w_prod = reduce(numpy.prod, self.width)
            ep = reduce(numpy.dot, (self.width, self.centre[t]*self.centre[t]))
            self.v_amp[t] = 2 * self.width[0] * numpy.pi(-self.sim.dim*0.5) * w_prod**0.5 * numpy.exp(2 * ep)
        return self.width[index] / self.width[0] * self.v_amp[t]


    def step_forward(self):
        '''
        Internal management of the time-integration scheme. The Gaussian time-integration employs a symmetric explicit time integration scheme of the form

        .. math ::
            
            \frac{f(x_0+h) + f(x_0-h)}{2h} \approx \left.\frac{df}{dx}\right\vert_{x=x_0}

        Except for the first timestep, in which the usual explicit time-integration scheme is used.

        '''
        if self.sim.t == 0:
            self.centre[1] = self.centre[0] + self.sim.tstep_val * self.d_centre[0]
            self.momentum[1] = self.momentum[0] + self.sim.tstep_val * self.d_momentum[0]
            self.phase[1] = self.phase[0] + self.sim.tstep_val * self.d_phase[0]
        else:
            self.centre[t+1] = self.centre[t-1] + 2 * sim.tstep_val * self.d_centre[t]
            self.momentum[t+1] = self.momentum[t-1] + 2 * sim.tstep_val * self.d_momentum[t]
            self.phase[t+1] = self.phase[t-1] + 2 * sim.tstep_val * self.d_phase[t]

    def copy(self):
        '''
        Creates a copy of the Gaussian function while keeping reference to the simulation and logger instance.

        Returns
        -------
        gc : libqdmmg.general.gaussian.Gaussian
            Copied Gaussian

        '''
        g = Gaussian(self.sim)
        g.centre = numpy.copy(self.centre)
        g.d_centre = numpy.copy(self.d_centre)
        g.momentum = numpy.copy(self.momentum)
        g.d_momentum = numpy.copy(self.d_momentum)
        g.phase = numpy.copy(self.phase)
        g.d_phase = numpy.copy(self.d_phase)
        g.v_amp = numpy.copy(self.v_amp)
        g.width = numpy.copy(self.width)
        return g
