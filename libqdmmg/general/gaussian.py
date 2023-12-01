'''

Container for General Gaussian used for wavepacket propagation

'''

import numpy
import libqdmmg.integrate as intor
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
    d_centre_v : 2D ndarray
        Array with shape (timesteps, dimensions). This array describes the quantum (variational) correction to the centre step.
    momentum : 2D ndarray
        Array with shape (timesteps, dimensions). This array contains the equivalent momenta at each timestep. Note that the actual momentum is given by 1j*momentum
    d_momentum : 2D ndarray
        Array with shape (timesteps, dimensions). This array is populated by the EOM generator and is extracted by the time-integration scheme to generate the next value in momentum.
    d_momentum_v : 2D ndarray
        Array with shape (timesteps, dimensions). This array describes the quantum (variational) correction to the momentum step.
    phase : 1D ndarray
        Array with shape (dimensions,). This array contains the equivalent phase at each timestep. Note that the actual phase is given by 1j*phase.
    d_phase : 1D ndarray
        Array with shape (dimensions,). This array is populated by the EOM generator and is extracted by the time-integration scheme to generate the next value in phase.
    d_phase_v : 1D ndarray
        Array with shape (dimensions,). This array describes the quantum (variational) correction to the momentum step.
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
        tsteps = sim.tsteps 
        self.sim = sim
        self.centre = numpy.zeros((tsteps, sim.dim))
        self.d_centre = numpy.zeros(self.centre.shape)
        self.d_centre_v = numpy.zeros(self.centre.shape)
        self.momentum = numpy.zeros((tsteps, sim.dim))
        self.d_momentum = numpy.zeros(self.momentum.shape)
        self.d_momentum_v = numpy.zeros(self.momentum.shape)
        self.phase = numpy.zeros(tsteps)
        self.d_phase = numpy.zeros(self.phase.shape)
        self.d_phase_v = numpy.zeros(self.phase.shape)
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

    def equals(self, other, t, tol=10**-2):
        '''
        This method calculates whether two Gaussians can be considered equal at some timestep t with given tolerance tol.

        Parameters
        ----------
        other : libqdmmg.general.gaussian.Gaussian
            The Gaussian to which this instance is to be compared.
        t : int
            The timestep
        tol : float, optional
            Absolute tolerance of comparison. Default 0.01
        
        Returns
        -------
        eq : bool
            Whether the Gaussians are considered equal within given tolerance.
        '''
        if numpy.linalg.norm(self.centre[t] - other.centre[t]) > tol:
            return False
        if numpy.linalg.norm(self.momentum[t] - other.momentum[t]) > tol:
            return False
        if abs(self.phase[t] - other.phase[t]) > tol:
            return False
        return True

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
        y = self.evaluate(x, t)
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
        wa = numpy.linalg.norm(self.width) - self.width
        ep = - reduce(numpy.dot, (wa, xs*xs))
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

        return 2 * numpy.pi**(-self.sim.dim * 0.5) * numpy.linalg.norm(self.width)**(self.sim.dim * 0.5 + 1)

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
            w_prod = numpy.prod(self.width, axis=0)
            ep = reduce(numpy.dot, (self.width, self.centre[t]*self.centre[t]))
            self.v_amp[t] = 2**(0.5*self.sim.dim+1) * self.width[0] * numpy.pi**(-self.sim.dim*0.5) * w_prod**0.5 * numpy.exp(2 * ep)
        return self.width[index] / self.width[0] * self.v_amp[t]

    def energy_kin(self, t):
        '''
        This method calculates the kinetic energy of the Gaussian at timestep t

        Parameters
        ----------
        t : int
            The timestep.

        Returns
        -------
        ke : float
            The kinetic energy at timestep t.
        '''
        return intor.int_request(self.sim, 'int_kinetic_gg', self, self, t).real

    def energy_pot(self, t):
        '''
        This method calculates the potential energy within locally harmonic approximation of the Gaussian at timestep t.

        Parameters
        ----------
        t : int
            The timestep.

        Returns
        -------
        pe : float
            The potential energy at timestep t.
        '''
        pot_intor = self.sim.potential.gen_potential_integrator()
        return pot_intor.int_request('int_gVg', self, self, t).real

    def energy_tot(self, t):
        '''
        This method calculates the total energy within locally harmonic approximation of the Gaussian at timestep t.

        Parameters
        ----------
        t : int
            timestep

        Returns
        -------
        te : float
            The total energy at timestep t.
        '''
        return self.energy_kin(t) + self.energy_pot(t)

    def interpolate(self, ti, returntype='tuple'):
        '''
        This method interpolates the centre, momentum and phase at a non-integer time"step". This is performed by either three-point or two-point interpolation. If the given time lays between integer timesteps t0 and t1 and t2 is a sensible timestep (i. e. t0 is smaller than the total number of timesteps minus two) then quadratic interpolation between t0, t1 and t2 is used. Otherwise, the method defaults to linear interpolation between t0 and t1. Note that the data for timesteps t0, t1 and t2 (if relevant) already needs to have been generated, otherwise nonsensical results are generated without warning. The returntype argument then dictates the type which should be returned. Specifically, the method differentiates between returning the interpolated centre and momentum as a tuple or as an ndarray. The phase is always returned as a float.

        Parameters
        ----------
        ti : float
            The time at which the values should be interpolated
        returntype : str, optional
            The type of returned object. If "tuple" is given, centre and momentum are returned as tuple, otherwise as ndarray. Default "tuple"
        '''
        t0 = int(ti)
        t1 = t0+1
        t2 = t0+2
        dt = ti - t0
        xa, xb, xc, pa, pb, pc = [numpy.zeros(self.sim.dim)] * 6
        ga, gb, gc = 0, 0, 0
        if t0 > self.sim.tsteps-3:
            xint = dt * self.centre[t0-1] + (1-dt) * self.centre[t0]
            pint = dt * self.momentum[t0-1] + (1-dt) * self.momentum[t0]
            gint = dt * self.phase[t0-1] + (1-dt) * self.phase[t0]
        else:
            # 3-Point forward interpolation
            xc = self.centre[t0]
            xb = -0.5 * self.centre[t2] + 2 * self.centre[t1] - 2 * self.centre[t0]
            xa = 0.5 * self.centre[t2] + self.centre[t0] - self.centre[t1]
            pc = self.momentum[t0]
            pb = -0.5 * self.momentum[t2] + 2 * self.momentum[t1] - 2 * self.momentum[t0]
            pa = 0.5 * self.momentum[t2] + self.momentum[t0] - self.momentum[t1]
            gc = self.phase[t0]
            gb = -0.5 * self.phase[t2] + 2 * self.phase[t1] - 2 * self.phase[t0]
            ga = 0.5 * self.phase[t2] + self.phase[t0] - self.phase[t1]
            xint = xa * dt*dt + xb * dt + xc
            pint = pa * dt*dt + pb * dt + pc
            gint = ga * dt*dt + gb * dt + gc
        if returntype == 'tuple':
            return tuple(xint), tuple(pint), gint
        return xint, pint, gint

    def step_forward(self, t):
        '''
        This method updates the centre points, momenta and phases with data given from the Time Integrator.

        Parameters
        ----------
        t : int
            The timestep.
        '''
        self.logger.debug2("Gaussian " + str(self) + " stepping forward in time.")
        self.centre[t+1] = self.sim.eom_intor.next_centre(self, t)
        self.momentum[t+1] = self.sim.eom_intor.next_momentum(self, t)
        self.phase[t+1] = self.sim.eom_intor.next_phase(self, t)
        self.phase[t+1] = (self.phase[t+1] + numpy.pi) % (2*numpy.pi) - numpy.pi
        self.logger.info("Centre:")
        for k in range(self.sim.dim):
            self.logger.info(" " * 10 + str(round(self.centre[t,k], 12)) + "  -->  " + str(round(self.centre[t+1,k], 12)))
        self.logger.info("")
        self.logger.info("Momentum:")
        for k in range(self.sim.dim):
            self.logger.info(" " * 10 + str(round(self.momentum[t,k], 12)) + "  -->  " + str(round(self.momentum[t+1,k], 12)))
        self.logger.info("")
        self.logger.info("Phase:")
        self.logger.info(" " * 10 + str(round(self.phase[t], 12)) + "  -->  " + str(round(self.phase[t+1], 12)))


    def copy(self, dephase=False):
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
        g.momentum = numpy.copy(self.momentum) * (not dephase)
        g.d_momentum = numpy.copy(self.d_momentum)
        g.phase = numpy.copy(self.phase) * (not dephase)
        g.d_phase = numpy.copy(self.d_phase)
        g.v_amp = numpy.copy(self.v_amp)
        g.width = numpy.copy(self.width)
        return g

