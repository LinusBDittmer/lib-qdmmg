'''

@author: Linus Bjarne Dittmer

'''

import numpy
import libqdmmg.general as g
from functools import reduce

class Potential:
    '''
    This class functions as an abstract blueprint for potential potentials. These should be defined as subclasses inheriting from Potential. The class features (apart from the constructor) two functions which are defined abstractly are are supposed to be overriden by subclasses. These are evaluate and gen_potential_integrator.

    Attributes
    ----------
    sim : libqdmmg.simulate.simulation.Simulation
        The main simulation instance holding all relevant information.
    logger : libqdmmg.general.logger.Logger
        The logger object for printing to the standard out.
    '''

    def __init__(self, sim):
        '''
        Constructor for the Potential class.

        Parameters
        ----------
        sim : libqdmmg.simulate.simulation.Simulation
            The main simulation instance holding all relevant information.
        '''
        self.sim = sim
        self.logger = sim.logger

    def evaluate(self, x):
        '''
        === To be overridden. ===

        This function is a blueprint for the evaluation of the potential to ensure that all subclasses possess this function. It returns the value of the potential at a given cartesian point.

        Parameters
        ----------
        x : 1D ndarray
            Array with shape (dimensions,), at which the potential should be evaluated.
        
        Returns
        -------
        pot_val : float
            Value of the potential at x.
        '''
        return 0

    def gen_potential_integrator(self):
        '''
        === To be overriden. ===

        This function is a blueprint for generation and setup of the potential integrator. It is defined abstractly to ensure that all subclasses possess this function.

        Returns
        -------
        pot_intor : libqdmmg.potential.potential.PotentialIntegrator
            A potential integrator instance.
        '''
        return None


class PotentialIntegrator:
    '''
    This class functions as a manager for integrating elementary integrals containing the potential. These functions match the structure of elementary analytical integrals except for the inclusion of the potential term and are defined abstractly.

    Attributes
    ----------
    potential : libqdmmg.potential.potential.Potential
        The potential whose integrals are handled.
    logger : libqdmmg.general.logger.Logger
        The logger instance for printing to the standard out.
    '''

    def __init__(self, potential):
        '''
        Constructor for the potential integrator class.

        Parameters
        ----------
        potential : libqdmmg.potentia.potential.Potential
            The potential whose integrals are handled.
        '''
        self.potential = potential
        self.logger = potential.sim.logger

    def int_request(self, request_string, *args, **kwargs):
        '''
        The interface method through which potential integrals are accessed. The provided request string is not case-sensitive and may include a leading backspace, without which it must be one of the following options:

        int_gVg
        int_gVxg
        int_gVx2g
        int_gVx2g_m
        int_gVx2g_d
        int_gVx3g
        int_gVx3g_m
        int_gVx3g_d
        int_uVg
        int_uVxg
        int_uVx2g
        int_uVx2g_m
        int_uVx2g_d
        int_uVx3g
        int_uVx3g_m
        int_uVx3g_d
        int_vVg
        int_vVxg
        int_vVx2g
        int_vVx2g_m
        int_vVx2g_d
        int_vVx3g
        int_vVx3g_m
        int_vVx3g_d

        Parameters
        ----------
        request_string : str
            The request string. See above for restrictions.
        args : list
            Arguments required for integral calculation. Must include at least two instances of libqdmmg.general.gaussian.Gaussian in the first two slots and the timestep as an integer in the third slot.
        kwargs : dict
            Keyword arguments required for integral calculation.

        Returns
        -------
        int_val : complex128
            The integral value.

        Raises
        ------
        AssertionError
            If an incorrect number of arguments is given.
        InvalidIntegralRequestStringException
            If the request string is not valid.
        '''
        rq = request_string.lower().strip()
        if rq[0] == '_':
            rq = rq[1:]
        argnum = len(args)

        assert argnum >= 3, f"Expected at least 3 arguments (g1, g2, t). Received {argnum}."

        if rq == 'int_gvg':
            return self._int_gVg(args)
        elif rq == 'int_uvg':
            return self._int_uVg(args)
        elif rq == 'int_vvg':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, vindex). Received {argnum}."
            return self._int_vVg(args)
        elif rq == 'int_gvxg':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, index). Received {argnum}."
            return self._int_gVxg(args, kwargs)
        elif rq == 'int_uvxg':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, index). Received {argnum}."
            return self._int_uVxg(args, kwargs)
        elif rq == 'int_vvxg':
            assert argnum >= 5, f"Expected at least 5 arguments (g1, g2, t, vindex, index). Received {argnum}."
            return self._int_vVxg(args, kwargs)
        elif rq == 'int_gvx2g_m':
            assert argnum >= 5, f"Expected at least 5 arguments (g1, g2, t, index1, index2). Received {argnum}."
            return self._int_gVx2g_mixed(args, kwargs)
        elif rq == 'int_uvx2g_m':
            assert argnum >= 5, f"Expected at least 5 arguments (g1, g2, t, index1, index2). Received {argnum}."
            return self._int_uVx2g_mixed(args, kwargs)
        elif rq == 'int_vvx2g_m':
            assert argnum >= 6, f"Expected at least 6 arguments (g1, g2, t, vindex, index1, index2). Received {argnum}."
            return self._int_vVx2g_mixed(args, kwargs)
        elif rq == 'int_gvx2g_d':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, index). Received {argnum}."
            return self._int_gVx2g_diag(args, kwargs)
        elif rq == 'int_uvx2g_d':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, index). Received {argnum}."
            return self._int_uVx2g_diag(args, kwargs)
        elif rq == 'int_vvx2g_d':
            assert argnum >= 5, f"Expected at least 5 arguments (g1, g2, t, vindex, index). Received {argnum}."
            return self._int_vVx2g_diag(args, kwargs)
        elif rq == 'int_gvx3g_m':
            assert argnum >= 5, f"Expected at least 5 arguments (g1, g2, t, index1, index2). Received {argnum}."
            return self._int_gVx3g_mixed(args, kwargs)    
        elif rq == 'int_uvx3g_m':
            assert argnum >= 5, f"Expected at least 5 arguments (g1, g2, t, index1, index2). Received {argnum}."
            return self._int_uVx3g_mixed(args, kwargs)
        elif rq == 'int_vvx3g_m':
            assert argnum >= 6, f"Expected at least 6 arguments (g1, g2, t, vindex, index1, index2). Received {argnum}."
            return self._int_vVx3g_mixed(args, kwargs)
        elif rq == 'int_gvx3g_d':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, index). Received {argnum}."
            return self._int_gVx3g_diag(args, kwargs)
        elif rq == 'int_uvx3g_d':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, index). Received {argnum}."
            return self._int_uVx3g_diag(args, kwargs)
        elif rq == 'int_vvx3g_d':
            assert argnum >= 5, f"Expected at least 5 arguments (g1, g2, t, vindex, index). Received {argnum}."
            return self._int_vVx3g_diag(args, kwargs)
        elif rq == 'int_gvx2g':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, index[, index2]). Received {argnum}."
            if argnum >= 5:
                if args[3] != args[4]:
                    return self._int_gVx2g_mixed(args, kwargs)
            return self._int_gVx2g_diag(args, kwargs)
        elif rq == 'int_uvx2g':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, index[, index2]). Received {argnum}."
            if argnum >= 5:
                if args[3] != args[4]:
                    return self._int_uVx2g_mixed(args, kwargs)
            return self._int_uVx2g_diag(args, kwargs)
        elif rq == 'int_vvx2g':
            assert argnum >= 5, f"Expected at least 5 arguments (g1, g2, t, vindex, index[, index2]). Received {argnum}."
            if argnum >= 6:
                if args[4] != args[5]:
                    return self._int_vVx2g_mixed(args, kwargs)
            return self._int_vVx2g_diag(args, kwargs)
        elif rq == 'int_gvx3g':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, index[, index2]). Received {argnum}."
            if argnum >= 5:
                if args[3] != args[4]:
                    return self._int_gVx3g_mixed(args, kwargs)
            return self._int_gVx3g_diag(args, kwargs)
        elif rq == 'int_uvx3g':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, index[, index2]). Received {argnum}."
            if argnum >= 5:
                if args[3] != args[4]:
                    return self._int_uVx3g_mixed(args, kwargs)
            return self._int_uVx3g_diag(args, kwargs)
        elif rq == 'int_vvx3g':
            assert argnum >= 5, f"Expected at least 5 arguments (g1, g2, t, vindex, index[, index2]). Received {argnum}."
            if argnum >= 6:
                if args[4] != args[5]:
                    return self._int_vVx3g_mixed(args, kwargs)
            return self._int_vVx3g_diag(args, kwargs)
        else:
            raise g.IIRSException(rq, "")

    def _int_gVg(self, g1, g2, t):
        '''
        === To be overridden. ===

        Integral of the function g1 * V * g2.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            First gaussian.
        g2 : libqdmmg.general.gaussian.Gaussian
            Second gaussian.
        t : int
            Timestep index.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_gVxg(self, g1, g2, t, index):
        '''
        === To be overridden. ===

        Integral of the function g1 * V * x_(index) * g2.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            First gaussian.
        g2 : libqdmmg.general.gaussian.Gaussian
            Second gaussian.
        t : int
            Timestep index.
        index : int
            Index of the direction in which the linear function is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_gVx2g_mixed(self, g1, g2, t, index1, index2):
        '''
        === To be overridden. ===

        Integral of the function g1 * V * x_(index1) * x_(index2) * g2. Note that index1 != index2, refer to _int_gVx2g_diag otherwise.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            First gaussian.
        g2 : libqdmmg.general.gaussian.Gaussian
            Second gaussian.
        t : int
            Timestep index.
        index1 : int
            Index of the direction in which the first linear factor is constructed.
        index2 : int
            Index of the direction in which the second linear factor is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_gVx2g_diag(self, g1, g2, t, index):
        '''
        === To be overridden. ===

        Integral of the function g1 * V * x_(index)^2 * g2.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            First gaussian.
        g2 : libqdmmg.general.gaussian.Gaussian
            Second gaussian.
        t : int
            Timestep index.
        index : int
            Index of the direction in which the quadratic function is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_gVx3g_mixed(self, g1, g2, t, index1, index2):
        '''
        === To be overridden. ===

        Integral of the function g1 * V * x_(index1) * x_(index2)^2 * g2. Note that index1 != index2, refer to _int_gVx3g_diag otherwise.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            First gaussian.
        g2 : libqdmmg.general.gaussian.Gaussian
            Second gaussian.
        t : int
            Timestep index.
        index1 : int
            Index of the direction in which the linear factor of the cubic polynomial is constructed.
        index2 : int
            Index of the direction in which the quadratic part of the cubic polynomial is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_gVx3g_diag(self, g1, g2, t, index):
        '''
        === To be overridden. ===

        Integral of the function g1 * V * x_(index)^3 * g2.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            First gaussian.
        g2 : libqdmmg.general.gaussian.Gaussian
            Second gaussian.
        t : int
            Timestep index.
        index : int
            Index of the direction in which the cubic function is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_uVg(self, g1, g2, t):
        '''
        === To be overridden. ===

        Integral of the function u1 * V * g2. u1 is the u-dual function of the Gaussian g1.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose u-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_uVxg(self, g1, g2, t, index):
        '''
        === To be overridden. ===

        Integral of the function u1 * V * x_(index) * g2. u1 is the u-dual function of the Gaussian g1.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose u-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.
        index : int
            Index of the direction in which the linear function is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_uVx2g_mixed(self, g1, g2, t, index1, index2):
        '''
        === To be overridden. ===

        Integral of the function u1 * V * x_(index1) * x_(index2) * g2. u1 is the u-dual function of the Gaussian g1. Note that index1 != index2, refer to _int_uVx2g_diag otherwise.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose u-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.
        index1 : int
            Index of the direction in which the first linear factor of the quadratic polynomial is constructed.
        index2 : int
            Index of the direction in which the second linear factor of the quadratic polynomial is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_uVx2g_diag(self, g1, g2, t, index):
        '''
        === To be overridden. ===

        Integral of the function u1 * V * x_(index)^2 * g2. u1 is the u-dual function of the Gaussian g1.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose u-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.
        index : int
            Index of the direction in which the quadratic function is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_uVx3g_mixed(self, g1, g2, t, index1, index2):
        '''
        === To be overridden. ===

        Integral of the function u1 * V * x_(index1) * x_(index2)^2 * g2. u1 is the u-dual function of the Gaussian g1. Note that index1 != index2, refer to _int_uVx3g_diag otherwise

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose u-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.
        index1 : int
            Index of the direction in which the linear part of the cubic polynomial is constructed.
        index2 : int
            Index of the direction in which the quadratic part of the cubic polynomial is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_uVx3g_diag(self, g1, g2, t, index):
        '''
        === To be overridden. ===

        Integral of the function u1 * V * x_(index)^3 * g2. u1 is the u-dual function of the Gaussian g1.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose u-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.
        index : int
            Index of the direction in which the linear function is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_vVg(self, g1, g2, t, vindex):
        '''
        === To be overridden. ===

        Integral of the function v1 * V * g2. v1 is the v-dual function of the Gaussian g1 with directional index vindex.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose v-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.
        vindex : int
            Directional index of the v-dual function.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_vVxg(self, g1, g2, t, vindex, index):
        '''
        === To be overridden. ===

        Integral of the function v1 * V * x_(index) * g2. v1 is the v-dual function of the Gaussian g1 with directional index vindex.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose v-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.
        vindex : int
            Directional index of the v-dual function.
        index : int
            Index of the direction in which the linear function is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_vVx2g_mixed(self, g1, g2, t, vindex, index1, index2):
        '''
        === To be overridden. ===

        Integral of the function v1 * V * x_(index1) * x_(index2) * g2. v1 is the v-dual function of the Gaussian g1 with directional index vindex. Note that index1 != index2, refer to _int_vVx2g_diag otherwise.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose v-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.
        vindex : int
            Directional index of the v-dual function.
        index1 : int
            Index of the direction in which the first linear factor is constructed.
        index2 : int
            Index of the direction in which the second linear factor is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_vVx2g_diag(self, g1, g2, t, vindex, index):
        '''
        === To be overridden. ===

        Integral of the function v1 * V * x_(index)^2 * g2. v1 is the v-dual function of the Gaussian g1 with directional index vindex.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose v-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.
        vindex : int
            Directional index of the v-dual function.
        index : int
            Index of the direction in which the quadratic function is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_vVx3g_mixed(self, g1, g2, t, vindex, index1, index2):
        '''
        === To be overridden. ===

        Integral of the function v1 * V * x_(index1) * x_(index2)^2 * g2. v1 is the v-dual function of the Gaussian g1 with directional index vindex. Note that index1 != index2, refer to _int_vVx3g_diag otherwise.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose v-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.
        vindex : int
            Directional index of the v-dual function.
        index1 : int
            Index of the direction in which the linear part of the cubic polynomial is constructed.
        index2 : int
            Index of the direction in which the quadratic part of the cubic polynomial is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0

    def _int_vVx3g_diag(self, g1, g2, t, vindex, index):
        '''
        === To be overridden. ===

        Integral of the function v1 * V * x_(index)^3 * g2. v1 is the v-dual function of the Gaussian g1 with directional index vindex.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            Gaussian whose v-dual function is used in the integrand
        g2 : libqdmmg.general.gaussian.Gaussian
            Gaussian which is used directly in the integrand
        t : int
            Timestep index.
        vindex : int
            Directional index of the v-dual function.
        index : int
            Index of the direction in which the cubic function is constructed.

        Returns
        -------
        int_val : complex128
            Integral value.
        '''
        return 0




