'''

@author: Linus Bjarne Dittmer

'''

import numpy
import libqdmmg.general as gen
import libqdmmg.integrate as intor
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
        self.reduced_mass = numpy.ones(sim.dim)
        self.reduced_charges = numpy.ones(sim.dim)

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

    def gradient(self, x):
        return self.num_gradient(x)

    def hessian(self, x):
        return self.num_hessian(x)

    def grad(self, x):
        return self.gradient(x)

    def hess(self, x):
        return self.hessian(x)

    def num_gradient(self, x, epsilon=10**-5):
        gradient = numpy.zeros(x.shape)
        for i in range(self.sim.dim):
            xp = numpy.copy(x)
            xm = numpy.copy(x)
            xp[i] += epsilon
            xm[i] -= epsilon
            gradient[i] = (self.evaluate(xp) - self.evaluate(xm)) / epsilon * 0.5
        return gradient

    def num_hessian(self, x, epsilon=10**-5):
        hess = numpy.zeros((self.sim.dim, self.sim.dim))
        for i in range(self.sim.dim):
            for j in range(self.sim.dim):
                if i > j: continue
                xpp = numpy.copy(x)
                xpm = numpy.copy(x)
                xmp = numpy.copy(x)
                xmm = numpy.copy(x)
                xpp[i] += epsilon
                xpp[j] += epsilon
                xpm[i] += epsilon
                xpm[j] -= epsilon
                xmp[i] -= epsilon
                xmp[j] += epsilon
                xmm[i] -= epsilon
                xmm[j] -= epsilon
                hess[i,j] = (self.evaluate(xpp) - self.evaluate(xmp) - self.evaluate(xpm) + self.evaluate(xmm)) / epsilon**2 * 0.25
        for i in range(self.sim.dim):
            for j in range(self.sim.dim):
                if i <= j: continue
                hess[i,j] = hess[j,i]
        return hess

    def gen_potential_integrator(self):
        '''
        === To be overriden. ===

        This function is a blueprint for generation and setup of the potential integrator. It is defined abstractly to ensure that all subclasses possess this function.

        Returns
        -------
        pot_intor : libqdmmg.potential.potential.PotentialIntegrator
            A potential integrator instance.
        '''
        return PotentialIntegrator(self)


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
        int_uVg
        int_uVxg
        int_vVg
        int_vVxg
        int_wVw
        int_wVg
        int_gVw

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

        if args[2] >= self.potential.sim.tsteps-1:
            a = list(args)
            a[2] = float(self.potential.sim.tsteps-1)
            args = tuple(a)

        if rq == 'int_gvg':
            return self._int_gVg(*args)
        elif rq == 'int_uvg':
            return self._int_uVg(*args)
        elif rq == 'int_vvg':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, vindex). Received {argnum}."
            return self._int_vVg(*args)
        elif rq == 'int_uvxg':
            assert argnum >= 4, f"Expected at least 4 arguments (g1, g2, t, index). Received {argnum}."
            return self._int_uVxg(*args, **kwargs)
        elif rq == 'int_vvxg':
            assert argnum >= 5, f"Expected at least 5 arguments (g1, g2, t, vindex, index). Received {argnum}."
            return self._int_vVxg(*args, **kwargs)
        elif rq == 'int_wvw':
            return self._int_wVw(*args)
        elif rq == 'int_wvg':
            return self._int_wVg(*args)
        elif rq == 'int_gvw':
            return self._int_gVw(*args)
        else:
            raise gen.IIRSException(rq, "")
    
    def _clean(self, i):
        if numpy.isnan(i):
            return 0.0
        if numpy.isinf(i):
            return 0.0
        return i

    def _int_wVw(self, w1, w2, t, r_taylor=None):
        int_val = 0.0
        coeffs1 = w1.get_coeffs(t)
        coeffs2 = w2.get_coeffs(t)
        if r_taylor is None:
            coeffs_t = numpy.concatenate((coeffs1, coeffs2))
            centres1 = numpy.array([g.centre[t] for g in w1.gaussians])
            centres2 = numpy.array([g.centre[t] for g in w2.gaussians])
            centres_t = numpy.concatenate((centres1, centres2))
            coeffs_t /= numpy.sum(coeffs_t)
            r_taylor = numpy.einsum('ji,j->i', centres_t, coeffs_t)
        for i in range(len(coeffs1)):
            for j in range(len(coeffs2)):
                int_val += coeffs1[i]*coeffs2[j]*self._int_gVg(w1.gaussians[i], w2.gaussians[j], t, r_taylor=r_taylor)
        return self._clean(int_val)

    def _int_wVg(self, w1, g2, t, r_taylor=None):
        int_val = 0
        coeffs = w1.get_coeffs(t)
        for i in range(len(coeffs)):
            int_val += coeffs[i]*self._int_gVg(w1.gaussians[i], g2, t, r_taylor=r_taylor)
        return self._clean(int_val)

    def _int_gVw(self, g1, w2, t, r_taylor=None):
        return self._int_wVg(w2, g1, t, r_taylor=r_taylor).conj()

    def _int_gVg(self, g1, g2, t, r_taylor=None):
        '''
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
        sim = self.potential.sim
        if r_taylor is None:
            tt = int(t)
            r_taylor = (g1.width * g1.centre[tt] + g2.width * g2.centre[tt]) / (g1.width + g2.width)
        v_const = self.potential.evaluate(r_taylor)
        grad = self.potential.gradient(r_taylor)
        hess = self.potential.hessian(r_taylor)
        gg = intor.int_request(sim, 'int_gg', g1, g2, t)
        v0 = v_const * gg
        gxg = numpy.zeros(sim.dim, dtype=numpy.complex128)
        gx2g = numpy.zeros((sim.dim, sim.dim), dtype=numpy.complex128)
        for i in range(len(gxg)):
            gxg[i] = intor.int_request(sim, 'int_gxg', g1, g2, t, i, m=gg, useM=True)
            for j in range(len(gxg)):
                gx2g[i,j] = intor.int_request(sim, 'int_gx2g', g1, g2, t, i, j, m=gg, useM=True)
        v1 = gxg - gg * r_taylor
        v2 = gx2g - numpy.einsum('k,m->mk', r_taylor, gxg) - numpy.einsum('m,k->mk', r_taylor, gxg) + numpy.einsum('m,k->mk', r_taylor, r_taylor) * gg

        return self._clean(v0 + numpy.dot(grad, v1) + 0.5 * numpy.einsum('ij,ji->', hess, v2))

    def _int_uVg(self, g1, g2, t, r_taylor=None):
        '''
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
        sim = self.potential.sim
        if r_taylor is None:
            r_taylor = (numpy.linalg.norm(g1.width) * g1.centre[t] - g1.width * g1.centre[t] + g2.width * g2.centre[t]) / (numpy.linalg.norm(g1.width) - g1.width + g2.width)
        v_const = self.potential.evaluate(r_taylor)
        grad = self.potential.gradient(r_taylor)
        hess = self.potential.hessian(r_taylor)
        ug = intor.int_request(sim, 'int_ug', g1, g2, t)
        v0 = v_const * ug
        uxg = numpy.zeros(sim.dim, dtype=numpy.complex128)
        ux2g = numpy.zeros((sim.dim, sim.dim), dtype=numpy.complex128)
        for i in range(len(uxg)):
            uxg[i] = intor.int_request(sim, 'int_uxg', g1, g2, t, i, m=ug, useM=True)
            for j in range(len(uxg)):
                ux2g[i,j] = intor.int_request(sim, 'int_ux2g', g1, g2, t, i, j, m=ug, useM=True)
        v1 = uxg - ug * r_taylor
        v2 = ux2g - numpy.einsum('k,m->mk', r_taylor, uxg) - numpy.einsum('m,k->mk', r_taylor, uxg) + numpy.einsum('m,k->mk', r_taylor, r_taylor) * ug
        return self._clean(v0 + numpy.dot(grad, v1) + 0.5 * numpy.einsum('ij,ij->', hess, v2))

    def _int_uVxg(self, g1, g2, t, index, r_taylor=None):
        '''
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
        sim = self.potential.sim
        if r_taylor is None:
            r_taylor = (numpy.linalg.norm(g1.width) * g1.centre[t] - g1.width * g1.centre[t] + g2.width * g2.centre[t]) / (numpy.linalg.norm(g1.width) - g1.width + g2.width)
        v_const = self.potential.evaluate(r_taylor)
        grad = self.potential.gradient(r_taylor)
        hess = self.potential.hessian(r_taylor)
        ug = intor.int_request(sim, 'int_ug', g1, g2, t)
        uxg = intor.int_request(sim, 'int_uxg', g1, g2, t, index, m=ug, useM=True)
        ux2g = numpy.zeros(sim.dim, dtype=numpy.complex128)
        ux3g = numpy.zeros((sim.dim, sim.dim), dtype=numpy.complex128)
        for i in range(sim.dim):
            ux2g[i] = intor.int_request(sim, 'int_ux2g', g1, g2, t, index, i, m=ug, useM=True)
            for j in range(sim.dim):
                ux3g[i,j] = intor.int_request(sim, 'int_ux3g', g1, g2, t, i, j, index, useM=True)
        v0 = v_const * uxg
        v1 = ux2g - uxg * r_taylor
        v2 = ux3g - numpy.einsum('k,m->mk', r_taylor, ux2g) - numpy.einsum('m,k->mk', r_taylor, ux2g) + numpy.einsum('m,k->mk', r_taylor, r_taylor) * uxg

        v1_t = 0.0
        for k in range(sim.dim):
            v1_t += grad[k] * (intor.int_request(sim, 'int_ux2g', g1, g2, t, index, k, m=ug, useM=True) - r_taylor[k] * intor.int_request(sim, 'int_uxg', g1, g2, t, index, m=ug, useM=True))

        v2_t = 0.0
        for m in range(sim.dim):
            for k in range(sim.dim):
                v2t1 = intor.int_request(sim, 'int_ux3g', g1, g2, t, k, index, m, m=ug, useM=True)
                v2t2 = intor.int_request(sim, 'int_ux2g', g1, g2, t, m, index, m=ug, useM=True) * r_taylor[k]
                v2t3 = intor.int_request(sim, 'int_ux2g', g1, g2, t, k, index, m=ug, useM=True) * r_taylor[m]
                v2t4 = intor.int_request(sim, 'int_uxg', g1, g2, t, index, m=ug, useM=True) * r_taylor[k] * r_taylor[m]
                v2_t += hess[m,k] * (v2t1 - v2t2 - v2t3 + v2t4)
        
        v1v = numpy.dot(grad, v1)
        v2v = numpy.einsum('ij,ji->', hess, v2)

        return self._clean(v0 + numpy.dot(grad, v1) + 0.5 * numpy.einsum('ij,ji->', hess, v2))
    
    def _int_vVg(self, g1, g2, t, vindex, r_taylor=None):
        '''
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
        sim = self.potential.sim
        if r_taylor is None:
            r_taylor = (- g1.width * g1.centre[t] + g2.width * g2.centre[t]) / (g1.width + g2.width)
        v_const = self.potential.evaluate(r_taylor)
        grad = self.potential.gradient(r_taylor)
        hess = self.potential.hessian(r_taylor)
        vg = intor.int_request(sim, 'int_vg', g1, g2, t, vindex)
        v0 = v_const * vg
        vxg = numpy.zeros(sim.dim, dtype=numpy.complex128)
        vx2g = numpy.zeros((sim.dim, sim.dim), dtype=numpy.complex128)
        for i in range(len(vxg)):
            vxg[i] = intor.int_request(sim, 'int_vxg', g1, g2, t, vindex, i, m=vg, useM=True)
            for j in range(len(vxg)):
                vx2g[i,j] = intor.int_request(sim, 'int_vx2g', g1, g2, t, vindex, i, j, m=vg, useM=True)
        v1 = vxg - vg * r_taylor
        v2 = vx2g - numpy.einsum('k,m->mk', r_taylor, vxg) - numpy.einsum('m,k->mk', r_taylor, vxg) + numpy.einsum('m,k->mk', r_taylor, r_taylor) * vg

        return self._clean(v0 + numpy.dot(grad, v1) + 0.5 * numpy.einsum('ij,ij->', hess, v2))
    
    def _int_vVxg(self, g1, g2, t, vindex, index, r_taylor=None):
        '''
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
        sim = self.potential.sim
        if r_taylor is None:
            r_taylor = (- g1.width * g1.centre[t] + g2.width * g2.centre[t]) / (g1.width + g2.width)
        v_const = self.potential.evaluate(r_taylor)
        grad = self.potential.gradient(r_taylor)
        hess = self.potential.hessian(r_taylor)
        vg = intor.int_request(sim, 'int_vg', g1, g2, t, vindex)
        vxg = intor.int_request(sim, 'int_vxg', g1, g2, t, vindex, index, m=vg, useM=True)
        vx2g = numpy.zeros(sim.dim, dtype=numpy.complex128)
        vx3g = numpy.zeros((sim.dim, sim.dim), dtype=numpy.complex128)
        for i in range(sim.dim):
            vx2g[i] = intor.int_request(sim, 'int_vx2g', g1, g2, t, vindex, i, index, m=vg, useM=True)
            for j in range(sim.dim):
                vx3g[i] = intor.int_request(sim, 'int_vx3g', g1, g2, t, vindex, i, j, index, m=vg, useM=True)
        v0 = v_const * vxg
        v1 = vx2g - vxg * r_taylor
        v2 = vx3g - numpy.einsum('k,m->mk', r_taylor, vx2g) - numpy.einsum('m,k->mk', r_taylor, vx2g) + numpy.einsum('m,k->mk', r_taylor, r_taylor) * vxg

        return self._clean(v0 + numpy.dot(grad, v1) + 0.5 * numpy.einsum('ij,ji->', hess, v2))


