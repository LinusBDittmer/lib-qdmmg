'''

Author: Linus Bjarne Dittmer


This script functions as a wrapper to handle analytical integration. It unpacks the received data in the type of Gaussians and timesteps for ease of caching.

'''

import numpy
import libqdmmg.integrate.atom_integrator as atom_intor

def get_gaussian_info(g, t):
    '''
    Extracts the centre, momentum and phase from the provided gaussian. This is relevant as a seamless interface between proper and interpolated timesteps. The function automatically resolves to the correct information retrieval method rby checking if an integer (or a float reasonably close to one) is provided in t or if t is somewhere between timesteps.

    Attributes
    ----------
    g : libqdmmg.general.gaussian.Gaussian
        The gaussian
    t : int or float
        The time or timestep.

    Returns
    -------
    gw : tuple
        The width of the Gaussian.
    gx : tuple
        The centre of the Gaussian at time(step) t.
    gp : tuple
        The momentum of the Gaussian at time(step) t.
    gg : float
        The phase of the Gaussian at time(step) t.
    '''
    if abs(round(t)-t) < 10**-5:
        t = int(round(t))
        return tuple(g.width), tuple(g.centre[t]), tuple(g.momentum[t]), g.phase[t]
    gx, gp, gg = g.interpolate(t)
    return tuple(g.width), gx, gp, gg

def int_gg(g1, g2, t):
    '''
    Calculates the integral of g1 * g2 over all of space.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    g1w, g1x, g1p, g1g = get_gaussian_info(g1, t)
    g2w, g2x, g2p, g2g = get_gaussian_info(g2, t)
    i = atom_intor.int_gg(g1w, g2w, g1x, g2x, g1p, g2p, g1g, g2g)
    if numpy.isnan(i) or numpy.isinf(i):
        return 0.0
    return i

def int_gxg(g1, g2, t, index, m=0.0, useM=False):
    '''
    Calculates the integral of g1 * x_(index) * g2 over all of space.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index : int
        Index of the direction along which the linear function is constructed.
    m : complex128, optional
        The value of the integral of g1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of g1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    g1w, g1x, g1p, g1g = get_gaussian_info(g1, t)
    g2w, g2x, g2p, g2g = get_gaussian_info(g2, t)
    i = atom_intor.int_gxg(g1w, g2w, g1x, g2x, g1p, g2p, g1g, g2g, index, m, useM)
    if numpy.isnan(i) or numpy.isinf(i):
        return 0.0
    return i


def int_gx2g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    '''
    Calculates the integral of g1 * x_(index1) * x_(index2) * g2 over all of space. Note that index1 != index2, refer to int_gx2g_diag otherwise

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index1 : int
        Index of the direction along which the first linear part of the quadratic polynomial is constructed.
    index2 : int
        Index of the direction along which the second linear part of the quadratic polynomial is constructed.
    m : complex128, optional
        The value of the integral of g1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of g1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    g1w, g1x, g1p, g1g = get_gaussian_info(g1, t)
    g2w, g2x, g2p, g2g = get_gaussian_info(g2, t)
    i = atom_intor.int_gx2g_mixed(g1w, g2w, g1x, g2x, g1p, g2p, g1g, g2g, index1, index2, m, useM)
    if numpy.isnan(i) or numpy.isinf(i):
        return 0.0
    return i


def int_gx2g_diag(g1, g2, t, index, m=0.0, useM=False):
    '''
    Calculates the integral of g1 * x_(index)^2 * g2 over all of space.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index : int
        Index of the direction along which the quadratic function is constructed.
    m : complex128, optional
        The value of the integral of g1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of g1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    g1w, g1x, g1p, g1g = get_gaussian_info(g1, t)
    g2w, g2x, g2p, g2g = get_gaussian_info(g2, t)
    i = atom_intor.int_gx2g_diag(g1w, g2w, g1x, g2x, g1p, g2p, g1g, g2g, index, m, useM)
    if numpy.isnan(i) or numpy.isinf(i):
        return 0.0
    return i


def int_gx3g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    '''
    Calculates the integral of g1 * x_(index1) * x_(index2)^2 * g2 over all of space. Note that index1 != index2, refer to int_gx3g_diag otherwise.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index1 : int
        Index of the direction along which the linear part of the cubic polynomial is constructed.
    index2 : int
        Index of the direction along which the quadratic part of the cubic polynomial is constructed.
    m : complex128, optional
        The value of the integral of g1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of g1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    g1w, g1x, g1p, g1g = get_gaussian_info(g1, t)
    g2w, g2x, g2p, g2g = get_gaussian_info(g2, t)
    i = atom_intor.int_gx3g_mixed(g1w, g2w, g1x, g2x, g1p, g2p, g1g, g2g, index1, index2, m, useM)
    if numpy.isnan(i) or numpy.isinf(i):
        return 0.0
    return i


def int_gx3g_diag(g1, g2, t, index, m=0.0, useM=False):
    '''
    Calculates the integral of g1 * x_(index)^3 * g2 over all of space.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index : int
        Index of the direction along which the cubic function is constructed.
    m : complex128, optional
        The value of the integral of g1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of g1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    g1w, g1x, g1p, g1g = get_gaussian_info(g1, t)
    g2w, g2x, g2p, g2g = get_gaussian_info(g2, t)
    i = atom_intor.int_gx3g_diag(g1w, g2w, g1x, g2x, g1p, g2p, g1g, g2g, index, m, useM)
    if numpy.isnan(i) or numpy.isinf(i):
        return 0.0
    return i

def int_gx3g_offdiag(g1, g2, t, index1, index2, index3, m=0.0, useM=False):
    '''
    Calculates the integral of g1 * x_(index1) * x_(index2) * x_(index3) * g2 over all of space. Note that index1 != index2, refer to int_gx3g_diag otherwise.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index1 : int
        Index of the direction along which the first linear part of the cubic polynomial is constructed.
    index2 : int
        Index of the direction along which the second linear part of the cubic polynomial is constructed.
    index3 : int
        Index of the direction along which the third linear part of the cubic polynomial is constructed.
    m : complex128, optional
        The value of the integral of g1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of g1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    g1w, g1x, g1p, g1g = get_gaussian_info(g1, t)
    g2w, g2x, g2p, g2g = get_gaussian_info(g2, t)
    i = atom_intor.int_gx3g_offdiag(g1w, g2w, g1x, g2x, g1p, g2p, g1g, g2g, index1, index2, index3, m, useM)
    if numpy.isnan(i) or numpy.isinf(i):
        return 0.0
    return i

def int_ug(g1, g2, t):
    '''
    Calculates the integral of u1 * g2 over all of space, where u1 is the u dual function of tuple(g1.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_ug(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t])
    if numpy.isnan(i):
        return 0.0
    return i

def int_uxg(g1, g2, t, index, m=0.0, useM=False):
    '''
    Calculates the integral of u1 * x_(index) * g2 over all of space, where u1 is the u dual function of tuple(g1.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index : int
        Index of the direction along which the linear function is constructed.
    m : complex128, optional
        The value of the integral of u1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of u1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_uxg(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], index, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i

def int_ux2g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    '''
    Calculates the integral of u1 * x_(index1) * x_(index2) * g2 over all of space, where u1 is the u dual function of tuple(g1. Note that index1 != index2, refer to int_ux2g_diag otherwise.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index1 : int
        Index of the direction along which the first linear part of the quadratic polynomial is constructed.
    m : complex128, optional
        The value of the integral of u1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of u1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_ux2g_mixed(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], index1, index2, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i

def int_ux2g_diag(g1, g2, t, index, m=0.0, useM=False):
    '''
    Calculates the integral of u1 * x_(index)^2 * g2 over all of space, where u1 is the u dual function of tuple(g1.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index : int
        Index of the direction along which the quadratic function is constructed.
    m : complex128, optional
        The value of the integral of u1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of u1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_ux2g_diag(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], index, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i

def int_ux3g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    '''
    Calculates the integral of u1 * x_(index1) * x_(index2)^2 * g2 over all of space, where u1 is the u dual function of tuple(g1.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index1 : int
        Index of the direction along which the linear part of the cubic polynomial is constructed.
    index2 : int
        Index of the direction along which the quadratic part of the cubic polynomial is constructed.
    m : complex128, optional
        The value of the integral of u1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of u1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_ux3g_mixed(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], index1, index2, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i

def int_ux3g_diag(g1, g2, t, index, m=0.0, useM=False):
    '''
    Calculates the integral of u1 * x_(index)^3 * g2 over all of space, where u1 is the u dual function of tuple(g1.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index : int
        Index of the direction along which the cubic function is constructed.
    m : complex128, optional
        The value of the integral of u1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of u1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_ux3g_diag(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], index, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i

def int_ux3g_offdiag(g1, g2, t, index1, index2, index3, m=0.0, useM=False):
    '''
    Calculates the integral of u1 * x_(index1) * x_(index2) * x_(index3) * g2 over all of space, where u1 is the u dual function of tuple(g1.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index1 : int
        Index of the direction along which the first linear part of the cubic polynomial is constructed.
    index2 : int
        Index of the direction along which the second linear part of the cubic polynomial is constructed.
    index3 : int
        Index of the direction along which the third linear part of the cubic polynomial is constructed.
    m : complex128, optional
        The value of the integral of u1 * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of u1 * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_ux3g_offdiag(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], index1, index2, index3, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i

def int_vg(g1, g2, t, vindex):
    '''
    Calculates the integral of v1_(vindex) * g2 over all of space, where v1 is the v dual function of g1 with vindex as direction index.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_vg(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], vindex)
    if numpy.isnan(i):
        return 0.0
    return i

def int_vxg(g1, g2, t, vindex, index, m=0.0, useM=False):
    '''
    Calculates the integral of v1_(vindex) * x_(index) * g2 over all of space, where v1 is the v dual function of g1 with vindex as direction index.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index : int
        Index of the direction along which the linear function is constructed.
    m : complex128, optional
        The value of the integral of v1_(vindex) * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of v1_(vindex) * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_vxg(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], vindex, index, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i

def int_vx2g_mixed(g1, g2, t, vindex, index1, index2, m=0.0, useM=False):
    '''
    Calculates the integral of v1_(vindex) * x_(index1) * x_(index2) * g2 over all of space, where v1 is the v dual function of g1 with vindex as direction index. Note that index1 != index2, refer to int_vx2g_diag otherwise.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index1 : int
        Index of the direction along which the first linear part of the quadratic function is constructed.
    index2 : int
        Index of the direction along which the second linear part of the quadratic function is constructed.
    m : complex128, optional
        The value of the integral of v1_(vindex) * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of v1_(vindex) * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_vx2g_mixed(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], vindex, index1, index2, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i

def int_vx2g_diag(g1, g2, t, vindex, index, m=0.0, useM=False):
    '''
    Calculates the integral of v1_(vindex) * x_(index)^2 * g2 over all of space, where v1 is the v dual function of g1 with vindex as direction index.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index : int
        Index of the direction along which the quadratic function is constructed.
    m : complex128, optional
        The value of the integral of v1_(vindex) * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of v1_(vindex) * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_vx2g_diag(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], vindex, index, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i

def int_vx3g_mixed(g1, g2, t, vindex, index1, index2, m=0.0, useM=False):
    '''
    Calculates the integral of v1_(vindex) * x_(index1) * x_(index2)^2 * g2 over all of space, where v1 is the v dual function of g1 with vindex as direction index.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index1 : int
        Index of the direction along which the linear part of the cubic polynomial is constructed.
    index2 : int
        Index of the direction along which the quadratic part of the cubic polynomial is constructed.
    m : complex128, optional
        The value of the integral of v1_(vindex) * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of v1_(vindex) * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_vx3g_mixed(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], vindex, index1, index2, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i

def int_vx3g_diag(g1, g2, t, vindex, index, m=0.0, useM=False):
    '''
    Calculates the integral of v1_(vindex) * x_(index)^3 * g2 over all of space, where v1 is the v dual function of g1 with vindex as direction index.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index : int
        Index of the direction along which the cubic function is constructed.
    m : complex128, optional
        The value of the integral of v1_(vindex) * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of v1_(vindex) * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_vx3g_diag(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], vindex, index, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i

def int_vx3g_offdiag(g1, g2, t, vindex, index1, index2, index3, m=0.0, useM=False):
    '''
    Calculates the integral of v1_(vindex) * x_(index1) * x_(index2) * x_(index3) * g2 over all of space, where v1 is the v dual function of g1 with vindex as direction index.

    Parameters
    ----------
    g1 : libqdmmg.general.gaussian.Gaussian
        The first Gaussian
    g2 : libqdmmg.general.gaussian.Gaussian
        The second Gaussian
    t : int
        Timestep index
    index1 : int
        Index of the direction along which the first linear part of the cubic polynomial is constructed.
    index2 : int
        Index of the direction along which the second linear part of the cubic polynomial is constructed.
    index3 : int
        Index of the direction along which the third linear part of the cubic polynomial is constructed.
    m : complex128, optional
        The value of the integral of v1_(vindex) * g2. Default is 0.0
    useM : bool, optional
        Whether the value of the integral of v1_(vindex) * g2 should be taken from the given parameter. If false, it is computed in runtime. Default False

    Returns
    -------
    int_val : complex128
        The value of the integral.
    '''
    i = atom_intor.int_vx3g_offdiag(tuple(g1.width), tuple(g2.width), tuple(g1.centre[t]), tuple(g2.centre[t]), tuple(g1.momentum[t]), tuple(g2.momentum[t]), g1.phase[t], g2.phase[t], vindex, index1, index2, index3, m, useM)
    if numpy.isnan(i):
        return 0.0
    return i


