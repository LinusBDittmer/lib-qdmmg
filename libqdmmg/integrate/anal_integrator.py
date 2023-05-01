'''

Author: Linus Bjarne Dittmer

This script contains the explicit formulas for the analytical integrals described in anal_integration_handler.py
'''

import numpy
from functools import reduce


def int_gg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase):
    '''
    This function calculates the integral of the product of two gaussians over all space.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase

    Returns
    -------
    integral : complex128
        Integral of g1*g2 over all of space.

    '''
    # parallelise with OpenMP in C
    cvec = g1width + g2width
    cvec_inv = 1 / cvec
    vvec = 2 * reduce(numpy.multiply, (g1width, g1centre)) + 2 * reduce(numpy.multiply, (g2width, g2centre)) + 1j*(g2momentum - g1momentum)
    vvec2 = reduce(numpy.multiply, (vvec, vvec))

    c1 = -reduce(numpy.dot, (g1width, g1centre*g1centre)) - reduce(numpy.dot, (g2width, g2centre*g2centre)) + 1j*(g2phase - g1phase) 
    c2 = 0.25 * reduce(numpy.dot, (cvec_inv, vvec2))
    ca = numpy.power(numpy.prod(cvec_inv), 0.5)
    return numpy.exp(c1 + c2) * ca * (3.141592654)**(len(cvec) * 0.5)

def int_gxg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, index, m=0.0, useM=False):
    '''
    This function calculates the integral of the function g1 * x_(index) * g2 over all of space.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    index : int
        Index of the coordinate along which the linear function is constructed.
    m : complex128, optional
        Value of the integral of g1 * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for g1 * g2 given as a parameter is to be used. If False, the value is recalculated. Default False

    Returns
    -------
    integral : complex128
        Integral of g1 * x_(index) * g2 over all of space.

    '''
    # parallelise with OpenMP in C
    cval = g1width[index] + g2width[index]
    vval = 2 * g1width[index] * g1centre[index] + 2 * g2width[index] * g2centre[index] + 1j*(g2momentum[index] - g1momentum[index])
    if not useM:
        m = int_gg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase)
    return 0.5 * m * vval / cval

def int_gx2g_mixed(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, index1, index2, m=0.0, useM=False):
    '''
    This function calculates the integral of the function g1 * x_(index1) * x_(index2) * g2 over all of space. Note that index1 != index2. In that case, use int_gx2g_diag instead.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    index1 : int
        Index of the coordinate along which the first linear part of the mixed quadratic polynomial is constructed.
    index2 : int
        Index of the coordinate along which the second linear part of the mixed quadratic polynomial is constructed.
    m : complex128, optional
        Value of the integral of g1 * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for g1 * g2 given as a parameter is to be used. If False, the value is recalculated. Default False

    Returns
    -------
    integral : complex128
        Integral of g1 * x_(index1) * x_(index2) * g2 over all of space.

    '''
    v1 = 2 * g1width[index1] * g1centre[index1] + 2 * g2width[index1] * g2centre[index1] + 1j*(g2momentum[index1] - g1momentum[index1])
    v2 = 2 * g1width[index2] * g1centre[index2] + 2 * g2width[index2] * g2centre[index2] + 1j*(g2momentum[index2] - g1momentum[index2])
    c1 = 1.0 / (g1width[index1] + g2width[index1])
    c2 = 1.0 / (g1width[index2] + g2width[index2])
    if not useM:
        m = int_gg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase)
    return (0.25 * v1 * v2 * c1 * c2 + 0.5 * c1 + 0.5 * c2) * m

def int_gx2g_diag(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, index, m=0.0, useM=False):
    '''
    This function calculates the integral of the function g1 * x_(index)^2 * g2 over all of space.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    index : int
        Index of the coordinate along which the quadratic polynomial is constructed.
    m : complex128, optional
        Value of the integral of g1 * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for g1 * g2 given as a parameter is to be used. If False, the value is recalculated. Default False

    Returns
    -------
    integral : complex128
        Integral of g1 * x_(index)^2 * g2 over all of space.
    
    '''
    v = 2 * g1width[index] * g1centre[index] + 2 * g2width[index] * g2centre[index] + 1j*(g2momentum[index] - g1momentum[index])
    c = 1.0 / (g1width[index] + g2width[index])
    if not useM:
       m = int_gg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase)
    return (0.25 * v * v * c * c + 0.5 * c) * m

def int_gx3g_mixed(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, index1, index2, m=0.0, useM=False):
    '''
    This function calculates the integral of the function g1 * x_(index1) * x_(index2)^2 * g2 over all of space. Note that index1 != index2 for this function, refer to int_gx3g_diag otherwise.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    index1 : int
        Index of the coordinate along which the linear part of the cubic polynomial is constructed.
    index2 : int
        Index of the coordinate along which the quadratic part of the cubic polynomial is constructed.
    m : complex128, optional
        Value of the integral of g1 * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for g1 * g2 given as a parameter is to be used. If False, the value is recalculated. Default False

    Returns
    -------
    integral : complex128
        Integral of g1 * x_(index1) * x_(index2)^2 * g2 over all of space.

    '''
    v1 = 2 * g1width[index1] * g1centre[index1] + 2 * g2width[index1] * g2centre[index1] + 1j*(g2momentum[index1] - g1momentum[index1])
    v2 = 2 * g1width[index2] * g1centre[index2] + 2 * g2width[index2] * g2centre[index2] + 1j*(g2momentum[index2] - g1momentum[index2])
    c1 = 1.0 / (g1width[index1] + g2width[index1])
    c2 = 1.0 / (g1width[index2] + g2width[index2])
    if not useM:
        m = int_gg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase)
    return 0.25 * (v2*c1*c2 + 2 * v2*c2*c2 + 0.5*v1*v2*v2*c1*c2*c2) * m

def int_gx3g_diag(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, index, m=0.0, useM=False):
    '''
    This function calculates the integral of g1 * x_(index)^3 * g2 over all of space.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    index : int
        Index of the coordinate along which the cubic function is constructed.
    m : complex128, optional
        Value of the integral of g1 * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for g1 * g2 given as a parameter is to be used. If False, the value is recalculated. Default False

    Returns
    -------
    integral : complex128
        Integral of g1 * x_(index)^3 * g2 over all of space.

    '''
    v = 2 * g1width[index] * g1centre[index] + 2 * g2width[index] * g2centre[index] + 1j*(g2momentum[index] - g1momentum[index])
    c = 1.0 / (g1width[index] + g2width[index])
    if not useM:
        m = int_gg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase)
    return 0.25 * (3*v*c*c + 0.25*v*v*v*c*c*c) * m


def int_ug(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase):
    '''
    This function calculates the integral of the function u1 * g2 over all of space where u1 is the u dual function to g1.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase

    Returns
    -------
    integral : complex128
        Integral of u1 * g2 over all of space.
    '''
    cvec2 = numpy.linalg.norm(g1width) - g1width
    cvec_inv = 1.0 / (cvec2 + g2width)
    vvec1_prec = 2 * cvec2 * g1centre + 2 * g2width * g2centre + 1j * g2momentum
    vvec1 = 0.25 * reduce(numpy.dot, (cvec_inv, vvec1_prec*vvec1_prec))
    vvec2 = reduce(numpy.dot, (cvec2, g1centre*g1centre)) + reduce(numpy.dot, (g2width, g2centre*g2centre))
    ep = vvec1 - vvec2 + 1j * g2phase
    factor1 = numpy.sqrt(reduce(numpy.prod, cvec_inv))
    return 2 * numpy.linalg.norm(g1width)**(len(g1width)*0.5) * factor1 * numpy.exp(ep)


def int_uxg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, index, m=0.0, useM=False):
    '''
    This function calculates the integral of the function u1 * x_(index) * g2 over all of space where u1 is the u dual function to g1.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    index : int
        Index of the coordinate along which the linear function is constructed.
    m : complex128, optional
        Value of the integral of u1 * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for u1 * g2 given as a parameter is to be used. If False, the value is recalculated. Default False


    Returns
    -------
    integral : complex128
        Integral of u1 * x_(index) * g2 over all of space.

    '''
    c2 = numpy.linalg.norm(g1width) - g1width[index]
    c_inv = 1.0 / (c2 + g2width[index])
    p = 2 * c2 * g1centre[index] + 2 * g2width[index] * g2centre[index] + 1j * g2momentum[index]
    if not useM:
        m = int_ug(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase)
    return 0.5 * p * c_inv * m

def int_ux2g_mixed(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, index1, index2, m=0.0, useM=False):
    '''
    This function calculates the integral of the function u1 * x_(index1) * x_(index2) * g2 over all of space where u1 is the u dual function to g1. Note that index1 != index2, otherwise refer to int_ux2g_diag

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    index1 : int
        Index of the coordinate along which the first linear part of the quadratic polynomial is constructed.
    index2 : int
        Index of the coordinate along which the second linear part of the quadratic polynomial is constructed.
    m : complex128, optional
        Value of the integral of u1 * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for u1 * g2 given as a parameter is to be used. If False, the value is recalculated. Default False


    Returns
    -------
    integral : complex128
        Integral of u1 * x_(index1) * x_(index2) * g2 over all of space.

    '''
    c1 = numpy.linalg.norm(g1width) - g1width[index1]
    c_inv1 = 1.0 / (c1 + g2width[index1])
    c2 = c1 + g1width[index1] - g1width[index2]
    c_inv2 = 1.0 / (c2 + g2width[index2])
    p1 = 2*c1*g1centre[index1] + 2*g2width[index1]*g2centre[index1] + 1j*g2momentum[index1]
    p2 = 2*c2*g1centre[index2] + 2*g2width[index2]*g2centre[index2] + 1j*g2momentum[index2]
    if not useM:
        m = int_ug(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase)
    return (0.25*p1*p2*c_inv1*c_inv2 + 0.5*c_inv1 + 0.5*c_inv2) * m

def int_ux2g_diag(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, index, m=0.0, useM=False):
    '''
    This function calculates the integral of the function u1 * x_(index)^2 * g2 over all of space where u1 is the u dual function to g1.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    index : int
        Index of the coordinate along which the quadratic polynomial is constructed.
    m : complex128, optional
        Value of the integral of u1 * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for u1 * g2 given as a parameter is to be used. If False, the value is recalculated. Default False


    Returns
    -------
    integral : complex128
        Integral of u1 * x_(index)^2 * g2 over all of space.

    '''
    c = numpy.linalg.norm(g1width) - g1width[index]
    c_inv = 1.0 / (c + g2width[index])
    p = 2*c*g1centre[index] + 2*g2width[index]*g2centre[index] + 1j*g2momentum[index]
    if not useM:
        m = int_ug(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase)
    return (0.25*p*p*c*c + 0.5*c) * m

def int_ux3g_mixed(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, index1, index2, m=0.0, useM=False):
    '''
    This function calculates the integral of the function u1 * x_(index1) * x_(index2)^2 * g2 over all of space where u1 is the u dual function to g1. Note that index1 != index2, otherwise refer to int_ux3g_diag

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    index1 : int
        Index of the coordinate along which the linear part of the cubic polynomial is constructed.
    index2 : int
        Index of the coordinate along which the quadratic part of the cubic polynomial is constructed.
    m : complex128, optional
        Value of the integral of u1 * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for u1 * g2 given as a parameter is to be used. If False, the value is recalculated. Default False


    Returns
    -------
    integral : complex128
        Integral of u1 * x_(index1) * x_(index2)^2 * g2 over all of space.

    '''
    c1 = numpy.linalg.norm(g1width) - g1width[index1]
    c_inv1 = 1.0 / (c1 + g2width[index1])
    c2 = c1 + g1width[index1] - g1width[index2]
    c_inv2 = 1.0 / (c2 + g2width[index2])
    p1 = 2*c1*g1centre[index1] + 2*g2width[index1]*g2centre[index1] + 1j*g2momentum[index1]
    p2 = 2*c2*g1centre[index2] + 2*g2width[index2]*g2centre[index2] + 1j*g2momentum[index2]
    if not useM:
        m = int_ug(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase)
    return 0.25 * (p2*c_inv1*c_inv2 + 2*p2*c_inv2*c_inv2 + 0.5*p1*p2*p2*c_inv1*c_inv2*c_inv2) * m

def int_ux3g_diag(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, index, m=0.0, useM=False):
    '''
    This function calculates the integral of the function u1 * x_(index)^3 * g2 over all of space where u1 is the u dual function to g1.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    index : int
        Index of the coordinate along which the cubic polynomial is constructed.
    m : complex128, optional
        Value of the integral of u1 * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for u1 * g2 given as a parameter is to be used. If False, the value is recalculated. Default False


    Returns
    -------
    integral : complex128
        Integral of u1 * x_(index)^3 * g2 over all of space.

    '''
    c = numpy.linalg.norm(g1width) - g1width[index]
    c_inv = 1.0 / (c + g2width[index])
    p = 2*c*g1centre[index] + 2*g2width[index]*g2centre[index] + 1j*g2momentum[index]
    if not useM:
        m = int_ug(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase)
    return 0.25 * (3*p*c_inv*c_inv + 0.5 * p*p*p*c_inv*c_inv*c_inv) * m


def int_vg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, vindex):
    '''
    This function calculates the integral of the function v1_(vindex) * g2 over all of space where v1_(vindex) is the v dual function to g1 in the direction with index vindex.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    vindex : int
        Index of the direction in which the v dual function is to be constructed.

    Returns
    -------
    integral : complex128
        Integral of v1_(vindex) * g2 over all of space.
    '''
    cvec_inv = 1.0 / (g1width + g2width)
    prefactor1 = numpy.sqrt(reduce(numpy.prod, (g2width*cvec_inv)))
    p = 2 * g2width * g2centre - 2 * g1width * g1centre + 1j * g2momentum
    ep1 = reduce(numpy.dot, (g1width, g1centre*g1centre)) - reduce(numpy.dot, (g2width, g2centre*g2centre))
    ep2 = reduce(numpy.dot, (cvec_inv, p*p))
    return 2 * g1width[vindex] * prefactor1 * numpy.exp(ep1 + 0.25 * ep2 + 1j*g2phase)

def int_vxg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, vindex, index, m=0.0, useM=False):
    '''
    This function calculates the integral of the function v1_(vindex) * x_(index) * g2 over all of space where v1_(vindex) is the v dual function to g1 in the direction with index vindex.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    vindex : int
        Index of the direction in which the v dual function is to be constructed.
    index : int
        Index of the coordinate along which the linear polynomial is constructed.
    m : complex128, optional
        Value of the integral of v1_(vindex) * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for v1_(vindex) * g2 given as a parameter is to be used. If False, the value is recalculated. Default False


    Returns
    -------
    integral : complex128
        Integral of v1_(vindex) * x_(index) * g2 over all of space.
    '''
    c_inv = 1.0 / (g1width[index] + g2width[index])
    p = 2 * g1width[index] * g1centre[index] + 2 * g2width[index] * g2centre[index] + 1j * g2phase
    if not useM:
        m = int_vg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, vindex)
    return 0.5 * p * c_inv * m

def int_vx2g_mixed(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, vindex, index1, index2, m=0.0, useM=False):
    '''
    This function calculates the integral of the function v1_(vindex) * x_(index1) * x_(index2) * g2 over all of space where v1_(vindex) is the v dual function to g1 in the direction with index vindex. Note that index1 != index2, otherwise refer to int_vx2g_diag

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    vindex : int
        Index of the direction in which the v dual function is to be constructed.
    index1 : int
        Index of the coordinate along which the first linear part of the quadratic polynomial is constructed.
    index2 : int
        Index of the coordinate along which the second linear part of the quadratic polynomial is constructed.
    m : complex128, optional
        Value of the integral of v1_(vindex) * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for v1_(vindex) * g2 given as a parameter is to be used. If False, the value is recalculated. Default False

    Returns
    -------
    integral : complex128
        Integral of v1_(vindex) * x_(index1) * x_(index2) * g2 over all of space.
    '''
    c_inv1 = 1.0 / (g1width[index1] + g2width[index1])
    c_inv2 = 1.0 / (g1width[index2] + g2width[index2])
    p1 = 2*g1width[index1]*g1centre[index1] + 2*g2width[index1]*g2centre[index1] + 1j*g2phase
    p2 = 2*g1width[index2]*g1centre[index2] + 2*g2width[index2]*g2centre[index2] + 1j+g2phase
    if not useM:
        m = int_vg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, vindex)
    return (0.25*p1*p2*c_inv1*c_inv2 + 0.5*c_inv1 + 0.5*c_inv2) * m

def int_vx2g_diag(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, vindex, index, m=0.0, useM=False):
    '''
    This function calculates the integral of the function v1_(vindex) * x_(index)^2 * g2 over all of space where v1_(vindex) is the v dual function to g1 in the direction with index vindex.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    vindex : int
        Index of the direction in which the v dual function is to be constructed.
    index : int
        Index of the coordinate along which the quadratic polynomial is constructed.
    m : complex128, optional
        Value of the integral of v1_(vindex) * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for v1_(vindex) * g2 given as a parameter is to be used. If False, the value is recalculated. Default False


    Returns
    -------
    integral : complex128
        Integral of v1_(vindex) * x_(index)^2 * g2 over all of space.
    '''
    c_inv = 1.0 / (g1width[index] + g2width[index])
    p = 2*g1width[index]*g1centre[index] + 2*g2width[index]*g2centre[index] + 1j*g2phase
    if not useM:
        m = int_vg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, vindex)
    return (0.25*p*p*c_inv*c_inv + 0.5*c_inv) * m

def int_vx3g_mixed(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, vindex, index1, index2, m=0.0, useM=False):
    '''
    This function calculates the integral of the function v1_(vindex) * x_(index1) * x_(index2)^2 * g2 over all of space where v1_(vindex) is the v dual function to g1 in the direction with index vindex. Note that index1 != index2, otherwise refer to int_vx3g_diag.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    vindex : int
        Index of the direction in which the v dual function is to be constructed.
    index1 : int
        Index of the coordinate along which the linear part of the cubic polynomial is constructed.
    index2 : int
        Index of the coordinate along which the quadratic part of the cubic polynomial is constructed.
    m : complex128, optional
        Value of the integral of v1_(vindex) * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for v1_(vindex) * g2 given as a parameter is to be used. If False, the value is recalculated. Default False


    Returns
    -------
    integral : complex128
        Integral of v1_(vindex) * x_(index1) * x_(index2)^2 * g2 over all of space.
    '''
    c_inv1 = 1.0 / (g1width[index1] + g2width[index1])
    c_inv2 = 1.0 / (g1width[index2] + g2width[index2])
    p1 = 2*g1width[index1]*g1centre[index1] + 2*g2width[index1]*g2centre[index1] + 1j*g2phase
    p2 = 2*g1width[index2]*g1centre[index2] + 2*g2width[index2]*g2centre[index2] + 1j+g2phase
    if not useM:
        m = int_vg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, vindex)
    return 0.25 * (p2*c_inv1*c_inv2 + 2*p2*c_inv2*c_inv2 + 0.5*p1*p2*p2*c_inv1*c_inv2*c_inv2) * m

def int_vx3g_diag(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, vindex, index, m=0.0, useM=False):
    '''
    This function calculates the integral of the function v1_(vindex) * x_(index)^3 * g2 over all of space where v1_(vindex) is the v dual function to g1 in the direction with index vindex.

    Parameters
    ----------
    g1width : 1D ndarray
        Width coefficients of the first Gaussian.
    g2width : 1D ndarray
        Width coefficients of the second Gaussian.
    g1centre : 1D ndarray
        Centre of the first Gaussian.
    g2centre : 1D ndarray
        Centre of the second Gaussian.
    g1momentum : 1D ndarray
        Equivalent momentum of the first Gaussian. Note that the actual momentum is given by 1j*g1momentum
    g2momentum : 1D ndarray
        Equivalent momentum of the second Gaussian. Note that the actual momentum is given by 1j*g2momentum
    g1phase : float32
        Equivalent phase of the first Gaussian. Note that the actual phase is given by 1j*g1phase
    g2phase : float32
        Equivalent phase of the second Gaussian. Note that the actual phase is given by 1j*g2phase
    vindex : int
        Index of the direction in which the v dual function is to be constructed.
    index : int
        Index of the coordinate along which the cubic polynomial is constructed.
    m : complex128, optional
        Value of the integral of v1_(vindex) * g2. Default 0.0
    useM : bool, optional
        Whether the value for m and thus the integral for v1_(vindex) * g2 given as a parameter is to be used. If False, the value is recalculated. Default False


    Returns
    -------
    integral : complex128
        Integral of v1_(vindex) * x_(index)^3 * g2 over all of space.
    ''' 
    c_inv = 1.0 / (g1width[index] + g2width[index])
    p = 2*g1width[index]*g1centre[index] + 2*g2width[index]*g2centre[index] + 1j*g2phase
    if not useM:
        m = int_vg(g1width, g2width, g1centre, g2centre, g1momentum, g2momentum, g1phase, g2phase, vindex)
    return 0.25 * (3*p*c_inv*c_inv + 0.5*p*p*p*c_inv*c_inv*c_inv) * m
