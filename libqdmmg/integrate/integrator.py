'''

Integration management manager

'''

import numpy
from functools import reduce

def int_gg(g1, g2, t):
    # parallelise with OpenMP in C
    cvec = g1.width + g2.width
    cvec_inv = reduce(numpy.divide, (numpy.ones(cvec.shape), cvec))
    vvec = 2 * reduce(numpy.multiply, (g1.width, g1.centre[t])) + 2 * reduce(numpy.multiply, (g2.width, g2.centre[t])) + 1j*(g2.momentum[t] - g1.momentum[t])
    vvec2 = reduce(numpy.multiply, (vvec, vvec))

    c1 = -reduce(numpy.dot, (g1.width, g1.centre[t]*g1.centre[t])) - reduce(numpy.dot, (g2.width, g2.centre[t]*g2.centre[t])) + 1j*(g2.phase[t] - g1.phase[t]) 
    c2 = 0.25 * reduce(numpy.dot, (cvec_inv, vvec2))
    ca = numpy.power(numpy.prod(cvec_inv), 0.5)
    return numpy.exp(c1 + c2) * ca * (3.141592654)**(len(cvec) * 0.5)

def int_gxg(g1, g2, t, index, m=0.0, useM=False):
    # parallelise with OpenMP in C
    cval = g1.width[index] + g2.width[index]
    vval = 2 * g1.width[index] * g1.centre[t,index] + 2 * g2.width[index] * g2.centre[t,index] + 1j*(g2.momentum[t,index] - g1.momentum[t,index])
    if not useM:
        m = int_gg(g1, g2, t)
    return 0.5 * m * vval / cval

def int_gx2g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    return 0

def int_gx2g_diag(g1, g2, t, index, m=0.0, useM=False):
    return 0

def int_gx3g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    return 0

def int_gx3g_diag(g1, g2, t, index, m=0.0, useM=False):
    return 0


def int_ug(g1, g2, t):
    return 0

def int_uxg(g1, g2, t, index, m=0.0, useM=False):
    return 0

def int_ux2g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    return 0

def int_ux2g_diag(g1, g2, t, index, m=0.0, useM=False):
    return 0

def int_ux3g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    return 0

def int_ux3g_diag(g1, g2, t, index, m=0.0, useM=False):
    return 0


def int_vg(g1, g2, t, vindex):
    return 0

def int_vxg(g1, g2, t, vindex, index, m=0.0, useM=False):
    return 0

def int_vx2g_mixed(g1, g2, t, vindex, index1, index2, m=0.0, useM=False):
    return 0

def int_vx2g_diag(g1, g2, t, vindex, index, m=0.0, useM=False):
    return 0

def int_vx3g_mixed(g1, g2, t, vindex, index1, index2, m=0.0, useM=False):
    return 0

def int_vx3g_diag(g1, g2, t, vindex, index, m=0.0, useM=False):
    return 0
