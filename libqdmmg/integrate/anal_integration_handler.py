'''

Integration management manager

'''

import libqdmmg.integrate.anal_integrator as anal_intor

def int_gg(g1, g2, t):
    return anal_intor.int_gg(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t])

def int_gxg(g1, g2, t, index, m=0.0, useM=False):
    return anal_intor.int_gxg(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], index, m, useM)

def int_gx2g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    return anal_intor.int_gx2g_mixed(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], index1, index2, m, useM)

def int_gx2g_diag(g1, g2, t, index, m=0.0, useM=False):
    return anal_intor.int_gx2g_diag(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], index, m, useM)

def int_gx3g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    return anal_intor.int_gx3g_mixed(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], index1, index2, m, useM)

def int_gx3g_diag(g1, g2, t, index, m=0.0, useM=False):
    return anal_intor.int_gx3g_diag(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], index, m, useM)


def int_ug(g1, g2, t):
    return anal_intor.int_ug(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t])

def int_uxg(g1, g2, t, index, m=0.0, useM=False):
    return anal_intor.int_uxg(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], index, m, useM)

def int_ux2g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    return anal_intor.int_ux2g_mixed(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], index1, index2, m, useM)

def int_ux2g_diag(g1, g2, t, index, m=0.0, useM=False):
    return anal_intor.int_ux2g_diag(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], index, m, useM)

def int_ux3g_mixed(g1, g2, t, index1, index2, m=0.0, useM=False):
    return anal_intor.int_ux3g_mixed(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], index1, index2, m, useM)

def int_ux3g_diag(g1, g2, t, index, m=0.0, useM=False):
    return anal_intor.int_ux3g_diag(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], index, m, useM)


def int_vg(g1, g2, t, vindex):
    return anal_intor.int_vg(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], vindex)

def int_vxg(g1, g2, t, vindex, index, m=0.0, useM=False):
    return anal_intor.int_vxg(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], vindex, index, m, useM)

def int_vx2g_mixed(g1, g2, t, vindex, index1, index2, m=0.0, useM=False):
    return anal_intor.int_vx2g_mixed(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], vindex, index1, index2, m, useM)

def int_vx2g_diag(g1, g2, t, vindex, index, m=0.0, useM=False):
    return anal_intor.int_vx2g_diag(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], vindex, index, m, useM)

def int_vx3g_mixed(g1, g2, t, vindex, index1, index2, m=0.0, useM=False):
    return anal_intor.int_vx3g_mixed(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], vindex, index1, index2, m, useM)

def int_vx3g_diag(g1, g2, t, vindex, index, m=0.0, useM=False):
    return anal_intor.int_vx3g_diag(g1.width, g2.width, g1.centre[t], g2.centre[t], g1.momentum[t], g2.momentum[t], g1.phase[t], g2.phase[t], vindex, index, m, useM)




