'''

Class that contains all the equations of motion in abstract form.

author: Linus Bjarne Dittmer

'''
import numpy

def eom_init_centre(sim, pot, g, t):
    dx_bar = g.momentum[t] / pot.reduced_mass
    return dx_bar

def eom_init_momentum(sim, pot, g, t):
    dp = numpy.diag(pot.hessian(g.centre[t])) - pot.gradient(g.centre[t]) - 4 * g.width * g.width * g.centre[t] / pot.reduced_mass
    return dp

def eom_init_phase(sim, pot, g, t):
    hess = pot.hessian(g.centre[t])
    grad = pot.gradient(g.centre[t])
    dg0_vec = 4 * g.width * g.width * g.centre[t] * g.centre[t] + 2 * g.width - g.momentum[t] * g.momentum[t] - 2 * g.width
    dg0_vec *= 0.5 / pot.reduced_mass
    dg0_vec -= grad * g.centre[t] + 0.25 * numpy.diag(hess) / g.width
    dg0 = numpy.sum(dg0_vec)
    dg1 = - 0.5 * numpy.einsum('m,mk,k->', g.centre[t], hess, g.centre[t])
    dg2 = - pot.evaluate(g.centre[t])
    return dg0 + dg1 + dg2

def eom_centre(sim, pot, g, t):
    return 0

def eom_init_momentum(sim, pot, g, t):
    return 0

def eom_init_phase(sim, pot, g, t):
    return 0
