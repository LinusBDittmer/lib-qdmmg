'''

Class that contains all the equations of motion in abstract form.

author: Linus Bjarne Dittmer

'''
import numpy
import libqdmmg.integrate as intor

def eom_init_centre(sim, pot, g, t):
    dx_bar = g.momentum[t] / pot.reduced_mass
    return dx_bar

def eom_init_momentum(sim, pot, g, t):
    dp = numpy.diag(pot.hessian(g.centre[t])) * g.centre[t] - pot.gradient(g.centre[t]) - 4 * g.width * g.width * g.centre[t] / pot.reduced_mass
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
    return eom_init_centre(sim, pot, g, t)

def eom_momentum(sim, pot, g, t):
    return eom_init_momentum(sim, pot, g, t)

def eom_phase(sim, pot, g, t):
    return eom_init_phase(sim, pot, g, t)

def eom_coefficient(sim, pot, g, t):
    wp = sim.previous_wavefunction
    logger = sim.logger
    a_coeff = sim.active_coeffs[t]
    pot_intor = pot.gen_potential_integrator()
    ovlp_gg = intor.int_request(sim, 'int_ovlp_gg', g, g, t)
    ovlp_gw = intor.int_request(sim, 'int_ovlp_gw', g, wp, t)
    dovlp_gg = intor.int_request(sim, 'int_dovlp_gg', g, g, t)
    dovlp_gw = intor.int_request(sim, 'int_dovlp_gw', g, wp, t)
    dovlp_wg = intor.int_request(sim, 'int_dovlp_wg', wp, g, t)
    dovlp_ww = intor.int_request(sim, 'int_dovlp_ww', wp, wp, t)

    e_g = g.energy_tot(t)
    e_wp = wp.energy_tot(t)
    e_coupling = intor.int_request(sim, 'int_kinetic_gw', g, wp, t)
    e_coupling += pot_intor.int_request('int_gVw', g, wp, t)
    b_coeff = numpy.sqrt(1 - a_coeff*a_coeff)
    ab_ratio = a_coeff / b_coeff

    a_term = dovlp_gg + dovlp_wg + 1j*(e_g - ab_ratio * e_coupling.conj())
    b_term = dovlp_gw + dovlp_ww + 1j*(e_coupling - ab_ratio * e_wp)
    inv_term = 2 * ab_ratio * ovlp_gw.real - ovlp_gg - ab_ratio*ab_ratio

    logger.debug3(f"A Term      : {a_term}")
    logger.debug3(f"B Term      : {b_term}")
    logger.debug3(f"A           : {a_coeff}")
    logger.debug3(f"B           : {b_coeff}")
    logger.debug3(f"Inv Term    : {inv_term}")
    logger.debug3(f"AB Ratio    : {ab_ratio}")

    d_coeff = ((a_coeff * a_term + b_coeff * b_term) / inv_term).real
    logger.debug3(f"Coefficient differential : {d_coeff}")
    return d_coeff
