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

def eom_centre(sim, pot, g1, t):
    return eom_init_centre(sim, pot, g1, t)
    wp = sim.previous_wavefunction
    wp_size = len(wp.gaussians)
    logger = sim.logger
    a_coeff = sim.active_coeffs[t]
    b_coeff = numpy.sqrt(1 - a_coeff*a_coeff)
    wp_coeff = wp.get_coeffs(t)
    wp_d_coeff = wp.get_d_coeffs(t)
    pot_intor = pot.gen_potential_integrator()

    # Simplification of potential electronic structure calls
    r_taylor = None
    r_taylor = g1.centre[t]

    # Construction of Tensors uxg, ux2g and semi-diagonal ux3g for each g in wp
    int_ug_tensor = numpy.zeros(wp_size, dtype=numpy.complex128)
    int_uxg_tensor = numpy.zeros((wp_size, sim.dim), dtype=numpy.complex128)
    int_ux2g_tensor = numpy.zeros((wp_size, sim.dim, sim.dim), dtype=numpy.complex128)
    # Only integrals of type u x y^2 g are relevant
    int_ux3g_tensor = numpy.zeros((wp_size, sim.dim, sim.dim), dtype=numpy.complex128)

    # Only potential integrals of type u V g and u x V g are relevant
    int_uVg_tensor = numpy.zeros(wp_size, dtype=numpy.complex128)
    int_uVxg_tensor = numpy.zeros((wp_size, sim.dim), dtype=numpy.complex128)
    
    # Tensors of wp parameters for easy indexing
    centre_tensor = numpy.zeros((wp_size, sim.dim))
    width_tensor = numpy.zeros((wp_size, sim.dim))
    momentum_tensor = numpy.zeros((wp_size, sim.dim))
    phase_tensor = numpy.zeros(wp_size)
    d_centre_tensor = numpy.zeros((wp_size, sim.dim))
    d_momentum_tensor = numpy.zeros((wp_size, sim.dim))
    d_phase_tensor = numpy.zeros(wp_size)

    # Construction of tensors
    for g, gaussian in enumerate(wp.gaussians):
        g2 = gaussian.copy()
        g2.momentum[t] -= g1.momentum[t]
        g2.phase[t] -= g1.phase[t]
        
        int_ug_tensor[g] = intor.int_request(sim, 'int_ug', g1, g2, t)
        int_uVg_tensor[g] = pot_intor.int_request('int_uVg', g1, g2, t, r_taylor=r_taylor)

        width_tensor[g] = gaussian.width
        centre_tensor[g] = gaussian.centre[t]
        momentum_tensor[g] = gaussian.momentum[t]
        phase_tensor[g] = gaussian.phase[t]
        d_centre_tensor[g] = gaussian.d_centre[t]
        d_momentum_tensor[g] = gaussian.d_momentum[t]
        d_phase_tensor[g] = gaussian.d_phase[t]

        for i in range(sim.dim):
            int_uxg_tensor[g,i] = intor.int_request(sim, 'int_uxg', g1, g2, t, i, m=int_ug_tensor[g], useM=True)
            int_uVxg_tensor[g,i] = pot_intor.int_request('int_uVxg', g1, g2, t, i, r_taylor=r_taylor)
            for j in range(sim.dim):
                int_ux2g_tensor[g,i,j] = intor.int_request(sim, 'int_ux2g', g1, g2, t, i, j, m=int_ug_tensor[g], useM=True)
                int_ux3g_tensor[g,i,j] = intor.int_request(sim, 'int_ux3g', g1, g2, t, i, j, j, m=int_ug_tensor[g], useM=True)
    
    '''    
    # Combined Tensors for even easier indexing
    int_linear_shift = numpy.zeros((wp_size, sim.dim), dtype=numpy.complex128)
    int_quadratic_shift = numpy.zeros((wp_size, sim.dim, sim.dim), dtype=numpy.complex128)
    int_cubic_shift = numpy.zeros((wp_size, sim.dim, sim.dim), dtype=numpy.complex128)
    
    for n in range(wp_size):
        for l in range(sim.dim):
            int_linear_shift[n,l] = int_uxg_tensor[n,l] - g1.centre[t,l] * int_ug_tensor[n]
            for k in range(sim.dim):
                int_quadratic_shift[n,l,k] = int_ux2g_tensor[n,l,k] - g1.centre[t,l] * int_uxg_tensor[n,k] - centre_tensor[n,k] * int_uxg_tensor[n,l] + centre_tensor[n,k] * g1.centre[t,l] * int_ug_tensor[n]
                int_cubic_shift[n,l,k] = int_ux3g_tensor[n,l,k] - 2*centre_tensor[n,k]*int_ux2g_tensor[n,l,k] + centre_tensor[n,k]*centre_tensor[n,k]*int_uxg_tensor[n,l] - g1.centre[t,l]*(int_ux2g_tensor[n,k,k] - 2*centre_tensor[n,k]*int_uxg_tensor[n,k] + centre_tensor[n,k]*centre_tensor[n,k]*int_ug_tensor[n])
    '''

    # Self-contribution term
    self_cont = g1.momentum[t] / (a_coeff * pot.reduced_mass)

    # Correctional contributions

    # External contribution 1: Correction by crossover with previous time derivative
    ext_cont1 = numpy.zeros(sim.dim, dtype=numpy.complex128)
    ext_cont1 += 0.25 * numpy.einsum('l,n,nl->l', 1/g1.width, wp_d_coeff, int_uxg_tensor)
    ext_cont1 -= 0.25 * numpy.einsum('l,n,l,n->l', 1/g1.width, wp_d_coeff, g1.centre[t], int_ug_tensor)
    ext_cont1 += 0.5 * numpy.einsum('n,nk,l,nk,nlk->l', wp_coeff, width_tensor, 1/g1.width, d_centre_tensor, int_ux2g_tensor)
    ext_cont1 -= 0.5 * numpy.einsum('n,nk,l,nk,l,nk->l', wp_coeff, width_tensor, 1/g1.width, d_centre_tensor, g1.centre[t], int_uxg_tensor)
    ext_cont1 -= 0.5 * numpy.einsum('n,nk,l,nk,nk,nl->l', wp_coeff, width_tensor, 1/g1.width, d_centre_tensor, centre_tensor, int_uxg_tensor)
    ext_cont1 += 0.5 * numpy.einsum('n,nk,l,nk,l,nk,n->l', wp_coeff, width_tensor, 1/g1.width, d_centre_tensor, g1.centre[t], centre_tensor, int_ug_tensor)
    ext_cont1 += 0.25j * numpy.einsum('n,l,nk,nlk->l', wp_coeff, 1/g1.width, d_momentum_tensor, int_ux2g_tensor)
    ext_cont1 -= 0.25j * numpy.einsum('n,l,nk,l,nk->l', wp_coeff, 1/g1.width, d_momentum_tensor, g1.centre[t], int_uxg_tensor)
    ext_cont1 += 0.25j * numpy.einsum('n,l,n,nl->l', wp_coeff, 1/g1.width, d_phase_tensor, int_uxg_tensor)
    ext_cont1 -= 0.25j * numpy.einsum('n,l,n,l,n->l', wp_coeff, 1/g1.width, d_phase_tensor, g1.centre[t], int_ug_tensor)
    '''
    ext_cont1 += 0.25 * numpy.einsum('l,n,nl->l', 1/g1.width, wp_d_coeff, int_linear_shift)
    ext_cont1 += 0.5 * numpy.einsum('n,nk,l,nk,nlk->l', wp_coeff, width_tensor, 1/g1.width, d_centre_tensor, int_quadratic_shift)
    ext_cont1 += 0.25j * numpy.einsum('n,l,nk,nlk->l', wp_coeff, 1/g1.width, d_momentum_tensor, int_ux2g_tensor)
    ext_cont1 -= 0.25j * numpy.einsum('n,l,nk,l,nk->l', wp_coeff, 1/g1.width, d_momentum_tensor, g1.centre[t], int_uxg_tensor)
    ext_cont1 += 0.25j * numpy.einsum('n,l,n,nl->l', wp_coeff, 1/g1.width, d_phase_tensor, int_linear_shift)
    '''

    correction1 = -b_coeff/a_coeff * (ext_cont1 + ext_cont1.conj())
    
    # External contribution 2: Correction by crossover with previous wavefunction via hamiltonian
    ext_cont2 = numpy.zeros(sim.dim, dtype=numpy.complex128)
    ext_cont2 += -0.5 * numpy.einsum('n,k,l,nk,nlk->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor*width_tensor, int_ux3g_tensor)
    ext_cont2 += numpy.einsum('n,k,l,nk,nk,nlk->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor*width_tensor, centre_tensor, int_ux2g_tensor)
    ext_cont2 -= 0.5 * numpy.einsum('n,k,l,nk,nk,nl->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor*width_tensor, centre_tensor*centre_tensor, int_uxg_tensor)
    ext_cont2 += 0.5 * numpy.einsum('n,k,l,nk,l,nkk->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor*width_tensor, g1.centre[t], int_ux2g_tensor)
    ext_cont2 -= numpy.einsum('n,k,l,nk,l,nk,nk->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor*width_tensor, g1.centre[t], width_tensor, int_uxg_tensor)
    ext_cont2 += 0.5 * numpy.einsum('n,k,l,nk,l,nk,n->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor*width_tensor, g1.centre[t], centre_tensor*centre_tensor, int_ug_tensor)
    ext_cont2 += 0.5j * numpy.einsum('n,k,l,nk,nk,nlk->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor, momentum_tensor, int_ux2g_tensor)
    ext_cont2 -= 0.5j * numpy.einsum('n,k,l,nk,nk,l,nk->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor, momentum_tensor, g1.centre[t], int_uxg_tensor)
    ext_cont2 -= 0.5j * numpy.einsum('n,k,l,nk,nk,nk,nl->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor, momentum_tensor, centre_tensor, int_uxg_tensor)
    ext_cont2 += 0.5j * numpy.einsum('n,k,l,nk,nk,l,nk,n->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor, momentum_tensor, g1.centre[t], centre_tensor, int_ug_tensor)
    ext_cont2 += 0.125 * numpy.einsum('n,k,l,nk,nl->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, 2*width_tensor + momentum_tensor*momentum_tensor, int_uxg_tensor)
    ext_cont2 -= 0.125 * numpy.einsum('n,k,l,nk,l,n->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, 2*width_tensor + momentum_tensor*momentum_tensor, g1.centre[t], int_ug_tensor)
    ext_cont2 += 0.25 * numpy.einsum('n,l,nl->l', wp_coeff, 1/g1.width, int_uVxg_tensor)
    ext_cont2 -= 0.25 * numpy.einsum('n,l,l,n->l', wp_coeff, 1/g1.width, g1.centre[t], int_uVg_tensor)
    '''
    ext_cont2 += -0.5 * numpy.einsum('n,k,l,nk,nlk->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor*width_tensor, int_cubic_shift)
    ext_cont2 += 0.5j * numpy.einsum('n,k,l,nk,nk,nlk->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, width_tensor, momentum_tensor, int_quadratic_shift)
    ext_cont2 += 0.125 * numpy.einsum('n,k,l,nk,nl->l', wp_coeff, 1/pot.reduced_mass, 1/g1.width, 2*width_tensor + momentum_tensor*momentum_tensor, int_linear_shift)
    ext_cont2 += 0.25 * numpy.einsum('n,l,nl->l', wp_coeff, 1/g1.width, int_uVxg_tensor)
    ext_cont2 += 0.25 * numpy.einsum('n,l,l,n->l', wp_coeff, 1/g1.width, g1.centre[t], int_uVg_tensor)
    '''

    correction2 = -1j*b_coeff/a_coeff*(ext_cont2 - ext_cont2.conj())

    # External contribution 3: Correction by crossover with influence coefficient derivative

    # In the first iteration, the influence coefficient differential is approximated as zero
    d_a_coeff = 0
    # In the second iteration, the influence coefficient differential is copied from the previous iteration
    if t == 1:
        d_a_coeff = sim.d_active_coeffs[0]
    # In subsequent iterations, the influence coefficient differential is extrapolated from the previous iteration via a quadratic fit
    elif t > 1:
        d_a_coeff = 2*sim.d_active_coeffs[t-1] - sim.d_active_coeffs[t-2]

    # Debug override: Take d_a_coeff from previous iteration
    if t > 0:
        d_a_coeff = sim.d_active_coeffs[t-1]

    ext_cont3 = numpy.zeros(sim.dim, dtype=numpy.complex128)
    ext_cont3 += numpy.einsum('l,n,nl->l', 1/g1.width, wp_coeff, int_uxg_tensor)
    ext_cont3 -= numpy.einsum('l,n,l,n->l', 1/g1.width, wp_coeff, g1.centre[t], int_ug_tensor)

    '''
    ext_cont3 = 2 * numpy.einsum('n,l,nl->l', wp_coeff, 1/g1.width, int_linear_shift.real)
    '''

    correction3 = 0.25 * d_a_coeff / b_coeff * (ext_cont3 + ext_cont3.conj())

    logger.debug3(f"Ext Cont 1 : {2*ext_cont1.real}")
    logger.debug3(f"Ext Cont 2 : {2*ext_cont2.imag}")
    logger.debug3(f"Ext Cont 3 : {ext_cont3.real}")
    logger.debug3(f"Correction 1 : {correction1}")
    logger.debug3(f"Correction 2 : {correction2}")
    logger.debug3(f"Correction 3 : {correction3}")
    logger.debug3(f"Corrective sum : {correction1+correction2+correction3}")
    logger.debug3(f"Relative contribution : {abs((correction1+correction2+correction3)/(correction1+correction2+correction3+self_cont))}")

    #return eom_init_centre(sim, pot, g1, t)
    return (self_cont + correction1 + correction2 + correction3).real


def eom_momentum(sim, pot, g1, t):
    return eom_init_momentum(sim, pot, g1, t)
    wp = sim.previous_wavefunction
    wp_size = len(wp.gaussians)
    logger = sim.logger
    a_coeff = sim.active_coeffs[t]
    b_coeff = numpy.sqrt(1 - a_coeff*a_coeff)
    wp_coeff = wp.get_coeffs(t)
    wp_d_coeff = wp.get_d_coeffs(t)
    pot_intor = pot.gen_potential_integrator()

    # Simplification of potential electronic structure calls
    r_taylor = None
    r_taylor = g1.centre[t]

    # Construction of Tensors vxg, vx2g and semi-diagonal vx3g for each g in wp
    int_vg_tensor = numpy.zeros(wp_size, dtype=numpy.complex128)
    int_vxg_tensor = numpy.zeros((wp_size, sim.dim), dtype=numpy.complex128)
    int_vx2g_tensor = numpy.zeros((wp_size, sim.dim, sim.dim), dtype=numpy.complex128)
    # Only integrals of type u x y^2 g are relevant
    int_vx3g_tensor = numpy.zeros((wp_size, sim.dim, sim.dim), dtype=numpy.complex128)

    # Only potential integrals of type v x V g are relevant
    int_vVxg_tensor = numpy.zeros((wp_size, sim.dim), dtype=numpy.complex128)

    # Tensors of wp parameters for easy indexing
    centre_tensor = numpy.zeros((wp_size, sim.dim))
    width_tensor = numpy.zeros((wp_size, sim.dim))
    momentum_tensor = numpy.zeros((wp_size, sim.dim))
    phase_tensor = numpy.zeros(wp_size)
    d_centre_tensor = numpy.zeros((wp_size, sim.dim))
    d_momentum_tensor = numpy.zeros((wp_size, sim.dim))
    d_phase_tensor = numpy.zeros(wp_size)

    # Construction of tensors
    for g, gaussian in enumerate(wp.gaussians):
        g2 = gaussian.copy()
        g2.momentum[t] -= g1.momentum[t]
        g2.phase[t] -= g1.phase[t]

        int_vg_tensor[g] = intor.int_request(sim, 'int_vg', g1, g2, t, 0)

        width_tensor[g] = gaussian.width
        centre_tensor[g] = gaussian.centre[t]
        momentum_tensor[g] = gaussian.momentum[t]
        phase_tensor[g] = gaussian.phase[t]
        d_centre_tensor[g] = gaussian.d_centre[t]
        d_momentum_tensor[g] = gaussian.d_momentum[t]
        d_phase_tensor[g] = gaussian.d_phase[t]

        for i in range(sim.dim):
            int_vxg_tensor[g,i] = intor.int_request(sim, 'int_vxg', g1, g2, t, 0, i, m=int_vg_tensor[g], useM=True)
            int_vVxg_tensor[g,i] = pot_intor.int_request('int_vVxg', g1, g2, t, 0, i, r_taylor=r_taylor)
            for j in range(sim.dim):
                int_vx2g_tensor[g,i,j] = intor.int_request(sim, 'int_vx2g', g1, g2, t, 0, i, j, m=int_vg_tensor[g], useM=True)
                int_vx3g_tensor[g,i,j] = intor.int_request(sim, 'int_vx3g', g1, g2, t, 0, i, j, j, m=int_vg_tensor[g], useM=True)

    # Self contribution
    self_cont = 0.5 * (pot.gradient(r_taylor) - numpy.einsum('ll,l->l', pot.hessian(r_taylor), r_taylor))
    
    # Contribution by crossover with previous time derivative
    ext_cont1 = numpy.zeros(sim.dim, dtype=numpy.complex128)
    ext_cont1 += 1 / g1.width[0] * numpy.einsum('n,l,nl->l', wp_d_coeff, g1.width, int_vxg_tensor)
    ext_cont1 += 2 / g1.width[0] * numpy.einsum('n,nk,nk,l,nkl->l', wp_coeff, width_tensor, d_centre_tensor, g1.width, int_vx2g_tensor)
    ext_cont1 -= 2 / g1.width[0] * numpy.einsum('n,nk,nk,nk,l,nl->l', wp_coeff, width_tensor, d_centre_tensor, centre_tensor, g1.width, int_vxg_tensor)
    ext_cont1 += 1j / g1.width[0] * numpy.einsum('n,nk,l,nkl->l', wp_coeff, d_momentum_tensor, g1.width, int_vx2g_tensor)
    ext_cont1 += 1j / g1.width[0] * numpy.einsum('n,n,l,nl->l', wp_coeff, d_phase_tensor, g1.width, int_vxg_tensor)

    correction1 = 0.5j * b_coeff / a_coeff * (ext_cont1 + ext_cont1.conj())

    # Contribution by crossover with previous hamiltonian matrix elements
    ext_cont2 = numpy.zeros(sim.dim, dtype=numpy.complex128)
    ext_cont2 += -2 / g1.width[0] * numpy.einsum('n,k,nk,l,nlk->l', wp_coeff, 1/pot.reduced_mass, width_tensor*width_tensor, g1.width, int_vx3g_tensor)
    ext_cont2 += 4 / g1.width[0] * numpy.einsum('n,k,nk,nk,l,nkl->l', wp_coeff, 1/pot.reduced_mass, width_tensor*width_tensor, centre_tensor, g1.width, int_vx2g_tensor)
    ext_cont2 += -2 / g1.width[0] * numpy.einsum('n,k,nk,nk,l,nl->l', wp_coeff, 1/pot.reduced_mass, width_tensor*width_tensor, centre_tensor*centre_tensor, g1.width, int_vxg_tensor)
    ext_cont2 += 2j / g1.width[0] * numpy.einsum('n,k,nk,nk,l,nlk->l', wp_coeff, 1/pot.reduced_mass, width_tensor, momentum_tensor, g1.width, int_vx2g_tensor)
    ext_cont2 += -2j / g1.width[0] * numpy.einsum('n,k,nk,nk,nk,l,nl->l', wp_coeff, 1/pot.reduced_mass, width_tensor, momentum_tensor, centre_tensor, g1.width, int_vxg_tensor)
    ext_cont2 += 0.5 / g1.width[0] * numpy.einsum('n,k,nk,l,nl->l', wp_coeff, 1/pot.reduced_mass, 2*width_tensor + momentum_tensor*momentum_tensor, g1.width, int_vxg_tensor)
    ext_cont2 += 1 / g1.width[0] * numpy.einsum('n,nl->l', wp_coeff, int_vVxg_tensor)

    correction2 = 0.5j * b_coeff / a_coeff * (ext_cont2 + ext_cont2.conj())

    # External contribution 3: Correction by crossover with influence coefficient derivative

    # In the first iteration, the influence coefficient differential is approximated as zero
    d_a_coeff = 0
    # In the second iteration, the influence coefficient differential is copied from the previous iteration
    if t == 1:
        d_a_coeff = sim.d_active_coeffs[0]
    # In subsequent iterations, the influence coefficient differential is extrapolated from the previous iteration via a quadratic fit
    elif t > 1:
        d_a_coeff = 2*sim.d_active_coeffs[t-1] - sim.d_active_coeffs[t-2]

    # Debug override: Take d_a_coeff from previous iteration
    if t > 0:
        d_a_coeff = sim.d_active_coeffs[t-1]

    ext_cont3 = numpy.einsum('n,nl->l', wp_coeff, int_vxg_tensor)

    correction3 = -0.5j * d_a_coeff / b_coeff * (ext_cont3 - ext_cont3.conj())

    return eom_init_momentum(sim, pot, g1, t)
    return (self_cont + correction1 + correction2 + correction3).real

def eom_phase(sim, pot, g1, t):
    return eom_init_phase(sim, pot, g1, t)
    wp = sim.previous_wavefunction
    wp_size = len(wp.gaussians)
    logger = sim.logger
    a_coeff = sim.active_coeffs[t]
    b_coeff = numpy.sqrt(1 - a_coeff*a_coeff)
    wp_coeff = wp.get_coeffs(t)
    wp_d_coeff = wp.get_d_coeffs(t)
    pot_intor = pot.gen_potential_integrator()

    # Simplification of potential electronic structure calls
    r_taylor = None
    r_taylor = g1.centre[t]
    e_pot = pot.evaluate(r_taylor)
    grad = pot.gradient(r_taylor)
    hess = pot.hessian(r_taylor)

    # Construction of Tensors vg, vxg and diagonal vx2g for each g in wp
    int_vg_tensor = numpy.zeros(wp_size, dtype=numpy.complex128)
    int_vxg_tensor = numpy.zeros((wp_size, sim.dim), dtype=numpy.complex128)
    int_vx2g_tensor = numpy.zeros((wp_size, sim.dim), dtype=numpy.complex128)

    # Only potential integrals of type v V g are relevant
    int_vVg_tensor = numpy.zeros(wp_size, dtype=numpy.complex128)

    # Tensors of wp parameters for easy indexing
    centre_tensor = numpy.zeros((wp_size, sim.dim))
    width_tensor = numpy.zeros((wp_size, sim.dim))
    momentum_tensor = numpy.zeros((wp_size, sim.dim))
    phase_tensor = numpy.zeros(wp_size)
    d_centre_tensor = numpy.zeros((wp_size, sim.dim))
    d_momentum_tensor = numpy.zeros((wp_size, sim.dim))
    d_phase_tensor = numpy.zeros(wp_size)
   
    # Construction of tensors
    for g, gaussian in enumerate(wp.gaussians):
        g2 = gaussian.copy()
        g2.momentum[t] -= g1.momentum[t]
        g2.phase[t] -= g1.phase[t]

        int_vg_tensor[g] = intor.int_request(sim, 'int_vg', g1, g2, t, 0)
        int_vVg_tensor[g] = pot_intor.int_request('int_vVg', g1, g2, t, 0, r_taylor=r_taylor)

        width_tensor[g] = gaussian.width
        centre_tensor[g] = gaussian.centre[t]
        momentum_tensor[g] = gaussian.momentum[t]
        phase_tensor[g] = gaussian.phase[t]
        d_centre_tensor[g] = gaussian.d_centre[t]
        d_momentum_tensor[g] = gaussian.d_momentum[t]
        d_phase_tensor[g] = gaussian.d_phase[t]

        for i in range(sim.dim):
            int_vxg_tensor[g,i] = intor.int_request(sim, 'int_vxg', g1, g2, t, 0, i, m=int_vg_tensor[g], useM=True)
            int_vx2g_tensor[g,i] = intor.int_request(sim, 'int_vx2g', g1, g2, t, 0, i, i, m=int_vg_tensor[g], useM=True)

    # Self contribution
    self_cont = numpy.dot(g1.width, 1/pot.reduced_mass) + 0.5 * numpy.dot(4*g1.width*g1.width*g1.centre[t]*g1.centre[t] - 2*g1.width - g1.momentum[t]*g1.momentum[t], 1/pot.reduced_mass)
    self_cont += -e_pot + numpy.dot(grad, r_taylor) - 0.5 * numpy.einsum('i,ij,j->', r_taylor, hess, r_taylor) - 0.0625 * numpy.dot(numpy.diag(hess), 1/g1.width)

    # First correction
    ext_cont1 = numpy.dot(wp_d_coeff, int_vg_tensor)
    ext_cont1 += 2 * numpy.einsum('n,nk,nk,nk->', wp_coeff, width_tensor, d_centre_tensor, int_vxg_tensor)
    ext_cont1 -= 2 * numpy.einsum('n,nk,nk,nk,n->', wp_coeff, width_tensor, d_centre_tensor, centre_tensor, int_vg_tensor)
    ext_cont1 += 1j * numpy.einsum('n,nk,nk->', wp_coeff, d_momentum_tensor, int_vxg_tensor)
    ext_cont1 += 1j * numpy.einsum('n,n,n->', wp_coeff, d_phase_tensor, int_vg_tensor)

    correction1 = 0.25j * b_coeff / (a_coeff * g1.width[0]) * (ext_cont1 - numpy.conj(ext_cont1))

    # Second correction
    ext_cont2 = -2 * numpy.einsum('n,k,nk,nk->', wp_coeff, 1/pot.reduced_mass, width_tensor*width_tensor, int_vx2g_tensor)
    ext_cont2 += 4 * numpy.einsum('n,k,nk,nk,nk->', wp_coeff, 1/pot.reduced_mass, width_tensor*width_tensor, centre_tensor, int_vxg_tensor)
    ext_cont2 -= 2 * numpy.einsum('n,k,nk,nk,n->', wp_coeff, 1/pot.reduced_mass, width_tensor*width_tensor, centre_tensor*centre_tensor, int_vg_tensor)
    
    ext_cont2 += 2j * numpy.einsum('n,k,nk,nk,nk->', wp_coeff, 1/pot.reduced_mass, width_tensor, momentum_tensor, int_vxg_tensor)
    ext_cont2 -= 2j * numpy.einsum('n,k,nk,nk,nk,n->', wp_coeff, 1/pot.reduced_mass, width_tensor, momentum_tensor, centre_tensor, int_vg_tensor)
    ext_cont2 += 0.5 * numpy.einsum('n,k,nk,n->', wp_coeff, 1/pot.reduced_mass, 2*width_tensor+momentum_tensor*momentum_tensor, int_vg_tensor)
    ext_cont2 += numpy.dot(wp_coeff, int_vVg_tensor)

    correction2 = -0.25 * b_coeff / (a_coeff * g1.width[0]) * (ext_cont2 + numpy.conj(ext_cont2))
    
     # In the first iteration, the influence coefficient differential is approximated as zero
    d_a_coeff = 0
    # In the second iteration, the influence coefficient differential is copied from the previous iteration
    if t == 1:
        d_a_coeff = sim.d_active_coeffs[0]
    # In subsequent iterations, the influence coefficient differential is extrapolated from the previous iteration via a quadratic fit
    elif t > 1:
        d_a_coeff = 2*sim.d_active_coeffs[t-1] - sim.d_active_coeffs[t-2]

    # Debug override: Take d_a_coeff from previous iteration
    if t > 0:
        d_a_coeff = sim.d_active_coeffs[t-1]

    # Third correction
    ext_cont3 = numpy.dot(wp_coeff, int_vg_tensor)

    correction3 = -0.25j * d_a_coeff / (b_coeff * g1.width[0]) * (ext_cont3 - numpy.conj(ext_cont3))

    #return eom_init_phase(sim, pot, g1, t)
    return (self_cont + correction1 + correction2 + correction3).real

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
