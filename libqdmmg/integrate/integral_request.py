'''

@author: Linus Bjarne Dittmer

'''

import numpy
import libqdmmg.integrate.atom_integration_handler as atom_intor
import libqdmmg.general as gen

ELEM_INTEGRALS = ['int_gg', 'int_gxg', 'int_gx2g', 'int_gx2g_m', 'int_gx2g_d', 'int_gx3g', 'int_gx3g_m', 'int_gx3g_d', 'int_gx3g_od', 'int_ug', 'int_uxg', 'int_ux2g', 'int_ux2g_m', 'int_ux2g_d', 'int_ux3g', 'int_ux3g_m', 'int_ux3g_d', 'int_ux3g_od', 'int_vg', 'int_vxg', 'int_vx2g', 'int_vx2g_m', 'int_vx2g_d', 'int_vx3g', 'int_vx3g_m', 'int_vx3g_d', 'int_vx3g_od']
COMP_INTEGRALS = ['int_ovlp_gg', 'int_ovlp_wg', 'int_ovlp_gw', 'int_ovlp_ww', 'int_dovlp_gg', 'int_dovlp_wg', 'int_dovlp_gw', 'int_dovlp_ww', 'int_kinetic_gg', 'int_kinetic_gw', 'int_kinetic_wg', 'int_kinetic_ww']


def int_request(sim, request_string, *args, **kwargs):
    '''
    This function is the main interface for accessing analytical integrals. These are requested by providing the main simulation instance and a request string, which can be either a composite or elementary analytic integral as well as the respective arguments and keyword arguments. Note that the request string is not case-sensitive. Whenever an analytical integral is required, it should be accessed through this function. The following options are allowed:

    int_ovlp_gg
    int_ovlp_wg
    int_ovlp_gw
    int_ovlp_ww
    int_dovlp_gg
    int_dovlp_gw
    int_dovlp_wg
    int_dovlp_ww
    int_kinetic_gg
    int_kinetic_gw
    int_kinetic_wg
    int_kinetic_gw
    int_kinetic_ww
    int_gg
    int_gxg
    int_gx2g
    int_gx2g_d
    int_gx2g_m
    int_gx3g
    int_gx3g_d
    int_gx3g_m
    int_gx3g_od
    int_ug
    int_uxg
    int_ux2g
    int_ux2g_d
    int_ux2g_m
    int_ux3g
    int_ux3g_d
    int_ux3g_m
    int_ux3g_od
    int_vg
    int_vxg
    int_vx2g
    int_vx2g_d
    int_vx2g_m
    int_vx3g
    int_vx3g_d
    int_vx3g_m
    int_vx3g_od


    Parameters
    ----------
    sim : libqdmmg.simulate.simulation.Simulation
        The main Simulation instance holding all relevant information
    request_string : str
        The Request String. See above for allowed values.
    args : list
        List of further arguments for integral calculation
    kwargs : dict
        List of further keyword arguments for integral calculation.

    Returns
    -------
    int_val : complex128
        The integral value.

    Raises
    ------
    InvalidIntegralRequestStringException
        If the request string does not match any valid argument.
    '''
    rq = request_string.strip().lower()

    if rq in ELEM_INTEGRALS:
        return int_atom_request(request_string, *args, **kwargs)
    elif rq in COMP_INTEGRALS:
        return int_composite_request(sim, request_string, *args, **kwargs)
    else:
        raise gen.IIRSException(rq, "")


def int_composite_request(sim, request_string, *args, **kwargs):
    '''
    This function handles requests for analytical integral calculations of composite integrals. These are performed by combining the respective elementary integrals. The following request strings are allowed:

    int_ovlp_gg
    int_ovlp_gw
    int_ovlp_ww
    int_dovlp_gg
    int_dovlp_gw
    int_dovlp_wg
    int_dovlp_ww
    int_kinetic_gg
    int_kinetic_gw
    int_kinetic_wg
    int_kinetic_ww
    
    Parameters
    ----------
    sim : libqdmmg.simulate.simulation.Simulation
        The main Simulation instance holding all relevant information
    request_string : str
        The Request String. See above for allowed values.
    args : list
        List of further arguments for integral calculation.
    kwargs : dict
        List of additional keyword arguments for integral calculation

    Returns
    -------
    int_val : complex128
        The integral value.

    Raises
    ------
    InvalidIntegralRequestStringException
        If the request string does not match any valid argument.
    AssertionError
        If the given arguments are insufficient.
    '''

    rq = request_string.strip().lower()
    argnum = len(args)
    assert argnum >= 3, f"All composite integrals require at least three arguments. Received {argnum}"
    assert isinstance(args[2], int) or isinstance(args[2], float), f"Timestep index must be given as integer in third argument slot. Received {type(args[2])}"
    t = args[2]

    if '_gg' in rq or '_gw' in rq:
        assert isinstance(args[0], gen.gaussian.Gaussian), f"For integrals of type -gg or -gw, the first argument must be a Gaussian. Received {type(args[0])}"
    if '_gg' in rq or '_wg' in rq:
        assert isinstance(args[1], gen.gaussian.Gaussian), f"For integrals of type -gg or -wg, the second argument must be a Gaussian. Received {type(args[1])}"
    if '_wg' in rq or '_ww' in rq:
        assert isinstance(args[0], gen.wavepacket.Wavepacket), f"For integrals of type -wg or -ww, the first argument must be a Wavepacket. Received {type(args[0])}"
    if '_gw' in rq or '_ww' in rq:
        assert isinstance(args[1], gen.wavepacket.Wavepacket), f"For integrals of type -gw or -ww, the second argument must be a Wavepacket. Received {type(args[1])}"

    g1centre, g1momentum, g1phase, g2centre, g2momentum, g2phase = None, None, 0, None, None, 0
    if abs(t - int(t)) < 10**-5:
        if '_gg' in rq or '_gw' in rq:
            g1 = args[0]
            g1centre = g1.centre[t]
            g1momentum = g1.momentum[t]
            g1phase = g1.phase[t]
        if '_wg' in rq or '_gg' in rq:
            g2 = args[1]
            g2centre = g2.centre[t]
            g2momentum = g2.momentum[t]
            g2phase = g2.phase[t]
    else:
        if '_gg' in rq or '_gw' in rq:
            g1centre, g1momentum, g1phase = args[0].interpolate(t, returntype='ndarray')
        if '_wg' in rq or '_gg' in rq:
            g2centre, g2momentum, g2phase = args[1].interpolate(t, returntype='ndarray')


    if rq == 'int_ovlp_gg':
        return int_atom_request('int_gg', *args, **kwargs)
    elif rq == 'int_ovlp_wg':
        wp = args[0]
        g = args[1]
        coeffs = wp.get_coeffs(t)
        int_gg_tensor = numpy.zeros(coeffs.shape, dtype=numpy.complex128)
        for i in range(len(coeffs)):
            int_gg_tensor[i] = int_atom_request('int_gg', wp.gaussians[i], g, t)
        return numpy.dot(coeffs, int_gg_tensor)
    elif rq == 'int_ovlp_gw':
        return int_composite_request(sim, 'int_ovlp_wg', args[1], args[0], t, **kwargs).conj()
    elif rq == 'int_ovlp_ww':
        wp1 = args[0]
        wp2 = args[1]
        coeffs1 = wp1.get_coeffs(t)
        coeffs2 = wp2.get_coeffs(t)
        int_gg_tensor = numpy.zeros((len(coeffs1), len(coeffs2)), dtype=numpy.complex128)
        for i in range(len(coeffs1)):
            for j in range(len(coeffs2)):
                int_gg_tensor[i,j] = int_atom_request('int_gg', wp1.gaussians[i], wp2.gaussians[j], t)
        return numpy.einsum('i,ij,j', coeffs1, int_gg_tensor, coeffs2)
    elif rq == 'int_dovlp_gg':
        g1 = args[0]
        g2 = args[1]
        gg = int_atom_request('int_gg', g1, g2, t)
        gxg = numpy.zeros(sim.dim, dtype=numpy.complex128)
        for i in range(sim.dim):
            gxg[i] = int_atom_request('int_gxg', g1, g2, t, i)
        dovlp_vec1 = gxg - g2centre * gg
        dovlp1 = 2 * numpy.dot(g2.width*g2.d_centre[int(t)], dovlp_vec1)
        dovlp2 = numpy.dot(g2.d_momentum[int(t)], gxg)
        return dovlp1 + 1j*(dovlp2 + g2.d_phase[int(t)]*gg)
    elif rq == 'int_dovlp_wg':
        wp = args[0]
        g1 = args[1]
        coeffs = wp.get_coeffs(t)
        int_val = 0.0
        for i in range(len(coeffs)):
            int_val += coeffs[i]*int_composite_request(sim, 'int_dovlp_gg', wp.gaussians[i], g1, t)
        return int_val
    elif rq == 'int_dovlp_gw':
        g1 = args[0]
        wp = args[1]
        coeffs = wp.get_coeffs(t)
        coeffs_t = numpy.copy(coeffs)
        d_coeffs = wp.get_d_coeffs(t)
        int_val = 0.0
        for i in range(len(coeffs)):
            int_val += d_coeffs[i] * int_atom_request('int_gg', g1, wp.gaussians[i], t) + coeffs[i] * int_composite_request(sim, 'int_dovlp_gg', g1, wp.gaussians[i], t)
        return int_val
    elif rq == 'int_dovlp_ww':
        wp1 = args[0]
        wp2 = args[1]
        coeffs = wp1.get_coeffs(t)
        int_val = 0.0
        for i in range(len(coeffs)):
            int_val += coeffs[i] * int_composite_request(sim, 'int_dovlp_gw', wp1.gaussians[i], wp2, t)
        return int_val
    elif rq == 'int_kinetic_gg':
        g1, g2 = args[0], args[1]
        int_gg = int_atom_request('int_gg', g1, g2, t)
        int_gxg = numpy.zeros(sim.dim, dtype=numpy.complex128)
        int_gx2g = numpy.zeros(sim.dim, dtype=numpy.complex128)
        for i in range(sim.dim):
            int_gxg[i] = int_atom_request('int_gxg', g1, g2, t, i, m=int_gg, useM=True)
            int_gx2g[i] = int_atom_request('int_gx2g_d', g1, g2, t, i, m=int_gg, useM=True)
        kin_vec = numpy.zeros(sim.dim, dtype=numpy.complex128)
        kin_vec += 4 * g2.width * g2.width * (int_gx2g - 2 * g2centre * int_gxg + g2centre * g2centre * int_gg)
        kin_vec -= 4j * g2.width * g2momentum * (int_gxg - g2centre * int_gg)
        kin_vec -= (2 * g2.width + g2momentum * g2momentum) * int_gg
        kin = numpy.dot(kin_vec, 1.0 / sim.potential.reduced_mass)
        return -0.5 * kin
    elif rq == 'int_kinetic_wg':
        wp, g1 = args[0], args[1]
        coeffs = wp.get_coeffs(t)
        kin_gauss = numpy.zeros(coeffs.shape, dtype=numpy.complex128)
        for i in range(len(coeffs)):
            kin_gauss[i] = int_composite_request(sim, 'int_kinetic_gg', wp.gaussians[i], g1, t)
        return numpy.dot(coeffs, kin_gauss)
    elif rq == 'int_kinetic_gw':
        g1, wp = args[0], args[1]
        coeffs = wp.get_coeffs(t)
        kin_gauss = numpy.zeros(coeffs.shape, dtype=numpy.complex128)
        for i in range(len(coeffs)):
            kin_gauss[i] = int_composite_request(sim, 'int_kinetic_gg', g1, wp.gaussians[i], t)
        return numpy.dot(coeffs, kin_gauss)
    elif rq == 'int_kinetic_ww':
        wp1, wp2 = args[0], args[1]
        coeffs1 = wp1.get_coeffs(t)
        coeffs2 = wp2.get_coeffs(t)
        kin_gauss = numpy.zeros((len(coeffs1), len(coeffs2)), dtype=numpy.complex128)
        for i in range(len(coeffs1)):
            for j in range(len(coeffs2)):
                kin_gauss[i,j] = int_composite_request(sim, 'int_kinetic_gg', wp1.gaussians[i], wp2.gaussians[j], t)
        return numpy.einsum('i,ij,j', coeffs1, kin_gauss, coeffs2)
    else:
        raise gen.IIRSException(rq, "comp")



def int_atom_request(request_string, *args, **kwargs):
    '''
    This function handles requests for analytical integral calculations of elementary integrals. The following request strings are allowed:

    int_gg
    int_gxg
    int_gx2g
    int_gx2g_d
    int_gx2g_m
    int_gx3g
    int_gx3g_d
    int_gx3g_m
    int_gx3g_od
    int_ug
    int_uxg
    int_ux2g
    int_ux2g_d
    int_ux2g_m
    int_ux3g
    int_ux3g_d
    int_ux3g_m
    int_ux3g_od
    int_vg
    int_vxg
    int_vx2g
    int_vx2g_d
    int_vx2g_m
    int_vx3g
    int_vx3g_d
    int_vx3g_m
    int_vx3g_od

    Parameters
    ----------
    request_string : str
        The Request String. See above for allowed values.
    args : list
        Additional arguments for integral calculation. Must at least contain two instances of libqdmmg.general.gaussian.Gaussian and one integer timestep.
    kwargs : dict
        Additional keyword arguments for integral calculation.

    Returns
    -------
    int_val : complex128
        The integral value.

    Raises
    ------
    AssertionError
        If an the number of arguments or their type is incorrect.
    InvalidIntegralRequestStringException
        If the request string does not match a valid argument.
    '''
    
    if len(args) == 0:
        raise Exception('No integral arguments given.')

    rq = request_string.strip().lower()
    while len(args) == 2:
        args = args[0]
    argnum = len(args)

    # Prescreening
    # All Elementary integrals have g1, g2, t
    assert argnum >= 3, f"All integrals require at least three arguments (g1, g2, t). Received {len(args)}"
    assert isinstance(args[0], gen.gaussian.Gaussian) and isinstance(args[1], gen.gaussian.Gaussian), f"Two Gaussians must be given as the first two arguments"
    assert isinstance(args[2], int) or isinstance(args[2], float), f"Timestep must be given as index in the third argument"

    g1 = args[0]
    g2 = args[1]
    t = args[2]
    vindex = 0

    if '_v' in rq:
        assert argnum >= 4, f"Integrals of type v- must have at least four arguments (g1, g2, t, vindex). Received {argnum}"
        assert isinstance(args[3], int), f"Integrals of type v- must have vindex given as fourth argument. Received type {type(args[3])}"
        vindex = args[3]

    if rq == 'int_gg':
        return atom_intor.int_gg(g1, g2, t)
    elif rq == 'int_gxg':
        assert argnum >= 4, f"Integrals of type gxg must have at least four arguments (g1, g2, t, index). Received {argnum}"
        return atom_intor.int_gxg(g1, g2, t, args[3], kwargs)
    elif rq == 'int_gx2g':
        assert argnum >= 4, f"Integrals of type gx2g and gx2g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        if argnum >= 5:
            if args[3] != args[4]:
                return atom_intor.int_gx2g_mixed(g1, g2, t, args[3], args[4], kwargs)
        return atom_intor.int_gx2g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_gx2g_d':
        assert argnum >= 4, f"Integrals of type gx2g and gx2g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        return atom_intor.int_gx2g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_gx2g_m':
        assert argnum >= 5, f"Integrals of type gx2g_m must have at least five arguments (g1, g2, t, index1, index2), Received {argnum}"
        return atom_intor.int_gx2g_mixed(g1, g2, t, args[3], args[4], kwargs)
    elif rq == 'int_gx3g':
        assert argnum >= 4, f"Integrals of type gx3g and gx3g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        if argnum >= 6:
            if args[3] != args[4] and args[4] != args[5]:
                return atom_intor.int_gx3g_offdiag(g1, g2, t, args[3], args[4], args[5], kwargs)
            elif args[3] == args[4] and args[4] != args[5]:
                return atom_intor.int_gx3g_mixed(g1, g2, t, args[5], args[3], kwargs)
            elif args[3] == args[5] and args[3] != args[4]:
                return atom_intor.int_gx3g_mixed(g1, g2, t, args[4], args[5], kwargs)
            elif args[4] == args[5] and args[5] != args[3]:
                return atom_intor.int_gx3g_mixed(g1, g2, t, args[3], args[4], kwargs)
        if argnum == 5:
            if args[3] != args[4]:
                return atom_intor.int_gx3g_mixed(g1, g2, t, args[3], args[4], kwargs)
        return atom_intor.int_gx3g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_gx3g_d':
        assert argnum >= 4, f"Integrals of type gx3g and gx3g_d must have at least four arguments (g1, g2, t, index1, [index2, index3]), Received {argnum}"
        return atom_intor.int_gx3g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_gx3g_m':
        assert argnum >= 5, f"Integrals of type gx3g_m must have at least five arguments (g1, g2, t, index1, index2), Received {argnum}"
        return atom_intor.int_gx3g_mixed(g1, g2, t, args[3], args[4], kwargs)
    elif rq == 'int_gx3g_od':
        assert argnum >= 6, f"Integrals of type gx3g_m must have at least five arguments (g1, g2, t, index1, index2, index3), Received {argnum}"
        return atom_intor.int_gx3g_offdiag(g1, g2, t, args[3], args[4], args[5], kwargs)
    elif rq == 'int_ug':
        return atom_intor.int_ug(g1, g2, t)
    elif rq == 'int_uxg':
        assert argnum >= 4, f"Integrals of type uxg must have at least four arguments (g1, g2, t, index1), Received {argnum}"
        return atom_intor.int_uxg(g1, g2, t, args[3], kwargs) 
    elif rq == 'int_ux2g':
        assert argnum >= 4, f"Integrals of type ux2g and ux2g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        if argnum >= 5:
            if args[3] != args[4]:
                return atom_intor.int_ux2g_mixed(g1, g2, t, args[3], args[4], kwargs)
        return atom_intor.int_ux2g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_ux2g_d':
        assert argnum >= 4, f"Integrals of type ux2g and ux2g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        return atom_intor.int_ux2g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_ux2g_m':
        assert argnum >= 5, f"Integrals of type ux2g_m must have at least five arguments (g1, g2, t, index1, index2), Received {argnum}"
        return atom_intor.int_ux2g_mixed(g1, g2, t, args[3], args[4], kwargs)
    elif rq == 'int_ux3g':
        assert argnum >= 4, f"Integrals of type ux3g and ux3g_d must have at least four arguments (g1, g2, t, index1, [index2, index3]), Received {argnum}"
        if argnum >= 6:
            if args[3] != args[4] and args[4] != args[5] and args[3] != args[5]:
                return atom_intor.int_ux3g_offdiag(g1, g2, t, args[3], args[4], args[5], kwargs)
            elif args[3] == args[4] and args[4] != args[5]:
                return atom_intor.int_ux3g_mixed(g1, g2, t, args[5], args[3], kwargs)
            elif args[3] == args[5] and args[3] != args[4]:
                return atom_intor.int_ux3g_mixed(g1, g2, t, args[4], args[5], kwargs)
            elif args[4] == args[5] and args[5] != args[3]:
                return atom_intor.int_ux3g_mixed(g1, g2, t, args[3], args[4], kwargs)
        if argnum == 5:
            if args[3] != args[4]:
                return atom_intor.int_ux3g_mixed(g1, g2, t, args[3], args[4], kwargs)
        return atom_intor.int_ux3g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_ux3g_d':
        assert argnum >= 4, f"Integrals of type ux3g and ux3g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        return atom_intor.int_ux3g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_ux3g_m':
        assert argnum >= 5, f"Integrals of type ux3g_m must have at least five arguments (g1, g2, t, index1, index2), Received {argnum}"
        return atom_intor.int_ux3g_mixed(g1, g2, t, args[3], args[4], kwargs)
    elif rq == 'int_ux3g_od':
        assert argnum >= 6, f"Integrals of type ux3g_od must have at least six arguments (g1, g2, t, index1, index2, index3), Received {argnum}"
        return atom_intor.int_ux3g_offdiag(g1, g2, t, args[3], args[4], args[5], kwargs)
    elif rq == 'int_vg':
        return atom_intor.int_vg(g1, g2, t, vindex)
    elif rq == 'int_vxg':
        assert argnum >= 5, f"Integrals of type vxg must have at least five arguments (g1, g2, t, vindex, index), Received {argnum}"
        return atom_intor.int_vxg(g1, g2, t, vindex, args[4], kwargs)
    elif rq == 'int_vx2g':
        assert argnum >= 5, f"Integrals of type vx2g and vx2g_d must have at least five arguments (g1, g2, t, vindex, index1, [index2]), Received {argnum}"
        if argnum >= 6:
            if args[4] != args[5]:
                return atom_intor.int_vx2g_mixed(g1, g2, t, vindex, args[4], args[5], kwargs)
        return atom_intor.int_vx2g_diag(g1, g2, t, vindex, args[4], kwargs)
    elif rq == 'int_vx2g_d':
        assert argnum >= 5, f"Integrals of type vx2g and vx2g_d must have at least five arguments (g1, g2, t, vindex, index1, [index2]), Received {argnum}"
        return atom_intor.int_vx2g_diag(g1, g2, t, vindex, args[4], kwargs)
    elif rq == 'int_vx2g_m':
        assert argnum >= 6, f"Integrals of type vx2g_m must have at least six arguments (g1, g2, t, vindex, index1, index2), Received {argnum}"
        return atom_intor.int_vx2g_mixed(g1, g2, t, vindex, args[4], args[5], kwargs)
    elif rq == 'int_vx3g':
        assert argnum >= 5, f"Integrals of type vx3g, vx3g_d or vx3g_od must have at least six arguments (g1, g2, t, vindex, index1, [index2, index3]), Received {argnum}"
        if argnum >= 7:
            if args[4] != args[5] and args[5] != args[6]:
                return atom_intor.int_vx3g_offdiag(g1, g2, t, vindex, args[4], args[5], args[6], kwargs)
            elif args[4] == args[5] and args[5] != args[6]:
                return atom_intor.int_vx3g_mixed(g1, g2, t, vindex, args[6], args[4], kwargs)
            elif args[4] == args[6] and args[4] != args[5]:
                return atom_intor.int_vx3g_mixed(g1, g2, t, vindex, args[5], args[4], kwargs)
            elif args[5] == args[6] and args[4] != args[6]:
                return atom_intor.int_vx3g_mixed(g1, g2, t, vindex, args[4], args[5], kwargs)
        if argnum == 6:
            if args[4] != args[5]:
                return atom_intor.int_vx3g_mixed(g1, g2, t, vindex, args[4], args[5], kwargs)
        return atom_intor.int_vx3g_diag(g1, g2, t, vindex, args[4], kwargs)
    elif rq == 'int_vx3g_d':
        assert argnum >= 5, f"Integrals of type vx3g and vx3g_d must have at least five arguments (g1, g2, t, vindex, index1, [index2, index3]), Received {argnum}"
        return atom_intor.int_vx3g_diag(g1, g2, t, vindex, args[4], kwargs)
    elif rq == 'int_vx3g_m':
        assert argnum >= 5, f"Integrals of type vx3g_m must have at least six arguments (g1, g2, t, vindex, index1, index2), Received {argnum}"
        return atom_intor.int_vx3g_mixed(g1, g2, t, vindex, args[4], args[5], kwargs)
    elif rq == 'int_vx3g_od':
        assert argnum >= 5, f"Integrals of type vx3g_od must have at least seven arguments (g1, g2, t, vindex, index1, index2, index3), Received {argnum}"
        return atom_intor.int_vx3g_offdiag(g1, g2, t, vindex, args[4], args[5], args[6], kwargs)
    else:
        raise gen.IIRSException(rq, "elem")

