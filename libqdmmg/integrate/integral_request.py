'''

@author: Linus Bjarne Dittmer

'''

import libqdmmg.integrate.anal_integration_handler as anal_intor
import libqdmmg.general as g

ELEM_INTEGRALS = ['int_gg', 'int_gxg', 'int_gx2g', 'int_gx2g_m', 'int_gx2g_d', 'int_gx3g', 'int_gx3g_m', 'int_gx3g_d', 'int_ug', 'int_uxg', 'int_ux2g', 'int_ux2g_m', 'int_ux2g_d', 'int_ux3g', 'int_ux3g_m', 'int_ux3g_d', 'int_vg', 'int_vxg', 'int_vx2g', 'int_vx2g_m', 'int_vx2g_d', 'int_vx3g', 'int_vx3g_m', 'int_vx3g_d']
COMP_INTEGRALS = ['int_ovlp', 'int_ovlp_prev', 'int_dovlp', 'int_dovlp_prev', 'int_dovlp_prev2', 'int_dovlp_prevprev', 'int_kinetic', 'int_kinetic_prev', 'int_kinetic_coupling']


def int_request(sim, request_string, *args, **kwargs):
    '''
    This function is the main interface for accessing analytical integrals. These are requested by providing the main simulation instance and a request string, which can be either a composite or elementary analytic integral as well as the respective arguments and keyword arguments. Note that the request string is not case-sensitive. Whenever an analytical integral is required, it should be accessed through this function. The following options are allowed:

    int_ovlp
    int_ovlp_prev
    int_dovlp
    int_dovlp_prev
    int_dovlp_prev2
    int_dovlp_prevprev
    int_kinetic
    int_kinetic_prev
    int_kinetic_coupling
    int_gg
    int_gxg
    int_gx2g
    int_gx2g_d
    int_gx2g_m
    int_gx3g
    int_gx3g_d
    int_gx3g_m
    int_ug
    int_uxg
    int_ux2g
    int_ux2g_d
    int_ux2g_m
    int_ux3g
    int_ux3g_d
    int_ux3g_m
    int_vg
    int_vxg
    int_vx2g
    int_vx2g_d
    int_vx2g_m
    int_vx3g
    int_vx3g_d
    int_vx3g_m


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
        return int_elem_request(request_string, args, kwargs)
    elif rq in COMP_INTEGRALS:
        return int_composite_request(sim, request_string, args, kwargs)
    else:
        raise g.IIRSException(rq, "")


def int_composite_request(sim, request_string, *args, **kwargs):
    '''
    This function handles requests for analytical integral calculations of composite integrals. These are performed by combining the respective elementary integrals. The following request strings are allowed:

    int_ovlp
    int_ovlp_prev
    int_dovlp
    int_dovlp_prev
    int_dovlp_prev2
    int_dovlp_prevprev
    int_kinetic
    int_kinetic_prev
    int_kinetic_coupling
    
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
    '''

    rq = request_string.strip().lower()

    if rq == 'int_ovlp':
        return 0
    elif rq == 'int_ovlp_prev':
        return 0
    elif rq == 'int_dovlp':
        return 0
    elif rq == 'int_dovlp_prev':
        return 0
    elif rq == 'int_dovlp_prev2':
        return 0
    elif rq == 'int_dovlp_doubleprev':
        return 0
    elif rq == 'int_kinetic':
        return 0
    elif rq == 'int_kinetic_prev':
        return 0
    elif rq == 'int_kinetic_coupling':
        return 0
    else:
        raise g.IIRSException(rq, "comp")



def int_elem_request(request_string, *args, **kwargs):
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
    int_ug
    int_uxg
    int_ux2g
    int_ux2g_d
    int_ux2g_m
    int_ux3g
    int_ux3g_d
    int_ux3g_m
    int_vg
    int_vxg
    int_vx2g
    int_vx2g_d
    int_vx2g_m
    int_vx3g
    int_vx3g_d
    int_vx3g_m

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
    args = args[0][0]
    argnum = len(args)

    # Prescreening
    # All Elementary integrals have g1, g2, t
    assert argnum >= 3, f"All integrals require at least three arguments (g1, g2, t). Received {len(args)}"
    assert isinstance(args[0], g.gaussian.Gaussian) and isinstance(args[1], g.gaussian.Gaussian), f"Two Gaussians must be given as the first two arguments"
    assert isinstance(args[2], int), f"Timestep must be given as index in the third argument"

    g1 = args[0]
    g2 = args[1]
    t = args[2]
    vindex = 0

    if '_v' in rq:
        assert argnum >= 4, f"Integrals of type v- must have at least four arguments (g1, g2, t, vindex). Received {argnum}"
        assert isinstance(args[3], int), f"Integrals of type v- must have vindex given as fourth argument. Received type {type(args[3])}"
        vindex = args[3]


    if rq == 'int_gg':
        return anal_intor.int_gg(g1, g2, t)
    elif rq == 'int_gxg':
        assert argnum >= 4, f"Integrals of type gxg must have at least four arguments (g1, g2, t, index). Received {argnum}"
        return anal_intor.int_gxg(g1, g2, t, args[3], kwargs)
    elif rq == 'int_gx2g':
        assert argnum >= 4, f"Integrals of type gx2g and gx2g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        if argnum >= 5:
            if args[3] != args[4]:
                return anal_intor.int_gx2g_mixed(g1, g2, t, args[3], args[4], kwargs)
        return anal_intor.int_gx2g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_gx2g_d':
        assert argnum >= 4, f"Integrals of type gx2g and gx2g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        return anal_intor.int_gx2g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_gx2g_m':
        assert argnum >= 5, f"Integrals of type gx2g_m must have at least five arguments (g1, g2, t, index1, index2), Received {argnum}"
        return anal_intor.int_gx2g_mixed(g1, g2, t, args[3], args[4], kwargs)
    elif rq == 'int_gx3g':
        assert argnum >= 4, f"Integrals of type gx3g and gx3g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        if argnum >= 5:
            if args[3] != args[4]:
                return anal_intor.int_gx3g_mixed(g1, g2, t, args[3], args[4], kwargs)
        return anal_intor.int_gx3g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_gx3g_d':
        assert argnum >= 4, f"Integrals of type gx3g and gx3g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        return anal_intor.int_gx3g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_gx3g_m':
        assert argnum >= 5, f"Integrals of type gx3g_m must have at least five arguments (g1, g2, t, index1, index2), Received {argnum}"
        return anal_intor.int_gx3g_mixed(g1, g2, t, args[3], args[4], kwargs)
    elif rq == 'int_ug':
        return anal_intor.int_ug(g1, g2, t)
    elif rq == 'int_uxg':
        assert argnum >= 4, f"Integrals of type uxg must have at least four arguments (g1, g2, t, index1), Received {argnum}"
        return anal_intor.int_uxg(g1, g2, t, args[3], kwargs) 
    elif rq == 'int_ux2g':
        assert argnum >= 4, f"Integrals of type ux2g and ux2g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        if argnum >= 5:
            if argnum[3] != argnum[4]:
                return anal_intor.int_ux2g_mixed(g1, g2, t, args[3], args[4], kwargs)
        return anal_intor.int_ux2g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_ux2g_d':
        assert argnum >= 4, f"Integrals of type ux2g and ux2g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        return anal_intor.int_ux2g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_ux2g_m':
        assert argnum >= 5, f"Integrals of type ux2g_m must have at least five arguments (g1, g2, t, index1, index2), Received {argnum}"
        return anal_intor.int_ux2g_mixed(g1, g2, t, args[3], args[4], kwargs)
    elif rq == 'int_ux3g':
        assert argnum >= 4, f"Integrals of type ux3g and ux3g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        if argnum >= 5:
            if args[3] != args[4]:
                return anal_intor.int_ux3g_mixed(g1, g2, t, args[3], args[4], kwargs)
        return anal_intor.int_ux3g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_ux3g_d':
        assert argnum >= 4, f"Integrals of type ux3g and ux3g_d must have at least four arguments (g1, g2, t, index1, [index2]), Received {argnum}"
        return anal_intor.int_ux3g_diag(g1, g2, t, args[3], kwargs)
    elif rq == 'int_ux3g_m':
        assert argnum >= 5, f"Integrals of type ux3g_m must have at least five arguments (g1, g2, t, index1, index2), Received {argnum}"
        return anal_intor.int_ux3g_mixed(g1, g2, t, args[3], args[4], kwargs)
    elif rq == 'int_vg':
        return anal_intor.int_vg(g1, g2, t, vindex)
    elif rq == 'int_vxg':
        assert argnum >= 5, f"Integrals of type vxg must have at least five arguments (g1, g2, t, vindex, index), Received {argnum}"
        return anal_intor.int_vxg(g1, g2, t, vindex, args[4], kwargs)
    elif rq == 'int_vx2g':
        assert argnum >= 5, f"Integrals of type vx2g and vx2g_d must have at least five arguments (g1, g2, t, vindex, index1, [index2]), Received {argnum}"
        if argnum >= 6:
            if args[4] != args[5]:
                return anal_intor.int_vx2g_mixed(g1, g2, t, vindex, args[4], args[5], kwargs)
        return anal_intor.int_vx2g_diag(g1, g2, t, vindex, args[4], args[5], kwargs)
    elif rq == 'int_vx2g_d':
        assert argnum >= 5, f"Integrals of type vx2g and vx2g_d must have at least five arguments (g1, g2, t, vindex, index1, [index2]), Received {argnum}"
        return anal_intor.int_vx2g_diag(g1, g2, t, vindex, args[4], kwargs)
    elif rq == 'int_vx2g_m':
        assert argnum >= 6, f"Integrals of type vx2g_m must have at least six arguments (g1, g2, t, vindex, index1, index2), Received {argnum}"
        return anal_intor.int_vx2g_mixed(g1, g2, t, vindex, args[4], args[5], kwargs)
    elif rq == 'int_vx3g':
        assert argnum >= 5, f"Integrals of type vx3g and vx3g_d must have at least five arguments (g1, g2, t, vindex, index1, [index2]), Received {argnum}"
        if argnum >= 6:
            if args[4] != args[5]:
                return anal_intor.int_vx3g_mixed(g1, g2, t, vindex, args[4], args[5], kwargs)
        return anal_intor.int_vx3g_diag(g1, g2, t, vindex, args[4], args[5], kwargs)
    elif rq == 'int_vx3g_d':
        assert argnum >= 5, f"Integrals of type vx3g and vx3g_d must have at least five arguments (g1, g2, t, vindex, index1, [index2]), Received {argnum}"
        return anal_intor.int_vx3g_diag(g1, g2, t, vindex, args[4], args[5], kwargs)
    elif rq == 'int_vx3g_m':
        assert argnum >= 5, f"Integrals of type vx3g_m must have at least six arguments (g1, g2, t, vindex, index1, index2), Received {argnum}"
        return anal_intor.int_vx3g_mixed(g1, g2, t, vindex, args[4], kwargs)
    else:
        raise g.IIRSException(rq, "elem")

