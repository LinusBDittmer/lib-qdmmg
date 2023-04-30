'''

Integration Package


'''

import libqdmmg.integrate.integral_request as int_req
import libqdmmg.integrate.grid as grid
import libqdmmg.integrate.numerical_integrator as num_intor

def int_request(sim, request_string, *args, **kwargs):
    return int_req.int_request(sim, request_string, args, kwargs)

def int_composite_request(sim, request_string, *args, **kwargs):
    return int_req.int_composite_request(sim, request_string, args, kwargs)

def int_elem_request(sim, request_string, *args, **kwargs):
    return int_req.int_elem_request(sim, request_string, args, kwargs)

def Grid(sim, resolution):
    return grid.Grid(sim, resolution)

def NumericalIntegrator(sim):
    return num_intor.NumericalIntegrator(sim)


