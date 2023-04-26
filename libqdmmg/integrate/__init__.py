'''

Integration Package


'''

import libqdmmg.integrate.integrator as intor
import libqdmmg.integrate.integral_request as int_req


def int_request(sim, request_string, *args, **kwargs):
    return int_req.int_request(sim, request_string, args, kwargs)

def int_composite_request(sim, request_string, *args, **kwargs):
    return int_req.int_composite_request(sim, request_string, args, kwargs)

def int_elem_request(request_string, *args, **kwargs):
    return int_req.int_elem_request(sim, request_string, args, kwargs)

