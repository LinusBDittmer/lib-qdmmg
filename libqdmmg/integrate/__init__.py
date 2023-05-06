'''

Integration Package


'''

import libqdmmg.integrate.integral_request as int_req

def int_request(sim, request_string, *args, **kwargs):
    return int_req.int_request(sim, request_string, args, kwargs)

def int_composite_request(sim, request_string, *args, **kwargs):
    return int_req.int_composite_request(sim, request_string, args, kwargs)

def int_atom_request(sim, request_string, *args, **kwargs):
    return int_req.int_atom_request(sim, request_string, args, kwargs)



