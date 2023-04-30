'''

@author: Linus Bjarne Dittmer

'''

import numpy
from functools import reduce

class Potential:
    

    def __init__(self, sim):
        self.sim = sim
        self.logger = sim.logger

    def evaluate(self, x):
        '''
        To be overridden
        '''
        return 0

    def gen_potential_integrator(self):
        '''
        To be overriden
        '''
        return None


class PotentialIntegrator:

    def __init__(self, potential):
        self.potential = potential
        self.logger = potential.sim.logger

    def int_request(self, request_string, *args, **kwargs):
        rq = request_string.lower().strip()
        if rq[0] == '_':
            rq = rq[1:]
        argnum = len(args)

        assert argnum >= 3, f"Expected at least 3 arguments (g1, g2, t). Received {argnum}."

        if rq == 'int_gVg':
            return self._int_gVg(args)


    def _int_gVg(self, g1, g2, t):
        '''
        To be overridden
        '''
        return 0

    def _int_gVxg(self, g1, g2, t, index):
        '''
        To be overridden
        '''
        return 0

    def _int_gVx2g_mixed(self, g1, g2, t, index1, index2):
        '''
        To be overridden
        '''
        return 0

    def _int_gVx2g_diag(self, g1, g2, t, index):
        '''
        To be overridden
        '''
        return 0

    def _int_gVx3g_mixed(self, g1, g2, t, index1, index2):
        '''
        To be overridden
        '''
        return 0

    def _int_gVx3g_diag(self, g1, g2, t, index):
        '''
        To be overridden
        '''
        return 0

    def _int_uVg(self, g1, g2, t):
        '''
        To be overridden
        '''
        return 0

    def _int_uVxg(self, g1, g2, t, index):
        '''
        To be overridden
        '''
        return 0

    def _int_uVx2g_mixed(self, g1, g2, t, index1, index2):
        '''
        To be overridden
        '''
        return 0

    def _int_uVx2g_diag(self, g1, g2, t, index):
        '''
        To be overridden
        '''
        return 0

    def _int_uVx3g_mixed(self, g1, g2, t, index1, index2):
        '''
        To be overridden
        '''
        return 0

    def _int_uVx3g_diag(self, g1, g2, t, index):
        '''
        To be overridden
        '''
        return 0

    def _int_vVg(self, g1, g2, t, vindex):
        '''
        To be overridden
        '''
        return 0

    def _int_vVxg(self, g1, g2, t, vindex, index):
        '''
        To be overridden
        '''
        return 0

    def _int_vVx2g_mixed(self, g1, g2, t, vindex, index1, index2):
        '''
        To be overridden
        '''
        return 0

    def _int_vVx2g_diag(self, g1, g2, t, vindex, index):
        '''
        To be overridden
        '''
        return 0

    def _int_vVx3g_mixed(self, g1, g2, t, vindex, index1, index2):
        '''
        To be overridden
        '''
        return 0

    def _int_vVx3g_diag(self, g1, g2, t, vindex, index):
        '''
        To be overridden
        '''
        return 0




