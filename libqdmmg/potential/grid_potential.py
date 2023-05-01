'''

@author Linus Bjarne Dittmer

'''

import numpy
import scipy.interpolate
import json
import libqdmmg.integrate as intor
from libqdmmg.potential.potential import Potential, PotentialIntegrator

class GridPotential(Potential):
    '''
    This class 
    '''

    def __init__(self, sim, filename):
        super().__init__(sim)
        self.filename = filename
        self.load_grid()

    def load_grid(self):
        gridfile_content = ""
        with open(self.filename, 'w') as f:
            gridfile_content = f.readlines()
        gridfile_dict = json.loads(gridfile_content)
        assert self.validate_gridfile(gridfile_dict), f"Unexpected gridfile format, refer to proper structure."
        self.save_grid_data(gridfile_dict)


    def validate_gridfile(self, gridfile_dict):
        keyset = [k.lower() for k in gridfile_dict]
        keyset_c = [k for k in gridfile_dict]
        if not 'name' in keyset:
            return False
        if not 'reduced_mass' in keyset:
            return False
        if not len(gridfile_dict[keyset_c[keyset.index('reduced_mass')]]) != self.sim.dim:
            return False
        if not 'gridpoints' in keyset:
            return False
        gridpoints = gridfile_dict[keyset_c[keyset.index('reduced_mass')]]
        g_keyset = [k.lower() for k in gridpoints]
        g_keyset_c = [k for k in gridpoints]
        if not 'data' in g_keyset:
            return False
        if not len(gridpoints[g_keyset_c[g_keyset.index('data')]]) % 2 == 0:
            return False
        return True

    def save_grid_data(self, gridfile_dict):
        keyset = [k.lower() for k in gridfile_dict]
        keyset_c = [k for k in gridfile_dict]
        self.name = gridfile_dict[keyset_c[keyset.index('name')]]
        self.reduced_mass = gridfile_dict[keyset_c[keyset.index('reduced_mass')]]
        self.grid_data = {}
        gridpoints = gridfile_dict[keyset_c[keyset.index('gridpoints')]]
        kg = [k for k in gridpoints]
        k = [k.lower() for k in kg]
        data = gridpoints[kg[k.index('data')]]
        grid_point_list = []
        grid_point_vals = []
        for j in range(int(0.5*len(data))):
            grid_point_list.append(numpy.array(data[2*j]))
            grid_point_vals.append(numpy.array(data[2*j+1]))
        self.grid_point_list = numpy.array(grid_point_list)
        self.grid_point_vals = numpy.array(grid_point_vals)

    def evaluate(self, x):
        return scipy.interpolate.griddata(self.grid_point_list, self.grid_point_vals, x, method='cubic')

    def gen_potential_integrator(self):
        return GridPotentialIntegrator(self)


class GridPotentialIntegrator(PotentialIntegrator):

    def __init__(self, potential, resolution):
        super().__init__(potential)
        assert isinstance(self.potential, GridPotential), f"Grid Potential Integrator is only for Grid Potentials or inheriting classes. Received class of type {type(potential)}"
        self.resolution = resolution
        self.num_intor = intor.NumericalIntegrator(self.potential.sim)
        self.num_intor.bind_potential(self.pot)

    def prepare_integration(self, g1, g2, t, int_type='int_gVg', index1=0, index2=0):
        self.grid = intor.Grid(self.potential.sim, self.resoultion)
        if '_u' in int_type:
            self.grid.define_by_ug(g1, g2, t)
            self.num_intor.bind_function(g1.evaluateU)
        elif '_v' in int_type:
            self.grid.define_by_vg(g1, g2, t)
            self.num_intor.bind_function(g1.evaluateV)
        else:
            self.grid.define_by_two_gaussians(g1, g2, t)
            self.num_intor.bind_function(g1.evaluate)
        self.num_intor.bind_grid(self.grid)
        self.num_intor.bind_function3(g2.evaluate)

        def fx(x, t):
            return x[index1]
        def fx2_m(x, t):
            return x[index1]*x[index2]
        def fx2_d(x, t):
            return x[index1]*x[index1]
        def fx3_m(x, t):
            return x[index1]*x[index2]*x[index2]
        def fx3_d(x, t):
            return x[index1]*x[index1]*x[index1]

        if 'Vxg' in int_type:
            self.num_intor.bind_function2(fx)
        elif 'Vx2g_mixed' in int_type:
            self.num_intor.bind_function2(fx2_m)
        elif 'Vx2g_diag' in int_type:
            self.num_intor.bind_function2(fx2_d)
        elif 'Vx3g_mixed' in int_type:
            self.num_intor.bind_function2(fx3_m)
        elif 'Vx3g_diag' in int_type:
            self.num_intor.bind_function2(fx3_d)


    def _int_gVg(self, g1, g2, t):
        self.prepare_integration(g1, g2, t)
        return self.num_intor.integrate(t)

    def _int_gVxg(self, g1, g2, t, index):
        self.prepare_integration(g1, g2, t, int_type='int_gVxg', index1=index)
        return self.num_intor.integrate(t)

    def _int_gVx2g_mixed(self, g1, g2, t, index1, index2):
        self.prepare_integration(g1, g2, t, int_type='int_gVx2g_mixed', index1=index1, index2=index2)
        return self.num_intor.integrate(t)

    def _int_gVx2g_diag(self, g1, g2, t, index):
        self.prepare_integration(g1, g2, t, int_type='int_gVx2g_diag', index1=index)
        return self.num_intor.integrate(t)

    def _int_gVx3g_mixed(self, g1, g2, t, index1, index2):
        self.prepare_integration(g1, g2, t, int_type='int_gVx3g_mixed', index1=index1, index2=index2)
        return self.num_intor.integrate(t)

    def _int_gVx3g_diag(self, g1, g2, t, index):
        self.prepare_integration(g1, g2, t, int_type='int_gVx3g_diag', index1=index)
        return self.num_intor.integrate(t)

    
    def _int_uVg(self, g1, g2, t):
        self.prepare_integration(g1, g2, t)
        return self.num_intor.integrate(t)

    def _int_uVxg(self, g1, g2, t, index):
        self.prepare_integration(g1, g2, t, int_type='int_uVxg', index1=index)
        return self.num_intor.integrate(t)

    def _int_uVx2g_mixed(self, g1, g2, t, index1, index2):
        self.prepare_integration(g1, g2, t, int_type='int_uVx2g_mixed', index1=index1, index2=index2)
        return self.num_intor.integrate(t)

    def _int_uVx2g_diag(self, g1, g2, t, index):
        self.prepare_integration(g1, g2, t, int_type='int_uVx2g_diag', index1=index)
        return self.num_intor.integrate(t)

    def _int_uVx3g_mixed(self, g1, g2, t, index1, index2):
        self.prepare_integration(g1, g2, t, int_type='int_uVx3g_mixed', index1=index1, index2=index2)
        return self.num_intor.integrate(t)

    def _int_uVx3g_diag(self, g1, g2, t, index):
        self.prepare_integration(g1, g2, t, int_type='int_uVx3g_diag', index1=index)
        return self.num_intor.integrate(t)


    def _int_vVg(self, g1, g2, t, vindex):
        self.prepare_integration(g1, g2, t)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)

    def _int_vVxg(self, g1, g2, t, vindex, index):
        self.prepare_integration(g1, g2, t, int_type='int_vVxg', index1=index)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)

    def _int_vVx2g_mixed(self, g1, g2, t, vindex, index1, index2):
        self.prepare_integration(g1, g2, t, int_type='int_vVx2g_mixed', index1=index1, index2=index2)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)

    def _int_vVx2g_diag(self, g1, g2, t, vindex, index):
        self.prepare_integration(g1, g2, t, int_type='int_vVx2g_diag', index1=index)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)

    def _int_vVx3g_mixed(self, g1, g2, t, vindex, index1, index2):
        self.prepare_integration(g1, g2, t, int_type='int_vVx3g_mixed', index1=index1, index2=index2)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)

    def _int_vVx3g_diag(self, g1, g2, t, vindex, index):
        self.prepare_integration(g1, g2, t, int_type='int_vVx3g_diag', index1=index)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)
