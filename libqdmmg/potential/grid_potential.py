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
    This class is a subclass of Potential to represent potentials that are given by a grid. These include physical potentials that are imported from external sources. GridPotentials require numerical integration. All relevant data is given in JSON format, specifically the following:

    {
        "name": "Potential Name",
        "reduced_mass": [1.0, 2.0],
        "gridpoints": {
           "data": [[0.0, 0.0], 0.5, [0.0, 1.0], 0.6, [1.0, 0.0], 0.6, [1.0, 1.0], 0.8]
        }
    }

    The JSON descriptors are not case-sensitive. Additional attributes are discarded.

    Attributes
    ----------
    filename : str
        Path or file name to the gridfile of .json format including the .json ending
    name : str
        Name of the Potential
    reduced_mass : 1D ndarray
        Array of shape (dimensions,) that includes the reduced masses along each coordinate
    grid_point_list : 2D ndarray
        Array of shape (num of points, dimensions) that includes all points at which the potential is given.
    grid_point_vals : 1D ndarray
        Array of shape (num of points,) that includes the values of the potential such that for each i V(grid_point_list[i]) = grid_point_vals[i]
    '''

    def __init__(self, sim, filename):
        '''
        Constructor for the GridPotential class.

        Parameters
        ----------
        sim : libqdmmg.simulate.simulation.Simulation
            Main simulation instance holding all relevant information.
        filename : str
            Path or filename to the gridefile of .json format including the .json ending
        '''
        super().__init__(sim)
        self.filename = filename
        self.load_grid()

    def load_grid(self):
        '''
        This function is invoked by the construtor and loads the data found in the gridfile. Then, the data is validated to have the correct format and content type, after which the data is saved into the class variables name, reduced_mass, grid_point_list, grid_point_vals.

        Raises
        ------
        AssertionError
            If the validation failes
        '''
        gridfile_content = ""
        with open(self.filename, 'w') as f:
            gridfile_content = f.readlines()
        gridfile_dict = json.loads(gridfile_content)
        assert self.validate_gridfile(gridfile_dict), f"Unexpected gridfile format, refer to proper structure."
        self.save_grid_data(gridfile_dict)


    def validate_gridfile(self, gridfile_dict):
        '''
        This function validates structure and data type of the loaded grid data.

        Parameters
        ----------
        gridfile_dict : dict
            The dictionary containing the JSON file data.

        Returns
        -------
        valid : bool
            Whether the gridfile_dict possesses the correct structure and content.
        '''
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
        '''
        This functions saves the received data into the local variables name, reduced_mass, grid_point_list and grid_point_vals

        Parameters
        ----------
        gridfile_dict : dict
            Dictionaary containing the JSON file data.
        '''
        keyset = [k.lower() for k in gridfile_dict]
        keyset_c = [k for k in gridfile_dict]
        self.name = gridfile_dict[keyset_c[keyset.index('name')]]
        self.reduced_mass = gridfile_dict[keyset_c[keyset.index('reduced_mass')]]
        gridpoints = gridfile_dict[keyset_c[keyset.index('gridpoints')]]
        kg = [k for k in gridpoints]
        k = [k.lower() for k in kg]
        data = gridpoints[kg[k.index('data')]]
        grid_point_list = []
        grid_point_vals = []
        for j in range(int(0.5*len(data))):
            grid_point_list.append(numpy.array(data[2*j]))
            grid_point_vals.append(float(data[2*j+1]))
        self.grid_point_list = numpy.array(grid_point_list)
        self.grid_point_vals = numpy.array(grid_point_vals)

    def evaluate(self, x):
        '''
        === Overriden from superclass. ===

        This function evaluates the potential at the given point x using linear interpolation.

        Parameters
        ----------
        x : 1D ndarray
            Array of shape (dimensions,) at which the potential is to be evaluated.

        Returns
        -------
        pot_val : complex128
            Value of the potential at the given point.
        '''
        return scipy.interpolate.griddata(self.grid_point_list, self.grid_point_vals, x, method='linear')

    def gen_potential_integrator(self):
        '''
        === Overriden from superclass. ===

        This function creates an instance of GridPotentialIntegrator for management of numerical integration.

        Returns
        -------
        gpi : libqdmmg.potential.grid_potential.GridPotentialIntegrator
            An instance of GridPotentialIntegrator linked to the calling instance of GridPotential
        '''
        return GridPotentialIntegrator(self)


class GridPotentialIntegrator(PotentialIntegrator):
    '''
    This class handles integration of a GridPotential using Numerical Integration methods.

    Attributes
    ----------
    resolution : int
        Resolution of the Integration grid.
    num_intor : libqdmmg.integrate.numerical_integrator.NumericalIntegrator
        Instance of NumericalIntegrator that handles numerical integration.
    grid : libqdmmg.integrate.grid.Grid
        Numerical integration grid
    '''


    def __init__(self, potential, resolution):
        '''
        Constructor for the GridPotentialIntegrator.

        Parameters
        ----------
        potential : libqdmmg.potential.potential.Potential
            Parent Potential which is included in integration.
        resolution : int
            Resolution of the integration grid.

        Raises
        ------
        AssertionError
            If the given potential is not a GridPotential
        '''
        super().__init__(potential)
        assert isinstance(self.potential, GridPotential), f"Grid Potential Integrator is only for Grid Potentials or inheriting classes. Received class of type {type(potential)}"
        self.resolution = resolution
        self.num_intor = intor.NumericalIntegrator(self.potential.sim)
        self.num_intor.bind_potential(self.pot)
        self.grid = None

    def prepare_integration(self, g1, g2, t, int_type='int_gVg', index1=0, index2=0):
        '''
        This function prepares the grid and numerical integrator for subsequent integration.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            The first Gaussian which is either used directly in the integration or whose u- or v-dual Gaussians are used.
        g2 : libqdmmg.general.gaussian.Gaussian
            The second Gaussian which is used directly in the integration.
        t : int
            Timestep index.
        int_type : str, optional
            Integration Request string. The classification is required for some steps in the initialisation of the grid and numerical integrator. Default is "int_gVg"
        index1 : int, optional
            First directional index, if required. If only one index is required, this index is used. Default 0
        index2 : int, optional
            Second directional index. Default 0
        '''
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
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t)
        return self.num_intor.integrate(t)

    def _int_gVxg(self, g1, g2, t, index):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_gVxg', index1=index)
        return self.num_intor.integrate(t)

    def _int_gVx2g_mixed(self, g1, g2, t, index1, index2):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_gVx2g_mixed', index1=index1, index2=index2)
        return self.num_intor.integrate(t)

    def _int_gVx2g_diag(self, g1, g2, t, index):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_gVx2g_diag', index1=index)
        return self.num_intor.integrate(t)

    def _int_gVx3g_mixed(self, g1, g2, t, index1, index2):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_gVx3g_mixed', index1=index1, index2=index2)
        return self.num_intor.integrate(t)

    def _int_gVx3g_diag(self, g1, g2, t, index):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_gVx3g_diag', index1=index)
        return self.num_intor.integrate(t)

    
    def _int_uVg(self, g1, g2, t):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t)
        return self.num_intor.integrate(t)

    def _int_uVxg(self, g1, g2, t, index):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_uVxg', index1=index)
        return self.num_intor.integrate(t)

    def _int_uVx2g_mixed(self, g1, g2, t, index1, index2):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_uVx2g_mixed', index1=index1, index2=index2)
        return self.num_intor.integrate(t)

    def _int_uVx2g_diag(self, g1, g2, t, index):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_uVx2g_diag', index1=index)
        return self.num_intor.integrate(t)

    def _int_uVx3g_mixed(self, g1, g2, t, index1, index2):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_uVx3g_mixed', index1=index1, index2=index2)
        return self.num_intor.integrate(t)

    def _int_uVx3g_diag(self, g1, g2, t, index):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_uVx3g_diag', index1=index)
        return self.num_intor.integrate(t)


    def _int_vVg(self, g1, g2, t, vindex):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)

    def _int_vVxg(self, g1, g2, t, vindex, index):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_vVxg', index1=index)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)

    def _int_vVx2g_mixed(self, g1, g2, t, vindex, index1, index2):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_vVx2g_mixed', index1=index1, index2=index2)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)

    def _int_vVx2g_diag(self, g1, g2, t, vindex, index):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_vVx2g_diag', index1=index)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)

    def _int_vVx3g_mixed(self, g1, g2, t, vindex, index1, index2):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_vVx3g_mixed', index1=index1, index2=index2)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)

    def _int_vVx3g_diag(self, g1, g2, t, vindex, index):
        '''
        === Overriden from superclass. ===

        See Also
        --------
        Potential :
            Superclass
        '''
        self.prepare_integration(g1, g2, t, int_type='int_vVx3g_diag', index1=index)
        return g1.width[vindex] / g1.width[0] * self.num_intor.integrate(t)
