'''

@author Linus Bjarne Dittmer

'''

import numpy

class Grid:
    '''
    This class is a blueprint for a numerical integration grid utilised by numerical integration routines. In the current version, it is orthonormal and permits only equidistant points.

    Attributes
    ----------
    sim : libqdmmg.simulate.simulation.Simulation
        Main simulation object. Functions as a masterclass holding all relevant information
    resolution : int
        Number of points in the grid along each axis.
    gridaxes : 2D ndarray
        An array with shape (dimensions, resolution) such that the i-th axis is given by gridaxes[i]. All griddata can be extrapolated from this array.

    '''

    def __init__(self, sim, resolution):
        '''
        Constructor for the Grid class.

        Parameters
        ----------
        sim : libqdmmg.simulate.simulation.Simulation
            Main simulation object. Functions as a masterclass holding all relevant information
        resoultion : int
            Number of points in the grid along each axis.

        '''
        self.sim = sim
        self.resolution = resolution

    def define_by_gaussian(self, g1, t):
        '''
        This method is used to construct the grid to be optimal for integration of the Gaussian g1 at timestep t. This is achieved by constructing the grid around the centre of g1 at timestep t such that the grid spans 3 standard deviations in every direction from the centre. This leaves an error of approximately 0.01 % per dimension.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            The Gaussian around which the grid is to be constructed.
        t : int
            The timestep at which the Gaussian data should be used.

        See Also
        --------
        define_by_two_gaussians : Construct the grid by the product of two Gaussians.
        define_by_ug : Construct the grid by the product of a Gaussian and a u-dual Gaussian.
        define_by_vg : Construct the grid by the product of a Gaussian and a v-dual Gaussian.
        '''
        centre = g1.centre[t]
        halfwidth = 3.0 / g1.width
        self.gridaxes = numpy.array([numpy.linspace(centre[i]-halfwidth[i], centre[i]+halfwidth[i], num=self.resolution) for i in range(self.sim.dim)])

    def define_by_two_gaussians(self, g1, g2, t):
        '''
        This method is used to construct the grid to be optimal for integration of the Gaussian product g1 * g2 at timestep t. This is achieved by constructing the grid around the centre of g1 * g2 at timestep t such that the grid spans 3 standard deviations in every direction from the centre. This leaves an error of approximately 0.01 % per dimension.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            The first Gaussian.
        g2 : libqdmmg.general.gaussian.Gaussian
            The second Gaussian.
        t : int
            The timestep at which the Gaussian data should be used.

        See Also
        --------
        define_by_gaussian: Construct the grid by one Gaussian.
        define_by_ug : Construct the grid by the product of a Gaussian and a u-dual Gaussian.
        define_by_vg : Construct the grid by the product of a Gaussian and a v-dual Gaussian.
        '''
        halfwidth = 3.0 / (g1.width + g2.width)
        centre = (g1.width * g1.centre + g2.width * g2.centre) / (g1.width + g2.width)
        self.gridaxes = numpy.array([numpy.linspace(centre[i]-halfwidth[i], centre[i]+halfwidth[i], num=self.resolution) for i in range(self.sim.dim)])

    def define_by_ug(self, g1, g2, t):
        '''
        This method is used to construct the grid to be optimal for integration of the product u1 * g2 of a Gaussian g2 with a u-dual Gaussian of g1 at timestep t. This is achieved by constructing the grid around the centre of u1 * g2 at timestep t such that the grid spans 3 standard deviations in every direction from the centre. This leaves an error of approximately 0.01 % per dimension. Since the u-dual Gaussian of g1 shares a centre with g1, this is equivalent to treating it like a standard Gaussian.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            The Gaussian whose u-dual function is to be used.
        g2 : libqdmmg.general.gaussian.Gaussian
            The Gaussian which is to be used directly.
        t : int
            The timestep at which the Gaussian data should be used.

        See Also
        --------
        define_by_gaussian: Construct the grid by one Gaussian.
        define_by_two_gaussians : Construct the grid by the product of two Gaussians.
        define_by_vg : Construct the grid by the product of a Gaussian and a v-dual Gaussian.
        '''
        w = numpy.linalg.norm(g1.width) - g1.width
        halfwidth = 3.0 / (w + g2.width)
        centre = (w * g1.centre + g2.width * g2.centre) / (w + g2.width)
        self.gridaxes = numpy.array([numpy.linspace(centre[i]-halfwidth[i], centre[i]+halfwidth[i], num=self.resolution) for i in range(self.sim.dim)])

    def define_by_vg(self, g1, g2, t):
        '''
        This method is used to construct the grid to be optimal for integration of the product v1 * g2 of a Gaussian g2 with a v-dual Gaussian of g1 at timestep t. The directional index of v1 does not portray any significance since it only affects the scaling. This is achieved by constructing the grid around the centre of v1 * g2 at timestep t such that the grid spans 3 standard deviations in every direction from the centre. This leaves an error of approximately 0.01 % per dimension. Since the v-dual Gaussian of g1 is opposite to g1 with respect to the coordinate origin and shares its width array, this is equivalent to treating v1 like a copy of g1 merely with flipped origin.

        Parameters
        ----------
        g1 : libqdmmg.general.gaussian.Gaussian
            The Gaussian whose u-dual function is to be used.
        g2 : libqdmmg.general.gaussian.Gaussian
            The Gaussian which is to be used directly.
        t : int
            The timestep at which the Gaussian data should be used.

        See Also
        --------
        define_by_gaussian: Construct the grid by one Gaussian.
        define_by_two_gaussians : Construct the grid by the product of two Gaussians.
        define_by_ug : Construct the grid by the product of a Gaussian and a u-dual Gaussian.
        '''
        vg = g1.copy()
        vg.centre *= -1
        self.define_by_two_gaussians(vg, g2, t)

    def gridpoint(self, indices):
        '''
        This method generates a point in cartesian space from given index coordinates.

        Parameters
        ----------
        indices : 1D ndarray
            Array of shape (dimensions,) and dtype int32 containing the point indices
        
        Returns
        -------
        point : 1D ndarray
            The generated point with shape (dimensions,) and dtype float32
        '''
        return numpy.array([self.gridaxes[j,indices[j]] for j in range(self.sim.dim)])

    def pointweight(self, indices, t):
        '''
        This method calculates the pointweight as the inverse point density. This is used in numerical integration as the measure of the n-volume surrounding each point necessary for said integration.

        Parameters
        ----------
        indices : 1D ndarray
            Array of shape (dimensions,) and dtype int32 containing the point indices.
        t : int
            Timestep index

        Returns
        -------
        weight : float32
            Pointweight at index (indices) and timestep t
        '''
        return abs(numpy.prod(self.gridaxes[:,1]-self.gridaxes[:,0]))



if __name__ == '__main__':
    import libqdmmg.simulate as sim
    import libqdmmg.general as gen


    s = sim.Simulation(10, 0.1, dim=3)
    grid = Grid(s, 10)
    g = gen.Gaussian(s)

    grid.define_by_gaussian(g, 0)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                print(grid.gridpoint((i,j,k)))
