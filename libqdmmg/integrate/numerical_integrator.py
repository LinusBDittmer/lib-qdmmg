'''

@author Linus Bjarne Dittmer

'''


import numpy


class NumericalIntegrator:
    '''
    This class is the active integrator for numerical integration of potentials. It is employed by the potential integration handler and explicitly performs numerical integration. It has four functions (function, function2, function3, potential) that are multiplied together to form the integrand.

    Attributes
    ----------
    sim : libqdmmg.simulate.simulation.Simulation
        The main simulation instance holding all relevant information
    grid : libqdmmg.integrate.grid.Grid
        The integration grid used for constructing points in cartesian space and managing integration weights.
    function : callable
        The first factor in the integrand. Must accept (x, t) as only non-keyword arguments with x as a 1D ndarray with shape (dimensions,) and t as an integer timestep index and must return exactly one numerical value. If left unitialised, this will be self.one
    function2 : callable
        See function.
    function3 : callable
        See function.
    potential : callable
        See function. Is usually the potential.
    '''

    def __init__(self, sim):
        '''
        Constructor for the Numerical Integrator class.

        Parameters
        ----------
        sim : libqdmmg.simulate.simulation.Simulation
            The man simulation instance holding all relevant information.
        '''
        self.sim = sim
        self.grid = None
        self.function = self.one
        self.function2 = self.one
        self.function3 = self.one
        self.potential = self.one

    def bind_grid(self, grid):
        '''
        This function is used to define the internal integration grid. It is an effective setter for the self.grid variable.

        Parameters
        ----------
        grid : libqdmmg.integrate.grid.Grid
            The new integration grid.
        '''
        self.grid = grid

    def bind_function(self, func):
        '''
        This function is used to define the first function that is used to form the integrand product. The callable function must receive a 1D ndarray of shape (dimension,) in the first argument slot and the timestep index as an integer in the second argument slot. No other non-keyword arguments are permitted. Additionally, the function must return exactly one numerical value (usually complex128 or float32)

        Parameters
        ----------
        func : callable
            The function. See above for restrictions

        See Also
        --------
        bind_function2 : set the second function
        bind_function3 : set the third function
        bind_potential : set the potential
        '''
        self.function = func

    def bind_function2(self, func):
        '''
        This function is used to define the second function in the integrand product. See bind_function for details.

        Parameters
        ----------
        func : callable
            The function. See above for restrictions

        See Also
        --------
        bind_function : set the first function
        bind_function3 : set the third function
        bind_potential : set the potential
        '''
        self.function2 = func

    def bind_function3(self, func):
        '''
        This function is used to define the third function in the integrand product. See bind_function for details.

        Parameters
        ----------
        func : callable
            The function. See above for restrictions

        See Also
        --------
        bind_function : set the first function
        bind_function2 : set the second function
        bind_potential : set the potential
        '''
        self.function3 = func

    def bind_potential(self, potential):
        '''
        This function is used to define the fourth function (usually the potential) in the integrand product. See bind_function for details.

        Parameters
        ----------
        potential : callable
            The function. See above for restrictions

        See Also
        --------
        bind_function : set the first function
        bind_function2 : set the second function
        bind_function3 : set the third function
        '''
        self.potential = potential

    def one(self, x, t):
        '''
        A default function that functions as a placeholder in the self.function, self.function2, self.function3 and self.potential attributes. Always returns 1

        Parameters
        ----------
        x : 1D ndarray
            The position variable with shape (dimensions,)
        t : int
            The timestep index

        Returns
        -------
        one : float32
            Just the number 1.
        '''
        return 1.0

    def grid_eval(self, index, t):
        '''
        Evaluate the integrand on the grid at the gridpoint with index (index) at timestep t.

        Parameters
        ----------
        index : indexable
            A list, tuple or 1D ndarray with shape (dimensions,) containing only integers between 0 and grid.resolution-1 that reflects the gridpoint index.
        t : int
            The timestep index

        Returns
        -------
        val : complex128
            The value of the integrand at the specified gridpoint.
        '''
        p = self.grid.gridpoint(index)
        return self.function(p, t) * self.function2(p, t) * self.function3(p, t) * self.potential.evaluate(p)

    def integrate(self, t):
        '''
        Perform numeric integration over the grid.

        Parameters
        ----------
        t : int
            The timestep index

        Returns
        -------
        int_val : complex128
            The result of numerical integration

        Raises
        ------
        AssertionError
            If the grid is None
        '''
        assert self.grid is not None, f"Grid required for integration, currently grid is None"

        index = numpy.zeros(self.sim.dim, dtype=numpy.int32)
        int_val = 0.0
        for indexnumber in range(self.grid.resolution**self.sim.dim):
            int_val += self.grid_eval(index, t) * self.grid.pointweight(index, t)

            index[0] += 1
            for i in range(self.sim.dim-1):
                if index[i] >= self.grid.resolution:
                    index[i] = 0
                    index[i+1] += 1
                else:
                    break

        return int_val


if __name__ == '__main__':
    import libqdmmg.simulate as sim
    import libqdmmg.integrate as intor
    import libqdmmg.general as gen
    s = sim.Simulation(1, 1, dim=3)
    g = intor.Grid(s, 10)
    gauss = gen.Gaussian(s)
    g.define_by_gaussian(gauss, 0)
    ni = NumericalIntegrator(s)
    ni.bind_grid(g)
    ni.bind_function(gauss.evaluate)
    print(ni.integrate(0))
    print(numpy.pi**(1.5))

