'''

@author: Linus Bjarne Dittmer

'''


class Potential:
    

    def __init__(self, sim):
        self.sim = sim
        self.logger = sim.logger

    def evaluate(self, x):
        '''
        To be overridden
        '''
        return 0


class GridPotential(Potential):

    def __init__(self, sim, filename):
        super().__init__(sim)
        self.filename = filename
        self.load_grid()

    def load_grid(self):
        self.grid = numpy.zeros([10]*sim.dim)


        


