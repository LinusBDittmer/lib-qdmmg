'''

author: Linus Bjarne Dittmer

This script contains classes for handling custom errors and exceptions. These are called in situations not covered by traditional python exceptions and include:

    InvalidIntegralRequestStringException (IIRSException):
    This is called when the function libqdmmg.integrate.int_request(...) is called with an invalid request string.

    SimulationNotRunException (SNRException):
    This is called when analysis code is attempted to run on a simulation that has not yet generated or imported a wavefunction.

    InvalidJSONFlagException:
    This is called when an import from a JSON file fails.

'''


class InvalidIntegralRequestStringException(Exception):
    '''
    This class is an Exception Class for Invalid Integral Strings. If the function libqdmmg.integrate.int_request(...) is called and the provided integral request string is not valid, this exception is thrown.

    Attributes
    ----------
    rq : str
        The invalid request string.
    int_class : str
        The type of request.

    '''

    def __init__(self, rq, int_class, *args):
        super().__init__(args)
        self.rq = rq
        self.int_class = int_class.lower().strip()

    def __str__(self):
        a = ""
        if self.int_class == 'elem':
            a = "elementary "
        elif self.int_class == 'comp':
            a = "composite "
        return f"The received integral request string is not a valid {a}integral descriptor." 

class SimulationNotRunException(Exception):
    '''
    This class is an Exception that is thrown whenever something is called that requires a complete wavefunction, but the wavefunction has not been generated or imported yet.

    Attributes
    ----------
    sim : libqdmmg.simulate.simulation.Simulation
        The simulation.

    '''


    def __init__(self, sim, *args):
        super().__init__(args)
        self.sim = sim

    def __str__(self):
        return f"Simulation was not run yet, no data can be given."

class InvalidJSONFlagException(Exception):
    '''
    This class in an Exception that is called upon error of import of JSON files.

    Attributes
    ----------
    path : str
        The path to the JSON file.

    '''

    def __init__(self, path, *args):
        super().__init__(args)
        self.path = path
    
    def __str__(self):
        return f"The JSON file loaded from {self.path} does not contain a Wavepacket."

