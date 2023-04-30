'''

Basic logging class and functionality

'''

SILENT = 0
ERROR = 1
ERROR_AND_WARNING = 2
INFO = 3
DEBUG_1 = 4
DEBUG_2 = 5
DEBUG_3 = 6

class Logger:
    '''
    Logger class for regulated output to the standard out and standard err.

    Attributes
    ----------
    _verbose : int, private
        See variables above, _verbose defines the level of output. Options are
        SILENT: no output
        ERROR: only error messages are printed
        ERROR_AND_WARNING: error messages and warnings are printed
        INFO: informative messages, error messages and warnings are printed. This is the recommended setting for typical use.
        DEBUG_1: enables additinoal printing of debug level 1
        DEBUG_2: enables additional printing of debug level 1 and 2
        DEBUG_3: enables additional printing of debug level 1, 2 and 3

    '''

    def __init__(self, verbose=-1):
        '''
        Constructor for the Logger class. See doc for more information.
        '''
        self._verbose = verbose

    def info(self, msg):
        '''
        Log informative message. This is only printed if _verbose is at least 3 (INFO).

        Parameters
        ----------
        msg : object
            Message to be printed. This can be any datatype that can be converted to a string via str(...)
        '''
        msg = str(msg)
        if self._verbose >= INFO:
            print(msg)

    def error(self, msg, exitCode=0):
        '''
        Log error message. This is only printed if _verbose is at least 1 (ERROR). Note that if an exit code is given, the program is not terminated directly.

        Parameters
        ----------
        msg : object
            Message to be printed. This can be any datatype that can be converted to a string via str(...)
        exitCode : int, optional
            Exit code if program exits. This is only printed if non-zero.
        '''
        msg = str(msg)
        if self._verbose >= ERROR:
            print('\033[1;31m ' + msg + ' \033[0;0m')
            if exitCode != 0:
                print()
                print('\033[1;31m PROGRAM EXITED UNEXPECTEDLY WITH CODE ' + str(exitCode) + ' \033[0;0m')

    def warn(self, msg):
        '''
        Log warning. This is only printed if _verbose is at least 2 (ERROR_AND_WARNING).

        Parameters
        ----------
        msg : object
            Message to be printed. This can be any datatype that can be converted to a string via str(...)
        '''
        msg = str(msg)
        if self._verbose >= ERROR_AND_WARNING:
            print('\033[3;33m ' + msg + ' \033[0;0m')

    def debug1(self, msg):
        '''
        Debug message of level 1. This is only printed if _verbose is at least 4 (DEBUG_1).

        Parameters
        ----------
        msg : object
            Message to be printed. This can be any datatype that can be converted to a string via str(...)
        '''
        msg = str(msg)
        if self._verbose >= DEBUG_1:
            print(msg)

    def debug2(self, msg):
        '''
        Debug message of level 2. This is only printed if _verbose is at least 5 (DEBUG_2).

        Parameters
        ----------
        msg : object
            Message to be printed. This can be any datatype that can be converted to a string via str(...)
        '''
        msg = str(msg)
        if self._verbose >= DEBUG_2:
            print(msg)

    def debug3(self, msg):
        '''
        Debug message of level 3. This is only printed if _verbose is 6 (DEBUG_3).

        Parameters
        ----------
        msg : object
            Message to be printed. This can be any datatype that can be converted to a string via str(...)
        '''
        msg = str(msg)
        if self._verbose >= DEBUG_3:
            print(msg)




'''

Logger generation function

'''

def new_logger(sim):
    '''
    Encapsulation of the Logger constructor. This is to avoid mutation errors regarding the verbose level and thus to avoid discrepancies between simulation.verbose and simulation.logger._verbose.

    Parameters
    ----------
    sim : libqdmmg.simulate.simulation.Simulation
        Main instance of the simulation.

    Returns
    -------
    logger : libqdmmg.general.logger.Logger
        Logger instance.
    '''
    return Logger(sim.verbose)

