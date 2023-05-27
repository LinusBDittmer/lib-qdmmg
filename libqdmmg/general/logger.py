'''

Basic logging class and functionality

'''

import libqdmmg
import atexit
import sys

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
        self._header = False
        self._coda = False
        libqdmmg.register_logger(self)
        if libqdmmg.is_logger_cardinal(self):
            atexit.register(self.print_coda)

    def print_header(self):
        if self._header or not libqdmmg.is_logger_cardinal(self): return
        version = libqdmmg.__version__
        dirname = __file__[:__file__.rfind('/')]
        with open(dirname+'/res/header.txt') as f:
            c = "".join(f.readlines())
            c = c.replace("{version}", version)
            print(c)
        width = 80
        s = "=" * int(0.5 * (width - len(" OUTPUT ")))
        print(s + " OUTPUT " + s + "\n\n")
        self._header = True

    def print_coda(self):
        if self._verbose == 0 or self._coda or not libqdmmg.is_logger_cardinal(self): 
            return
        exec_details = ""
        dirname = __file__[:__file__.rfind('/')]
        if libqdmmg.error_on_exit():
            with open(dirname+'/res/coda_error.txt') as f:
                c = "".join(f.readlines())
                c = c.replace("{exec_details}", exec_details)
                print(c)
        else:
            with open(dirname+'/res/coda.txt') as f:
                c = "".join(f.readlines())
                c = c.replace("{exec_details}", exec_details)
                print(c)
        self._coda = True


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
            self.print_header()
            print(msg)

    def important(self, msg):
        '''
        Log important message. This is printed if _verbose is not 0 (SILENT).
        
        Parameters
        ----------
        msg : object
            Message to be printed. This can be any datatype that can be converted to a string via str(...)
        '''
        msg = str(msg)
        if self._verbose > SILENT:
            self.print_header()
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
            self.print_header()
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
            self.print_header()
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
            self.print_header()
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
            self.print_header()
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
            self.print_header()
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


if __name__ == '__main__':
    import libqdmmg.simulate
    s = libqdmmg.simulate.Simulation(2, 1, verbose=4)
    logger = new_logger(s)
    logger.info("This is a test run for the logging system. No important information is displayed here.")

