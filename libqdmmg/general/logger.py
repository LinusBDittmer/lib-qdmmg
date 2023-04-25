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

    def __init__(self, verbose=-1):
        self._verbose = verbose

    def info(self, msg):
        msg = str(msg)
        if self._verbose >= INFO:
            print(msg)

    def error(self, msg, exitCode=0):
        msg = str(msg)
        if self._verbose >= ERROR:
            print('\033[1;31m ' + msg + ' \033[0;0m')
            if exitCode != 0:
                print()
                print('\033[1;31m PROGRAM EXITED UNEXPECTEDLY WITH CODE ' + str(exitCode) + ' \033[0;0m')

    def warn(self, msg):
        msg = str(msg)
        if self._verbose >= ERROR_AND_WARNING:
            print('\033[3;33m ' + msg + ' \033[0;0m')

    def debug1(self, msg):
        msg = str(msg)
        if self._verbose >= DEBUG_1:
            print(msg)

    def debug2(self, msg):
        msg = str(msg)
        if self._verbose >= DEBUG_2:
            print(msg)

    def debug3(self, msg):
        msg = str(msg)
        if self._verbose >= DEBUG_3:
            print(msg)




'''

Logger generation function

'''

def new_logger(sim):
    return Logger(sim.verbose)

