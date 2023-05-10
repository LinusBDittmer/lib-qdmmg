'''

General Init File


'''
__version__ = '0.0.1'
_hooks = None
_cardinal_logger = None
import sys

# Custom Error hook for proper output termination

def register_logger(logger):
    global _cardinal_logger
    if _cardinal_logger is None:
        _cardinal_logger = logger
        _cardinal_logger.print_header()


def is_logger_cardinal(logger):
    global _cardinal_logger
    if _cardinal_logger is None:
        return True
    return logger == _cardinal_logger

def error_on_exit():
    global _hooks
    return not ((_hooks.exit_code is None or _hooks.exit_code == 0) and (_hooks.exception is None))

class ExitHooks:

    def __init__(self):
        self.exit_code = None
        self.exception = None
        self.hook()

    def hook(self):
        self._orig_exit = sys.exit
        self._orig_exc_handler = sys.excepthook
        sys.exit = self.exit
        sys.excepthook = self.exc_handler

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args):
        self.exception = exc
        self._orig_exc_handler(exc_type, exc, args)


# Hook custom exception for script coda to system
_hooks = ExitHooks()
