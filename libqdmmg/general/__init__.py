'''

General Package Init File. Handles general stuff.

'''


import libqdmmg.general.logger as log
import libqdmmg.general.gaussian as gaussian
import libqdmmg.general.wavepacket as wavepacket
import libqdmmg.general.exceptions as exceptions

def new_logger(sim):
    return log.new_logger(sim)

def Gaussian(sim, centre=None, width=None, momentum=None, phase=0.0):
    return gaussian.Gaussian(sim, centre=centre, width=width, momentum=momentum, phase=phase)

def Wavepacket(sim):
    return wavepacket.Wavepacket(sim)

def InvalidIntegralRequestStringException(rq, int_class, *args):
    return exceptions.InvalidIntegralRequestStringException(rq, int_class, args)

def SimulationNotRunException(sim, *args):
    return exceptions.SimulationNotRunException(sim, args)

def SNRException(sim, *args):
    return SimulationNotRunException(sim, args)

def IIRSException(rq, int_class, *args):
    return InvalidIntegralRequestStringException(rq, int_class, args)
