'''

General Package Init File. Handles general stuff.

'''


import libqdmmg.general.logger as log
import libqdmmg.general.gaussian as gaussian
import libqdmmg.general.wavepacket as wavepacket

def new_logger(sim):
    return log.new_logger(sim)

def Gaussian(sim, centre=None, width=None, momentum=None, phase=0.0):
    return gaussian.Gaussian(sim, centre=centre, width=width, momentum=momentum, phase=phase)

def Wavepacket(sim, g1):
    return wavepacket.Wavepacket(sim, g1)
