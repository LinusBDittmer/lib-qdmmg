'''

@author Linus Bjarne Dittmer

'''


import libqdmmg.potential.potential as pot
import libqdmmg.potential.harmonic_oscillator as ho
import libqdmmg.potential.double_quadratic_well as dqw
import libqdmmg.potential.trial_potential as tp

def HarmonicOscillator(sim, forces):
    return ho.HarmonicOscillator(sim, forces)

def DoubleQuadraticWell(sim, quartic=None, cubic=None, quadratic=None, linear=None, constant=None):
    return dqw.DoubleQuadraticWell(sim, quartic, cubic, quadratic, linear, constant)

def TrialPotential(sim, grad, forces):
    return tp.TrialPotential(sim, grad, forces)
