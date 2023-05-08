'''

@author Linus Bjarne Dittmer

'''


import libqdmmg.potential.potential as pot
import libqdmmg.potential.harmonic_oscillator as ho
import libqdmmg.potential.trial_potential as tp

def HarmonicOscillator(sim, forces):
    return ho.HarmonicOscillator(sim, forces)

def TrialPotential(sim, grad, forces):
    return tp.TrialPotential(sim, grad, forces)
