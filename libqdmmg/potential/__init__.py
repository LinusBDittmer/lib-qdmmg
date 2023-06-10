'''

@author Linus Bjarne Dittmer

'''


import libqdmmg.potential.potential as pot
import libqdmmg.potential.harmonic_oscillator as ho
import libqdmmg.potential.double_quadratic_well as dqw
import libqdmmg.potential.trial_potential as tp
import libqdmmg.potential.henon_heiles as hh
import libqdmmg.potential.coupled_harmonic_oscillator as cho
import libqdmmg.potential.mol_potential as molpot

def HarmonicOscillator(sim, forces):
    return ho.HarmonicOscillator(sim, forces)

def DoubleQuadraticWell(sim, quartic=None, cubic=None, quadratic=None, linear=None, constant=None):
    return dqw.DoubleQuadraticWell(sim, quartic, cubic, quadratic, linear, constant)

def TrialPotential(sim, grad, forces):
    return tp.TrialPotential(sim, grad, forces)

def HenonHeiles(sim, omega, l):
    return hh.HenonHeiles(sim, omega, l)

def CoupledHarmonicOscillator(sim, forces, coupling):
    return cho.CoupledHarmonicOscillator(sim, forces, coupling)

def MolecularPotential(sim, eq_geometry, rounding=2, theory='rhf', xc='b3lyp', basis='sto-3g', charge=0, multiplicity=1):
    return molpot.MolecularPotential(sim, eq_geometry, rounding, theory, xc, basis, charge, multiplicity)
