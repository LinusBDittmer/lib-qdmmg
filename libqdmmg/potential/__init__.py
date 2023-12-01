'''

@author Linus Bjarne Dittmer

'''


import libqdmmg.potential.potential as pot
import libqdmmg.potential.harmonic_oscillator as ho
import libqdmmg.potential.double_quadratic_well as dqw
import libqdmmg.potential.henon_heiles as hh
import libqdmmg.potential.coupled_harmonic_oscillator as cho
import libqdmmg.potential.mol_potential as molpot

def HarmonicOscillator(sim, forces):
    return ho.HarmonicOscillator(sim, forces)

def DoubleQuadraticWell(sim, quartic=1.0, quadratic=0.0, shift=None, coupling=0.0):
    return dqw.DoubleQuadraticWell(sim, quartic, quadratic, shift, coupling)

def HenonHeiles(sim, omega, l):
    return hh.HenonHeiles(sim, omega, l)

def CoupledHarmonicOscillator(sim, forces, coupling):
    return cho.CoupledHarmonicOscillator(sim, forces, coupling)

def MolecularPotential(sim, eq_geometry, rounding=4, theory='rhf', xc='b3lyp', basis='sto-3g', charge=0, multiplicity=1):
    return molpot.MolecularPotential(sim, eq_geometry, rounding, theory, xc, basis, charge, multiplicity)

def from_xyz(path):
    content = None
    with open(path, 'r') as f:
        content = f.readlines()
    read_multcharge = False
    geometry = []
    mult = 1
    charge = 0
    for line in content:
        if len(line.strip()) == 0:
            continue
        if not read_multcharge:
            charge = int(line.strip().split()[0])
            mult = int(line.strip().split()[1])
            read_multcharge = True
            continue
        atom = line.strip().split()
        geometry.append(atom[0])
        geometry.append(float(atom[1]))
        geometry.append(float(atom[2]))
        geometry.append(float(atom[3]))
    return tuple(geometry), charge, mult
