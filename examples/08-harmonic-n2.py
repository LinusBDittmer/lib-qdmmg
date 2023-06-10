import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot

s = sim.Simulation(500, 1.0, dim=1, verbose=6, generations=3)
molpot = pot.MolecularPotential(s, ("H", 0.0, 0.0, 0.0, "H", 2.0, 0.0, 0.0))
s.bind_potential(molpot.to_harmonic_oscillator())
s.gen_wavefunction()

plot.density_plots(s.get_wavefunction(), name='n2_ho_amplitude')
