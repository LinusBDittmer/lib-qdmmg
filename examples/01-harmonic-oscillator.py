import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot

s = sim.Simulation(500, 1.0, dim=1, verbose=6, generations=3)
p = pot.HarmonicOscillator(s, 38.85*numpy.ones(1))
p.reduced_mass = 1850 * numpy.ones(1)
s.bind_potential(p)
s.gen_wavefunction()
exp.export_to_json(s.get_wavefunction(), 'trial.json')

plot.density_plots(s.get_wavefunction(), name='ho_amplitude_plot')
plot.density_plot_ascii(s.get_wavefunction(), 'test_density.txt', 0, 'density', width=150, height=300)
plot.density_plot_ascii(s.get_wavefunction(), 'test_amp.txt', 0, 'amplitude', width=150, height=300)
