import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot

s = sim.Simulation(1000, 0.0005, dim=1, verbose=6, generations=5)
p = pot.DoubleQuadraticWell(s, quartic=0.1*numpy.ones(1), quadratic=numpy.ones(1))
s.bind_potential(p)
s.gen_wavefunction()
exp.export_to_json(s.get_wavefunction(), 'trial.json')

plot.density_plots(s.get_wavefunction(), name='dqw_density_plot')
plot.density_plot_ascii(s.get_wavefunction(), 'test_density.txt', 0, 'density', width=150, height=300)
plot.density_plot_ascii(s.get_wavefunction(), 'test_amp.txt', 0, 'amplitude', width=150, height=300)
