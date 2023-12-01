import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot
import libqdmmg.properties as prop

s = sim.Simulation(200, 10.0, dim=2, verbose=7, generations=3)
p = pot.HarmonicOscillator(s, 38.85*numpy.ones(2))
p.reduced_mass = numpy.array([1989, 29452])
s.bind_potential(p)
s.gen_wavefunction()
exp.export_to_json(s.get_wavefunction(), 'trial.json')
autocorrelation = prop.Autocorrelation(s)
fauto = prop.FourierAutocorrelation(s, autocorrelation)
autocorrelation.kernel()
fauto.kernel()

plot.linear_plots(autocorrelation, fauto, name='ho_properties')
plot.density_plots(s.get_wavefunction(), name='ho_amplitude_real', drawtype='amplitude', amptype='real')
#plot.density_plot_ascii(s.get_wavefunction(), 'test_density.txt', 0, 'density', width=150, height=300)
#plot.density_plot_ascii(s.get_wavefunction(), 'test_amp.txt', 0, 'amplitude', width=150, height=300)
