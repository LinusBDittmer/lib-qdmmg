import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot
import libqdmmg.properties as prop

dim = 2
s = sim.Simulation(1000, 1.0, dim=2, verbose=3, generations=5)
p = pot.DoubleQuadraticWell(s, quartic=0.032515876, quadratic=4.552682*10**-3, shift=numpy.array([0.264958, 0.264958]), coupling=0*10**-5)
p.reduced_mass = 0.5*18.3615 * numpy.ones(dim)
s.bind_potential(p)
s.gen_wavefunction()
exp.export_to_json(s.get_wavefunction(), 'trial.json')
autocorrelation = prop.Autocorrelation(s)
autocorrelation.kernel()
fauto = prop.FourierAutocorrelation(s, autocorrelation)
fauto.kernel()
norm = prop.Norm(s)
norm.kernel(obj=s.get_wavefunction().zerotime_wp())
populations = prop.Populations(s)
populations.kernel()

plot.density_plots(s.get_wavefunction(), name='dqw_density_plot', drawtype='amplitude')
plot.linear_plots(autocorrelation, fauto, populations, name='dqw_properties')
#plot.density_plot_ascii(s.get_wavefunction(), 'test_density.txt', 0, 'density', width=150, height=300)
#plot.density_plot_ascii(s.get_wavefunction(), 'test_amp.txt', 0, 'amplitude', width=150, height=300)
