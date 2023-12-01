import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot
import libqdmmg.properties as prop

s = sim.Simulation(100, 7.5, dim=1, verbose=9, generations=3)
p = pot.MolecularPotential(s, ("H", 0.0, 0.0, 1.8025074284, "Li", 0.0, 0.0, -1.0525074284), rounding=3, basis='sto-3g')
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

plot.density_plots(s.get_wavefunction(), name='lih_density_plot', drawtype='amplitude')
plot.linear_plots(autocorrelation, fauto, populations, name='lih_properties')
#plot.density_plot_ascii(s.get_wavefunction(), 'test_density.txt', 0, 'density', width=150, height=300)
#plot.density_plot_ascii(s.get_wavefunction(), 'test_amp.txt', 0, 'amplitude', width=150, height=300)
