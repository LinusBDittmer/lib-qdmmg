import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot
import libqdmmg.properties as prop
import libqdmmg.integrate as intor

numpy.set_printoptions(linewidth=500, precision=5)

dist_in_a = 0.2
barrier_kcal = 0.25

d_bohr = 1.88973 * dist_in_a
barrier_ha = 0.001593601 * barrier_kcal

a = 16 * barrier_ha / d_bohr**4
b = 8 * barrier_ha / d_bohr**2

dim = 2
#a = 0.255940
#b = 0.028561
#a = 32.007279
#b = 0.142856
dist = 0.5*(2*b/a)**0.5
s = sim.Simulation(25, 5.0, dim=2, verbose=3, generations=3)
p = pot.DoubleQuadraticWell(s, quartic=a, quadratic=b, shift=numpy.array([dist, dist]), coupling=0*0.005)
p.reduced_mass = 1836.15 * numpy.ones(dim)
#p.reduced_mass = 183.615 * numpy.ones(dim)
#p.reduced_mass = numpy.array([1989.0, 1947.0])
s.bind_potential(p)
s.gen_wavefunction()
s.redo_coefficients()
#exp.export_to_json(s.get_wavefunction(), 'trial.json')
autocorrelation = prop.Autocorrelation(s)
autocorrelation.kernel()
fauto = prop.FourierAutocorrelation(s, autocorrelation)
fauto.kernel()
norm = prop.Norm(s)
norm.kernel()
norm2 = prop.Norm(s)
norm2.kernel(obj=s.get_wavefunction().zerotime_wp())
populations = prop.Populations(s)
populations.kernel()


plot.density_plots(s.get_wavefunction(), name='dqw_density_plot', drawtype='amplitude')
plot.linear_plots(autocorrelation, fauto, populations, name='dqw_properties')
#plot.density_plot_ascii(s.get_wavefunction(), 'test_density.txt', 0, 'density', width=150, height=300)
#plot.density_plot_ascii(s.get_wavefunction(), 'test_amp.txt', 0, 'amplitude', width=150, height=300)
