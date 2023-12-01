import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot
import libqdmmg.properties as prop

dim = 2
s = sim.Simulation(500, 0.5, dim=dim, verbose=3, generations=1)
p = pot.CoupledHarmonicOscillator(s, 38*numpy.ones(dim), 10)
p.reduced_mass = 2000 * numpy.ones(dim)
s.bind_potential(p)
s.gen_wavefunction()
exp.export_to_json(s.get_wavefunction(), 'trial.json')
kin_energy = prop.KineticEnergy(s)
pot_energy = prop.PotentialEnergy(s)
pot_energy.kernel()
tot_energy = prop.TotalEnergy(s, kin_energy, pot_energy)
#displacement = prop.AverageDisplacement(s)
kin_energy.kernel()
tot_energy.kernel()
#displacement.kernel()
autocorrelation = prop.Autocorrelation(s)
autocorrelation.kernel()

plot.linear_plots(autocorrelation, tot_energy, name='cho_autocorrelation')
plot.density_plots(s.get_wavefunction(), name='cho_density', drawtype='density')
plot.density_plots(s.get_wavefunction(), name='cho_amplitude', drawtype='amplitude', amptype='real')
#plot.density_plot_ascii(s.get_wavefunction(), 'test_density.txt', 0, 'density', width=150, height=300)
#plot.density_plot_ascii(s.get_wavefunction(), 'test_amp.txt', 0, 'amplitude', width=150, height=300)
