import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot
import libqdmmg.properties as prop

s = sim.Simulation(500, 0.5, dim=1, verbose=7, generations=3)
p = pot.HarmonicOscillator(s, 38.85*numpy.ones(1))
p.reduced_mass = 1836 * numpy.ones(1)
s.bind_potential(p)
s.gen_wavefunction()
exp.export_to_json(s.get_wavefunction(), 'trial.json')
kin_energy = prop.KineticEnergy(s)
pot_energy = prop.PotentialEnergy(s)
pot_energy.kernel()
tot_energy = prop.TotalEnergy(s, kin_energy, pot_energy)
displacement = prop.AverageDisplacement(s)
kin_energy.kernel()
tot_energy.kernel()
displacement.kernel()

plot.linear_plots(kin_energy, pot_energy, tot_energy, name='ho_properties')
plot.density_plots(s.get_wavefunction(), name='ho_density', drawtype='density')
plot.density_plots(s.get_wavefunction(), name='ho_amplitude_real', drawtype='amplitude', amptype='real')
plot.density_plots(s.get_wavefunction(), name='ho_amplitude_imag', drawtype='amplitude', amptype='imag')
#plot.density_plot_ascii(s.get_wavefunction(), 'test_density.txt', 0, 'density', width=150, height=300)
#plot.density_plot_ascii(s.get_wavefunction(), 'test_amp.txt', 0, 'amplitude', width=150, height=300)
