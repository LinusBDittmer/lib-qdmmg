import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot
import libqdmmg.properties as prop

dt = 0.5
fs = 10

s = sim.Simulation(sim.fs_to_tsteps(fs, dt), dt, dim=6, verbose=3, generations=3)
p = pot.HenonHeiles(s, 1.0, 0.111803)
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

plot.linear_plots(kin_energy, pot_energy, tot_energy, name='hh_properties')
plot.density_plots(s.get_wavefunction(), name='hh_density', drawtype='density')
plot.density_plots(s.get_wavefunction(), name='hh_amplitude', drawtype='amplitude', amptype='real')
#plot.density_plot_ascii(s.get_wavefunction(), 'test_density.txt', 0, 'density', width=150, height=300)
#plot.density_plot_ascii(s.get_wavefunction(), 'test_amp.txt', 0, 'amplitude', width=150, height=300)
