import numpy
import matplotlib.pyplot as plt

import libqdmmg.simulate as sim
import libqdmmg.general as gen
import libqdmmg.potential as pot
import libqdmmg.export as exp
import libqdmmg.plotting as plot
import libqdmmg.properties as prop

s = sim.Simulation(500, 2.0, dim=6, verbose=9, generations=3)
geometry, charge, mult = pot.from_xyz('/export/home/ldittmer/Documents/Linus/MMG/workspace/examples/benchmarking/H2OF-/geometry/base.xyz')
p = pot.MolecularPotential(s, geometry, charge=charge, multiplicity=mult, rounding=4, basis='6-31g', theory='rhf')
s.bind_potential(p)
s.gen_wavefunction()
exp.export_to_json(s.get_wavefunction(), '/export/home/ldittmer/Documents/Linus/MMG/workspace/examples/benchmarking/H2OF-/wavefunction.json')
autocorrelation = prop.Autocorrelation(s)
autocorrelation.kernel()
fauto = prop.FourierAutocorrelation(s, autocorrelation)
fauto.kernel()
norm = prop.Norm(s)
norm.kernel(obj=s.get_wavefunction().zerotime_wp())
populations = prop.Populations(s)
populations.kernel()

plot.density_plots(s.get_wavefunction(), name='/export/home/ldittmer/Documents/Linus/MMG/workspace/examples/benchmarking/H2OF-/h2of_amplitude', drawtype='amplitude')
plot.linear_plots(autocorrelation, fauto, name='/export/home/ldittmer/Documents/Linus/MMG/workspace/examples/benchmarking/H2OF-/h2of_properties')
#plot.density_plot_ascii(s.get_wavefunction(), 'test_density.txt', 0, 'density', width=150, height=300)
#plot.density_plot_ascii(s.get_wavefunction(), 'test_amp.txt', 0, 'amplitude', width=150, height=300)
