from pyscf import gto, scf, hessian
from pyscf.hessian import thermo
import numpy

numpy.set_printoptions(linewidth=250)

mol = gto.M(atom="O 1.0 0.0 0.0; C 0.0 0.0 0.0; O -1.0 0.0 0.0", basis='6-31g', verbose=4)
charges = mol._atm[:,0]
mf = scf.RHF(mol)
mf.kernel()
hess = mf.Hessian().kernel()
eq_info = thermo.harmonic_analysis(mol, hess)
reduced_mass = eq_info['reduced_mass']
normal_modes = eq_info['norm_mode']
normal_modes = numpy.einsum('ijk,i->ijk', normal_modes, numpy.sqrt(reduced_mass))
relative_charges = numpy.einsum('ijk,j->ijk', normal_modes, charges)
relative_charges = numpy.einsum('ijk->ij', relative_charges) / 3
relative_charges = numpy.einsum('ij->i', relative_charges)
print(relative_charges)
print(normal_modes)
