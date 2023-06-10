from pyscf import gto, scf, hessian
from pyscf.hessian import thermo
import numpy

numpy.set_printoptions(linewidth=250)

mol = gto.M(atom="H 1.0 0.0 0.0; H 0.0 0.0 0.0; He 0.0 1.0 0.0; He 0.0 -1.0 0.0", basis='6-31g', verbose=4)
mf = scf.RHF(mol)
mf.kernel()
hess = mf.Hessian().kernel()
freq_info = thermo.harmonic_analysis(mol, hess)
norm_mode = freq_info['norm_mode']

