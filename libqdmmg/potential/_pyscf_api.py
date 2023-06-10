'''

This class is the connection to PySCF.

'''

from pyscf import gto, scf, dft
from pyscf.hessian import thermo


def eval_call(atom, basis, charge, spin, theory='rhf', xc='', verbose=0):
    mol = gto.M(atom=atom, charge=charge, spin=spin, verbose=verbose, basis=basis)
    theo = get_theory(theory, mol, xc=xc)
    theo.kernel()
    sp_energy = theo.energy_tot()
    return sp_energy

def grad_call(atom, basis, charge, spin, theory='rhf', xc='', verbose=0):
    mol = gto.M(atom=atom, charge=charge, spin=spin, verbose=verbose, basis=basis)
    theo = get_theory(theory, mol, xc=xc)
    theo.kernel()
    grad = theo.Gradients().kernel()
    return grad

def hess_call(atom, basis, charge, spin, theory='rhf', xc='', verbose=0):
    mol = gto.M(atom=atom, charge=charge, spin=spin, verbose=verbose, basis=basis)
    theo = get_theory(theory, mol, xc=xc)
    theo.kernel()
    hess = theo.Hessian().kernel()
    return hess

def joint_call(atom, basis, charge, spin, theory='rhf', xc='', verbose=0):
    mol = gto.M(atom=atom, charge=charge, spin=spin, verbose=verbose, basis=basis)
    theo = get_theory(theory, mol, xc=xc)
    theo.kernel()
    sp_energy = theo.energy_tot()
    grad = theo.Gradients().kernel()
    hess = theo.Hessian().kernel()
    return sp_energy, grad, hess

def eq_info(atom, basis, charge, spin, theory='rhf', xc='', verbose=0):
    mol = gto.M(atom=atom, charge=charge, spin=spin, verbose=verbose, basis=basis)
    theo = get_theory(theory, mol, xc=xc)
    theo.kernel()
    hess = theo.Hessian().kernel()
    eq_info = thermo.harmonic_analysis(mol, hess)
    return eq_info, hess

def get_theory(theory, mol, xc=''):
    if theory == 'rhf':
        return scf.RHF(mol)
    elif theory == 'uhf':
        return scf.UHF(mol)
    elif theory == 'rks':
        return dft.RKS(mol, xc=xc)
    elif theory == 'uks':
        return dft.UKS(mol, xc=xc)
