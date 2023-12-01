'''

@author Linus Bjarne Dittmer


'''

import numpy
import libqdmmg.potential._pyscf_api as _pyscf_api
import libqdmmg.potential as pot
from libqdmmg.potential.potential import Potential, PotentialIntegrator


BOHR_TO_ANG = 0.529177249
U_TO_ME = 1822.888486212016

class MolecularPotential(Potential):

    def __init__(self, sim, eq_geometry, rounding=4, theory='rhf', xc='b3lyp', basis='sto-3g', charge=0, multiplicity=1):
        super().__init__(sim)
        self.atom_labels = tuple([eq_geometry[i] for i in range(0, len(eq_geometry), 4)])
        self.eq_geometry = tuple([g for i,g in enumerate(eq_geometry) if i % 4 > 0])
        self.rounding = rounding
        self.theory = theory
        self.xc = xc
        self.basis = basis
        self.charge = charge
        self.multiplicity = multiplicity
        self.data = {}

        self.gen_equilibrium_data()

    def gen_equilibrium_data(self):

        # Hessian is given as a rank 4 tensor (Atom Number 1, Atom Number 2, XYZ of Atom 1, XYZ of Atom 2)
        # Normal modes are given as a rank 3 tensor (Mode index, Atom Number, XYZ)
        # Normal modes are not normalised, but mass-weighted
        # Normalisation can be performed by multiplying with sqrt(m_reduced)
        # Hessian has to be normalised by division with reduced mass
        # Harmonic Frequencies can be calculated with the normalised quantities 

        eq_info, self.eq_hessian, self.eq_energy, charges = _pyscf_api.eq_info(self.build_atom_string(), self.basis, self.charge, self.multiplicity-1, theory=self.theory, xc=self.xc)
        self.eq_hessian /= BOHR_TO_ANG**2
        self.norm_modes = eq_info['norm_mode']
        self.reduced_mass = eq_info['reduced_mass'] * U_TO_ME
        self.norm_modes = numpy.einsum('ijk,i->ijk', self.norm_modes, numpy.sqrt(self.reduced_mass))
        self.reduced_charges = numpy.einsum('ijk,j->ijk', self.norm_modes, charges)
        self.reduced_charges = numpy.einsum('ijk->i', self.reduced_charges) / 3
        self.eq_hessian = self.convert_hessian(self.eq_hessian)

        assert len(self.norm_modes) == self.sim.dim, f"Incorrect number of dimensions specified in Simulation object. Should be {len(self.norm_modes)} but is {self.sim.dim}"

    def convert_hessian(self, hess):
        return numpy.einsum('iab,acbd,jcd->ij', self.norm_modes, hess, self.norm_modes) #/ self.reduced_mass

    def convert_gradient(self, grad):
        return numpy.einsum('iab,ab->i', self.norm_modes, grad) #/ self.reduced_mass

    def harmonic_frequencies(self, hess, unit='cm-1'):
        ks = numpy.diag(self.eq_hessian) / self.reduced_mass
        prefactors = ks / abs(ks)
        ks = abs(ks)
        fs = 2 * numpy.sqrt(ks) * prefactors
        if unit == 'cm-1':
            fs *= 219474.63
        elif unit == 'ev':
            fs *= 27.211399
        return fs

    def build_atom_string(self, geom=None):
        if geom is None:
            geom = self.eq_geometry
        else:
            displacement = numpy.reshape(numpy.einsum('ijk,i->jk', self.norm_modes, geom), len(self.eq_geometry))
            geom = self.eq_geometry + displacement

        geom_str = ""
        for i in range(0, len(self.atom_labels)):
            g0 = round(geom[3*i]*BOHR_TO_ANG, self.rounding)
            g1 = round(geom[3*i+1]*BOHR_TO_ANG, self.rounding)
            g2 = round(geom[3*i+2]*BOHR_TO_ANG, self.rounding)
            geom_str += self.atom_labels[i] + " " + str(g0) + " " + str(g1) + " " + str(g2) + "; "
        return geom_str[:-2]

    def get_eff_point(self, x):
        return tuple(numpy.round(numpy.array(x), self.rounding))

    def calc_new_point(self, x):
        x = self.get_eff_point(x)
        print(f"Distance : {x}")
        print(f"Geometry : {self.build_atom_string(geom=x)}")
        sp, g, h = _pyscf_api.joint_call(self.build_atom_string(geom=x), self.basis, self.charge, self.multiplicity-1, theory=self.theory, xc=self.xc)
        g = self.convert_gradient(g)
        h = self.convert_hessian(h)
        self.data[x] = (sp, g, h)

    def evaluate(self, x):
        x = self.get_eff_point(x)
        if not x in self.data:
            self.calc_new_point(x)
        return self.data[x][0] - self.eq_energy

    def gradient(self, x):
        x = self.get_eff_point(x)
        if not x in self.data:
            self.calc_new_point(x)
        return self.data[x][1]

    def hessian(self, x):
        x = self.get_eff_point(x)
        if not x in self.data:
            self.calc_new_point(x)
        return self.data[x][2]

    def to_harmonic_oscillator(self):
        forces = numpy.diag(self.eq_hessian)
        ho = pot.HarmonicOscillator(self.sim, forces)
        ho.reduced_mass = self.reduced_mass
        return ho

    def gen_potential_integrator(self):
        return MolecularIntegrator(self)

class MolecularIntegrator(PotentialIntegrator):
    
    def __init__(self, molpot):
        super().__init__(molpot)
        assert isinstance(molpot, MolecularPotential), f"Only molecular potential permitted, received {type(molpot)}"

if __name__ == '__main__':
    import libqdmmg.simulate as sim
    s = sim.Simulation(2, 0.1, dim=1)
    molpot = MolecularPotential(s, ("C", 0.0, 0.0, 0.0, "O", 0.0, 1.128 / BOHR_TO_ANG, 0.0, "O", 0.0, -1.128 / BOHR_TO_ANG, 0.0))

