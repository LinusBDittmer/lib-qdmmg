import time
import numpy
import scipy.sparse
import libqdmmg.potential
import matplotlib.pyplot as plt

class Gridcheck:

    def __init__(self, sim, resolution, extent, shift=0.0, use_proj=True, states=50):
        # Shift has to be positive such that no negative eigenvalues occur, since davidson does not find them
        self.sim = sim
        self.resolution = resolution
        self.extent = extent
        self.shift = shift
        self.use_proj = use_proj
        self.states = states

    def kernel(self, nvec=5):
        dim = self.sim.dim
        dx2 = ((self.extent[1] - self.extent[0]) / self.resolution)**2

        gridaxis = numpy.linspace(self.extent[0], self.extent[1], self.resolution) 

        self.construct_potgrid(gridaxis)
        
        #energies, eigvecs = davidson.davidson_solver(self, (self.resolution**2, self.resolution**2), nvec, use_proj=self.use_proj)
        energies, eigvecs = self.find_eigvals_eigvecs()
        energies -= self.shift
        energies = energies.real

        initial_coeffs = self.proj_initial_wf(eigvecs, gridaxis, dx2)
       
        # Construction of exponentials matrix. Shape: (t, N)
        timesteps = numpy.linspace(0.0, (self.sim.tsteps-1) * self.sim.tstep_val, self.sim.tsteps)
        freqmat = numpy.einsum('i,j->ij', timesteps, energies)
        freqmat = numpy.exp(-1j * freqmat)
        vecscales = numpy.einsum('i,ij->ij', initial_coeffs, eigvecs.T)
        print(vecscales.shape)
        coeffs = numpy.einsum('ij,jk->ik', freqmat, vecscales)
        print(coeffs.shape)

        norm = numpy.sum(abs(coeffs[0])**2 * dx2)
        coeffs /= numpy.sqrt(norm)

        autocorrelation = numpy.zeros(coeffs.shape[0], dtype=numpy.complex128)
        for t in range(coeffs.shape[0]):
            c = coeffs[t]
            autocorrelation[t] = numpy.sum(c.conj() * coeffs[0] * dx2)

        return autocorrelation

    def find_eigvals_eigvecs(self):
        print(f"Constructing Matrices...")
        diag = numpy.ones(self.resolution)
        diags = numpy.array([diag, -2*diag, diag])
        D_matrix = scipy.sparse.spdiags(diags, numpy.array([-1, 0, 1]), self.resolution, self.resolution)
        kinetic_op = -0.5 * scipy.sparse.kronsum(D_matrix, D_matrix)
        U_matrix = scipy.sparse.diags(self.potgrid.reshape(self.resolution**2), (0))
        hamiltonian = kinetic_op + U_matrix

        print(f"Diagonalising Hamiltonian...")
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(hamiltonian, k=self.states, which='SM')
        e_scale = self.sim.potential.reduced_mass[0] * ((self.extent[1] - self.extent[0]) / self.resolution)**2
        eigenvalues /= e_scale
        eigenvalues -= eigenvalues[0]

        print(f"First 10 Eigenvalues: {eigenvalues[:10]}")

        return eigenvalues, eigenvectors

    def proj_initial_wf(self, eigvecs, gridaxes, dx2):
        initial_gaussian = self.sim.generate_new_gaussian(-1)
        initial_grid = numpy.zeros((self.resolution, self.resolution), dtype=numpy.complex128)

        for i, x in enumerate(gridaxes):
            for j, y in enumerate(gridaxes):
                initial_grid[i,j] = initial_gaussian.evaluate(numpy.array((x, y)), 0)

        initial_vec = initial_grid.reshape(self.resolution**2)
        initial_coeffs, res = numpy.linalg.lstsq(eigvecs, initial_vec, rcond=None)[0:2]

        print(initial_coeffs[:10])

        resvec = eigvecs @ initial_coeffs - initial_vec
        resvec = resvec**2
        res = numpy.sum(resvec * dx2)
        print(f"Difference of Density: {res}")

        return initial_coeffs



    def hdotpsi(self, psiset):
        if len(psiset.shape) == 1:
            psi = psiset
        else:
            psidot = numpy.zeros(psiset.shape, dtype=numpy.complex128)
            for i in range(psidot.shape[1]):
                psidot[:,i] = self.hdotpsi(psiset[:,i])
            return psidot
        psimat = numpy.reshape(psi, (self.resolution, self.resolution))
        pdist = (self.extent[1] - self.extent[0]) / self.resolution
        laplacian = numpy.zeros_like(psimat, dtype=numpy.complex128)
        # Compute the Laplacian for the interior points
        laplacian[1:-1, 1:-1] = psimat[2:, 1:-1] + psimat[:-2, 1:-1] + psimat[1:-1, 2:] + psimat[1:-1, :-2] - 4 * psimat[1:-1, 1:-1]
        laplacian /= pdist**2
        kinetic = -0.5 / self.sim.potential.reduced_mass[0] * laplacian
        dot_prod = kinetic + self.potgrid * psimat + self.shift * psimat
        return numpy.reshape(dot_prod, psi.shape)

    def construct_potgrid(self, gridaxes, mass_dist_weight=True):
        # Only viable for 2D systems
        self.potgrid = numpy.zeros((self.resolution, self.resolution))

        for i, x in enumerate(gridaxes):
            for j, y in enumerate(gridaxes):
                self.potgrid[i,j] = self.sim.potential.evaluate((x, y))

        if mass_dist_weight:
            dx = (self.extent[1] - self.extent[0]) / self.resolution
            fac = self.sim.potential.reduced_mass[0] * dx**2
            self.potgrid *= fac

if __name__ == '__main__':
    import libqdmmg.simulate as sim
    import libqdmmg.potential as pot
    dist_in_a = 0.2
    barrier_kcal = 0.25

    d_bohr = 1.88973 * dist_in_a
    barrier_ha = 0.001593601 * barrier_kcal

    a = 16 * barrier_ha / d_bohr**4
    b = 8 * barrier_ha / d_bohr**2
    a = 0.255940 * 100
    b = 0.028561 * 100

    dist = 0.5*(2*b/a)**0.5

    stepsize=0.5
    steps = sim.fs_to_tsteps(100, stepsize)

    s = sim.Simulation(steps, stepsize, dim=2)
    p = pot.DoubleQuadraticWell(s, quartic=a, quadratic=b, shift=numpy.array([dist, dist]), coupling=0.0)
    #p = pot.HarmonicOscillator(s, forces=numpy.array((0.1, 0.1)))
    p.reduced_mass = numpy.array((1836.15, 1836.15))
    s.bind_potential(p)
    
    gc = Gridcheck(s, 1000, (-5.0, 5.0), states=250)
    auto = gc.kernel()
    tspace = numpy.linspace(0, s.tsteps*s.tstep_val, num=s.tsteps)
    for j, t in enumerate(tspace):
        print(f"Autocorrelation at timestep {j} / time {t}: {auto[j]}") 

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(tspace, abs(auto))
    plt.savefig("test.png", dpi=150, bbox_inches='tight')

