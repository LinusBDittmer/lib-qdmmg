'''

Integration Handler for EOM

'''

import numpy
import scipy
import libqdmmg.integrate as intor

class EOM_Integrator:

    def __init__(self, sim, micro_steps):
        self.sim = sim
        self.stepsize = sim.tstep_val
        self.micro_steps = micro_steps
        self.logger = sim.logger
        self.coeff_damping = 0.05
        self.coeff_damping_shift = 0.05

    def next_centre(self, gaussian, t):
        return None

    def next_momentum(self, gaussian, t):
        return None

    def next_phase(self, gaussian, t):
        return 0

    def next_coefficient(self, coeff, d_coeff, t):
        return 0

    def setup_renew_coefficients(self, gs):
        glen = len(gs)
        pot_intor = self.sim.potential.gen_potential_integrator()
        hs = numpy.zeros((self.sim.tsteps, glen, glen), dtype=numpy.complex128)
        ovlps = numpy.zeros((self.sim.tsteps, glen, glen), dtype=numpy.complex128)
        dovlps = numpy.zeros((self.sim.tsteps, glen, glen), dtype=numpy.complex128)
        for t in range(self.sim.tsteps):
            self.logger.info(f"Precalculating Matrix elements for t = {t}")
            for i in range(glen):
                for j in range(glen):
                    hs[t,i,j] = intor.int_request(self.sim, 'int_kinetic_gg', gs[i], gs[j], t)
                    hs[t,i,j] += pot_intor.int_request('int_gVg', gs[i], gs[j], t)
                    ovlps[t,i,j] = intor.int_request(self.sim, 'int_ovlp_gg', gs[i], gs[j], t)
                    dovlps[t,i,j] = intor.int_request(self.sim, 'int_dovlp_gg', gs[i], gs[j], t)
        self.h_eigvecs = numpy.zeros(hs.shape, dtype=numpy.complex128)
        self.ovlp_eigvecs = numpy.zeros(hs.shape, dtype=numpy.complex128)
        self.dovlp_schurvecs = numpy.zeros(hs.shape, dtype=numpy.complex128)
        self.h_eigvals = numpy.zeros((self.sim.tsteps, glen))
        self.ovlp_eigvals = numpy.zeros((self.sim.tsteps, glen))
        self.dovlp_schurvals = numpy.zeros((self.sim.tsteps, glen, glen), dtype=numpy.complex128)
        for t in range(self.sim.tsteps):
            self.h_eigvals[t], self.h_eigvecs[t] = numpy.linalg.eigh(hs[t])
            self.ovlp_eigvals[t], self.ovlp_eigvecs[t] = numpy.linalg.eigh(ovlps[t])
            self.dovlp_schurvals[t], self.ovlp_schurvecs[t] = scipy.linalg.schur(dovlps[t])

    def renew_coefficients(self, gs, c0, t):
        pot_intor = self.sim.potential.gen_potential_integrator()
        glen = len(gs)
        c0 = numpy.array(c0, dtype=numpy.complex128)
        dt = self.stepsize
        n = len(c0)
        h = numpy.zeros((glen, glen), dtype=numpy.complex128)
        ovlp = numpy.zeros(h.shape, dtype=numpy.complex128)
        dovlp = numpy.zeros(h.shape, dtype=numpy.complex128)
        for i in range(glen):
            for j in range(glen):
                h[i,j] = intor.int_request(self.sim, 'int_kinetic_gg', gs[i], gs[j], t)
                h[i,j] += pot_intor.int_request('int_gVg', gs[i], gs[j], t)
                #ovlp[i,j] = intor.int_request(self.sim, 'int_ovlp_gg', gs[i], gs[j], t2)
                #dovlp[i,j] = intor.int_request(self.sim, 'int_dovlp_gg', gs[i], gs[j], t2)

        def microstep(t_micro, dt_micro, c0_micro):
            s_micro = numpy.zeros(h.shape, dtype=numpy.complex128)
            ds_micro = numpy.zeros(h.shape, dtype=numpy.complex128)
            for i in range(glen):
                for j in range(glen):
                    s_micro[i,j] = intor.int_request(self.sim, 'int_ovlp_gg', gs[i], gs[j], t_micro)
                    ds_micro[i,j] = intor.int_request(self.sim, 'int_dovlp_gg', gs[i], gs[j], t_micro)

            h_mat = -1j * (h - 1j * ds_micro)
            cnorm = numpy.sqrt(numpy.dot(numpy.dot(s_micro, c0_micro), c0_micro.conj()))
            ovlp_sqrt = scipy.linalg.sqrtm(s_micro)
            ovlp_sqrt_inv = scipy.linalg.pinv(ovlp_sqrt, atol=10**-5, rcond=None)
            c0_orth = numpy.dot(ovlp_sqrt, c0_micro / cnorm.real)
            h_orth = numpy.dot(ovlp_sqrt_inv, numpy.dot(h_mat, ovlp_sqrt_inv))
            h_orth = 0.5 * (h_orth - h_orth.conj().T)
            h_orth_eff = numpy.array(h_orth * dt_micro, dtype=numpy.complex128)
            prop_orth = scipy.linalg.expm(h_orth_eff)
            c1_orth = numpy.dot(prop_orth, c0_orth)
            c1 = numpy.dot(ovlp_sqrt_inv, c1_orth)
            # Debugging
            #print(f"NO Antihermitian Propagator residue: {numpy.linalg.norm(h_mat + h_mat.conj().T)}")
            #print(f"Antihermitian Propagator residue: {numpy.linalg.norm(h_orth + h_orth.conj().T)}")
            #print(f"Orthogonal Propagator residue: {numpy.linalg.norm(prop_orth @ prop_orth.conj().T - numpy.eye(n))}")
            #print(f"C0 Norm: {numpy.dot(c0_orth.conj(), c0_orth)}, Res: {numpy.dot(c0_orth.conj(), c0_orth) - 1}")
            #print(f"C1 Norm: {numpy.dot(c1_orth.conj(), c1_orth)}, Res: {numpy.dot(c1_orth.conj(), c1_orth) - 1}")
            #print(f"C0 Indirect Norm: {numpy.dot(numpy.dot(s_micro, c0_micro), c0_micro.conj())}")
            #print(f"C1 Indirect Norm: {numpy.dot(numpy.dot(s_micro, c1), c1.conj())}")


            return c1

        n_micro = self.micro_steps * (self.sim.generations + 1)
        ts_micro = list(numpy.linspace(t, t+dt, num=n_micro+1))[1:]
        c1 = numpy.array(numpy.copy(c0), dtype=numpy.complex128)
        for t_micro in ts_micro:
            c1 = microstep(t_micro, dt / n_micro, c1)
            
        return c1


    def renew_coefficients4(self, gs, c0, t):
        pot_intor = self.sim.potential.gen_potential_integrator()
        glen = len(gs)
        c0 = numpy.array(c0, dtype=numpy.complex128)
        dt = self.stepsize
        n = len(c0)
        t2 = t+1
        if t >= self.sim.tsteps-1:
            t2 = t
        h = numpy.zeros((glen, glen), dtype=numpy.complex128)
        ovlp = numpy.zeros(h.shape, dtype=numpy.complex128)
        dovlp = numpy.zeros(h.shape, dtype=numpy.complex128)
        for i in range(glen):
            for j in range(glen):
                h[i,j] = intor.int_request(self.sim, 'int_kinetic_gg', gs[i], gs[j], t2)
                h[i,j] += pot_intor.int_request('int_gVg', gs[i], gs[j], t2)
                ovlp[i,j] = intor.int_request(self.sim, 'int_ovlp_gg', gs[i], gs[j], t2)
                dovlp[i,j] = intor.int_request(self.sim, 'int_dovlp_gg', gs[i], gs[j], t2)

        h_mat = -1j * (h - 1j * dovlp)
        ovlp_sqrt = scipy.linalg.sqrtm(ovlp)
        ovlp_sqrt_inv = scipy.linalg.pinv(ovlp_sqrt, atol=10**-5, rcond=None)
        c0_orth = numpy.dot(ovlp_sqrt, c0)
        h_orth = numpy.dot(ovlp_sqrt_inv, numpy.dot(h_mat, ovlp_sqrt_inv))
        h_orth = 0.5 * (h_orth - h_orth.conj().T)
        h_orth_eff = numpy.array(h_orth * dt, dtype=numpy.complex128)
        prop_orth = scipy.linalg.expm(h_orth_eff)
        c1_orth = numpy.dot(prop_orth, c0_orth)
        c1 = numpy.dot(ovlp_sqrt_inv, c1_orth)
        
        # Debugging
        print(f"NO Antihermitian Propagator residue: {numpy.linalg.norm(h_mat + h_mat.conj().T)}")
        print(f"Antihermitian Propagator residue: {numpy.linalg.norm(h_orth + h_orth.conj().T)}")
        print(f"Orthogonal Propagator residue: {numpy.linalg.norm(prop_orth @ prop_orth.conj().T - numpy.eye(n))}")
        print(f"C0 Norm: {numpy.dot(c0_orth.conj(), c0_orth)}, Res: {numpy.dot(c0_orth.conj(), c0_orth) - 1}")
        print(f"C1 Norm: {numpy.dot(c1_orth.conj(), c1_orth)}, Res: {numpy.dot(c1_orth.conj(), c1_orth) - 1}")
        print(f"C0 Indirect Norm: {numpy.dot(numpy.dot(ovlp, c0), c0.conj())}")
        print(f"C1 Indirect Norm: {numpy.dot(numpy.dot(ovlp, c1), c1.conj())}")

        return c1

    def renew_coefficients3(self, gs, c0, t):
        pot_intor = self.sim.potential.gen_potential_integrator()
        glen = len(gs)
        c0 = numpy.array(c0, dtype=numpy.complex128)
        dt = self.stepsize
        n = len(c0)
        h = numpy.zeros((glen, glen), dtype=numpy.complex128)
        ovlp = numpy.zeros(h.shape, dtype=numpy.complex128)
        dovlp = numpy.zeros(h.shape, dtype=numpy.complex128)
        for i in range(glen):
            for j in range(glen):
                h[i,j] = intor.int_request(self.sim, 'int_kinetic_gg', gs[i], gs[j], t)
                h[i,j] += pot_intor.int_request('int_gVg', gs[i], gs[j], t)
                ovlp[i,j] = intor.int_request(self.sim, 'int_ovlp_gg', gs[i], gs[j], t)
                dovlp[i,j] = intor.int_request(self.sim, 'int_dovlp_gg', gs[i], gs[j], t)

        h_mat = -1j * (h - 1j * dovlp)
        h_eff = numpy.zeros((2*n, 2*n))
        s_eff = numpy.zeros((2*n, 2*n))
        c0_eff = numpy.concatenate((c0.real, c0.imag))
        h_eff[:n,:n] = h_mat.real
        h_eff[n:,n:] = h_mat.real
        h_eff[:n,n:] = -h_mat.imag
        h_eff[n:,:n] = h_mat.imag
        s_eff[:n,:n] = ovlp.real
        s_eff[n:,n:] = ovlp.real
        s_eff[:n,n:] = -ovlp.imag
        s_eff[n:,:n] = ovlp.imag
        s_eff_inv = scipy.linalg.pinv(s_eff, atol=10**-3, rtol=0.0)
        p0_eff = numpy.dot(s_eff_inv, h_eff)
        p_eff = scipy.linalg.expm(p0_eff * dt)
        print(f"Propagator determinant residue: {numpy.linalg.det(p_eff) - 1}")
        print(f"Propagator orthogonality residue: {numpy.linalg.norm(p_eff @ p_eff.T - numpy.eye(2*n))}")
        c1_eff = numpy.dot(p_eff, c0_eff)
        c1 = c1_eff[:n] + 1j*c1_eff[n:]
        dc = c1 - c0
        signs = dc / numpy.abs(dc)
        damped_dc = numpy.sign(dc) * self.coeff_damping * (1 - numpy.exp(-abs(dc) / self.coeff_damping))

        print(f"Effective Inverse Ovlp Determinant: {numpy.linalg.det(s_eff_inv)}")
        print(f"True Ovlp Determinant: {numpy.linalg.det(ovlp)}")
        print(f"Anti-Hermitian Propagator residue: {numpy.linalg.norm(h_mat + h_mat.T)}")
        return c0 + damped_dc


    def renew_coefficients2(self, gs, c0, t):
        if t == 0 and False:
            self.setup_renew_coefficients(gs)
        pot_intor = self.sim.potential.gen_potential_integrator()
        glen = len(gs)
        dt = self.stepsize
        h0 = numpy.zeros((glen, glen), dtype=numpy.complex128)
        h1 = numpy.zeros(h0.shape, dtype=numpy.complex128)
        ovlp0 = numpy.zeros(h0.shape, dtype=numpy.complex128)
        ovlp1 = numpy.zeros(h0.shape, dtype=numpy.complex128)
        dovlp0 = numpy.zeros(h0.shape, dtype=numpy.complex128)
        dovlp1 = numpy.zeros(h0.shape, dtype=numpy.complex128)
        for i in range(glen):
            for j in range(glen):
                h0[i,j] = intor.int_request(self.sim, 'int_kinetic_gg', gs[i], gs[j], t)
                h0[i,j] += pot_intor.int_request('int_gVg', gs[i], gs[j], t)
                h1[i,j] = intor.int_request(self.sim, 'int_kinetic_gg', gs[i], gs[j], t+1)
                h1[i,j] += pot_intor.int_request('int_gVg', gs[i], gs[j], t+1)
                ovlp0[i,j] = intor.int_request(self.sim, 'int_ovlp_gg', gs[i], gs[j], t)
                ovlp1[i,j] = intor.int_request(self.sim, 'int_ovlp_gg', gs[i], gs[j], t+1)
                dovlp0[i,j] = intor.int_request(self.sim, 'int_dovlp_gg', gs[i], gs[j], t)
                dovlp1[i,j] = intor.int_request(self.sim, 'int_dovlp_gg', gs[i], gs[j], t+1)
        '''
        # Hamiltonian Interpolation
        h0eigvals, h0eigvecs = numpy.linalg.eigh(h0)
        h1eigvals, h1eigvecs = numpy.linalg.eigh(h1)
        h0rotmat = scipy.linalg.logm(h1eigvecs @ h0eigvecs.conj().T)
        hrot = lambda tx: scipy.linalg.expm(h0rotmat*tx) @ h0eigvecs
        h = lambda tx: (hrot(tx) @ numpy.diag(tx*h0eigvals + (1-tx)*h1eigvals) @ hrot(tx).conj().T).real
        # Overlap Interpolation
        s0eigvals, s0eigvecs = numpy.linalg.eigh(ovlp0)
        s1eigvals, s1eigvecs = numpy.linalg.eigh(ovlp1)
        s0rotmat = scipy.linalg.logm(s1eigvecs @ s0eigvecs.conj().T)
        srot = lambda tx: scipy.linalg.expm(s0rotmat*tx) @ s0eigvecs
        ovlp = lambda tx: srot(tx) @ numpy.diag(tx*s0eigvals + (1-tx)*s1eigvals) @ srot(tx).conj().T
        # Differential Overlap Interpolation
        t0eigvals, t0eigvecs = scipy.linalg.schur(dovlp0)
        t1eigvals, t1eigvecs = scipy.linalg.schur(dovlp1)
        t0rotmat = scipy.linalg.logm(t1eigvecs @ t0eigvecs.conj().T)
        trot = lambda tx: scipy.linalg.expm(t0rotmat*tx) @ t0eigvecs
        dovlp = lambda tx: trot(tx) @ (tx*t0eigvals + (1-tx)*t1eigvals) @ trot(tx).conj().T
        '''

        def intp_mat(tx):
            tx = int(tx)
            h = numpy.zeros((glen, glen), dtype=numpy.complex128)
            ovlp = numpy.zeros(h0.shape, dtype=numpy.complex128)
            dovlp = numpy.zeros(h0.shape, dtype=numpy.complex128)
            for i in range(glen):
                for j in range(glen):
                    h[i,j] = intor.int_request(self.sim, 'int_kinetic_gg', gs[i], gs[j], tx)
                    h[i,j] += pot_intor.int_request('int_gVg', gs[i], gs[j], tx)
                    ovlp[i,j] = intor.int_request(self.sim, 'int_ovlp_gg', gs[i], gs[j], tx)
                    dovlp[i,j] = intor.int_request(self.sim, 'int_dovlp_gg', gs[i], gs[j], tx)
            #print("Potential Hamiltonian:")
            #print(h)
            #print("Overlap:")
            #print(ovlp)
            #print("Differential Overlap:")
            #print(dovlp)
            return h, ovlp, dovlp

        def func(tx, b):
            a = b.view(numpy.complex128)
            h, ovlp, dovlp = h0, ovlp0, dovlp0
            #h, ovlp, dovlp = intp_mat(t+tx)
            #h, ovlp, dovlp = inpt_mat(t)
            #da = -1j * numpy.linalg.lstsq(ovlp, numpy.dot(h - 1j*dovlp, a))[0]
            #da = -1j * numpy.linalg.solve(ovlp, numpy.dot(h - 1j*dovlp, a))
            #residue = numpy.dot(ovlp, da) + 1j * numpy.dot(h - 1j*dovlp, a)
            #print(f"Residue: {numpy.linalg.norm(residue)}")
            h_eff = -1j*numpy.dot(h - 1j*dovlp, a)
            # Solving linear equation explicitly for real and imaginary part.
            h_vec = numpy.concatenate((h_eff.real, h_eff.imag))
            n = a.shape[0]
            # Overlap rewritten
            s_eff = numpy.zeros((2*n, 2*n))
            s_eff[:n,:n] = ovlp.real
            s_eff[n:,n:] = ovlp.real
            s_eff[n:,:n] = -ovlp.imag
            s_eff[:n,n:] = ovlp.imag
            # Solve equation via pseudoinverse with SVD
            s_eff_inv = scipy.linalg.pinv(s_eff, atol=10**-3, rcond=None)
            da_eff = numpy.dot(s_eff_inv, h_vec)
            #da_eff = numpy.linalg.lstsq(s_eff, h_vec)[0]
            da = da_eff[:n] + da_eff[n:] * 1j
            print(f"Difference in A: {da}")
            # Damping
            mnorm = numpy.amax(abs(da))
            if mnorm >= 1:
                print("Turning on Damping")
                da *= numpy.exp(-self.coeff_damping * (mnorm - self.coeff_damping_shift)**2)
            
            return da.view(numpy.float64)
                    
            #ovlpinv = numpy.linalg.inv(ovlp)
            #return ovlpinv @ (h - 1j*dovlp)
        
        # Solution approach as in AIMS
        c0 = numpy.array(c0, dtype=numpy.complex128)
        #ncoeffs = scipy.integrate.odeint(func, c0.view(numpy.float64), (0, 1))
        #ncoeffs = ncoeffs[-1]
        subtimesteps = 4 * (self.sim.generations + 1)
        tstep_tuple = tuple(numpy.linspace(0, 1, num=subtimesteps))
        ncoeffs = scipy.integrate.solve_ivp(func, (0., 1.), c0.view(numpy.float64), method='RK45', t_eval=tstep_tuple)
        ncoeffs = ncoeffs.y.T[-1]
        ncoeffs = numpy.ascontiguousarray(ncoeffs).view(numpy.complex128)

        # Propagator approach, requires explicit inversion of overlap
        #propagator = -1j*scipy.integrate.quad_vec(func, 0, 1)[0]
        #ncoeffs = numpy.dot(scipy.linalg.expm(propagator), c0)
        return ncoeffs

class EOM_EulerIntegrator(EOM_Integrator):

    def __init__(self, sim, micro_steps=4):
        super().__init__(sim, micro_steps)

    def next_centre(self, gaussian, t):
        return gaussian.centre[t] + self.stepsize * (gaussian.d_centre[t] + gaussian.d_centre_v[t])

    def next_momentum(self, gaussian, t):
        return gaussian.momentum[t] + self.stepsize * (gaussian.d_momentum[t] + gaussian.d_momentum_v[t])

    def next_phase(self, gaussian, t):
        return gaussian.phase[t] + self.stepsize * (gaussian.d_phase[t] + gaussian.d_phase_v[t])

    def next_coefficient(self, coeff, d_coeff, t):
        return coeff[t] + self.stepsize * d_coeff[t]


class EOM_AdamsBashforth(EOM_Integrator):

    def __init__(self, sim, order=10, micro_steps=4):
        super().__init__(sim, micro_steps)
        self.order = order
        self.coefficients = [None]*order
        self.calculate_coeffs()

    def calculate_coeffs(self):
        def abpol(x, j, s):
            val = 1.0
            for i in range(s+1):
                if i == j: continue
                val *= x+i
            return val

        for s in range(self.order):
            self.coefficients[s] = numpy.zeros(s+1)
            for j in range(s+1):
                self.coefficients[s][j] = scipy.integrate.quad(abpol, 0, 1, args=(j,s))[0]
                self.coefficients[s][j] *= (-1)**j / (scipy.special.factorial(j) * scipy.special.factorial(s-j))

    def get_effective_coeffs(self, t):
        return self.coefficients[min(self.order-1, t)]

    def next_centre(self, gaussian, t):
        eff_coeff = self.get_effective_coeffs(t)
        relevant_centres = gaussian.d_centre[max(0, t-self.order+1):t+1] + gaussian.d_centre_v[max(0, t-self.order+1):t+1]
        unadapted_shift = numpy.einsum('j,jn->n', eff_coeff, relevant_centres)
        return gaussian.centre[t] + self.stepsize * unadapted_shift

    def next_momentum(self, gaussian, t):
        eff_coeff = self.get_effective_coeffs(t)
        relevant_momenta = gaussian.d_momentum[max(0, t-self.order+1):t+1] + gaussian.d_momentum_v[max(0, t-self.order+1):t+1]
        unadapted_shift = numpy.einsum('j,jn->n', eff_coeff, relevant_momenta)
        return gaussian.momentum[t] + self.stepsize * unadapted_shift

    def next_phase(self, gaussian, t):
        eff_coeff = self.get_effective_coeffs(t)
        relevant_phases = gaussian.d_phase[max(0, t-self.order+1):t+1] + gaussian.d_phase_v[max(0, t-self.order+1):t+1]
        unadapted_shift = numpy.dot(eff_coeff, relevant_phases)
        return gaussian.phase[t] + self.stepsize * unadapted_shift

    def next_coefficient(self, coeff, d_coeff, t):
        eff_coeff = self.get_effective_coeffs(t)
        relevant_coeffs = d_coeff[max(0, t-self.order+1):t+1]
        unadapted_shift = self.stepsize * numpy.dot(eff_coeff, relevant_coeffs)
        adapted_shift = self.get_energy_conservation_factor(coeff[t], unadapted_shift, t) * unadapted_shift
        return coeff[t] + adapted_shift

    def get_energy_conservation_factor(self, coeff, shift, t, scale=10000):
        return 1.0
        if abs(shift / self.stepsize) < 10**-5:
            return 1.0
        g1 = self.sim.active_gaussian
        wp = self.sim.previous_wavefunction
        pot_intor = self.sim.potential.gen_potential_integrator()
        eg0 = g1.energy_tot(t)
        eg1 = g1.energy_tot(t+1)
        ewp0 = wp.energy_tot(t)
        ewp1 = wp.energy_tot(t+1)
        w0 = intor.int_request(self.sim, 'int_kinetic_gw', g1, wp, t)
        w1 = intor.int_request(self.sim, 'int_kinetic_gw', g1, wp, t+1)
        s0 = intor.int_request(self.sim, 'int_ovlp_gg', g1, g1, t)
        s1 = intor.int_request(self.sim, 'int_ovlp_gg', g1, g1, t+1)
        st0 = intor.int_request(self.sim, 'int_ovlp_gw', g1, wp, t)
        st1 = intor.int_request(self.sim, 'int_ovlp_gw', g1, wp, t+1)
        
        def obj(x):
            a0 = coeff
            a1 = coeff + shift*(x+1)
            b0 = numpy.sqrt(a0*a0*(st0*st0-s0)+1) - a0*st0
            b1 = numpy.sqrt(a1*a1*(st1*st1-s1)+1) - a1*st1
            return scale*(a0*a0*eg0 + 2*a0*b0*w0 + b0*b0*ewp0 - a1*a1*eg1 - 2*a1*b1*w1 - b1*b1*ewp1) / self.stepsize

        def jacobj(x):
            a1 = coeff + shift*(x+1)
            b1 = numpy.sqrt(a1*a1*(st1*st1-s1)+1) - a1*st1
            da1 = shift
            db1 = shift * (st1*st1-s1) * a1 / numpy.sqrt(a1*a1*(st1*st1-s1)) - shift*st1
            return -2*scale*(a1*da1*eg1 + w1*(da1*b1+db1*a1) + ewp1*b1*db1) / self.sim.tstep_val

        root_obj = scipy.optimize.root_scalar(obj, x0=0.0, fprime=jacobj, method='newton', rtol=10**-7)
        self.sim.logger.debug3(f"Energy Conservation remainder : {obj(root_obj.root)}")
        if root_obj.converged:
            if root_obj.root > -2 and root_obj.root < 2 and not numpy.isnan(root_obj.root):
                return root_obj.root.real+1
        return 1.0

class EOM_Matexp(EOM_Integrator):

    def __init__(self, sim, order=4, aux_order=3, aux_thresh=5, micro_steps=4):
        super().__init__(sim, micro_steps)
        self.order = order
        self.eom_aux = EOM_AdamsBashforth(sim, order=aux_order, micro_steps=micro_steps)
        self.aux_thresh = aux_thresh
        self.eom_euler = EOM_EulerIntegrator(sim, micro_steps=micro_steps)
    
    def build_matrix(self, x, p):
        mat_dim = self.sim.dim*2
        block_size = self.sim.dim
        mat = numpy.zeros((mat_dim, mat_dim))
        # Matrix strucutre
        # XX XP
        # PX PP

        # Only PX, P1 and XP are non-zero
        # XP Block
        mat[:block_size,block_size:block_size*2] = numpy.diag(1/self.sim.potential.reduced_mass)
        # PX Block
        h = self.sim.potential.hessian(x)
        w = self.sim.active_gaussian.width
        # Testing to see if covariance addition changes kinks in autocorrelation
        mat[block_size:,:block_size] = -h #+ numpy.diag(numpy.diag(h))
        #mat[block_size:,:block_size] += numpy.diag(- 4 * w * w / self.sim.potential.reduced_mass)
        return mat

    def get_timestep(self, x0, p0, x_var, p_var, prop_mat):
        d = self.sim.dim
        dvec = numpy.concatenate((x_var, p_var))
        zvec = numpy.concatenate((x0, p0))
        grad = self.sim.potential.gradient(x0)
        hess = self.sim.potential.hessian(x0)
        self.logger.debug1(f"Gradient : {grad}")
        self.logger.debug1(f"Hessian-Centre Contraction : {numpy.einsum('ii,i->i', hess, x0)}")
        self.logger.debug1(f"Hessian : {hess}")
        dvec[d:] += - grad + numpy.dot(hess, x0)
        self.logger.debug1(f"Property Matrix determinant : {numpy.linalg.det(prop_mat)}")

        #if abs(numpy.linalg.det(prop_mat)) > 10**-15:
        self.logger.debug1(f"Calculating EOM Integration by MatExp Algorithm")
        svec = numpy.linalg.lstsq(prop_mat, dvec, rcond=None)[0]
        #svec = numpy.dot(numpy.linalg.inv(prop_mat), dvec)
        sol = numpy.dot(scipy.linalg.expm(prop_mat * self.sim.tstep_val), (zvec + svec)) - svec
        return sol[:d], sol[d:]
       
        # Legacy code for SciPy Integration
        
        self.logger.debug1(f"Calculating EOM Integration by Direct SciPy Integration")

        def func(y, t):
            return numpy.dot(prop_mat, y) + dvec

        tspace = numpy.linspace(0.0, self.sim.tstep_val, num=3)
        sol = scipy.integrate.odeint(func, zvec, tspace)
        return sol[-1,:d], sol[-1,d:]


    def gen_taylor_coeffs(self, array):
        r = numpy.arange(-len(array)*self.sim.tstep_val, 0, num=len(array))
        pchip = scipy.interpolate.PchipInterpolator(r, array, extrapolate=True)
        taylor = scipy.interpolate.approximate_taylor_polynomial(pchip, -2*self.sim.tstep_val, self.order, self.sim.tstep_val*2)
        return taylor.coeffs

    def next_coefficient(self, coeff, d_coeff, t):
        if t < self.aux_thresh:
            return self.eom_aux.next_coefficient(coeff, d_coeff, t)
        wp = self.sim.previous_wavefunction
        g = self.sim.active_gaussian
        pot_intor = self.sim.potential.gen_potential_integrator()
        ovlp_gg = intor.int_request(self.sim, 'int_ovlp_gg', g, g, t)
        ovlp_gw = intor.int_request(self.sim, 'int_ovlp_gw', g, wp, t)
        dovlp_gg = intor.int_request(self.sim, 'int_dovlp_gg', g, g, t)
        dovlp_gw = intor.int_request(self.sim, 'int_dovlp_gw', g, wp, t)
        dovlp_wg = intor.int_request(self.sim, 'int_dovlp_wg', wp, g, t)
        dovlp_ww = intor.int_request(self.sim, 'int_dovlp_ww', wp, wp, t)
        a0 = self.sim.active_coeffs[t]
        e_g = g.energy_tot(t)
        e_wp = wp.energy_tot(t)
        e_coupling = intor.int_request(self.sim, 'int_kinetic_gw', g, wp, t)
        e_coupling += pot_intor.int_request('int_gVw', g, wp, t)

        def cnew(t, y):
            a_coeff = y+a0
            b_coeff = -a_coeff * ovlp_gw.real + numpy.sqrt(a_coeff*a_coeff*(ovlp_gw.real*ovlp_gw.real - ovlp_gg) + 1)
            b_breve = -ovlp_gw.real + a_coeff*(ovlp_gw.real*ovlp_gw.real - ovlp_gg) / numpy.sqrt(a_coeff*a_coeff*(ovlp_gw.real*ovlp_gw.real - ovlp_gg) + 1)

            a_term = dovlp_gg + dovlp_wg + 1j*(e_g - b_breve * e_coupling.conj())
            b_term = dovlp_gw + dovlp_ww + 1j*(e_coupling - b_breve * e_wp)
            inv_term = 2 * b_breve * ovlp_gw.real - ovlp_gg - b_breve * b_breve
            d_coeff = (a_coeff * a_term + b_coeff * b_term) / inv_term
            return d_coeff.real

        solver = scipy.integrate.ode(cnew).set_integrator('vode', method='bdf')
        solver.set_initial_value(0, 0)
        cn = solver.integrate(self.sim.tstep_val)
        #if abs(cn) > numpy.sqrt(abs(coeff[t])) / self.sim.tstep_val:
        #    return self.eom_aux.next_coefficient(coeff, d_coeff)
        if numpy.isnan(cn) or numpy.isinf(cn) or not solver.successful():
            return self.eom_aux.next_coefficient(coeff, d_coeff, t)
        if cn < 0 and cn > -10**-9:
            cn = -10**-9
        elif cn >= 0 and cn < 10**-9:
            cn = 10**-9
        return cn

    def next_centre(self, gaussian, t):
        if t < self.aux_thresh:
            return self.eom_aux.next_centre(gaussian, t)
        momentum = gaussian.momentum[t]
        centre = gaussian.centre[t]
        prop_mat = self.build_matrix(centre, momentum)
        dx = self.get_timestep(centre, momentum, gaussian.d_centre_v[t], gaussian.d_momentum_v[t], prop_mat)[0]
        if not numpy.isnan(dx).any():
            return dx
        return self.eom_aux.next_centre(gaussian, t)

    def next_momentum(self, gaussian, t):
        if t < self.aux_thresh:
            return self.eom_aux.next_momentum(gaussian, t)
        momentum = gaussian.momentum[t]
        centre = gaussian.centre[t]
        prop_mat = self.build_matrix(centre, momentum)
        dp = self.get_timestep(centre, momentum, gaussian.d_centre_v[t], gaussian.d_momentum_v[t], prop_mat)[1]
        if not numpy.isnan(dp).any():
            return dp
        return self.eom_aux.next_momentum(gaussian, t)


    def next_phase(self, gaussian, t):
        if t < self.aux_thresh:
            return self.eom_aux.next_phase(gaussian, t)
        d = self.sim.dim
        x0 = gaussian.centre[t]
        p0 = gaussian.momentum[t]
        x_var = gaussian.d_centre_v[t]
        p_var = gaussian.d_momentum_v[t]
        prop_mat = self.build_matrix(x0, p0)
        dvec = numpy.concatenate((x_var, p_var))
        zvec = numpy.concatenate((x0, p0))
        v0 = self.sim.potential.evaluate(x0)
        grad = self.sim.potential.gradient(x0)
        hess = self.sim.potential.hessian(x0)
        dvec[d:] += numpy.dot(hess, x0) - grad

        if abs(numpy.linalg.det(prop_mat)) < 10**-12:
            return self.eom_aux.next_phase(gaussian, t)

        def gamma(tau):
            svec = numpy.dot(numpy.linalg.inv(prop_mat), dvec)
            sol = numpy.dot(scipy.linalg.expm(prop_mat * tau), (zvec - svec)) - svec
            p = sol[d:]
            x = sol[:d]
            minv = 1/self.sim.potential.reduced_mass
            g = 2 * numpy.dot(gaussian.width*gaussian.width*x*x, minv) - 0.5 * numpy.dot(p*p,minv) - 1.5*numpy.einsum('i,ij,j', x, hess, x) - numpy.dot(x, grad) + numpy.einsum('i,ij,j', x, hess, x0) - 0.25 * numpy.dot(numpy.diag(hess), 1/gaussian.width) - v0 + gaussian.d_phase_v[t]
            return g
        
        sol = scipy.integrate.quad(gamma, 0, self.sim.tstep_val)
        return sol[0] + gaussian.phase[t]


