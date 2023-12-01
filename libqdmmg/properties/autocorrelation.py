'''

Autocorrelation function

'''

from libqdmmg.properties.property import Property
import libqdmmg.integrate as intor
import numpy
import scipy

class Autocorrelation(Property):

    def __init__(self, sim):
        super().__init__(sim, "Autocorrelation", dtype=float)
        self.zerotime = sim.previous_wavefunction.zerotime_wp()

    def compute(self, obj, t):
        obj = self.wavepacketify(obj)
        a = intor.int_request(self.sim, 'int_ovlp_ww', self.zerotime, obj, t)
        return ((numpy.conj(a) * a).real)**0.5


class FourierAutocorrelation(Property):

    def __init__(self, sim, auto):
        super().__init__(sim, "Fourier Autocorrelation", dtype=float)
        self.auto = auto

    def kernel(self, obj=None):
        if obj is None:
            obj = self.sim.previous_wavefunction
        else:
            obj = self.wavepacketify(obj)

        self.values = scipy.fft.fft(self.auto.get())
        self.values[0] = 0
        self.values = self.values[:int(0.5*self.sim.tsteps)+1]
        self.values = scipy.interpolate.interp1d(numpy.arange(len(self.values)), self.values)(numpy.linspace(0.0, len(self.values)-1, num=self.sim.tsteps))

class Populations(Property):

    def __init__(self, sim):
        super().__init__(sim, "Populations", dtype=float, shape=tuple([len(sim.previous_wavefunction.gaussians)]))
        self.gd = len(sim.previous_wavefunction.gaussians)
        self.ovlp = numpy.zeros((self.gd, self.gd), dtype=numpy.complex128)
        self.islog = True

    def compute(self, obj, t):
        obj = self.wavepacketify(obj)
        for i in range(self.gd):
            for j in range(self.gd):
                self.ovlp[i,j] = intor.int_request(self.sim, 'int_ovlp_gg', self.sim.previous_wavefunction.gaussians[i], self.sim.previous_wavefunction.gaussians[j], t)
        ortho = scipy.linalg.sqrtm(self.ovlp)
        w, s, vh = scipy.linalg.svd(self.ovlp)
        nat_pop = numpy.dot(ortho, obj.get_coeffs(t))
        return (nat_pop.conj() * nat_pop).real
