'''

Lib-QDMMG Simulate Package


'''

import libqdmmg.simulate.simulation as sim
import libqdmmg.simulate.eom_master as eom_master
import libqdmmg.simulate.eom_integrator as eom_intor
import libqdmmg.simulate.gaussian_strategy as gstrat

def Simulation(tsteps, tstep_val, verbose=0, dim=1, generations=0, qcutoff=0.1, micro_steps=4):
    return sim.Simulation(tsteps, tstep_val, verbose, dim, generations, qcutoff, micro_steps)

def EOM_Master(sim, stepping=-1, invert_stepping=False, qcutoff=0.1):
    return eom_master.EOM_Master(sim, stepping, invert_stepping, qcutoff)

def EOM_EulerIntegrator(sim, micro_steps=4):
    return eom_intor.EOM_EulerIntegrator(sim, micro_steps)

def EOM_AdamsBashforth(sim, order=5, micro_steps=4):
    return eom_intor.EOM_AdamsBashforth(sim, order=order, micro_steps=micro_steps)

def EOM_Pade(sim, m_order=4, n_order=2, micro_steps=4):
    return eom_intor.EOM_Pade(sim, m_order, n_order, micro_steps)

def EOM_Matexp(sim, order=2, auxorder=3, micro_steps=4):
    return eom_intor.EOM_Matexp(sim, order, auxorder, micro_steps)

def GaussianGridStrategy(sim, gridsize, gridres, gridshift, prio_size=10):
    return gstrat.GaussianGridStrategy(sim, gridsize, gridres, gridshift, prio_size)

def GaussianSingleRandomStrategy(sim, centre_mu=1.0, centre_sigma=0.5, width_mu=1.0, width_sigma=0.3, init_full=True):
    return gstrat.GaussianSingleRandomStrategy(sim, centre_mu, centre_sigma, width_mu, width_sigma, init_full)

def fs_to_tsteps(fs, tstep_val):
    return int(fs / (0.02418884326 * tstep_val))+1
