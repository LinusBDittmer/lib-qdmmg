'''

Lib-QDMMG Simulate Package


'''

import libqdmmg.simulate.simulation as sim
import libqdmmg.simulate.eom_master as eom_master
import libqdmmg.simulate.eom_integrator as eom_intor

def Simulation(tsteps, tstep_val, verbose=0, dim=1, generations=0):
    return sim.Simulation(tsteps, tstep_val, verbose, dim, generations)

def EOM_Master(sim, stepping=-1, invert_stepping=False, qcutoff=0.1):
    return eom_master.EOM_Master(sim, stepping, invert_stepping, qcutoff)

def EOM_EulerIntegrator(sim):
    return eom_intor.EOM_EulerIntegrator(sim)

def EOM_AdamsBashforth(sim, order=5):
    return eom_intor.EOM_AdamsBashforth(sim, order=order)

def EOM_Pade(sim, m_order=4, n_order=2):
    return eom_intor.EOM_Pade(sim, m_order, n_order)

def fs_to_tsteps(fs, tstep_val):
    return int(fs / (0.02418884326 * tstep_val))+1
