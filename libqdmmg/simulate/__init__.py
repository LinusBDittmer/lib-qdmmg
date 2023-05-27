'''

Lib-QDMMG Simulate Package


'''

import libqdmmg.simulate.simulation as sim
import libqdmmg.simulate.eom_master as eom_master
import libqdmmg.simulate.eom_integrator as eom_intor

def Simulation(tsteps, tstep_val, verbose=0, dim=1, generations=0):
    return sim.Simulation(tsteps, tstep_val, verbose, dim, generations)

def EOM_Master(sim):
    return eom_master.EOM_Master(sim)

def EOM_EulerIntegrator(sim):
    return eom_intor.EOM_EulerIntegrator(sim)

def EOM_AdamsBashforth(sim, order=5):
    return eom_intor.EOM_AdamsBashforth(sim, order=order)
