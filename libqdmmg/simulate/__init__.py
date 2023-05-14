'''

Lib-QDMMG Simulate Package


'''

import libqdmmg.simulate.simulation as sim
import libqdmmg.simulate.eom_master as eom_master

def Simulation(tsteps, tstep_val, verbose=0, dim=1, generations=0):
    return sim.Simulation(tsteps, tstep_val, verbose, dim, generations)

def EOM_Master(s):
    return eom_master.EOM_Master(s)
