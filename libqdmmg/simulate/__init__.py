'''

Lib-QDMMG Simulate Package


'''

import libqdmmg.simulate.simulation as sim


def Simulation(tsteps, tstep_val, verbose=0, dim=1):
    return sim.Simulation(tsteps, tstep_val, verbose, dim)
