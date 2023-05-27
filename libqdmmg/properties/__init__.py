'''

Module for computing properties

'''

import libqdmmg.properties.property as prop

def KineticEnergy(sim):
    return prop.KineticEnergy(sim)

def PotentialEnergy(sim):
    return prop.PotentialEnergy(sim)

def TotalEnergy(sim, kinetic_energy, potential_energy):
    return prop.TotalEnergy(sim, kinetic_energy, potential_energy)

def AverageDisplacement(sim):
    return prop.AverageDisplacement(sim)


