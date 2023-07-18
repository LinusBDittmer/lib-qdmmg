'''

Module for computing properties

'''

import libqdmmg.properties.property as prop
import libqdmmg.properties.autocorrelation as auto

def KineticEnergy(sim):
    return prop.KineticEnergy(sim)

def PotentialEnergy(sim):
    return prop.PotentialEnergy(sim)

def TotalEnergy(sim, kinetic_energy, potential_energy):
    return prop.TotalEnergy(sim, kinetic_energy, potential_energy)

def AverageDisplacement(sim):
    return prop.AverageDisplacement(sim)

def Norm(sim):
    return prop.Norm(sim)

def Autocorrelation(sim):
    return auto.Autocorrelation(sim)

def FourierAutocorrelation(sim, a):
    return auto.FourierAutocorrelation(sim, a)

def Populations(sim):
    return auto.Populations(sim)
