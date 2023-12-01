# lib-qdmmg

This library is an implementation the Matching Marching Gaussians Quantum Dynamics algorithm. The Matching Marching Gaussians (MMG) algorithm is a method for direct Born-Oppenheimer dynamics. 

## Installation

Currently, only installation through `git clone` is available. As lib-qdmmg has been developed using Python 3.11, it is not recommended to use an older Python version. Additionally, the following Python dependencies need to be installed:

| Dependency | Minimum Version |
|------------|-----------------|
| NumPy      | 1.25.1          |
| SciPy      | 1.11.1          |
| Matplotlib | 3.7.2           |
| PySCF      | 2.3.0           |

## Usage

lib-qdmmg is intended to be used analogously to PySCF. See the `examples` folder for more information. Generally, a calculation using MMG is divided into three parts:

1. Initialisation
2. Propagation
3. Finalisation and Analysis

For the initialisation, you first need to construct a `libqdmmg.simulate.simulation.Simulation` object, which requires you to specify the number of timesteps, the duration of a single step in Hartree-seconds and the number of dimensions of the simulation. Next, you are required to construct a `potential.potential.Potential` object and bind it to the simulation. These are the required steps of initialisation. 

Propagation is performed by calling the function `gen_wavefunction()` on the simulation object. 

Analysis can be performed freely using the functionality given by the `export` and `plotting` module.
