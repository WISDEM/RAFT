# RAFT - Response Amplitudes of Floating Turbines

RAFT is a Python code for frequency-domain analysis of floating wind turbines. It constitutes the "Level 1" of modeling fidelity in the [WEIS](https://weis.readthedocs.io/en/latest/index.html) toolset for floating wind turbine controls co-design. Once completed, RAFT will provide frequency-domain modeling of full floating wind turbine systems including turbine aerodynamics, controls, select structural degrees of freedom, platform hydrodynamics, and mooring response. RAFT is under active development and currently has the floating support structure components implemented. These components will be actively tested and revised as the model is verified and further developed. Upcoming development efforts focus turbine aerodynamics and control, and integration with the larger WEIS framework.

This project makes use of the HAMS (Hydrodynamic Analysis of Marine Structures) tool for boundary-element-method solution of the potential flow problem developed by Yingyi Liu and available at https://github.com/YingyiLiu/HAMS. We use a Python wrapper for it, [pyHAMS](https://github.com/WISDEM/pyHAMS).

## Getting Started

As RAFT is still a work in progress, we recomend prospective users and beta testers begin by running runRAFT.py, which is set to use one of three included example input files.

### Prerequisites

- Python 3
- MoorPy (available at https://github.com/NREL/MoorPy)
- pyHams (available at https://github.com/WISDEM/pyHAMS)
- CCBlade and WISDEM* (available at https://github.com/WISDEM/WISDEM)

\* RAFT uses CCBlade and currently requires additional related functions from the larger WISDEM code. We recommend installing WISDEM for the time being.

### Installation

Download/clone and install this RAFT repository as well as that of [MoorPy](https://github.com/NREL/MoorPy), [pyHAMS](https://github.com/WISDEM/pyHAMS), and [WISDEM](https://github.com/WISDEM/WISDEM). To install RAFT in development mode, go to its directory and run ```python setup.py develop``` or ```pip install -e .``` from the command line.

## Documentation

A dedicated documentation site is under development. In the meantime, comments in the code are the best guide for how RAFT works.
