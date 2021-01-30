# RAFT - Response Amplitudes of Floating Turbines

RAFT is a Python code for frequency-domain analysis of floating wind turbines. It constitutes the first level of modeling fidelity in the WEIS toolset for floating wind turbine controls co-design. Once completed, RAFT will provide frequency-domain modeling of full floating wind turbine systems including turbine aerodynamics, controls, select structural degrees of freedom, platform hydrodynamics, and mooring dynamics. RAFT is under development and currently has the floating support structure components implemented. These components will be actively tested and revised as the model is verified and further developed. Upcoming development efforts focus on inclusion of tower flexibility and turbine aerodynamics. 

This project makes use of the HAMS (Hydrodynamic Analysis of Marine Structures) tool developed by Yingyi Liu and available at https://github.com/YingyiLiu/HAMS.

## Getting Started

As RAFT is still a work in progress, prospective users and beta testers are recommended to begin by running runRAFT.py, which is set to use one of three included example input files.

### Prerequisites

- Python 3

### Installation

run ```python setup.py develop``` from cmd line to install the package as a developer

## Documentation

A dedicated documentation site is under development. In the meantime, comments in the code are the best guide for how RAFT works.

## License
Licensed under the Apache License, Version 2.0
