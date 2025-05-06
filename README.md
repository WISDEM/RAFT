# RAFT - Response Amplitudes of Floating Turbines

RAFT is a Python code for frequency-domain analysis of floating wind turbines. It constitutes the "Level 1" of modeling fidelity in the [WEIS](https://weis.readthedocs.io/en/latest/index.html) toolset for floating wind turbine controls co-design. RAFT provides frequency-domain modeling of full floating wind turbine systems including turbine aerodynamics, controls, rigid-body platform hydrodynamics, and mooring response. The rotor aerodynamics are provided by [CCBlade](https://github.com/WISDEM/CCBlade) and the mooring system is represented by NREL's new quasi-static mooring system model, [MoorPy](https://github.com/NREL/MoorPy). Potential-flow hydrodynamics can optionally be used via [pyHAMS](https://github.com/WISDEM/pyHAMS), a Python wrapper of the HAMS (Hydrodynamic Analysis of Marine Structures) tool for boundary-element-method solution of the potential flow problem developed by [Yingyi Liu](https://github.com/YingyiLiu/HAMS).

RAFT v1.0.0 includes the capabilities described above, and further development is planned to expand these capabilities. Documentation and verification efforts are ongoing. Please see [RAFT's documentation](https://openraft.readthedocs.io/en/latest/) for more information.


## Part of the WETO Stack

RAFT is primarily developed with the support of the U.S. Department of Energy and is part of the [WETO Software Stack](https://nrel.github.io/WETOStack). For more information and other integrated modeling software, see:
- [Portfolio Overview](https://nrel.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://nrel.github.io/WETOStack/_static/entry_guide/index.html)
- [Systems Engineering Workshop](https://nrel.github.io/WETOStack/workshops/user_workshops_2024.html#systems-engineering)
- [OpenFAST Workshop](https://nrel.github.io/WETOStack/workshops/user_workshops_2024.html#openfast-ecosystem)


## Best Practices for Using `conda`

Installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  RAFT requires [Anaconda 64-bit](https://www.anaconda.com/distribution/).  However, the `conda` command has begun to show its age and we now recommend the one-for-one replacement with the [Miniforge3 distribution](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3), which is much more lightweight and more easily solves for the RAFT package dependencies.


## Using RAFT as a Library

If you would like to use RAFT as a library, without direct interaction with the source code, then you can install it directly via `conda install raft` (preferred) or `pip install raft`.


## Getting Started

New users of RAFT as a standalone model are recommended to begin by looking at the input file and script provided in the examples folder, and seeking further information from the [RAFT documentation](https://openraft.readthedocs.io/en/latest/). For use as part of the WEIS toolset, information will be provided once this capability is completed in the [WEIS documentation](https://weis.readthedocs.io/en/latest/). For now, the following will help get started with running RAFT as a standalone model.

RAFT uses a number of prerequisites, or other python package dependencies, to run its calculations. All of the prerequisites can be installed through the `conda` package manager, but power users may wish to install all of the NREL packages from source for customization and debugging.  Notable dependencies are:

- Python 3
- NumPy
- Matplotlib
- SciPy
- YAML
- [MoorPy](https://github.com/NREL/MoorPy)
- [pyHams](https://github.com/WISDEM/pyHAMS)
- [CCBlade](https://github.com/WISDEM/CCBlade) or [WISDEM](https://github.com/WISDEM/WISDEM)*

\* CCBlade is a module of WISDEM, but can be used separately. RAFT only requires CCBlade (and some additional related functions) out of the larger WISDEM code. Power users installing from source can install either CCBlade or WISDEM, but for highest efficiency, we recommend installing CCBlade, without the entire WISDEM installation.

Note that conda is required for the following installations.

## Installing All Dependencies with `conda`

To install all required python packages to run RAFT using `conda`, follow the steps below.

1. Navigate to a directory of your choosing on your local machine and clone this RAFT repository to that new directory (if you don't have the `git` command, start with `conda install git` in your "base" environment).

        (base) YOUR_PATH> git clone https://github.com/WISDEM/RAFT.git
        (base) YOUR_PATH> cd RAFT
    
    This will create new folder called "RAFT" that is a copy from the GitHub repository, located in your "YOUR_PATH" directory

2. Create a new python virtual environment based on the "environment.yml" file, which lists all the packages needed to run RAFT

        (base) YOUR_PATH/RAFT> conda env create --name raft-env -f environment.yml

    This will create a new python virtual environment called "raft-env"

3. Activate the new virtual environment

        (base) ANY_PATH> conda activate raft-env

    This will activate the newly created virtual environment, in which we will install the remaining dependencies.  After install is complete, when opening a new conda terminal that starts in the "base" environment, you can start with this "raft-env" activation command.

4. Install the RAFT package into the virtual environment

        (raft-env) YOUR_PATH/RAFT> pip install -e .

    This installs RAFT and all of its modules directly from your working directory in "editable" mode, meaning, if you save a change to your source code in the files in "YOUR_PATH/RAFT", future calls to RAFT modules will include those changes.

5. Test the installation

        (raft-env) YOUR_PATH/RAFT> pytest
    
    Running pytest should ensure everything was installed correctly
    
	

## Installing NREL Dependencies from Source

To install all external python packages using `conda` and NREL packages from source, follow the steps below. Since the NREL dependencies are also under development, this may also help resolve issues if one code gets out-of-sync with the others.

1. Navigate to a directory of your choosing on your local machine and clone this RAFT repository to that new directory (if you don't have the `git` command, start with `conda install git` in your "base" environment.

        (base) YOUR_PATH> git clone https://github.com/WISDEM/RAFT.git
        (base) YOUR_PATH> cd RAFT
    
    This will create new folder called "RAFT" that is a copy from the GitHub repository, located in your "YOUR_PATH" directory

2. Create a new python virtual environment based on the "environment-source.yml" file, which lists all the packages needed to run RAFT

        (base) YOUR_PATH/RAFT> conda env create --name raft-env -f environment-source.yml

    This will create a new python virtual environment called "raft-env"

3. Activate the new virtual environment

        (base) ANY_PATH> conda activate raft-env

    This will activate the newly created virtual environment, in which we will install the remaining dependencies.  After install is complete, when opening a new conda terminal that starts in the "base" environment, you can start with this "raft-env" activation command.

4. Install the necessary compilers and build tools for your system

        (raft-env) YOUR_PATH/RAFT> cd ..
        (raft-env) YOUR_PATH> pip install meson-python meson ninja cmake
        (raft-env) YOUR_PATH> conda install gfortran                        (Mac / Linux)
        (raft-env) YOUR_PATH> conda install m2w64-toolchain libpython       (Windows)


5. Install the NREL packages from source one at a time.

        (raft-env) YOUR_PATH> git clone https://github.com/WISDEM/CCBlade.git
        (raft-env) YOUR_PATH> git clone https://github.com/NREL/MoorPy.git
        (raft-env) YOUR_PATH> git clone https://github.com/WISDEM/pyHAMS.git
        (raft-env) YOUR_PATH> cd MoorPy
        (raft-env) YOUR_PATH/MoorPy> pip install -e .
        (raft-env) YOUR_PATH/MoorPy> cd ..
        (raft-env) YOUR_PATH> cd CCBlade
        (raft-env) YOUR_PATH/CCBlade> pip install -e .
        (raft-env) YOUR_PATH/CCBlade> cd ..
        (raft-env) YOUR_PATH> cd pyHAMS
        (raft-env) YOUR_PATH/pyHAMS> pip install --no-build-isolation -e .
        (raft-env) YOUR_PATH/pyHAMS> cd ..

    This installs MoorPy, CCBlade, and pyHAMS in "editable" mode, meaning, if you save a change to your source code in the files, future calls to RAFT modules will include those changes.

6. Install the RAFT package into the virtual environment

        (raft-env) YOUR_PATH> cd RAFT
        (raft-env) YOUR_PATH/RAFT> pip install -e .
    
    This installs RAFT and all of its modules directly from your working directory in "editable" mode, meaning, if you save a change to your source code in the files in "YOUR_PATH/RAFT", future calls to RAFT modules will include those changes.

This new "raft-env" should now be compatible to run RAFT standalone.

7. Test the installation

        (raft-env) YOUR_PATH/RAFT> pip install pytest
        (raft-env) YOUR_PATH/RAFT> cd tests
        (raft-env) YOUR_PATH/RAFT/tests> pytest test_fowt.py test_helpers.py test_member.py test_model.py test_rotor.py
    
    Running these specific test files should prove that the installation was successful. Other tests that include 'omdao' are only used for installations with WISDEM.

If you need to remove any virtual environment for any reason, you can run 

    conda env remove -n "name-of-the-virtual-environment"



## Documentation and Issues

Please see <https://weis.readthedocs.io/en/latest/> for documentation.

Questions and issues can be posted at <https://github.com/WISDEM/RAFT/issues>.

## License
RAFT is licensed under the Apache License, Version 2.0

