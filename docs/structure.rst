Model Structure
===============




RAFT Model Objects
------------------

RAFT represents a floating wind turbine system as a collection of different object types, as illustrated below.

.. image:: /images/objects.JPG
    :align: center
	
Each of these objects corresponds to a class in python. The Model class contains 
the full RAFT representation of a floating wind system. Within it, the FOWT class
describes the floating platform and turbine, while the mooring sytem is handled by
a separate model, `MoorPy <https://moorpy.readthedocs.io>`_. Within the FOWT class,
the floating platform and turbine tower are represented as a single rigid body 
composed of multiple Member objects that describe linear cylindrical or rectangular
geometries. The turbine rotor-nacelle assembly is represented by a Rotor class that 
describes lumped mass properties and distributed aerodynamic properties. These RAFT
objects are further described below.	

Model
^^^^^
An entire floating offshore wind turbine array can be stored in a "Model" object. A model includes a list of the FOWTs in the array,
the array-wide mooring system, and various modeling parameters, such as the design load cases (DLCs). These attributes are then run
to solve for the response amplitude operators (RAOs) of each FOWT in the array. There should only be one Model object per RAFT instance.

- Modeling frequencies (:math:`\omega`)
- a list of FOWTs
- the list of (x,y) coordinates for each FOWT in the Model
- the mooring system

FOWT
^^^^
A FOWT object holds all components of a floating wind turbine platform other than the mooring system. This includes the floating substructure, the turbine tower, and the turbine RNA.
The floating substructure and turbine tower can be described as a list of cylindrical or rectangular Members objects. The RNA is represented as a Rotor object.

- calcStatics: calculates the static properties of each Member in the FOWT to fill in its 6x6 matrices

  - mass, inertia, ballast, volume, hydrostatics, center of gravity

- calcBEM: generates a mesh for each member to calculate the hydrodynamic properties of each member in the FOWT by running pyHAMS
- calcTurbineConstants: calculates the aerodynamic added mass, damping, and forcing matrices of the rotor
- calcHydroConstants: calculates the hydrodynamic added mass and forcing terms of each member using a Morison equation approximation
- calcLinearizedTerms: calculates the hydrodynamic damping and forcing as a result from nonlinear drag

  - Needs the platform motions as input to calculate the relative velocity at each position 


Member
^^^^^^
A Member is any cylindrical or rectangular component of a FOWT. They are assumed to be completely rigid objects with no flexibility.
RAFT currently does not support structural flexibility

There are many attributes that are used to describe members.

- bottom node position; top node positions (used to define orientation)
- shape (circular or rectangular)
- length
- potMod (toggle to determine whether to calculate hydrodynamics using BEM or Morison)
- heading (duplicate members at various headings by rotating about the PRP)
- stations (a division of the member along its length either in units of length or from 0 to 1)
- diameters (either a scalar to apply at each station, or a list of len(stations))

  - side lengths (for rectangular members, with two values for each station)

- gamma (the angle of twist about the member's axial axis)
- thicknesses (either a scalar to apply at each station, or a list of len(stations))
- height of ballast
- density of ballast
- density of shell material (steel is normally 7850 kg/m^3, but can be approximated as 8500 kg/m^3 to account for auxilliary equipment)
- end caps and bulkheads (positions along the member, thicknesses, inner missing diameter)
- drag coefficients (axial, transverse1, transverse2, end)
- added mass coefficients (axial, transverse1, transverse2, end)
- meshing variables

Rotor
^^^^^
A Rotor object is used to describe the aerodynamics and controls of the FOWT. Linearized.

- height of the hub [m]
- shaft tilt [deg]
- overhang [m]
- rotor radius [m]
- wind speeds at hub
- rotor RPM
- blade pitch
- moment of inertia of the drivetrain
- control gains
- Airfoil setup

Most of these go into setting up a CCAirfoil and CCBlade object, which is then able to be run and the important parameters puled out
from calcAeroContributions and calcAeroServoContributions

- CCBlade returns "loads" and "derivs"

  - can extract the derivatives of thrust with respect to wind speed, rotor RPM, and blade pitch, as well as the derivatives of torque with respect to wind speed, rotor RPM, and blade pitch.

- torque and pitch control gains
- eventually return the aerodynamic forcing (IECKaimal), aerodynamic added mass, and aerodynamic damping



RAFT Heirarchy and Interfaces
------------------------------------

Each RAFT object is described by a class in a separate source file, and these classes interact following
a set heirarchy. Some of the classes also interact with external models. These relationships are 
outlined below.

.. image:: /images/hierarchy.JPG
    :align: center


`MoorPy <https://moorpy.readthedocs.io/en/latest/>`_ is a quasi-static mooring system modeler 
that provides the mooring system representation for RAFT. It supports floating platform linear
hydrostatic properties and constant external loads. For each load case, RAFT provides MoorPy
the system hydrostatic properties and mean wind loads, and MoorPy calculates the mean floating
platform offsets for RAFT.


`CCBlade <https://wisdem.readthedocs.io/en/latest/wisdem/ccblade/index.html>`_ is a blade-element-momentum 
theory model that provides the aerodynamic coefficients for RAFT. These coefficients account for the mean
aerodynamic forces and moments of the wind turbine rotor at each wind speed, as well as the derivatives 
of those forces and moments with respect to changes in relative wind speed, rotor speed, and blade pitch
angle. From this information, RAFT can compute the overall aerodynamic effects of the system's rigid-body
response, including accounting for the impact of turbine control.

`pyHAMS <https://github.com/WISDEM/pyHAMS>`_ is a Python wrapper of the HAMS (Hydrodynamic Analysis of 
Marine Structures) tool for boundary-element-method solution of the potential flow problem. HAMS is 
developed by Yingyi Liu and available at https://github.com/YingyiLiu/HAMS. HAMS provides the option
for RAFT to use potential-flow hydrodynamic coefficients instead of, or in addition to, strip theory.

