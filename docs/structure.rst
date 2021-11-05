Model Structure
===============


.. image:: hierarchy.jpg





RAFT Objects
---------------

RAFT models floating offshore wind turbines by creating Python objects to describe the system.

One of RAFT's unique features is its ability to model an array of floating offshore wind turbines (FOWTS). It does this by 
creating Python objects to describe different parts of the system. 



Each FOWT object is made up of a number of Member objects. Members can be cylinders or rectangular prisms that are part of the substructure or the turbine tower.
Each FOWT also contains a Rotor object to model the wind turbine rotor and blade effects.

Model
^^^^^
An entire floating offshore wind turbine array can be stored in a "Model" object. A model includes each FOWT, the array-wide mooring system, and various 
modeling parameters, such as the design load cases (DLCs). These attributes are then run to solve for the response amplitude operators (RAOs) of each FOWT in the array.
There should only be one MOdel object per RAFT instance.

- Modeling frequencies (math:: \omega)
- a list of FOWTs
- the list of (x,y) coordinates for each FOWT in the Model
- the mooring system

FOWTs
^^^^^

A FOWT object holds all components of a floating wind turbine platform other than the mooring system. This includes the floating substructure, the turbine tower, and the turbine RNA.
The floating substructure and turbine tower can be described as a list of cylindrical or rectangular Members objects. The RNA is represented as a Rotor object.

- calcStatics: calculates the static properties of the FOWT to fill in its 6x6 matrices
- - mass, inertia, ballast, volume, hydrostatics, center of gravity
- calcBEM: generates a mesh for each member to calculate the hydrodynamic properties of each member in the FOWT by running pyHAMS
- calcTurbineConstants: calculates the aerodynamic added mass, damping, and forcing matrices of the rotor
- calcHydroConstants: calculates the hydrodynamic added mass and forcing terms of each member using a Morison equation approximation, if desired
- calcLinearizedTerms: calculates the hydrodynamic damping and forcing as a result from nonlinear drag
- - Needs the platform motions as input to calculate the relative velocity at each position 


Members
^^^^^^^

A member is any cylindrical or rectangular component of a FOWT. They are assumed to be completely rigid objects with no flexibility.
There are many attributes that are used to describe members.

- bottom node position; top node positions (used to define orientation)
- shape (circular or rectangular)
- length
- potMod (toggle to determine whether to calculate hydrodynamics using BEM or not)
- heading (if the member needs to be duplicated by rotating about the PRP)
- stations (a division of the member along its length either in units of length or from 0 to 1)
- diameters (either a scalar to apply at each station, or a list of len(stations))
- - side lengths (for rectangular members, with two values for each side length for each station)
- gamma (the angle of twist about the member's axial axis)
- thicknesses (either a scalar to apply at each station, or a list of len(stations))
- height of ballast
- density of ballast
- density of shell material (steel is normally 7850 kg/m^3, but sometimes we say 8500 to account for auxilliary equipment)
- end caps and bulkheads (positions along the member, thicknesses, inner missing diameter like a donut)
- drag coefficients (axial, transverse1, transverse2, end)
- added mass coefficients (axial, transverse1, transverse2, end)
- meshing variables

Rotor
^^^^^