MHK Turbine Applications
========================

As part of the CT-Opt project, RAFT is being expanded to also support underwater
marine hydrokinetic (MHK) turbines. Just as with floating wind turbines, RAFT
supports frequency-domain modeling of the global response and linearized controlled
rotor dynamics of a moored, floating, MHK turbine system. This page provides 
information about using the under-development MHK capabilities of RAFT. Please
refer to the other pages for general usage, and then this page for specific 
usage changes needed for MHK applications.


Running RAFT
------------

Usage patterns for MHK applications are identical to those for floating wind turbine
applications. Refer to the :ref:`Usage and Workflow page<Usage and Workflow>` for more information.


Model Setup
^^^^^^^^^^^

The main differences for MHK applications are in how the design is set up in the input
dictionary or YAML file. Current speed, shear exponent, and heading must be entered in
the Case input section. And the rotor location must be specified beneath the seabed.


Additional Phenomena
--------------------

For MHK applications, RAFT simulates a number of additional phenomena. The features that have
been added are as follows.

- Rotor added mass
- Rotor buoyancy
- Multiple rotors with arbitrary positions and attachments
- Cavitation check
- Rotor gyroscopic reactions
- Mean current drag loads on floating structure


Inputs
------

(This section to be updated)

The input design YAML can be broken up into multiple parts. The following contains the various sections of an example
input file for the IEA 15MW turbine with the VolturnUS-S steel semi-submersible platform.

Modeling Settings
^^^^^^^^^^^^^^^^^

.. code-block:: python

    settings:                   # global Settings
        min_freq     :  0.005   #  [Hz]       lowest frequency to consider, also the frequency bin width 
        max_freq     :  0.40    #  [Hz]       highest frequency to consider
        XiStart      :   0      # sets initial amplitude of each DOF for all frequencies
        nIter        :  10      # sets how many iterations to perform in Model.solveDynamics()

Site Characteristics
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    site:
        water_depth : 200        # [m]      uniform water depth
        rho_water   : 1025.0     # [kg/m^3] water density
        rho_air     : 1.225      # [kg/m^3] air density
        mu_air      : 1.81e-05   #          air dynamic viscosity
        shearExp    : 0.12       #          shear exponent

Load Cases
^^^^^^^^^^

This section lists the environmental and operating conditions of each load case to be analyzed.

.. code-block:: python

    cases:
        keys : [wind_speed, wind_heading, turbulence, turbine_status, turbine_heading, wave_spectrum, wave_period, wave_height, wave_heading, current_speed, current_heading, current_turbulence  ]
        data :  #   m/s        deg    % or e.g. IIB_NTM    string            deg         string          (s)          (m)          (deg)         (m/s)           (deg)         % or e.g. IIB_NTM     
            -  [     0,         0,            0,         operating,           0,        JONSWAP,         12,          1,           0,            1,              0,                   0           ]
      
The reference height of current_speed depends on whether it is a MHK or floating wind application.
If the first (or only) rotor is underwater, then the current speed refers to the hub height of the first rotor.
Otherwise, the current speed is taken to be at the water surface.

Nonzero turbine headings are not yet supported but will be in the future.


Turbine
^^^^^^^

.. code-block:: python

    turbine:
        
        mRNA          :     991000        # [kg]      RNA mass 
        IxRNA         :          0        # [kg-m2]   RNA moment of inertia about local x axis (assumed to be identical to rotor axis for now, as approx) [kg-m^2]
        IrRNA         :          0        # [kg-m2]   RNA moment of inertia about local y or z axes [kg-m^2]
        xCG_RNA       :          0        # [m]       x location of RNA center of mass [m] (Actual is ~= -0.27 m)
        hHub          :        150.0      # [m]       hub height above water line [m]
        Fthrust       :       1500.0E3    # [N]       temporary thrust force to use
        
        I_drivetrain: 318628138.0         # [kg-m^2]  full rotor + drivetrain inertia as felt on the high-speed shaft
        
        nBlades     : 3                   #           number of blades
        Zhub        : 150.0               # [m]       hub height 
        Rhub        : 3.97                # [m]       hub radius 
        precone     : 4.0                 # [deg]
        shaft_tilt  : 6.0                 # [deg]
        overhang    : 12.0313             # [m]
        
		
        blade: 
            precurveTip : -4.0            # [m]
            presweepTip : 0.0             # [m] 
            Rtip        : 120.97          # [m]       rotor tip radius from axis

            geometry: 
            #          r        chord     theta     precurve  presweep  
              - [     8.004,    5.228,    15.474,    0.035,   0.000 ]
              - [    12.039,    5.321,    14.692,    0.084,   0.000 ]
              - [    16.073,    5.458,    13.330,    0.139,   0.000 ]
              - ...                                
              - [   104.832,    2.464,    -2.172,   -2.523,   0.000 ]
              - [   108.867,    2.283,    -2.108,   -2.864,   0.000 ]
              - [   112.901,    2.096,    -1.953,   -3.224,   0.000 ]
              - [   116.936,    1.902,    -1.662,   -3.605,   0.000 ]
			  
            airfoils: 
            #     station(rel)  airfoil name 
              - [   0.00000,   circular       ]
              - [   0.02000,   circular       ]
              - [   0.15000,   SNL-FFA-W3-500 ]
              - [   0.24517,   FFA-W3-360     ]
              - [   0.32884,   FFA-W3-330blend]
              - [   0.43918,   FFA-W3-301     ]
              - [   0.53767,   FFA-W3-270blend]
              - [   0.63821,   FFA-W3-241     ]
              - [   0.77174,   FFA-W3-211     ]
              - [   1.00000,   FFA-W3-211     ]


        airfoils: 
          - name               : circular
            relative_thickness : 1.0
            data:  # alpha       c_l         c_d         c_m  
              - [ -179.9087,    0.00010,    0.35000,   -0.00010 ] 
              - [  179.9087,    0.00010,    0.35000,   -0.00010 ] 
			  
          - name               : SNL-FFA-W3-500 
            relative_thickness : 0.5 
            data:  # alpha       c_l         c_d         c_m   
              - [ -179.9660,    0.00000,    0.08440,    0.00000 ] 
              - [ -170.0000,    0.44190,    0.08440,    0.31250 ] 
              - [ -160.0002,    0.88370,    0.12680,    0.28310 ] 
              - ...
              - [  179.9660,    0.00000,    0.08440,    0.00000 ] 			  
			  
          - ...

   
        pitch_control:
          GS_Angles: [0.06019804, 0.08713416, 0.10844806, 0.12685912, ... ]
          GS_Kp: [-0.9394215 , -0.80602855, -0.69555026, -0.60254912, ... ]
          GS_Ki: [-0.07416547, -0.06719673, -0.0614251 , -0.05656651, ... ]
          Fl_Kp: -9.35
        wt_ops:
            v: [3.0, 3.266896551724138, 3.533793103448276, 3.800689655172414, ... ]
            pitch_op: [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, ...]
            omega_op: [2.1486, 2.3397, 2.5309,  2.722, 2.9132, 3.1043, 3.2955, ...]
        gear_ratio: 1
        torque_control:
            VS_KP: -38609162.66552
            VS_KI: -4588245.18720
        
        
        tower:
            dlsMax    :  5.0                       # maximum node splitting section amount; can't be 0
        
            name      :  tower                     # [-]    an identifier 
            type      :  1                         # [-]    
            rA        :  [ 0, 0,  15]              # [m]    end A coordinates
            rB        :  [ 0, 0, 144.582]          # [m]    and B coordinates
            shape     :  circ                      # [-]    circular or rectangular
            gamma     :  0.0                       # [deg]  twist angle about the member's z-axis
            
            stations  :  [ 15,  28,  ...  144.5]   # [-]    location of stations along axis. Will be normalized such that start value maps to rA and end value to rB
            d         :  [ 10,  9.9, ...  6.5 ]    # [m]    diameters if circular or side lengths if rectangular (can be pairs)
            t         :  [ 0.08295,  0.0829,...]   # [m]    wall thicknesses (scalar or list of same length as stations)
            Cd        :  0.0                       # [-]    transverse drag coefficient       (optional, scalar or list of same length as stations)
            Ca        :  0.0                       # [-]    transverse added mass coefficient (optional, scalar or list of same length as stations)
            rho_shell :  7850                      # [kg/m3]   material density






Example MHK Turbine Case
------------------------

A rough example MHK turbine case has been added to the designs included in RAFT.
While a proper reference design is in development, this example can be used to
demonstrate the new features.



The figure below is generated by RAFT and shows the calcualted system 
equilibrium state in unloaded and loaded conditions (produced using the Model.plot method).

.. image:: /images/FOCTT.png
    :align: center

As with FOWTs, properties like natural frequencies and mode shapes can be calculated.

The plot below show the power spectral densities of select responses calculated from
a basic load case (produced using the Model.plotResponse method).

.. image:: /images/FOCTT_response.png
    :align: center
    :scale: 80 %
   

The table below shows the response statistics calculated by
RAFT for an example case.

==================  =========    ========   =========
Response channel     Average     RMS         Maximum
==================  =========    ========   =========
surge (m)            1.68e-02    6.30e-01    1.91e+00
sway (m)            -2.54e-08    2.92e-09   -2.54e-08
heave (m)           -1.34e+00    5.55e-01    3.22e-01
roll (deg)          -2.88e-10    1.23e-09    3.41e-09
pitch (deg)          1.16e-03    2.46e-01    7.41e-01
yaw (deg)           -4.67e-12    2.24e-10    6.69e-10
nacelle acc. (m/s)   0.00e+00    2.97e-01    0.00e+00
tower bending (Nm)   3.69e+04    5.46e+07    0.00e+00
rotor speed (RPM)    0.00e+00    0.00e+00    0.00e+00
blade pitch (deg)    0.00e+00    0.00e+00
rotor power          0.00e+00
line 1 tension (N)   2.61e+06    3.15e+04    2.71e+06
line 2 tension (N)   2.62e+06    2.45e+04    2.69e+06
line 3 tension (N)   2.62e+06    2.45e+04    2.69e+06
==================  =========    ========   =========





