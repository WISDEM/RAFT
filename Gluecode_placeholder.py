import pytest
import sys
import numpy as np
import matplotlib.pyplot as plt

# test local code; consider src layout in future to test installed code
sys.path.append('..')
import FrequencyDomain as fd
import MoorPy as mp

import MoorDesign as md   #MoorDesign is almost obsolete at this point so this is likely to change in the future 

#import Capytaine_placeholder as capy


import importlib
mp = importlib.reload(mp)
fd = importlib.reload(fd)




# >>>>> this is the test script to execture the frequency domain model <<<<<
def runFDmodel(wt_opt = None, model = 'OC3'):
    '''
        wt_opt : this is where the openMDAO data structure from gc_WT_InitModel.py would get passed in
    '''

    
    # >>>>>>>>>> the simpler case of nothing passed in, so just using manual model inputs specified below <<<<<<<<
    if wt_opt == None:
        
        memberStrings = []      # initialize an empty list to hold all the member strings
        RNAprops = {}           # initialize an empty dictionary to hold all RNA properties
        
        # turn off and on the wind thrust force
        #Fthrust = 800e3                              # peak thrust force, [N]
        Fthrust = 0
        
        if model=='OC3':
            # --- OC3 ---
            # ::::::::::::::::::::::::::: member inputs ::::::::::::::::::::::::::::::::
            
            # turbine Tower description ------------------
            
            # diameters and thicknesses linearly interpolated from dA[0] to dB[-1] and t[0] to t[-1]
            #                 number   type  shape  dA      dB      xa      ya     za     xb     yb      zb        tA       tB   l_fill  rho_fill rho_shell
            memberStrings.append(" 1     1   circ  6.500   6.237    0.0    0.0    10.00   0.0    0.0    17.76    0.0270   0.0262   0.0       0.0    8500  ")
            memberStrings.append(" 2     1   circ  6.237   5.974    0.0    0.0    17.76   0.0    0.0    25.52    0.0262   0.0254   0.0       0.0    8500  ")
            memberStrings.append(" 3     1   circ  5.974   5.711    0.0    0.0    25.52   0.0    0.0    33.28    0.0254   0.0246   0.0       0.0    8500  ")
            memberStrings.append(" 4     1   circ  5.711   5.448    0.0    0.0    33.28   0.0    0.0    41.04    0.0246   0.0238   0.0       0.0    8500  ")
            memberStrings.append(" 5     1   circ  5.448   5.185    0.0    0.0    41.04   0.0    0.0    48.80    0.0238   0.0230   0.0       0.0    8500  ")
            memberStrings.append(" 6     1   circ  5.185   4.922    0.0    0.0    48.80   0.0    0.0    56.56    0.0230   0.0222   0.0       0.0    8500  ")
            memberStrings.append(" 7     1   circ  4.922   4.659    0.0    0.0    56.56   0.0    0.0    64.32    0.0222   0.0214   0.0       0.0    8500  ")
            memberStrings.append(" 8     1   circ  4.659   4.396    0.0    0.0    64.32   0.0    0.0    72.08    0.0214   0.0206   0.0       0.0    8500  ")
            memberStrings.append(" 9     1   circ  4.396   4.133    0.0    0.0    72.08   0.0    0.0    79.84    0.0206   0.0198   0.0       0.0    8500  ")
            memberStrings.append("10     1   circ  4.133   3.870    0.0    0.0    79.84   0.0    0.0    87.60    0.0198   0.0190   0.0       0.0    8500  ")
            
            #memberStrings.append(" 1     1   circ  6.500   3.870    0.0    0.0    10.00   0.0    0.0    87.60    0.0270   0.0190   0.0     0.0     8500  ")
            
            # spar platform substructure description --------------
            # July 2020 weight-buoyancy balancing (either should work)
            #memberStrings.append("11     2   circ  9.400   9.400    0.0    0.0   -120.0   0.0    0.0   -12.00   0.0270   52.0    1850.0   8500  ")
            #memberStrings.append("11     2   circ  9.400   9.400    0.0    0.0   -120.0  0.0    0.0   -12.00   0.0660   41.4    2000.0  8500  ")
            # January 2021 weight-buoyancy balancing
            memberStrings.append("11     2   circ  9.400   9.400    0.0    0.0   -120.0   0.0    0.0   -12.00    0.0270   0.0270  52.0    1860.0    8500  ")
            
            #memberStrings.append("11     2   re  20/9.400   20/9.400    0.0    0.0   -120.0  0.0    0.0   -12.00   0.0660   41.4    2000.0  8500  45.0 ") # rectangular member test
            memberStrings.append("12     2   circ  9.400   6.500    0.0    0.0    -12.0   0.0    0.0    -4.00    0.0270   0.0270   0.0       0.0    8500  ")
            memberStrings.append("13     2   circ  6.500   6.500    0.0    0.0     -4.0   0.0    0.0    10.00    0.0270   0.0270   0.0       0.0    8500  ")
            
            
            # ::::::::::::::::::::::::::: turbine RNA description :::::::::::::::::::::::::::
            #                 Rotor          Nacelle
            mRNA    = 110000               + 240000      # RNA mass [kg]
            IxRNA   = 11776047*(1 + 1 + 1) + 115926      # RNA moment of inertia about local x axis (assumed to be identical to rotor axis for now, as approx) [kg-m^2]
            IrRNA   = 11776047*(1 +.5 +.5) + 2607890     # RNA moment of inertia about local y or z axes [kg-m^2]
            xCG_RNA = 0                                  # x location of RNA center of mass [m] (Actual is ~= -0.27 m)
            hHub    = 90.0                               # hub height above water line [m]
            
            #yawstiff = 98340000.0   # this shouldn't go in the RNA description, but I needed a dictionary variable
            yawstiff = 0
            
            
            # :::::::::::::::::::::::::::::: moorings inputs :::::::::::::::::::::::::::::::::::
            
            depth =         320.                                    # [m] from waterline
            type_string =   'main'                                  #
            anchorR =       853.87                                  # [m] from centerline
            fairR =         5.2                                     # [m] from centerline
            fair_depth =    70.                                     # [m] from waterline
            LineLength =    902.2                                   # [m]
            LineD =         0.09                                    # [m]
            dryMass_L =     77.7066                                 # [kg/m]
            EA =            384243000                               # [N]
            angle =         np.array([0, 2*np.pi/3, -2*np.pi/3])    # [rad]
            rho =           1025.0                                  # [kg/m^3]
            
            
            # make a single-type 3-line mooring system by calling a function in MoorDesign
            MooringSystem = md.make3LineSystem(depth, type_string, LineD, dryMass_L, EA, angle, anchorR, fair_depth, fairR, LineLength)
            # initialize the system
            MooringSystem.initialize()
        
        
        elif model=='test':
            # --- test ---
            # ::::::::::::::::::::::::::: member inputs ::::::::::::::::::::::::::::::::
            memberStrings.append("1  20  circ  10  10  0  0  -10  0  0  10  0.1 0.1  0  0.0  8500")
            
            # ::::::::::::::::::::::::::: turbine RNA description :::::::::::::::::::::::::::
            #                 Rotor          Nacelle
            mRNA    = 110000               + 240000      # RNA mass [kg]
            IxRNA   = 11776047*(1 + 1 + 1) + 115926      # RNA moment of inertia about local x axis (assumed to be identical to rotor axis for now, as approx) [kg-m^2]
            IrRNA   = 11776047*(1 +.5 +.5) + 2607890     # RNA moment of inertia about local y or z axes [kg-m^2]
            xCG_RNA = 0                                  # x location of RNA center of mass [m] (Actual is ~= -0.27 m)
            hHub    = 90.0                               # hub height above water line [m]
            
            
            # :::::::::::::::::::::::::::::: moorings inputs :::::::::::::::::::::::::::::::::::
            depth =         320.                                    # [m] from waterline
            type_string =   'main'                                  #
            anchorR =       853.87                                  # [m] from centerline
            fairR =         5.2                                     # [m] from centerline
            fair_depth =    70.                                     # [m] from waterline
            LineLength =    902.2                                   # [m]
            LineD =         0.09                                    # [m]
            dryMass_L =     77.7066                                 # [kg/m]
            EA =            384243000                               # [N]
            angle =         np.array([0, 2*np.pi/3, -2*np.pi/3])    # [rad]
            rho =           1025.0                                  # [kg/m^3]
            
            
            # make a single-type 3-line mooring system by calling a function in MoorDesign
            MooringSystem = md.make3LineSystem(depth, type_string, LineD, dryMass_L, EA, angle, anchorR, fair_depth, fairR, LineLength)
            # initialize the system
            MooringSystem.initialize()
            
        elif model=='OC4':
            # --- OC4 ---
            # ::::::::::::::::::::::::::: member inputs ::::::::::::::::::::::::::::::::
            
            # turbine Tower description ------------------
            
            # diameters and thicknesses linearly interpolated from dA[0] to dB[-1] and t[0] to t[-1]
            #                 number   type  shape  dA      dB      xa      ya     za     xb     yb      zb        tA       tB   l_fill  rho_fill rho_shell  gamma
            memberStrings.append(" 1     1   circ  6.500   6.237    0.0    0.0    10.00   0.0    0.0    17.76    0.0270   0.0262   0.0     0.0     8500  ")
            memberStrings.append(" 2     1   circ  6.237   5.974    0.0    0.0    17.76   0.0    0.0    25.52    0.0262   0.0254   0.0     0.0     8500  ")
            memberStrings.append(" 3     1   circ  5.974   5.711    0.0    0.0    25.52   0.0    0.0    33.28    0.0254   0.0246   0.0     0.0     8500  ")
            memberStrings.append(" 4     1   circ  5.711   5.448    0.0    0.0    33.28   0.0    0.0    41.04    0.0246   0.0238   0.0     0.0     8500  ")
            memberStrings.append(" 5     1   circ  5.448   5.185    0.0    0.0    41.04   0.0    0.0    48.80    0.0238   0.0230   0.0     0.0     8500  ")
            memberStrings.append(" 6     1   circ  5.185   4.922    0.0    0.0    48.80   0.0    0.0    56.56    0.0230   0.0222   0.0     0.0     8500  ")
            memberStrings.append(" 7     1   circ  4.922   4.659    0.0    0.0    56.56   0.0    0.0    64.32    0.0222   0.0214   0.0     0.0     8500  ")
            memberStrings.append(" 8     1   circ  4.659   4.396    0.0    0.0    64.32   0.0    0.0    72.08    0.0214   0.0206   0.0     0.0     8500  ")
            memberStrings.append(" 9     1   circ  4.396   4.133    0.0    0.0    72.08   0.0    0.0    79.84    0.0206   0.0198   0.0     0.0     8500  ")
            memberStrings.append("10     1   circ  4.133   3.870    0.0    0.0    79.84   0.0    0.0    87.60    0.0198   0.0190   0.0     0.0     8500  ")
            
            #memberStrings.append(" 1     1   circ  6.500   3.870    0.0    0.0    10.00   0.0    0.0    87.60    0.0270   0.0190   0.0     0.0     8500  ")
            
            """
            # spar platform substructure description --------------
            # Modeling the end caps to fit inside the ends of the cylindrical members (hydrostatics won't work this way because of the end cap definitions)
            # Main Column
            memberStrings.append("11    2    circ    6.5     6.5      0.0      0.0    -20.0     0.0      0.0    10.00   0.03    0.03     0.0         0.0   7850  ")
            # Upper Columns
            memberStrings.append("12    3    circ   12.0    12.0     14.43    25.0    -14.0    14.43    25.0    12.00   0.06    0.06     7.83     1025.0   7850  ")
            memberStrings.append("13    3    circ   12.0    12.0    -28.87     0.0    -14.0   -28.87     0.0    12.00   0.06    0.06     7.83     1025.0   7850  ")
            memberStrings.append("14    3    circ   12.0    12.0     14.43   -25.0    -14.0    14.43   -25.0    12.00   0.06    0.06     7.83     1025.0   7850  ")
            # Base Columns
            memberStrings.append("15    4    circ   24.0    24.0     14.43    25.0    -20.0    14.43    25.0   -14.00   0.06    0.06     5.1078   1025.0   7850  ")
            memberStrings.append("16    4    circ   24.0    24.0    -28.87     0.0    -20.0   -28.87     0.0   -14.00   0.06    0.06     5.1078   1025.0   7850  ")
            memberStrings.append("17    4    circ   24.0    24.0     14.43   -25.0    -20.0    14.43   -25.0   -14.00   0.06    0.06     5.1078   1025.0   7850  ")
            # Delta Upper Pontoons
            memberStrings.append("18    5    circ    1.6     1.6      9.20    22.0     10.0   -23.67     3.0    10.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("19    5    circ    1.6     1.6    -23.67    -3.0     10.0     9.2    -22.0    10.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("20    5    circ    1.6     1.6     14.43   -19.0     10.0    14.43    19.0    10.00   0.0175  0.0175   0.0         0.0   7850  ")
            # Delta Lower Pontoons
            memberStrings.append("21    6    circ    1.6     1.6      4.00    19.0    -17.0   -18.47     6.0   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("22    6    circ    1.6     1.6    -18.47    -6.0    -17.0     4.0    -19.0   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("23    6    circ    1.6     1.6     14.43   -13.0    -17.0    14.43    13.0   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            # Y Upper Pontoons
            memberStrings.append("24    7    circ    1.6     1.6      1.625    2.815   10.0    11.43    19.81   10.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("25    7    circ    1.6     1.6     -3.25     0.0     10.0   -22.87     0.0    10.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("26    7    circ    1.6     1.6      1.625   -2.815   10.0    11.43   -19.81   10.00   0.0175  0.0175   0.0         0.0   7850  ")
            # Y Lower Pontoons
            memberStrings.append("27    8    circ    1.6     1.6      1.625    2.815  -17.0     8.4     14.6   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("28    8    circ    1.6     1.6     -3.25     0.0    -17.0   -16.87     0.0   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("29    8    circ    1.6     1.6      1.625   -2.815  -17.0     8.4    -14.6   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            # Cross Braces
            memberStrings.append("30    9    circ    1.6     1.6      1.625    2.815  -16.2    11.43    19.81    9.13   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("31    9    circ    1.6     1.6     -3.25     0.0    -16.2   -22.87     0.0     9.13   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("32    9    circ    1.6     1.6      1.625   -2.815  -16.2    11.43   -19.81    9.13   0.0175  0.0175   0.0         0.0   7850  ")
            
            # Upper Column Top Caps
            memberStrings.append("33    10   circ   11.88   11.88    14.43    25.0     11.94   14.43    25.0    12.00   5.94    5.94     0.0         0.0   7850  ")
            memberStrings.append("34    10   circ   11.88   11.88   -28.87     0.0     11.94  -28.87     0.0    12.00   5.94    5.94     0.0         0.0   7850  ")
            memberStrings.append("35    10   circ   11.88   11.88    14.43   -25.0     11.94   14.43   -25.0    12.00   5.94    5.94     0.0         0.0   7850  ")
            # Upper Column Bottom Caps
            
            memberStrings.append("36    11   circ   11.88   11.88    14.43    25.0    -14.0    14.43    25.0   -13.94   5.94    5.94     0.0         0.0   7850  ")
            memberStrings.append("37    11   circ   11.88   11.88   -28.87     0.0    -14.0   -28.87     0.0   -13.94   5.94    5.94     0.0         0.0   7850  ")
            memberStrings.append("38    11   circ   11.88   11.88    14.43   -25.0    -14.0    14.43   -25.0   -13.94   5.94    5.94     0.0         0.0   7850  ")
            '''
            memberStrings.append("36    11   circ   12.00   12.00    14.43    25.0    -14.06    14.43    25.0   -14.0   6.00    6.00     0.0         0.0   7850  ")
            memberStrings.append("37    11   circ   12.00   12.00   -28.87     0.0    -14.06   -28.87     0.0   -14.0   6.00    6.00     0.0         0.0   7850  ")
            memberStrings.append("38    11   circ   12.00   12.00    14.43   -25.0    -14.06    14.43   -25.0   -14.0   6.00    6.00     0.0         0.0   7850  ")
            '''
            # Base Column Top Caps
            
            memberStrings.append("39    12   circ   23.88   23.88    14.43    25.0    -14.06   14.43    25.0   -14.00  11.94   11.94     0.0         0.0   7850  ")
            memberStrings.append("40    12   circ   23.88   23.88   -28.87     0.0    -14.06  -28.87     0.0   -14.00  11.94   11.94     0.0         0.0   7850  ")
            memberStrings.append("41    12   circ   23.88   23.88    14.43   -25.0    -14.06   14.43   -25.0   -14.00  11.94   11.94     0.0         0.0   7850  ")
            '''
            memberStrings.append("39    12   circ   23.88   23.88    14.43    25.0    -14.06   14.43    25.0   -14.00   5.94    5.94     0.0         0.0   7850  ")
            memberStrings.append("40    12   circ   23.88   23.88   -28.87     0.0    -14.06  -28.87     0.0   -14.00   5.94    5.94     0.0         0.0   7850  ")
            memberStrings.append("41    12   circ   23.88   23.88    14.43   -25.0    -14.06   14.43   -25.0   -14.00   5.94    5.94     0.0         0.0   7850  ")
            '''
            
            # Base Column Bottom Caps
            memberStrings.append("42    13   circ   23.88   23.88    14.43    25.0    -20.0    14.43    25.0   -19.94  11.94   11.94     0.0         0.0   7850  ")
            memberStrings.append("43    13   circ   23.88   23.88   -28.87     0.0    -20.0   -28.87     0.0   -19.94  11.94   11.94     0.0         0.0   7850  ")
            memberStrings.append("44    13   circ   23.88   23.88    14.43   -25.0    -20.0    14.43   -25.0   -19.94  11.94   11.94     0.0         0.0   7850  ")
            
            # Main Column Bottom Caps
            memberStrings.append("45    14   circ    6.44    6.44     0.0      0.0    -20.0     0.0      0.0   -19.97   3.22    3.22     0.0         0.0   7850  ")
            """
            
            # Modeling the end caps to fit on the outside of the ends of the cylindrical members, while still maintaining the cylinder's original length
            # Main Column
            memberStrings.append("11    2    circ    6.5     6.5      0.0      0.0    -19.97     0.0      0.0   10.00   0.03    0.03     0.0         0.0   7850  ")
            # Upper Columns
            memberStrings.append("12    3    circ   12.0    12.0     14.43    25.0    -13.94    14.43    25.0    11.94   0.06    0.06    7.77     1025.0   7850  ")
            memberStrings.append("13    3    circ   12.0    12.0    -28.86     0.0    -13.94   -28.86     0.0    11.94   0.06    0.06    7.77     1025.0   7850  ")
            memberStrings.append("14    3    circ   12.0    12.0     14.43   -25.0    -13.94    14.43   -25.0    11.94   0.06    0.06    7.77     1025.0   7850  ")
            # Base Columns
            memberStrings.append("15    4    circ   24.0    24.0     14.43    25.0    -19.94    14.43    25.0   -14.06   0.06    0.06    5.1018   1025.0   7850  ")
            memberStrings.append("16    4    circ   24.0    24.0    -28.86     0.0    -19.94   -28.86     0.0   -14.06   0.06    0.06    5.1018   1025.0   7850  ")
            memberStrings.append("17    4    circ   24.0    24.0     14.43   -25.0    -19.94    14.43   -25.0   -14.06   0.06    0.06    5.1018   1025.0   7850  ")
            # Delta Upper Pontoons
            memberStrings.append("18    5    circ    1.6     1.6      9.20    22.0     10.0   -23.67     3.0    10.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("19    5    circ    1.6     1.6    -23.67    -3.0     10.0     9.2    -22.0    10.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("20    5    circ    1.6     1.6     14.43   -19.0     10.0    14.43    19.0    10.00   0.0175  0.0175   0.0         0.0   7850  ")
            # Delta Lower Pontoons
            memberStrings.append("21    6    circ    1.6     1.6      4.00    19.0    -17.0   -18.47     6.0   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("22    6    circ    1.6     1.6    -18.47    -6.0    -17.0     4.0    -19.0   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("23    6    circ    1.6     1.6     14.43   -13.0    -17.0    14.43    13.0   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            # Y Upper Pontoons
            memberStrings.append("24    7    circ    1.6     1.6      1.625    2.815   10.0    11.43    19.81   10.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("25    7    circ    1.6     1.6     -3.25     0.0     10.0   -22.86     0.0    10.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("26    7    circ    1.6     1.6      1.625   -2.815   10.0    11.43   -19.81   10.00   0.0175  0.0175   0.0         0.0   7850  ")
            # Y Lower Pontoons
            memberStrings.append("27    8    circ    1.6     1.6      1.625    2.815  -17.0     8.4     14.6   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("28    8    circ    1.6     1.6     -3.25     0.0    -17.0   -16.87     0.0   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("29    8    circ    1.6     1.6      1.625   -2.815  -17.0     8.4    -14.6   -17.00   0.0175  0.0175   0.0         0.0   7850  ")
            # Cross Braces
            memberStrings.append("30    9    circ    1.6     1.6      1.625    2.815  -16.2    11.43    19.81    9.13   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("31    9    circ    1.6     1.6     -3.25     0.0    -16.2   -22.86     0.0     9.13   0.0175  0.0175   0.0         0.0   7850  ")
            memberStrings.append("32    9    circ    1.6     1.6      1.625   -2.815  -16.2    11.43   -19.81    9.13   0.0175  0.0175   0.0         0.0   7850  ")
            
            # Upper Column Top Caps
            memberStrings.append("33    10   circ   12.00   12.00    14.43    25.0     11.94   14.43    25.0    12.00   6.00    6.00     0.0         0.0   7850  ")
            memberStrings.append("34    10   circ   12.00   12.00   -28.86     0.0     11.94  -28.86     0.0    12.00   6.00    6.00     0.0         0.0   7850  ")
            memberStrings.append("35    10   circ   12.00   12.00    14.43   -25.0     11.94   14.43   -25.0    12.00   6.00    6.00     0.0         0.0   7850  ")
            # Upper Column Bottom Caps
            memberStrings.append("36    11   circ   12.00   12.00    14.43    25.0    -14.0    14.43    25.0   -13.94   6.00    6.00     0.0         0.0   7850  ")
            memberStrings.append("37    11   circ   12.00   12.00   -28.86     0.0    -14.0   -28.86     0.0   -13.94   6.00    6.00     0.0         0.0   7850  ")
            memberStrings.append("38    11   circ   12.00   12.00    14.43   -25.0    -14.0    14.43   -25.0   -13.94   6.00    6.00     0.0         0.0   7850  ")
            
            # Base Column Top Caps
            memberStrings.append("39    12   circ   24.00   24.00    14.43    25.0    -14.06   14.43    25.0   -14.00  12.00   12.00     0.0         0.0   7850  ")
            memberStrings.append("40    12   circ   24.00   24.00   -28.86     0.0    -14.06  -28.86     0.0   -14.00  12.00   12.00     0.0         0.0   7850  ")
            memberStrings.append("41    12   circ   24.00   24.00    14.43   -25.0    -14.06   14.43   -25.0   -14.00  12.00   12.00     0.0         0.0   7850  ")
            
            # Base Column Bottom Caps
            memberStrings.append("42    13   circ   24.00   24.00    14.43    25.0    -20.0    14.43    25.0   -19.94  12.00   12.00     0.0         0.0   7850  ")
            memberStrings.append("43    13   circ   24.00   24.00   -28.86     0.0    -20.0   -28.86     0.0   -19.94  12.00   12.00     0.0         0.0   7850  ")
            memberStrings.append("44    13   circ   24.00   24.00    14.43   -25.0    -20.0    14.43   -25.0   -19.94  12.00   12.00     0.0         0.0   7850  ")
            
            # Main Column Bottom Caps
            memberStrings.append("45    14   circ    6.50    6.50     0.0      0.0    -20.0     0.0      0.0   -19.97   3.25    3.25     0.0         0.0   7850  ")
            
            # Extra Fake Test Member to move the XCG closer to zero. Including this member erases the pitch moment from the mooring lines
            #memberStrings.append("46    15   circ   2.0    2.0      0.0      0.0    -10.0    1.311765      0.0   -10.0    0.01    0.05     0.0         0.0   7850")
            
            
            # ::::::::::::::::::::::::::: turbine RNA description :::::::::::::::::::::::::::
            #                 Rotor          Nacelle
            mRNA    = 110000               + 240000      # RNA mass [kg]
            IxRNA   = 11776047*(1 + 1 + 1) + 115926      # RNA moment of inertia about local x axis (assumed to be identical to rotor axis for now, as approx) [kg-m^2]
            IrRNA   = 11776047*(1 +.5 +.5) + 2607890     # RNA moment of inertia about local y or z axes [kg-m^2]
            xCG_RNA = 0                                  # x location of RNA center of mass [m] (Actual is ~= -0.27 m)
            hHub    = 90.0                               # hub height above water line [m]
            
            
            # :::::::::::::::::::::::::::::: moorings inputs :::::::::::::::::::::::::::::::::::
            
            depth = 200.
            type_string = 'main'
            anchorR = 837.6
            fairR = 40.868
            fair_depth = 14.
            LineLength = 835.5
            LineD = 0.0766
            dryMass_L = 113.35
            EA = 753.6e6
            angle =         np.array([np.pi, np.pi/3, -np.pi/3])    # [rad]
            rho =           1025.0                                  # [kg/m^3]
            
            # make a single-type 3-line mooring system by calling a function in MoorDesign
            MooringSystem = md.make3LineSystem(depth, type_string, LineD, dryMass_L, EA, angle, anchorR, fair_depth, fairR, LineLength)
            # initialize the system
            MooringSystem.initialize()
            
            
        elif model=='Volturnus':
            # --- Volturnus ---
            # ::::::::::::::::::::::::::: member inputs ::::::::::::::::::::::::::::::::
            
            # turbine Tower description ------------------
            
            #                 number   type  shape   dA      dB      xa      ya     za      xb     yb      zb           tA         tB   l_fill  rho_fill rho_shell
            memberStrings.append(" 1     1   circ  10.000   9.964    0.0    0.0    15.00    0.0    0.0    28.00     0.082954   0.082954   0.0     0.0     7850  ")
            memberStrings.append(" 2     1   circ   9.964   9.964    0.0    0.0    28.00    0.0    0.0    28.001    0.082954   0.083073   0.0     0.0     7850  ")
            memberStrings.append(" 3     1   circ   9.964   9.967    0.0    0.0    28.001   0.0    0.0    41.00     0.083073   0.083073   0.0     0.0     7850  ")
            memberStrings.append(" 4     1   circ   9.967   9.967    0.0    0.0    41.00    0.0    0.0    41.001    0.083073   0.082799   0.0     0.0     7850  ")
            memberStrings.append(" 5     1   circ   9.967   9.927    0.0    0.0    41.001   0.0    0.0    54.00     0.082799   0.082799   0.0     0.0     7850  ")
            memberStrings.append(" 6     1   circ   9.927   9.927    0.0    0.0    54.00    0.0    0.0    54.001    0.082799   0.029900   0.0     0.0     7850  ")
            memberStrings.append(" 7     1   circ   9.927   9.528    0.0    0.0    54.001   0.0    0.0    67.00     0.029900   0.029900   0.0     0.0     7850  ")
            memberStrings.append(" 8     1   circ   9.528   9.528    0.0    0.0    67.00    0.0    0.0    67.001    0.029900   0.027842   0.0     0.0     7850  ")
            memberStrings.append(" 9     1   circ   9.528   9.149    0.0    0.0    67.001   0.0    0.0    80.00     0.027842   0.027842   0.0     0.0     7850  ")
            memberStrings.append("10     1   circ   9.149   9.149    0.0    0.0    80.00    0.0    0.0    80.001    0.027842   0.025567   0.0     0.0     7850  ")
            memberStrings.append("11     1   circ   9.149   8.945    0.0    0.0    80.001   0.0    0.0    93.00     0.025567   0.025567   0.0     0.0     7850  ")
            memberStrings.append("12     1   circ   8.945   8.945    0.0    0.0    93.00    0.0    0.0    93.001    0.025567   0.022854   0.0     0.0     7850  ")
            memberStrings.append("13     1   circ   8.945   8.735    0.0    0.0    93.001   0.0    0.0   106.00     0.022854   0.022854   0.0     0.0     7850  ")
            memberStrings.append("14     1   circ   8.735   8.735    0.0    0.0   106.00    0.0    0.0   106.001    0.022854   0.020250   0.0     0.0     7850  ")
            memberStrings.append("15     1   circ   8.735   8.405    0.0    0.0   106.001   0.0    0.0   119.00     0.020250   0.020250   0.0     0.0     7850  ")
            memberStrings.append("16     1   circ   8.405   8.405    0.0    0.0   119.00    0.0    0.0   119.001    0.020250   0.018339   0.0     0.0     7850  ")
            memberStrings.append("17     1   circ   8.405   7.321    0.0    0.0   119.001   0.0    0.0   132.00     0.018339   0.018339   0.0     0.0     7850  ")
            memberStrings.append("18     1   circ   7.321   7.321    0.0    0.0   132.00    0.0    0.0   132.001    0.018339   0.021211   0.0     0.0     7850  ")
            memberStrings.append("19     1   circ   7.321   6.500    0.0    0.0   132.001   0.0    0.0   144.582    0.021211   0.021211   0.0     0.0     7850  ")
            
            # Main Column
            memberStrings.append("20    2    circ   10.0    10.0      0.0      0.0    -20.0      0.0      0.0     15.0    0.05    0.05    0.0     0.0     7850  ")
            # Upper Columns
            memberStrings.append("21    3    circ   12.5    12.5     25.875   44.817  -20.0     25.875   44.817   15.0    0.05    0.05    1.40   5000.0   7850  ")
            memberStrings.append("22    3    circ   12.5    12.5    -51.75     0.0    -20.0    -51.75     0.0     15.0    0.05    0.05    1.40   5000.0   7850  ")
            memberStrings.append("23    3    circ   12.5    12.5     25.875  -44.817  -20.0     25.875  -44.817   15.0    0.05    0.05    1.40   5000.0   7850  ")
            # Lower Rectangular Pontoons
            memberStrings.append("24    4    rect   12.5/7  12.5/7    2.5      4.33   -16.5     23.375   40.487  -16.5    0.05    0.05   43.0    1025.0   7850  0.0 ")
            memberStrings.append("25    4    rect   12.5/7  12.5/7   -5.0      0.0    -16.5    -45.5      0.0    -16.5    0.05    0.05   43.0    1025.0   7850  0.0 ")
            memberStrings.append("26    4    rect   12.5/7  12.5/7    2.5     -4.33   -16.5     23.375  -40.487  -16.5    0.05    0.05   43.0    1025.0   7850  0.0 ")
            # Upper Supports
            memberStrings.append("27    5    circ    0.91    0.91     2.5      4.33    14.545   23.375    40.487  14.545  0.01    0.01    0.0     0.0     7850  ")
            memberStrings.append("28    5    circ    0.91    0.91    -5.0      0.0     14.545  -45.5      0.0     14.545  0.01    0.01    0.0     0.0     7850  ")
            memberStrings.append("29    5    circ    0.91    0.91     2.5     -4.33    14.545   23.375   -40.487  14.545  0.01    0.01    0.0     0.0     7850  ")
            
            
            # ::::::::::::::::::::::::::: turbine RNA description :::::::::::::::::::::::::::
            mRNA    = 991000                             # RNA mass [kg]
            IxRNA = 0
            IrRNA = 0
            xCG_RNA = 0                                  # x location of RNA center of mass [m] (Actual is ~= -0.27 m)
            hHub    = 150.0                              # hub height above water line [m]
            
            
            # :::::::::::::::::::::::::::::: moorings inputs :::::::::::::::::::::::::::::::::::
            
            depth = 200.
            type_string = 'chain'
            anchorR = 837.6
            fairR = 58.
            fair_depth = 14.
            LineLength = 850.
            LineD = 0.185
            dryMass_L = 685
            EA = 3270e6
            angle =         np.array([np.pi, np.pi/3, -np.pi/3])    # [rad]
            rho =           1025.0                                  # [kg/m^3]
            
            # make a single-type 3-line mooring system by calling a function in MoorDesign
            MooringSystem = md.make3LineSystem(depth, type_string, LineD, dryMass_L, EA, angle, anchorR, fair_depth, fairR, LineLength)
            # initialize the system
            MooringSystem.initialize()
            
            
        elif model=='DTU-10MW':
            # --- DTU 10MW ---
            # ::::::::::::::::::::::::::: member inputs ::::::::::::::::::::::::::::::::
            
            # turbine Tower description ------------------
    
            # new version (done for OpenFAST reasons in the Shared Moorings Project)
            zStart =    11          # [m] from waterline
            zEnd =      114.7       # [m] from waterline
            # old version
            #zStart =    13          # [m] from waterline
            #zEnd =      116.63      # [m] from waterline
            
            nTowMem = 10
            z = np.round(np.linspace(zStart, zEnd, nTowMem+1), 2)
            
            #                 number   type  shape  dA      dB      xa      ya      za        xb     yb      zb      t     l_fill  rho_ballast
            memberStrings.append(" 1     1    circ  8.00    7.75    0.0    0.0  "+str(z[0])+" 0.0    0.0  "+str(z[1])+"  0.038   0.0    1025.0  ")
            memberStrings.append(" 2     1    circ  7.75    7.50    0.0    0.0  "+str(z[1])+" 0.0    0.0  "+str(z[2])+"  0.036   0.0    1025.0  ")
            memberStrings.append(" 3     1    circ  7.50    7.25    0.0    0.0  "+str(z[2])+" 0.0    0.0  "+str(z[3])+"  0.034   0.0    1025.0  ")
            memberStrings.append(" 4     1    circ  7.25    7.00    0.0    0.0  "+str(z[3])+" 0.0    0.0  "+str(z[4])+"  0.032   0.0    1025.0  ")
            memberStrings.append(" 5     1    circ  7.00    6.75    0.0    0.0  "+str(z[4])+" 0.0    0.0  "+str(z[5])+"  0.030   0.0    1025.0  ")    
            memberStrings.append(" 6     1    circ  6.75    6.50    0.0    0.0  "+str(z[5])+" 0.0    0.0  "+str(z[6])+"  0.028   0.0    1025.0  ")
            memberStrings.append(" 7     1    circ  6.50    6.25    0.0    0.0  "+str(z[6])+" 0.0    0.0  "+str(z[7])+"  0.026   0.0    1025.0  ")
            memberStrings.append(" 8     1    circ  6.25    6.00    0.0    0.0  "+str(z[7])+" 0.0    0.0  "+str(z[8])+"  0.024   0.0    1025.0  ")
            memberStrings.append(" 9     1    circ  6.00    5.75    0.0    0.0  "+str(z[8])+" 0.0    0.0  "+str(z[9])+"  0.022   0.0    1025.0  ")
            memberStrings.append("10     1    circ  5.75    5.50    0.0    0.0  "+str(z[9])+" 0.0    0.0  "+str(z[10])+" 0.020   0.0    1025.0  ")
            
    
            # spar platform substructure description --------------
            '''
            # Ballast members from Senu's sizing for Shared Moorings baseline design
            memberStrings.append("11     2  circ  14.75   14.75    0.0    0.0    -90.000   0.0    0.0    -85.200   0.046    4.800    3743.42  ")
            memberStrings.append("12     2  circ  14.75   14.75    0.0    0.0    -85.200   0.0    0.0    -75.708   0.046    9.492    3792.35  ")
            memberStrings.append("13     2  circ  14.75   14.75    0.0    0.0    -75.708   0.0    0.0    -72.734   0.046    2.974    1883.78  ")
            '''
            # Ballast members from Stein getting weight = displ (4.8 m of steel ballast, 9.5 m of concrete ballast, 3 m of water ballast)
            memberStrings.append("11     2  circ  14.75   14.75    0.0    0.0    -90.000   0.0    0.0    -85.200   0.046    4.800    7850.0   ")
            memberStrings.append("12     2  circ  14.75   14.75    0.0    0.0    -85.200   0.0    0.0    -75.708   0.046    9.492    2650.0   ")
            memberStrings.append("13     2  circ  14.75   14.75    0.0    0.0    -75.708   0.0    0.0    -72.734   0.046    2.974    1025.0   ")
    
            memberStrings.append("14     2  circ  14.75   14.75    0.0    0.0    -72.734   0.0    0.0    -20.000   0.046    0.000    1025.0   ")
            memberStrings.append("15     2  circ  14.75    8.00    0.0    0.0    -20.000   0.0    0.0     -5.000   0.063    0.000    1025.0   ")
            memberStrings.append("16     2  circ   8.00    8.00    0.0    0.0     -5.000   0.0    0.0      7.000   0.068    0.000    1025.0   ")
            memberStrings.append("17     2  circ   8.00    7.00    0.0    0.0      7.000   0.0    0.0     11.000   0.055    0.000    1025.0   ")
            
            
            # ::::::::::::::::::::::::::: turbine RNA description :::::::::::::::::::::::::::          
            mRotor      = 227962        #[kg]
            mNacelle    = 446036        #[kg]
            IxHub       = 325671        #[kg-m^2]
            IzNacelle   = 7326346       #[kg-m^2]
            IxBlades    = 45671252      #[kg-m^2] MOI value from FAST file, don't know where MOI is about. Assuming about the hub
            xCG_Hub     = -7.07         #[m] from yaw axis
            xCG_Nacelle = 2.687         #[m] from yaw axis
    
            mRNA = mRotor + mNacelle    #[kg]
            IxRNA = IxBlades*(1 + 1 + 1) + IxHub # RNA moment of inertia about local x axis (assumed to be identical to rotor axis for now, as approx) [kg-m^2]
            IrRNA = IxBlades*(1 + .5 + .5) + IzNacelle # RNA moment of inertia about local y or z axes [kg-m^2]
    
            xCG_RNA = ((mRotor*xCG_Hub)+(mNacelle*xCG_Nacelle))/(mRotor+mNacelle)          # x location of RNA center of mass [m]
    
            #hHub    = 119.0                          # hub height above water line [m]
            hHub    = 118.0                 # changed to fit FAST data back in late July 2020
            
            
            # :::::::::::::::::::::::::::::: moorings inputs :::::::::::::::::::::::::::::::::::
            
            bridle = 1      # flag to determine whether to use the bridle mooring configuration or not
            
            depth =         600.                                    # [m]
            rho =           1025.                                   # [kg/m^3]
            angle =         np.array([0, 2*np.pi/3, -2*np.pi/3])    # [rad]
            anchorR =       656.139                                 # [m]
            fair_depth =    21.                                     # [m]
            fairR =         7.875                                   # [m]
            LineLength =    868.5                                   # [m]
            
            if bridle==0:
                
                type_string =   'main'                              #
                LineD =         0.15                                # [m]
                wetMass_L =     4.401                               # [kg/m]
                dryMass_L = wetMass_L + (np.pi/4)*LineD**2*rho
                EA =            384243000                           # [N]     or try 100e6
                
            elif bridle==1:     # if a bridle mooring configuration is desired
    
                type_string =   ['chain','synth']                   #
                LineD =         [0.17971, 0.15]                     # [m]
                dryMass_L =     [200, 40.18]                        # [kg/m]
                EA =            [0, 121415000]                      # [N]     or try 100e6  
                EA[0] =         0.854e11*(LineD[0]**2)              # [N]
        
                chainLength =   80                                                                      # [m]
                bridleLength =  30                                                                      # [m]
                synthLength =   LineLength-chainLength-bridleLength                                     # [m]
        
                synthR = anchorR-0.9*chainLength                                                        # [m]
                synthZ = 0.95*depth                                                                     # [m]
                bridleR = ((synthR-fairR)/(synthLength+bridleLength))*bridleLength + fairR              # [m] by interpolation
                bridleZ = ((synthZ-fair_depth)/(synthLength+bridleLength))*bridleLength + fair_depth    # [m] by interpolation
    
            
            # make a 3-main line bridle (crow's foot) mooring system by calling a function in MoorDesign
            MooringSystem = md.makeBridleSystem(depth, type_string, LineD, dryMass_L, EA, angle, anchorR, fair_depth, fairR, LineLength, chainLength, bridleLength, synthLength, synthR, synthZ, bridleR, bridleZ)
            # initialize the system
            MooringSystem.initialize()
    
    
    
    else: #  >>>>>>>>>>>> Otherwise, this is the more realistic case where we have to process wt_opt to produce memberStrings and MooringSystem <<<<<<<
        
        # Members
        floating_init_options = modeling_options['floating']  # I need to include these because this is where we get name_member
        n_members = floating_init_options['members']['n_members'] 
        
        n_joints = len(wt_opt['floating.floating_joints.location'])
        rA = np.zeros([n_joints, 2])
        rB = np.zeros([n_joints, 2])
        for i in range(n_joints):
            joint_locs[i,:] = wt_opt['floating.floating_joints.location'][i,:]
        
        for i in range(n_members):
            name_member = floating_init_options['members']['name'][i]
            type = 2 # arbitrary value to designate that the member is part of the floating substructure
            
            dA = wt_opt['floating.floating_member_' + name_member + '.outer_diameter'][0]
            dB = wt_opt['floating.floating_member_' + name_member + '.outer_diameter'][1]
            # <<<<<<<< the IEA ontology paper says that the outer_diameter parameter describes two diameters at joints 1 and 2
            
            t = sum(wt_opt['floating.floating_member_' + name_member + '.layer_thickness'])
            # took the sum of this because we just want to know the total thickness to get to dB
            # <<<<<<<<< not sure if I summed it right because the thickness of each layer is [j,:] in gc_WT_InitModel
            
            if n_joints != n_members + 1:
                raise ValueError('There should be n_members+1 number of joints to use the right rA and rB values')
            rA = joint_locs[i,:]
            rB = joint_locs[i+1,:]
            
            # <<<<<<<<<<< Ballast section: PROBABLY WON'T WORK. JUST USING WHAT I WAS GIVEN
            v_fill = wt_opt['floating.floating_member_' + name_member + '.ballast_volume'] 
            rho_fill = wt_opt['floating.floating_member_' + name_member + '.ballast_material.rho']
    
            #dB_fill = (dBi-dAi)*(self.l_fill/self.l) + dAi       # interpolated diameter of member where the ballast is filled to
            #v_fill = (np.pi/4)*(1/3)*(dAi**2+dB_fill**2+dAi*dB_fill)*self.l_fill    #[m^3]
            # There's a way to solve for l_fill using the above equations given v_fill
            
            # Going to simplify and just take it as the proportion of length to volume
            dAi = dA - 2*t # assming the thickness is constant along the member with respect to the length
            dBi = dB - 2*t
            l = np.linalg.norm(rB-rA)
            v_mem = (np.pi/4)*(1/3)*(dAi**2+dBi**2+dAi*dBi)*l
            
            l_fill = l * v_fill/v_mem
            
            # plug variables into a Member in FrequencyDomain and append to the memberString list
                        # change to string in FD v
            memberStrings.append(fd.Member( str(name_member)+" "+str(type)+" "+str(dA)+" "+str(dB)+" "+str(rA[0])+" "+str(rA[1])+" "+str(rA[2])+\
                                     " "+str(rB[0])+" "+str(rB[1])+" "+str(rB[2])+" "+str(t)+" "+str(l_fill)+" "+str(rho_fill), nw))
        
        
        # Mooring System
        # Import modules - just putting them here for organization. Can move to the top whenever
        import sys
        sys.path.insert(1, '../MoorPy')
        import MoorPy as mp
        # reload the libraries each time in case we make any changes
        import importlib
        mp   = importlib.reload(mp)
        # <<<<<<<<<<<< Do I need to include these in this gluecode script even though they're included in FD?
        
        
        # Create a MoorPy system
        ms = mp.System()
        ms.depth = wt_opt['env.water_depth']
        
        # Add the line types that are provided in the wt_opt OpenMDAO object
        n_line_types = len(wt_opt['mooring.line_diameter'])
        for i in range(n_line_types):
            name = wt_opt['mooring.line_names'][i]
            d = wt_opt['mooring.line_diameter'][i]
            massden = wt_opt['mooring.line_mass_density'][i]
            EA = wt_opt['mooring.line_stiffness'][i]
            MBL = wt_opt['mooring.line_breaking_load'][i]
            cost = wt_opt['mooring.line_cost'][i]
            
            ms.LineTypes[name] = mp.LineType( name, d, massden, EA, MBL=MBL, cost=cost, notes="made in FrequencyDomain.py" )
            
        # Add the wind turbine platform reference point   <<<<<<<<<<<<<< Get values
        ms.addBody(0, PRP, m=mTOT, v=VTOT, rCG=rCG_TOT, AWP=AWP_TOT, rM=np.array([0,0,zMeta]), f6Ext=np.array([Fthrust,0,0, 0,Mthrust,0]))
        
        # Add points to the sytem
        for i in range(n_nodes):
            ID = wt_opt['mooring.node_id'][i]            # <<<<<<<<< not 100% on the syntax of these calls
            
            if wt_opt['mooring.node_type'][i] == 'fixed':
                type = 1
            elif wt_opt['mooring.node_type'][i] == 'vessel':
                type = -1
            elif wt_opt['mooring.node_type'][i] == 'connection':
                type = 0
            
            r = np.array( wt_opt['mooring.nodes_location'][i,:], dtype=float)
            # TODO - can add in other variables for the point like anchor ID, fairlead_type, node_mass, node_volume, drag area, added mass
            ms.PointList.append( mp.Point( ID, type, r ) )

            # attach body points to the body
            # the nodes_location array is relative to inertial frame if Fixed or Connect, but relative to platform frame if Vessel
            if type==-1:
                ms.BodyList[0].addPoint(ID, r)
            
        
        # Add and attach lines to the nodes of the system
        n_lines = len(wt_opt['mooring.unstretched_length'])
        for i in range(n_lines):
            ID = wt_opt['mooring.line_id'][i]
            LineLength = wt_opt['mooring.unstretched_length'][i]
            linetype = wt_opt['mooring.line_type'][i]
            
            ms.LineList.append( mp.Line( ID, LineLength, LineTypes[linetype] ) )
            
            node1 = wt_opt['mooring.node1_id']
            node2 = wt_opt['mooring.node2_id']
            # Run an if statement to make sure that node1 is the deeper point
            if ms.PointList[node1].r[2] < ms.PointList[node2].r[2]:
                pass
            elif ms.PointList[node1].r[2] > ms.PointList[node2].r[2]:
                node1 = node2
                node2 = node1
            else:
                pass # if the z value of both points is the same, then it doesn't matter
            
            ms.PointList[node1].addLine(ID, 0)
            ms.PointList[node2].addLine(ID, 1)
        
        # TODO - anchor types
        
        # Turn on the system
        ms.initialize()
        MooringSystem = ms






    # --- RNA ---
    # store the RNA data in the RNAprops dictionary
    RNAprops['mRNA'] = mRNA
    RNAprops['IxRNA'] = IxRNA
    RNAprops['IrRNA'] = IrRNA
    RNAprops['xCG_RNA'] = xCG_RNA
    RNAprops['hHub'] = hHub
    RNAprops['Fthrust'] = Fthrust
    if model=='OC3':
        RNAprops['yaw stiffness'] = yawstiff
    


    # --- BEM ---
    # (preprocessing step:) Generate and load BEM hydro data
    capyData = []
    
    capyTestFile = f'./test_data/mesh_converge_0.750_1.250.nc'

    w = np.arange(0.05, 2.8, 0.05)  # frequency range (to be set by modeling options yaml)
    
    '''
    # load or generate Capytaine data
    if capyDataExists:
        wDes, addedMass, damping, fEx = capy.read_capy_nc(capyTestFile, wDes=w)
    else:
        wCapy, addedMass, damping, fEx = capy.call_capy(meshFName, w)
        
    # package results to send to model
    capyData = (wCapy, addedMass, damping, fEx)
    '''

    
    # --- Create Model ---
    # now that memberStrings and MooringSystem are made on way or another, call the model 

    model = fd.Model(memberList=memberStrings, ms=MooringSystem, w=w, depth=depth, BEM=capyData, RNAprops=RNAprops)  # set up model

    model.setEnv(Hs=8, Tp=12, V=10, Fthrust=RNAprops['Fthrust'])  # set basic wave and wind info

    model.calcSystemProps()          # get all the setup calculations done within the model

    model.solveEigen()

    model.calcMooringAndOffsets()    # calculate the offsets for the given loading
    
    model.solveDynamics()            # put everything together and iteratively solve the dynamic response
    
    model.plot()
    
    plt.show()
    
    return model
    

if __name__ == "__main__":
    
    #runFDmodel()
    
    model = runFDmodel()
    #model = runFDmodel(model='OC4')
    #model = runFDmodel(model='test')
    #model = runFDmodel(model='Volturnus')
    
    
    fowt = model.fowtList[0]
    
    print('Tower Mass:          ',np.round(fowt.mtower,2),' kg')
    print('Tower CG:            ',np.round(fowt.rCG_tow[2],4),' m from SWL')
    print('Substructure Mass:   ',np.round(fowt.msubstruc,2),' kg')
    print('Substructure CG:     ',np.round(fowt.rCG_sub[2],4),' m from SWL')
    print('Steel Mass:          ',np.round(fowt.mshell,2),' kg')
    print('Ballast Mass:        ',np.round(fowt.mballast,2),' kg')
    print('Ballast Densities    ',fowt.pb,' kg/m^3')
    print('Total Mass:          ',np.round(fowt.M_struc[0,0],2),' kg')
    print('Total CG:            ',np.round(fowt.rCG_TOT[2],2),' m from SWL')
    print('Roll Inertia at PCM  ',np.round(fowt.I44,2),' kg-m^2')
    print('Pitch Inertia at PCM ',np.round(fowt.I55,2),' kg-m^2')
    print('Yaw Inertia at PCM   ',np.round(fowt.I66,2),' kg-m^2')
    print('Roll Inertia at PRP: ',np.round(fowt.I44B,2),' kg-m^2')
    print('Pitch Inertia at PRP:',np.round(fowt.I55B,2),' kg-m^2')
    print('Buoyancy (pgV):      ',np.round(fowt.V*fowt.env.g*fowt.env.rho,2),' N')
    print('Center of Buoyancy:  ',np.round(fowt.rCB[2],4),'m from SWL')
    print('C33:                 ',np.round(fowt.C_hydro[2,2],2),' N')
    print('C44:                 ',np.round(fowt.C_hydro[3,3],2),' Nm/rad')
    print('C55:                 ',np.round(fowt.C_hydro[4,4],2),' Nm/rad')
    print('F_lines: ',list(np.round(np.array(model.F_moor0),2)),' N')
    print('C_lines: ',model.C_moor0)
    
    print('A11/A22:             ',fowt.A_hydro_morison[0,0],' kg')
    print(fowt.A_hydro_morison[2,2])
    print(fowt.A_hydro_morison[3,3])
    print(fowt.A_hydro_morison[0,4])
    print(fowt.A_hydro_morison[1,3])
    
    
    mag = abs(fowt.F_hydro_iner/fowt.zeta)
    
    plt.plot(fowt.w, mag[0,:])
    plt.plot(fowt.w, mag[2,:])
    plt.plot(fowt.w, mag[4,:])
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Exciting Force [N')
    
    
    
    
    def pdiff(x,y):
        return (abs(x-y)/y)*100
    
    
    
    
    
