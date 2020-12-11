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
def runFDmodel(wt_opt = None, model = 'OC3-Hywind'):
    '''
        wt_opt : this is where the openMDAO data structure from gc_WT_InitModel.py would get passed in
    '''

    
    # >>>>>>>>>> the simpler case of nothing passed in, so just using manual model inputs specified below <<<<<<<<
    if wt_opt == None:
        
        memberStrings = []    # initialize an empty list to hold all the member strings
        
        if model=='OC3-Hywind':
            # ::::::::::::::::::::::::::: member inputs ::::::::::::::::::::::::::::::::
            
            # ------------------ turbine Tower description ------------------
            # diameters and thicknesses linearly interpolated from dA[0] to dB[-1] and t[0] to t[-1]
            #                 number   type    dA      dB      xa      ya     za     xb     yb      zb      t     l_fill  rho_ballast
    
            memberStrings.append(" 1     1    6.500   6.237    0.0    0.0    10.00   0.0    0.0    17.76   0.0270   0.0    1025.0  ")
            memberStrings.append(" 2     1    6.237   5.974    0.0    0.0    17.76   0.0    0.0    25.52   0.0262   0.0    1025.0  ")
            memberStrings.append(" 3     1    5.974   5.711    0.0    0.0    25.52   0.0    0.0    33.28   0.0254   0.0    1025.0  ")
            memberStrings.append(" 4     1    5.711   5.448    0.0    0.0    33.28   0.0    0.0    41.04   0.0246   0.0    1025.0  ")
            memberStrings.append(" 5     1    5.448   5.185    0.0    0.0    41.04   0.0    0.0    48.80   0.0238   0.0    1025.0  ")
            memberStrings.append(" 6     1    5.185   4.922    0.0    0.0    48.80   0.0    0.0    56.56   0.0230   0.0    1025.0  ")
            memberStrings.append(" 7     1    4.922   4.659    0.0    0.0    56.56   0.0    0.0    64.32   0.0222   0.0    1025.0  ")
            memberStrings.append(" 8     1    4.659   4.396    0.0    0.0    64.32   0.0    0.0    72.08   0.0214   0.0    1025.0  ")
            memberStrings.append(" 9     1    4.396   4.133    0.0    0.0    72.08   0.0    0.0    79.84   0.0206   0.0    1025.0  ")
            memberStrings.append("10     1    4.133   3.870    0.0    0.0    79.84   0.0    0.0    87.60   0.0198   0.0    1025.0  ")
    
            # ---------- spar platform substructure description --------------
            memberStrings.append("11     2    9.400   9.400    0.0    0.0   -120.0   0.0    0.0   -12.00   0.0270   52.0    1850.0  ")
            memberStrings.append("11     2    9.400   9.400    0.0    0.0   -120.0   0.0    0.0   -12.00   0.0660   41.4    2000.0  ")
            memberStrings.append("12     2    9.400   6.500    0.0    0.0    -12.0   0.0    0.0    -4.00   0.0270    0.0    1025.0  ")
            memberStrings.append("13     2    6.500   6.500    0.0    0.0     -4.0   0.0    0.0    10.00   0.0270    0.0    1025.0  ")
    
    
            # -------------------------- turbine RNA description ------------------------
            #                 Rotor          Nacelle
            mRNA    = 110000               + 240000      # RNA mass [kg]
            IxRNA   = 11776047*(1 + 1 + 1) + 115926      # RNA moment of inertia about local x axis (assumed to be identical to rotor axis for now, as approx) [kg-m^2]
            IrRNA   = 11776047*(1 +.5 +.5) + 2607890     # RNA moment of inertia about local y or z axes [kg-m^2]
            xCG_RNA = 0                                  # x location of RNA center of mass [m] (Actual is ~= -0.27 m)
            hHub    = 90.0                               # hub height above water line [m]
    
            Fthrust = 800e3                              # peak thrust force, [N]
            
            
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
            
    
        elif model=='DTU-10MW':
            # ::::::::::::::::::::::::::: member inputs ::::::::::::::::::::::::::::::::
            
            # ------------------ turbine Tower description ------------------
    
            # new version (done for OpenFAST reasons in the Shared Moorings Project)
            zStart =    11          # [m] from waterline
            zEnd =      114.7       # [m] from waterline
            # old version
            #zStart =    13          # [m] from waterline
            #zEnd =      116.63      # [m] from waterline
            
            nTowMem = 10
            z = np.round(np.linspace(zStart, zEnd, nTowMem+1), 2)
            
            #                 number   type    dA      dB      xa      ya      za        xb     yb      zb      t     l_fill  rho_ballast
            memberStrings.append(" 1     1    8.00    7.75    0.0    0.0  "+str(z[0])+" 0.0    0.0  "+str(z[1])+"  0.038   0.0    1025.0  ")
            memberStrings.append(" 2     1    7.75    7.50    0.0    0.0  "+str(z[1])+" 0.0    0.0  "+str(z[2])+"  0.036   0.0    1025.0  ")
            memberStrings.append(" 3     1    7.50    7.25    0.0    0.0  "+str(z[2])+" 0.0    0.0  "+str(z[3])+"  0.034   0.0    1025.0  ")
            memberStrings.append(" 4     1    7.25    7.00    0.0    0.0  "+str(z[3])+" 0.0    0.0  "+str(z[4])+"  0.032   0.0    1025.0  ")
            memberStrings.append(" 5     1    7.00    6.75    0.0    0.0  "+str(z[4])+" 0.0    0.0  "+str(z[5])+"  0.030   0.0    1025.0  ")
    
            memberStrings.append(" 6     1    6.75    6.50    0.0    0.0  "+str(z[5])+" 0.0    0.0  "+str(z[6])+"  0.028   0.0    1025.0  ")
            memberStrings.append(" 7     1    6.50    6.25    0.0    0.0  "+str(z[6])+" 0.0    0.0  "+str(z[7])+"  0.026   0.0    1025.0  ")
            memberStrings.append(" 8     1    6.25    6.00    0.0    0.0  "+str(z[7])+" 0.0    0.0  "+str(z[8])+"  0.024   0.0    1025.0  ")
            memberStrings.append(" 9     1    6.00    5.75    0.0    0.0  "+str(z[8])+" 0.0    0.0  "+str(z[9])+"  0.022   0.0    1025.0  ")
            memberStrings.append("10     1    5.75    5.50    0.0    0.0  "+str(z[9])+" 0.0    0.0  "+str(z[10])+" 0.020   0.0    1025.0  ")
            
    
            # ---------- spar platform substructure description --------------
            '''
            # Ballast members from Senu's sizing for Shared Moorings baseline design
            memberStrings.append("11     2    14.75   14.75    0.0    0.0    -90.000   0.0    0.0    -85.200   0.046    4.800    3743.42  ")
            memberStrings.append("12     2    14.75   14.75    0.0    0.0    -85.200   0.0    0.0    -75.708   0.046    9.492    3792.35  ")
            memberStrings.append("13     2    14.75   14.75    0.0    0.0    -75.708   0.0    0.0    -72.734   0.046    2.974    1883.78  ")
            '''
            # Ballast members from Stein getting weight = displ (4.8 m of steel ballast, 9.5 m of concrete ballast, 3 m of water ballast)
            memberStrings.append("11     2    14.75   14.75    0.0    0.0    -90.000   0.0    0.0    -85.200   0.046    4.800    7850.0   ")
            memberStrings.append("12     2    14.75   14.75    0.0    0.0    -85.200   0.0    0.0    -75.708   0.046    9.492    2650.0   ")
            memberStrings.append("13     2    14.75   14.75    0.0    0.0    -75.708   0.0    0.0    -72.734   0.046    2.974    1025.0   ")
    
            memberStrings.append("14     2    14.75   14.75    0.0    0.0    -72.734   0.0    0.0    -20.000   0.046    0.000    1025.0   ")
            memberStrings.append("15     2    14.75    8.00    0.0    0.0    -20.000   0.0    0.0     -5.000   0.063    0.000    1025.0   ")
            memberStrings.append("16     2     8.00    8.00    0.0    0.0     -5.000   0.0    0.0      7.000   0.068    0.000    1025.0   ")
            memberStrings.append("17     2     8.00    7.00    0.0    0.0      7.000   0.0    0.0     11.000   0.055    0.000    1025.0   ")
    
    
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


        ''' Other stuff that might be useful later
        # RNA
        mRNA = wt_opt['towerse.rna_mass']
        CG_RNA = wt_opt['towerse.rna_cg']
        I_RNA = wr['towerse.rna_I']
        
        if "loading" in modeling_options:
            for k in range(modeling_options["tower"]["nLC"]):
                kstr = "" if modeling_options["tower"]["nLC"] == 0 else str(k + 1)
                Fthrust = wt_opt["towerse.pre" + kstr + ".rna_F"]
                Mthrust = wt_opt["towerse.pre" + kstr + ".rna_M"]
                windspeed = wt_opt["towerse.wind" + kstr + ".Uref"]
        
        
        # Environmental Inputs
        g = 9.81            #[m/s^2]
        rho = wt_opt['env.rho_water']
        depth = wt_opt['env.water_depth']
        Hs = wt_opt['env.hsig_wave']
        Tp = wt_opt['env.Tsig_wave']
        '''


    # (preprocessing step:) Generate and load BEM hydro data
    capyData = []
    
    capyTestFile = f'./test_data/mesh_converge_0.750_1.250.nc'

    w = np.arange(0.1, 2.8, 0.01)  # frequency range (to be set by modeling options yaml)
    
    # load or generate Capytaine data
    if capyDataExists:
        wDes, addedMass, damping, fEx = capy.read_capy_nc(capyTestFile, wDes=w)
    else:
        wCapy, addedMass, damping, fEx = capy.call_capy(meshFName, w)
        
    # package results to send to model
    capyData = (wCapy, addedMass, damping, fEx)



    # now that memberStrings and MooringSystem are made on way or another, call the model 

    model = fd.Model(memberList=memberStrings, ms=MooringSystem, depth=depth, BEM=capyData)  # set up model

    model.setEnv(Hs=8, Tp=12, V=10)  # set basic wave and wind info

    model.calcSystemProps()          # get all the setup calculations done within the model

    model.calcMooringAndOffsets()    # calculate the offsets for the given loading
    
    model.solveDynamics()            # put everything together and iteratively solve the dynamic response
    
    plt.show()


if __name__ == "__main__":
    
    runFDmodel()
