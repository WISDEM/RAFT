import pytest
import sys
import numpy as np

# test local code; consider src layout in future to test installed code
sys.path.append('..')
import FrequencyDomain as fd
import Capytaine_placeholder as capy

capyTestFile = f'./test_data/mesh_converge_0.750_1.250.nc'


def test_read_capy_nc_fExInterpImagVals():
    wDes = np.arange(0.1, 2.8, 0.01)
    wDes, addedMassInterp, dampingInterp, fExInterp = capy.read_capy_nc(capyTestFile, wDes=wDes)
    refFExInterpImag = np.loadtxt(f'./ref_data/capytaine_integration/wDes-fExcitationInterpImag-surge.txt')
    assert max(abs(refFExInterpImag[:,1] - fExInterp[0,:].imag)) < 1e-12

def test_call_capy_addedMassShape():
    wRange = np.arange(0.1, 2.9, 0.1)
    meshFName = f'./test_data/float.gdf'
    wCapy, addedMass, damping, fEx = capy.call_capy(meshFName, wRange)
    assert addedMass.shape == (6, 6, 28)




# >>>>> this is the test script to execture the frequency domain model <<<<<
def runFDmodel(openMDAO_in = None):
    '''
        openMDAO_in : this is where the openMDAO data strucutre wt_opt from gc_WT_InitModel.py would get passed in
    '''

    
    # >>>>>>>>>> the simpler case of nothing passed in, so just using manual model inputs specified below <<<<<<<<
    if openMDAO_in =- None:


        # ::::::::::::::::::::::::::: member inputs ::::::::::::::::::::::::::::::::

        memberStrings = []

        # -------------------- OC3 Hywind Spar ----------------------
        '''
        # ------------------ turbine Tower description ------------------
        # diameters and thicknesses linearly interpolated from dA[0] to dB[-1] and t[0] to t[-1]
        #                      number   type    dA      dB      xa      ya     za     xb     yb      zb      t     l_fill  rho_ballast

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
        memberStrings.append("11     2    9.400   9.400    0.0    0.0    -120.   0.0    0.0    -12.0   0.0270   52.    1850.0  ")
        #memberStrings.append("11     2    9.400   9.400    0.0    0.0    -120.   0.0    0.0    -12.0   0.066   41.4    2000.0  ")
        memberStrings.append("12     2    9.400   6.500    0.0    0.0    -12.0   0.0    0.0    -4.00   0.0270   0.0    1025.0  ")
        memberStrings.append("13     2    6.500   6.500    0.0    0.0    -4.00   0.0    0.0    10.00   0.0270   0.0    1025.0  ")

        #memberStrings.append("1      2    6.500   6.500    0.0    0.0    -100.00   0.0    0.0    0.00   0.0270   0.0    1025.0  ")

        # -------------------------- turbine RNA description ------------------------
        # below are rough properties for NREL 5 MW reference turbine
        mRNA    = 110000              + 240000  # RNA mass [kg]
        IxRNA   = 11776047*(1 + 1 + 1) + 115926   # RNA moment of inertia about local x axis (assumed to be identical to rotor axis for now, as approx) [kg-m^2]
        IrRNA   = 11776047*(1 +.5 +.5) + 2607890   # RNA moment of inertia about local y or z axes [kg-m^2]
        xCG_RNA = 0                             # x location of RNA center of mass [m] (Close enough to -0.27 m)
        hHub    = 90.0                          # hub height above water line [m]

        Fthrust = 800e3  # peak thrust force, [N]
        '''


        # ------------------- DTU 10 MW Spar --------------------------

        # ------------------ turbine Tower description ------------------
        #                      number   type    dA      dB      xa      ya     za     xb     yb      zb      t     l_fill  rho_ballast

        # new version from 11-114.7 (for OpenFAST reasons)
        memberStrings.append(" 1     1    8.00    7.75    0.0    0.0    11.00    0.0    0.0    21.37    0.038   0.0    1025.0  ")
        memberStrings.append(" 2     1    7.75    7.50    0.0    0.0    21.37    0.0    0.0    31.74    0.036   0.0    1025.0  ")
        memberStrings.append(" 3     1    7.50    7.25    0.0    0.0    31.74    0.0    0.0    42.11    0.034   0.0    1025.0  ")
        memberStrings.append(" 4     1    7.25    7.00    0.0    0.0    42.11    0.0    0.0    52.48    0.032   0.0    1025.0  ")
        memberStrings.append(" 5     1    7.00    6.75    0.0    0.0    52.48    0.0    0.0    62.85    0.030   0.0    1025.0  ")

        memberStrings.append(" 6     1    6.75    6.50    0.0    0.0    62.85    0.0    0.0    73.22    0.028   0.0    1025.0  ")
        memberStrings.append(" 7     1    6.50    6.25    0.0    0.0    73.22    0.0    0.0    83.59    0.026   0.0    1025.0  ")
        memberStrings.append(" 8     1    6.25    6.00    0.0    0.0    83.59    0.0    0.0    93.96    0.024   0.0    1025.0  ")
        memberStrings.append(" 9     1    6.00    5.75    0.0    0.0    93.96    0.0    0.0   104.33    0.022   0.0    1025.0  ")
        memberStrings.append("10     1    5.75    5.50    0.0    0.0   104.33    0.0    0.0   114.70    0.020   0.0    1025.0  ")

        # =============================================================================
        # # old version from 13-116.63
        # memberStrings.append(" 1     1    8.00    7.75    0.0    0.0    13.000    0.0    0.0    23.363   0.038   0.0    1025.0  ")
        # memberStrings.append(" 2     1    7.75    7.50    0.0    0.0    23.363    0.0    0.0    33.726   0.036   0.0    1025.0  ")
        # memberStrings.append(" 3     1    7.50    7.25    0.0    0.0    33.726    0.0    0.0    44.089   0.034   0.0    1025.0  ")
        # memberStrings.append(" 4     1    7.25    7.00    0.0    0.0    44.089    0.0    0.0    54.452   0.032   0.0    1025.0  ")
        # memberStrings.append(" 5     1    7.00    6.75    0.0    0.0    54.452    0.0    0.0    64.815   0.030   0.0    1025.0  ")
        # 
        # memberStrings.append(" 6     1    6.75    6.50    0.0    0.0    64.815    0.0    0.0    75.178   0.028   0.0    1025.0  ")
        # memberStrings.append(" 7     1    6.50    6.25    0.0    0.0    75.178    0.0    0.0    85.541   0.026   0.0    1025.0  ")
        # memberStrings.append(" 8     1    6.25    6.00    0.0    0.0    85.541    0.0    0.0    95.904   0.024   0.0    1025.0  ")
        # memberStrings.append(" 9     1    6.00    5.75    0.0    0.0    95.904    0.0    0.0   106.267   0.022   0.0    1025.0  ")
        # memberStrings.append("10     1    5.75    5.50    0.0    0.0   106.267    0.0    0.0   116.630   0.020   0.0    1025.0  ")
        # =============================================================================

        # ---------- spar platform substructure description --------------

        # =============================================================================
        # Ballast members from Senu's sizing
        # memberStrings.append("11     2    14.75   14.75    0.0    0.0    -90.   0.0    0.0    -85.2   0.046   4.8    3743.42  ")
        # memberStrings.append("12     2    14.75   14.75    0.0    0.0    -85.2   0.0    0.0    -75.708   0.046   9.492    3792.35  ")
        # memberStrings.append("13     2    14.75   14.75    0.0    0.0    -75.708   0.0    0.0    -72.734   0.046   2.974    1883.78  ")
        # =============================================================================

        # Ballast members from Stein getting weight = displ
        memberStrings.append("11     2    14.75   14.75    0.0    0.0    -90.000   0.0    0.0    -85.200   0.046    4.800    7850.  ")
        memberStrings.append("12     2    14.75   14.75    0.0    0.0    -85.200   0.0    0.0    -75.708   0.046    9.492    2650.  ")
        memberStrings.append("13     2    14.75   14.75    0.0    0.0    -75.708   0.0    0.0    -72.734   0.046    2.974    1025.  ")

        memberStrings.append("14     2    14.75   14.75    0.0    0.0    -72.734   0.0    0.0    -20.   0.046   0.0    1025.  ")
        memberStrings.append("15     2    14.75    8.00    0.0    0.0    -20.   0.0    0.0    -5.   0.063   0.0    1025.0  ")
        memberStrings.append("16     2     8.00    8.00    0.0    0.0    -5.   0.0    0.0    7.   0.068   0.0    1025.0  ")
        memberStrings.append("17     2     8.00    7.00    0.0    0.0    7.   0.0    0.0    11.   0.055   0.0    1025.0  ")



        # :::::::::::::::::::::::::::::: moorings inputs :::::::::::::::::::::::::::::::::::

        # --------------- set up quasi-static mooring system and solve for mean offsets -------------------

        import MoorDesign as md

        # =============================================================================
        # # Inputs for OC3 Hywind
        # depth = 320.
        # type_string = 'main'
        # anchorR = 853.87
        # fairR = 5.2
        # fair_depth = 70.
        # LineLength = 902.2
        # LineD = 0.09
        # dryMass_L = 77.7066 #[kg/m]
        # EA = 384243000 #[N]
        # angle = np.array([0, 2*np.pi/3, -2*np.pi/3])
        # =============================================================================

        # Inputs for DTU 10 MW setup
        depth = 600. #[m]
        type_string = 'main'
        LineD = 0.15 #[m]
        wetMass_L = 4.401 #[kg/m]
        dryMass_L = wetMass_L + (np.pi/4)*LineD**2*rho
        EA = 384243000 #100e6 #[N]
        angle = np.array([0, 2*np.pi/3, -2*np.pi/3])
        anchorR = 656.139 #[m]
        fair_depth = 21. #[m]
        fairR = 7.875 #[m]
        LineLength = 868.5 #[m]




        MooringSystem = md.make3LineSystem(depth, type_string, LineD, dryMass_L, EA, angle, anchorR, fair_depth, fairR, LineLength)

        MooringSystem.BodyList[0].m = mTOT
        MooringSystem.BodyList[0].v = VTOT
        MooringSystem.BodyList[0].rCG = rCG_TOT
        MooringSystem.BodyList[0].AWP = AWP_TOT
        MooringSystem.BodyList[0].rM = np.array([0,0,zMeta])
        MooringSystem.BodyList[0].f6Ext = np.array([Fthrust,0,0, 0,Mthrust,0])

        MooringSystem.initialize()

        # --------------- Bridle Confirguration --------------------------
        # Inputs for DTU 10 MW setup
        depth = 600. #[m]
        type_string = ['chain','synth']
        LineD = [0.17971, 0.15]  #[m]
        #wetMass_L = 4.401 #[kg/m]
        #dryMass_L = wetMass_L + (np.pi/4)*LineD**2*rho
        dryMass_L = [200, 40.18]
        EA = [0, 121415000] #100e6 #[N]
        EA[0] = 0.854e11*(LineD[0]**2)
        angle = np.array([0, 2*np.pi/3, -2*np.pi/3])
        anchorR = 656.139 #[m]
        fair_depth = 21. #[m]
        fairR = 7.875 #[m]

        LineLength = 868.5 #[m]

        chainLength = 80
        bridleLength = 30
        synthLength = LineLength-chainLength-bridleLength

        synthR = anchorR-0.9*chainLength
        synthZ = 0.95*depth
        bridleR = ((synthR-fairR)/(synthLength+bridleLength))*bridleLength + fairR
        bridleZ = ((synthZ-fair_depth)/(synthLength+bridleLength))*bridleLength + fair_depth


        Bridle = md.makeBridleSystem(depth, type_string, LineD, dryMass_L, EA, angle, anchorR, fair_depth, fairR, LineLength, chainLength, bridleLength, synthLength, synthR, synthZ, bridleR, bridleZ)

        Bridle.BodyList[0].m = mTOT
        Bridle.BodyList[0].v = VTOT
        Bridle.BodyList[0].rCG = rCG_TOT
        Bridle.BodyList[0].AWP = AWP_TOT
        Bridle.BodyList[0].rM = np.array([0,0,zMeta])
        Bridle.BodyList[0].f6Ext = np.array([Fthrust,0,0, 0,Mthrust,0])

        Bridle.initialize()


        # If using the bridle mooring system rather than the original, do a rename so we can refer to it as MooringSystem going forward (otherwise comment the line out)
        MooringSystem = Bridle


    else: #  >>>>>>>>>>>> Otherwise, this is the more realistic case where we have to process wt_opt to produce memberStrings and MooringSystem <<<<<<<

        from gc_WT_InitModel import wt_opt
        wt = wt_opt
        
        
        # Members
        floating_init_options = modeling_options['floating']  # I need to include these because this is where we get name_member
        n_members = floating_init_options['members']['n_members'] 
        
        n_joints = len(wt['floating.floating_joints.location'])
        rA = np.zeros([n_joints, 2])
        rB = np.zeros([n_joints, 2])
        for i in range(n_joints):
            joint_locs[i,:] = wt['floating.floating_joints.location'][i,:]
        
        for i in range(n_members):
            name_member = floating_init_options['members']['name'][i]
            type = 2 # arbitrary value to designate that the member is part of the floating substructure
            
            dA = wt['floating.floating_member_' + name_member + '.outer_diameter'][0]
            dB = wt['floating.floating_member_' + name_member + '.outer_diameter'][1]
            # <<<<<<<< the IEA ontology paper says that the outer_diameter parameter describes two diameters at joints 1 and 2
            
            t = sum(wt['floating.floating_member_' + name_member + '.layer_thickness'])
            # took the sum of this because we just want to know the total thickness to get to dB
            # <<<<<<<<< not sure if I summed it right because the thickness of each layer is [j,:] in gc_WT_InitModel
            
            if n_joints != n_members + 1:
                raise ValueError('There should be n_members+1 number of joints to use the right rA and rB values')
            rA = joint_locs[i,:]
            rB = joint_locs[i+1,:]
            
            # <<<<<<<<<<< Ballast section: PROBABLY WON'T WORK. JUST USING WHAT I WAS GIVEN
            v_fill = wt['floating.floating_member_' + name_member + '.ballast_volume'] 
            rho_fill = wt['floating.floating_member_' + name_member + '.ballast_material.rho']
    
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
        ms.depth = wt['env.water_depth']
        
        # Add the line types that are provided in the wt_opt OpenMDAO object
        n_line_types = len(wt['mooring.line_diameter'])
        for i in range(n_line_types):
            name = wt['mooring.line_names'][i]
            d = wt['mooring.line_diameter'][i]
            massden = wt['mooring.line_mass_density'][i]
            EA = wt['mooring.line_stiffness'][i]
            MBL = wt['mooring.line_breaking_load'][i]
            cost = wt['mooring.line_cost'][i]
            
            ms.LineTypes[name] = mp.LineType( name, d, massden, EA, MBL=MBL, cost=cost, notes="made in FrequencyDomain.py" )
            
        # Add the wind turbine platform reference point   <<<<<<<<<<<<<< Get values
        ms.addBody(0, PRP, m=mTOT, v=VTOT, rCG=rCG_TOT, AWP=AWP_TOT, rM=np.array([0,0,zMeta]), f6Ext=np.array([Fthrust,0,0, 0,Mthrust,0]))
        
        # Add points to the sytem
        for i in range(n_nodes):
            ID = wt['mooring.node_id'][i]            # <<<<<<<<< not 100% on the syntax of these calls
            
            if wt['mooring.node_type'][i] == 'fixed':
                type = 1
            elif wt['mooring.node_type'][i] == 'vessel':
                type = -1
            elif wt['mooring.node_type'][i] == 'connection':
                type = 0
            
            r = np.array( wt['mooring.nodes_location'][i,:], dtype=float)
            # TODO - can add in other variables for the point like anchor ID, fairlead_type, node_mass, node_volume, drag area, added mass
            ms.PointList.append( mp.Point( ID, type, r ) )

            # attach body points to the body
            # the nodes_location array is relative to inertial frame if Fixed or Connect, but relative to platform frame if Vessel
            if type==-1:
                ms.BodyList[0].addPoint(ID, r)
            
        
        # Add and attach lines to the nodes of the system
        n_lines = len(wt['mooring.unstretched_length'])
        for i in range(n_lines):
            ID = wt['mooring.line_id'][i]
            LineLength = wt['mooring.unstretched_length'][i]
            linetype = wt['mooring.line_type'][i]
            
            ms.LineList.append( mp.Line( ID, LineLength, LineTypes[linetype] ) )
            
            node1 = wt['mooring.node1_id']
            node2 = wt['mooring.node2_id']
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
        mRNA = wt['towerse.rna_mass']
        CG_RNA = wt['towerse.rna_cg']
        I_RNA = wr['towerse.rna_I']
        
        if "loading" in modeling_options:
            for k in range(modeling_options["tower"]["nLC"]):
                kstr = "" if modeling_options["tower"]["nLC"] == 0 else str(k + 1)
                Fthrust = wt["towerse.pre" + kstr + ".rna_F"]
                Mthrust = wt["towerse.pre" + kstr + ".rna_M"]
                windspeed = wt["towerse.wind" + kstr + ".Uref"]
        

        
        
        # Environmental Inputs
        g = 9.81            #[m/s^2]
        rho = wt['env.rho_water']
        depth = wt['env.water_depth']
        Hs = wt['env.hsig_wave']
        Tp = wt['env.Tsig_wave']
        '''



    # >>>>>>> now that memberStrings and MooringSystem are made on way or another, call the model <<<<<<<<<
    fowt = fd.FOWT(memberStrings, MooringSystem)

