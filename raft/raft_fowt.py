# RAFT's floating wind turbine class

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator, griddata

import raft.member2pnl as pnl
from raft.helpers import *
from raft.raft_member import Member
from raft.raft_rotor import Rotor
import moorpy as mp

# Attempt to import pygmsh and meshmagick with warnings if not installed
try:
    import pygmsh
except ImportError:
    pygmsh = None

try:
    import meshmagick
except ImportError:
    meshmagick = None


# deleted call to ccblade in this file, since it is called in raft_rotor
# also ignoring changes to solveEquilibrium3 in raft_model and the re-addition of n=len(stations) in raft_member, based on raft_patch



class FOWT():
    '''This class comprises the frequency domain model of a single floating wind turbine'''

    def __init__(self, design, w, mpb, depth=600, x_ref=0, y_ref=0, heading_adjust=0):
        '''This initializes the FOWT object which contains everything for a single turbine's frequency-domain dynamics.
        The initializiation sets up the design description.

        Parameters
        ----------
        design : dict
            Dictionary of the design - must include sections for site, turbine, platform, and mooring
        w
            Array of frequencies to be used in analysis (rad/s)
        mpb
            A MoorPy Body object that represents this FOWT in MoorPy
        mooring : dict
            A dictionary describing the mooring system that is specific to this FOWT (to be stored inside this object)
        depth
            Water depth, positive-down. (m)
        x_ref : float
            Reference x location of this fowt in the array [m]
        y_ref : float
            Reference y location of this fowt in the array [m]
        heading_adjust : float
            Rotation to the heading of the platform and mooring system to be applied [deg]
        '''

        print("Making FOWT")
       
        # basic setup
        self.nDOF = 6
        self.nw = len(w)                                          # number of frequencies
        self.Xi0 = np.zeros( self.nDOF)                           # mean offsets of platform from its reference point [m, rad]
        self.Xi  = np.zeros([self.nDOF, self.nw], dtype=complex)  # complex response amplitudes as a function of frequency  [m, rad]
        self.heading_adjust = heading_adjust                      # rotation to the heading of the platform and mooring system to be applied [deg]
        
        self.design = design
        self.intersectMesh = design['platform'].get('intersectMesh', 0) # Default to 0 (no intersection mesh)
        self.characteristic_length_min = design['platform'].get('characteristic_length_min', 1.0)
        self.characteristic_length_max = design['platform'].get('characteristic_length_max', 3.0)
        if self.intersectMesh == 1:
            # make sure the mesh size are positive
            assert self.characteristic_length_min > 0, "characteristic_length_min must be positive"
            assert self.characteristic_length_max > 0, "characteristic_length_max must be positive"
        print(f"Mesh characteristic lengths: min={self.characteristic_length_min}, max={self.characteristic_length_max}")
        
        # position in the array
        self.x_ref = x_ref      # reference x position of the FOWT in the array [m]
        self.y_ref = y_ref      # reference y position of the FOWT in the array [m]
        self.r6 = np.zeros(6)   # mean position/orientation in absolute/array coordinates [m,rad]

        # Platform lumped inertias
        self.pointInertias = []        
        if 'pointInertias' in design['platform']:
            nPoints = len(design['platform']['pointInertias'])            
            for ip, pointInertia in enumerate(design['platform']['pointInertias']):
                self.pointInertias.append({'m': 0, 'inertia': np.zeros([6, 6]), 'r': np.zeros([3,])}) # mass, inertia, and position of each point mass to be added to the platform
                pointMass = getFromDict(pointInertia, 'mass', shape=0, default=0) # mass of the substructure [kg]
                self.pointInertias[-1]['m'] = pointMass
                auxInertia = getFromDict(pointInertia, 'moments_of_inertia', shape=6, default=[0,0,0]) # moments of inertia of the substructure [kg*m^2] - specified in the order Jxx, Jyy, Jzz, Jxy, Jxz, Jyz
                self.pointInertias[-1]['inertia'] = np.array([[pointMass, 0, 0, 0, 0, 0],
                                                              [0, pointMass, 0, 0, 0, 0],
                                                              [0, 0, pointMass, 0, 0, 0],
                                                              [0, 0, 0, auxInertia[0], auxInertia[3], auxInertia[4]],
                                                              [0, 0, 0, auxInertia[3], auxInertia[1], auxInertia[5]],
                                                              [0, 0, 0, auxInertia[4], auxInertia[5], auxInertia[2]]])
                self.pointInertias[-1]['r'] = getFromDict(pointInertia, 'location', shape=3, default=[0,0,0])
                        
        # count number of platform members
        self.nplatmems = 0
        for platmem in design['platform']['members']:
            if 'heading' in platmem:
                #print(platmem['heading'])
                self.nplatmems += len(platmem['heading'])
            else:
                self.nplatmems += 1
        
        # count numbers of rotors and their towers and copy some info
        if 'turbine' in design:
        
            self.nrotors = getFromDict(design['turbine'], 'nrotors', dtype=int, shape=0, default=1)
            if self.nrotors==1: design['turbine']['nrotors'] = 1
            
            if 'tower' in design['turbine']:
                if isinstance(design['turbine']['tower'], dict):                            # if a single tower is specified (rather than a list)
                    design['turbine']['tower'] = [design['turbine']['tower']]*self.nrotors  # replicate the tower info for each rotor
            
                self.ntowers = len(design['turbine']['tower'])
            
            else:
                self.ntowers = 0

            # copy over site info into turbine dictionary
            design['turbine']['rho_air'       ] = getFromDict(design['site'], 'rho_air', shape=0, default=1.225)
            design['turbine']['mu_air'        ] = getFromDict(design['site'], 'mu_air', shape=0, default=1.81e-05)
            design['turbine']['shearExp_air'  ] = getFromDict(design['site'], 'shearExp_air', shape=0, default=0.12)
            design['turbine']['rho_water'     ] = getFromDict(design['site'], 'rho_water', shape=0, default=1025.0)
            design['turbine']['mu_water'      ] = getFromDict(design['site'], 'mu_water', shape=0, default=1.0e-03)
            design['turbine']['shearExp_water'] = getFromDict(design['site'], 'shearExp_water', shape=0, default=0.12)
            
            # Nacelle geometry (for hydro): if a single one (dictionary) is listed,
            # replicate it into a list for all rotors.
            if 'nacelle' in design['turbine']:
                if isinstance(design['turbine']['nacelle'], dict): 
                    design['turbine']['nacelle'] = [design['turbine']['nacelle']]*self.nrotors
                        
        else: # no turbines/rotors case
        
            self.nrotors = 0  # this will ensure RAFT doesn't set up or look for any Rotor objects
            self.ntowers = 0
            
            # note: RNA descriptions are not defined because there is no turbine


        self.rotorList = []

        self.depth = depth

        self.w = np.array(w)
        self.dw = w[1]-w[0]         # frequency increment [rad/s]    
        
        self.k = np.array([waveNumber(w, self.depth) for w in self.w])  # wave number [m/rad]

        self.rho_water = getFromDict(design['site'], 'rho_water', default=1025.0)
        self.g         = getFromDict(design['site'], 'g'        , default=9.81)
        self.shearExp_water = getFromDict(design['site'], 'shearExp_water', default=0.12)
        
        self.potModMaster = getFromDict(design['platform'], 'potModMaster', dtype=int, default=0)
        dlsMax       = getFromDict(design['platform'], 'dlsMax'      , default=5.0)
        min_freq_BEM = getFromDict(design['platform'], 'min_freq_BEM', default=self.dw/2/np.pi)
        self.dw_BEM  = 2.0*np.pi*min_freq_BEM
        self.dz_BEM  = getFromDict(design['platform'], 'dz_BEM', default=3.0)
        self.da_BEM  = getFromDict(design['platform'], 'da_BEM', default=2.0)


        # member-based platform description
        self.memberList = []                                         # list of member objects

        for mi in design['platform']['members']:

            # prepare member info
            if self.potModMaster in [1]:
                mi['potMod'] = False
            elif self.potModMaster in [2,3]:
                mi['potMod'] = True
                
            if 'dlsMax' not in mi:    # apply the global dlsMax setting to any member that doesn't have its own setting
                mi['dlsMax'] = dlsMax

            headings = getFromDict(mi, 'heading', shape=-1, default=0.)
            mi['headings'] = headings   # differentiating the list of headings/copies of a member from the individual member's heading
            
            # create member object
            if np.isscalar(headings):
                self.memberList.append(Member(mi, self.nw, heading=headings+heading_adjust))
            else:
                for heading in headings:
                    self.memberList.append(Member(mi, self.nw, heading=heading+heading_adjust))
        
        
        # add tower(s) and nacelle(s) to member list if applicable
        if 'turbine' in design:
            if 'tower' in design['turbine']:
                for mem in design['turbine']['tower']:
                    self.memberList.append(Member(mem, self.nw))
            if 'nacelle' in design['turbine']:
                for mem in design['turbine']['nacelle']:
                    self.memberList.append(Member(mem, self.nw))
        #TODO: consider putting the tower somewhere else rather than in end of memberList <<<
        
        # array-level mooring system connection
        self.body = mpb                                              # reference to Body in mooring system corresponding to this turbine

        # this FOWT's own MoorPy system (may not be used)
        if 'mooring' in design and isinstance(design['mooring'], dict):

            self.ms = mp.System()
            self.ms.parseYAML(design['mooring'])
            
            # ensure proper setup with one coupled Body tied to this FOWT
            if len(self.ms.bodyList) == 0:
                self.ms.addBody(-1, [0,0,0,0,0,0]) # create a new body if needed
                for point in self.ms.pointList:
                    if point.type == -1:  # attached any coupled points to the body
                        self.ms.bodyList[0].attachPoint(point.number, point.r)
                        point.type = 1  # now indicate point is fixed (to the body)
                        
            elif len(self.ms.bodyList) == 1:
                self.ms.bodyList[0].type = -1  # ensure it's set to coupled type
            else:
                raise Exception("More than one body detected in FOWT mooring system.")
                
            # move mooring system according to the FOWT's reference position
            self.ms.transform(trans=[x_ref, y_ref], rot=heading_adjust)  
            self.ms.initialize()

        else:
            self.ms = None
        
        self.F_moor0 = np.zeros(6)     # mean mooring forces in a given scenario
        self.C_moor = np.zeros([6,6])  # mooring stiffness matrix in a given scenario

        if 'yaw_stiffness' in design['platform']:
            self.yawstiff = design['platform']['yaw_stiffness']       # If you're modeling OC3 spar, for example, import the manual yaw stiffness needed by the bridle config
        else:
            self.yawstiff = 0

        # build Rotor list
        for ir in range(self.nrotors):
            self.rotorList.append(Rotor(design['turbine'], self.w, ir))
        
        # initialize mean force arrays to zero, so the model can work before we calculate excitation
        self.f_aero0 = np.zeros([6, self.nrotors])
        # mean weight and hydro force arrays are set elsewhere. In future hydro could include current.
        self.D_hydro = np.zeros(6)      # initialize mean drag force from current - acts as a "static" force, like f_aero0

        # flag to signal whether any members will be modeled with BEM
        self.potMod = any([member['potMod']==True for member in design['platform']['members'] ])


        # initialize BEM arrays, whether or not a BEM solver is used
        # __, __, nWaveHeadings = getUniqueCaseHeadings(design['cases']['keys'], design['cases']['data'])  # identify how many wave headings need to be preprocessed
        self.A_BEM = np.zeros([6,6,self.nw], dtype=float)                 # hydrodynamic added mass matrix [kg, kg-m, kg-m^2]
        self.B_BEM = np.zeros([6,6,self.nw], dtype=float)                 # wave radiation drag matrix [kg, kg-m, kg-m^2]
        #self.X_BEM = np.zeros([self.nWaves, 6, self.nw], dtype=complex)               # linaer wave excitation force/moment coefficients vector [N, N-m]
        #self.F_BEM = np.zeros([self.nWaves, 6, self.nw], dtype=complex)               # linaer wave excitation force/moment complex amplitudes vector [N, N-m]
        # <<< TODO: Set maximum of amount of headingsm for X_BEM and F_BEM and use interpolation of hydrodynamics. Or not?

        # First-order linear hydro input file options (preexisting files option)
        # when potFirstOrder > 0
        self.potFirstOrder = getFromDict(design['platform'], 'potFirstOrder', dtype=int, default=0)
        if self.potFirstOrder==1:
            if not 'hydroPath' in design['platform']:
                raise Exception('If potFirstOrder==1, then hydroPath must be specified in the platform input.')
            self.hydroPath = design['platform']['hydroPath']
            self.readHydro() #self.hydroPath)
        

        # Add a flag to either not compute 2nd order hydro; read QTF; or compute it with a slender-body approximation
        self.potSecOrder = getFromDict(design['platform'], 'potSecOrder', dtype=int, default=0)
        if self.potSecOrder==1:
            if (not 'min_freq2nd' in design['platform']) or (not 'max_freq2nd' in design['platform']):
                raise Exception('If potSecOrder==1, then both min_freq2nd and max_freq2nd must be specified in the platform input.')
            min_freq2nd = design['platform']['min_freq2nd']
            max_freq2nd = design['platform']['max_freq2nd']
            df_freq2nd = min_freq2nd
            if ('df_freq2nd' in design['platform']):
                df_freq2nd = design['platform']['df_freq2nd']
            self.w1_2nd = np.arange(min_freq2nd, max_freq2nd+0.5*min_freq2nd, df_freq2nd)*2*np.pi
            self.w2_2nd = self.w1_2nd.copy()

            self.k1_2nd = np.zeros(len(self.w1_2nd))
            for i, w in enumerate(self.w1_2nd):
                self.k1_2nd[i] = waveNumber(w, self.depth)
            self.k2_2nd = self.k1_2nd.copy()

        elif self.potSecOrder==2:
            if not 'hydroPath' in design['platform']:
                raise Exception('If potSecOrder==2, then hydroPath must be specified in the platform input.')
            self.qtfPath = design['platform']['hydroPath'] + '.12d'
            self.readQTF(self.qtfPath)

        # Set output folder for QTFs
        self.outFolderQTF = None
        if 'outFolderQTF' in design['platform']:
            self.outFolderQTF = design['platform']['outFolderQTF']
    
    
    def setPosition(self, r6):
        '''Updates the FOWT's (mean) position including all components.
        
        r6 : float array
            6 DOF absolute position of FOWT [m, rad]
        '''
        
        # if offset provided, set things according to those positions, otherwise zero it
        self.r6 = r6
        self.Xi0 = self.r6 - np.array([self.x_ref, self.y_ref, 0, 0, 0, 0])
        
        # calculate and save a rotation/orientation matrix
        self.Rmat = rotationMatrix(*self.r6[3:])  # rotation matrix for fowt orientation
        
        # set the positions of the FOWT's members, rotors, and MoorPy system
        if self.ms:
            self.ms.bodyList[0].setPosition(self.r6)
        
        for rot in self.rotorList:
            rot.setPosition(r6=self.r6)
            
        for mem in self.memberList:
            mem.setPosition(r6=self.r6)
        
        # solve the mooring system equilibrium of this FOWT's own MoorPy system
        if self.ms:
            self.ms.solveEquilibrium(tol=0.05)
            self.C_moor = self.ms.getCoupledStiffnessA()
            self.F_moor0 = self.ms.bodyList[0].getForces(lines_only=True)
        

    def calcStatics(self):
        '''Computes the static properties of the FOWT in terms of mass and hydrostatic-related
        matrices and mean force vectors about the FOWT PRP in unrotated directions, based on
        the mean offsets of the FOWT, Xi0.
        '''

        rho = self.rho_water
        g   = self.g

        # structure-related arrays
        self.M_struc = np.zeros([6,6])                # structure/static mass/inertia matrix [kg, kg-m, kg-m^2]
        self.B_struc = np.zeros([6,6])                # structure damping matrix [N-s/m, N-s, N-s-m] (may not be used)
        self.C_struc = np.zeros([6,6])                # structure effective stiffness matrix [N/m, N, N-m]
        self.W_struc = np.zeros([6])                  # static weight vector [N, N-m]
        
        # hydrostatic arrays
        self.C_hydro = np.zeros([6,6])                # hydrostatic stiffness matrix [N/m, N, N-m]
        self.W_hydro = np.zeros(6)                    # buoyancy force/moment vector [N, N-m]  <<<<< not used yet


        # --------------- add in linear hydrodynamic coefficients here if applicable --------------------
        #[as in load them] <<<<<<<<<<<<<<<<<<<<<

        # --------------- Get general geometry properties including hydrostatics ------------------------

        # initialize some variables for running totals
        VTOT = 0.                   # Total underwater volume of all members combined
        m_all = 0.                   # Total mass of all members [kg]
        AWP_TOT = 0.                # Total waterplane area of all members [m^2]
        IWPx_TOT = 0                # Total waterplane moment of inertia of all members about x axis [m^4]
        IWPy_TOT = 0                # Total waterplane moment of inertia of all members about y axis [m^4]
        Sum_V_rCB = np.zeros(3)     # product of each member's buoyancy multiplied by center of buoyancy [m^4]
        Sum_AWP_rWP = np.zeros(2)   # product of each member's waterplane area multiplied by the area's center point [m^3]
        m_center_sum = np.zeros(3)  # product of each member's mass multiplied by its center of mass [kg-m] (Only considers the shell mass right now)

        self.m_sub = 0              # total mass of just the members that make up the substructure [kg]
        self.C_struc_sub = np.zeros([6,6])  # substructure effective stiffness matrix [N/m, N, N-m]
        self.M_struc_sub = np.zeros([6,6])  # total mass matrix of just the substructure about the PRP
        m_sub_sum = 0               # product of each substructure member's mass and CG, to be used to find the total substructure CG [kg-m]
        self.m_shell = 0             # total mass of the shells/steel of the members in the substructure [kg]
        mballast = []               # list to store the mass of the ballast in each of the substructure members [kg]
        pballast = []               # list to store the density of ballast in each of the substructure members [kg]
        self.mtower = np.zeros(self.ntowers)    # assume that the whole tower will always be one member
        self.rCG_tow = []

        memberList = [mem for mem in self.memberList if mem.name != 'nacelle']
        # loop through each member
        for i,mem in enumerate(memberList):

            # calculate member's orientation information (stored in the member and used in later steps)
            mem.setPosition(r6=self.r6)  # <<< is this redundant, assume fowt.setPosition has been called?
            
            # note: quantities in the following section are relative to the PRP (but with global direcions)

            # ---------------------- get member's mass and inertia properties ------------------------------
            # get member mass and inertia info (including mem.M_struc) <<< still split between converting to PRP in or out of these functions
            mass, center, m_shell, mfill, pfill = mem.getInertia(rPRP=self.r6[:3]) 

            # Calculate the mass matrix of the FOWT about the PRP
            self.W_struc += translateForce3to6DOF( np.array([0,0, -g*mass]), center )  # weight vector
            self.M_struc += mem.M_struc     # mass/inertia matrix about the PRP
            
            m_center_sum += center*mass     # product sum of the mass and center of mass to find the total center of mass [kg-m]

            # Tower calculations
            if mem.type <= 1:   # <<<<<<<<<<<< maybe find a better way to do the if condition
                self.mtower[i-self.nplatmems] = mass                  # mass of the tower [kg]
                self.rCG_tow.append(center)               # center of mass of the tower from the PRP [m]
            # Substructure calculations
            if mem.type > 1:
                self.m_sub += mass              # mass of the substructure
                self.M_struc_sub += mem.M_struc     # mass matrix of the substructure about the PRP
                m_sub_sum += center*mass        # product sum of the substructure members and their centers of mass [kg-m]
                self.m_shell += m_shell               # mass of the substructure shell material [kg]
                mballast.extend(mfill)              # list of ballast masses in each substructure member (list of lists) [kg]
                pballast.extend(pfill)              # list of ballast densities in each substructure member (list of lists) [kg/m^3]

            # -------------------- get each member's buoyancy/hydrostatic properties -----------------------

            Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = mem.getHydrostatics(rho=self.rho_water, g=self.g, rPRP=self.r6[:3])
            
            # add to fowt's mean force vector and stiffness matrix
            self.W_hydro += Fvec # translateForce3to6DOF( np.array([0,0, Fz]), mem.rA )  # buoyancy vector
            self.C_hydro += Cmat # translateMatrix6to6DOF(Cmat, mem.rA)                       # hydrostatic stiffness matrix
       
            # convert other metrics to also be about the PRP (platform reference point)
            VTOT    += V_UW    # add to total underwater volume of all members combined
            AWP_TOT += AWP
            IWPx_TOT += IWP + AWP*yWP**2
            IWPy_TOT += IWP + AWP*xWP**2
            Sum_V_rCB   += r_CB*V_UW
            Sum_AWP_rWP += np.array([xWP, yWP])*AWP
            
        # ------------- include buoyancy effects of underwater rotors (blades first, then nacelles) -------------
        # loop through each blade member to calculate rotor buoyancy forces (for underwater turbines)
        for i, rotor in enumerate(self.rotorList):

            if rotor.r3[2] < 0:      # only do this for underwater rotors
                
                for j in range(int(rotor.nBlades)):     # for each blade on the rotor

                    # ensure the blade azimuths are equally spaced apart (so that the specific heading values are arbitrary)
                    if all(np.mod(np.diff(rotor.azimuths, append=rotor.azimuths[0]),360) != np.mod(np.diff(rotor.azimuths, append=rotor.azimuths[0])[0], 360)):
                        raise ValueError("The azimuths of the blades need to be equally spaced apart")

                    for k,afmem in enumerate(rotor.bladeMemberList):    # for each airfoil member in the bladeMemberList

                        # store original positions of the airfoil member about the original rotor axis
                        rA_OG = afmem.rA0
                        rB_OG = afmem.rB0
                        rOG = np.vstack([rA_OG, rB_OG])

                        # set the heading, or azimuth angle, of the blade member
                        afmem.heading = rotor.azimuths[j]

                        # find the end nodes of the blade member about the global coordinates (e.g,, if rA_OG = [0,0,0] and rB_OG=[0,1,0], and heading=90, then rA=[0,0,0] and rB=[0,0,1])
                        r_new = rotor.getBladeMemberPositions(rotor.azimuths[j], rOG)

                        # save these end node positions in the blade member
                        afmem.rA0 = r_new[0,:]
                        afmem.rB0 = r_new[1,:]

                        # save the positions of the nodes for each blade
                        rotor.nodes[j,k,:] = afmem.rA0
                        if k==len(rotor.bladeMemberList)-1:     # if it's the last blade member, save it's rB position to the last position in the nodes array
                            rotor.nodes[j,k+1,:] = afmem.rB0

                        # find the actual orientation vectors of the blade member
                        afmem.setPosition()

                        # calculate the mass and inertial properties of the blade members
                        #mass, center, m_shell, mfill, pfill = afmem.getInertia()
                        # >>>>>> can be used later if actual rectangular mass properties are desired other than mRNA <<<<<<<<

                        # calculate hydrostatic properties of the blade (sub)member and add them to the system matrices
                        Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = afmem.getHydrostatics(rho=self.rho_water, g=self.g, rPRP=self.r6[:3])
                        
                        # outputs of getHydrostatics should already be about the PRP
                        self.W_hydro += Fvec # buoyancy vector
                        self.C_hydro += Cmat # hydrostatic stiffness matrix

                        VTOT    += V_UW    # add to total underwater volume of all members combined
                        AWP_TOT += AWP
                        IWPx_TOT += IWP + AWP*yWP**2
                        IWPy_TOT += IWP + AWP*xWP**2
                        Sum_V_rCB   += r_CB*V_UW
                        Sum_AWP_rWP += np.array([xWP, yWP])*AWP

                        # reset original rA and rB values of the airfoil member
                        afmem.rA0 = rA_OG
                        afmem.rB0 = rB_OG
                        afmem.setPosition()
                        
                        # Note: it might be possible to streamline the above using new capabilities in setPosition (but not sure).
        
        
        nacelleMemberList = [mem for mem in self.memberList if mem.name == 'nacelle']
        # include only hydrostatic properties of nacelles (inertia properties are stored in mRNA/IxRNA/IrRNA and used below)
        for mem in nacelleMemberList:

            # call getHydroStatics for nacelles
            Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = mem.getHydrostatics(rho=self.rho_water, g=self.g, rPRP=self.r6[:3])
            
            # add to fowt's mean force vector and stiffness matrix
            self.W_hydro += Fvec # translateForce3to6DOF( np.array([0,0, Fz]), mem.rA )  # buoyancy vector
            self.C_hydro += Cmat # translateMatrix6to6DOF(Cmat, mem.rA)                       # hydrostatic stiffness matrix
       
            # convert other metrics to also be about the PRP (platform reference point)
            VTOT    += V_UW    # add to total underwater volume of all members combined
            AWP_TOT += AWP
            IWPx_TOT += IWP + AWP*yWP**2
            IWPy_TOT += IWP + AWP*xWP**2
            Sum_V_rCB   += r_CB*V_UW
            Sum_AWP_rWP += np.array([xWP, yWP])*AWP
        
        
        # ------------------------- include RNA properties -----------------------------
        for i, rotor in enumerate(self.rotorList):
            
            # create mass/inertia matrix
            Mmat = np.diag([rotor.mRNA, rotor.mRNA, rotor.mRNA, 
                            rotor.IxRNA, rotor.IrRNA, rotor.IrRNA])
            
            # Rotate RNA mass matrix into the global orientation
            Mmat = rotateMatrix6(Mmat, rotor.R_q)  
            
            # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices
            self.W_struc += translateForce3to6DOF(np.array([0,0, -g*rotor.mRNA]), rotor.r_CG_rel )   # weight vector
            self.M_struc += translateMatrix6to6DOF(Mmat, rotor.r_CG_rel)                            # mass/inertia matrix
            m_center_sum += rotor.r_CG_rel*rotor.mRNA


        # ------------------------- include point inertia properties -----------------------------
        for ip, pointInertia in enumerate(self.pointInertias):            
            self.M_struc += translateMatrix6to6DOF(pointInertia['inertia'], pointInertia['r'])
            self.W_struc += translateForce3to6DOF( np.array([0,0, -g*pointInertia['m']]), pointInertia['r'] )
            m_center_sum += pointInertia['r']*pointInertia['m']

            self.m_sub += pointInertia['m']
            self.M_struc_sub += translateMatrix6to6DOF(pointInertia['inertia'], pointInertia['r'])     # mass matrix of the substructure about the PRP
            m_sub_sum += pointInertia['r']*pointInertia['m']        # product sum of the substructure members and their centers of mass [kg-m]


        # ----------- process inertia-related totals ----------------

        m_all = self.M_struc[0,0]             # total mass of all the members
        rCG_all = m_center_sum/m_all          # total CG of all the members
        
        self.rCG = rCG_all
        self.rCG_sub = m_sub_sum/self.m_sub   # solve for just the substructure mass and CG
        
        
  
        # get principal moments of inertia (about CG) 
        # note: these are likely only useful in the unrotated frame, i.e.
        # the first time calcStatics is called.
        
        # the mass matrix of the substructure about the substruc's CM
        M_sub = translateMatrix6to6DOF(self.M_struc_sub, -self.rCG_sub)  # -rCG_sub due to convention of the function
        
        # overall structure mass matrix about its CM
        M_all = translateMatrix6to6DOF(self.M_struc, -self.rCG)

        # could check that off-diagonals are approximately zero as an error check
        
        
        # Solve for the total mass of each type of ballast in the substructure
        self.pb = []                                                # empty list to store the unique ballast densities
        for i in range(len(pballast)):
            if pballast[i] != 0:                                    # if the value in pballast is not zero
                if self.pb.count(pballast[i]) == 0:                 # and if that value is not already in pb
                    self.pb.append(pballast[i])                     # store that ballast density value

        self.m_ballast = np.zeros(len(self.pb))                      # make an empty m_ballast list with len=len(pb)
        for i in range(len(self.pb)):                               # for each ballast density
            for j in range(len(mballast)):                          # loop through each ballast mass
                if float(pballast[j]) == float(self.pb[i]):   # but only if the index of the ballast mass (density) matches the value of pb
                    self.m_ballast[i] += mballast[j]                 # add that ballast mass to the correct index of mballast



        # ----------- process key hydrostatic-related totals for use in static equilibrium solution ------------------
                               # save the total underwater volume
        rCB_TOT = Sum_V_rCB/VTOT       # location of center of buoyancy on platform
        
        if VTOT==0: # if you're only working with members above the platform, like modeling the wind turbine
            zMeta = 0
        else:
            zMeta   = rCB_TOT[2] + IWPx_TOT/VTOT  # add center of buoyancy and BM=I/v to get z elevation of metecenter [m] (have to pick one direction for IWP)

        self.C_struc[3,3] = -m_all*g*rCG_all[2]
        self.C_struc[4,4] = -m_all*g*rCG_all[2]
        
        self.C_struc_sub[3,3] = -self.m_sub*g*self.rCG_sub[2]
        self.C_struc_sub[4,4] = -self.m_sub*g*self.rCG_sub[2]

        # add relevant properties to this turbine's MoorPy Body
        # >>> should double check proper handling of mean weight and buoyancy forces throughout model <<<
        if self.body:   # note: this is likely unused now <<<
            self.body.m = m_all
            self.body.v = VTOT
            self.body.rCG = rCG_all
            self.body.AWP = AWP_TOT
            self.body.rM = np.array([rCB_TOT[0], rCB_TOT[1], zMeta])    # now includes x and y coordinates for center of buoyancy
        #is there any risk of additional moments due to offset CB since MoorPy assumes CB at ref point? <<<
        self.rCB = rCB_TOT
        self.m = m_all
        self.V = VTOT
        self.AWP = AWP_TOT
        self.rM = np.array([rCB_TOT[0], rCB_TOT[1], zMeta])
      
        # save things in a dictionary now        
        self.props = {}
        self.props['m']     = self.m 
        self.props['m_sub'] = self.m_sub 
        self.props['v'] = self.V
        self.props['rCG']     = self.rCG
        self.props['rCG_sub'] = self.rCG_sub
        self.props['rCB'] = self.rCB
        self.props['AWP'] = self.AWP
        self.props['rM'] = self.rM
        self.props['Ixx'] = M_all[3,3]  # principale moments of inertia of entire structure
        self.props['Iyy'] = M_all[4,4]
        self.props['Izz'] = M_all[5,5]
        self.props['Ixx_sub'] = M_sub[3,3]  # principale moments of inertia of substructure
        self.props['Iyy_sub'] = M_sub[4,4]
        self.props['Izz_sub'] = M_sub[5,5]


    def calcBEM(self, dw=0, wMax=0, wInf=10.0, dz=0, da=0, dh=0, headings=[0], meshDir=os.path.join(os.getcwd(),'BEM'), dmin=1, dmax=3):
        '''This generates a mesh for the platform and runs a BEM analysis on it
        using pyHAMS. It can also write adjusted .1 and .3 output files suitable
        for use with OpenFAST.
        The mesh is only made for non-interesecting members flagged with potMod=1.
        Note that this method does not consider mean offsets when calculating coefficients.
        
        PARAMETERS
        ----------
        dw : float
            Optional specification of custom frequency increment (rad/s).
        wMax : float
            Optional specification of maximum frequency for BEM analysis (rad/s). Will only be
            used if it is greater than the maximum frequency used in RAFT.
        wInf : float
            Optional specification of large frequency to use as approximation for infinite
            frequency in pyHAMS analysis (rad/s).
        dz : float
            desired longitudinal panel size for potential flow BEM analysis (m)
        da : float
            desired azimuthal panel size for potential flow BEM analysis (m)
        headings : list of floats
            incident wave headings to be considered, default [0] (deg). <<<<<<<< capability still needs to be added, after adjustments to pyHAMS
        '''
        
        # try to make a mesh and run HAMS if we're supposed to (not in the case of potModMaster 3)
        if self.potMod and self.potModMaster in [0, 2]:
            
            import pyhams.pyhams as ph  # import PyHAMS only if we're going to use it

            # go through members to be modeled with BEM and calculated their nodes and panels lists
            nodes = []
            panels = []

            vertices = np.zeros([0,3])  # for GDF output

            dz = self.dz_BEM if dz==0 else dz  # allow override if provided
            da = self.da_BEM if da==0 else da
            #dh = self.dh_BEM if dh==0 else dh
            if self.intersectMesh == 0:
                for mem in self.memberList: 
                    if mem.potMod:          # >>>>>>>>>>>>>>>> now using for rectnagular member and the dimensions are hardcoded. need to integrate with the .yaml file input
                        if mem.shape == "circular":
                            pnl.meshMember(mem.stations, mem.d, mem.rA, mem.rB, dz_max=dz, da_max=da, savedNodes=nodes, savedPanels=panels)
                            # for GDF output
                            vertices_i = pnl.meshMemberForGDF(mem.stations, mem.d, mem.rA, mem.rB, dz_max=dz, da_max=da)
                            vertices = np.vstack([vertices, vertices_i])  # append the member's vertices to the master list
                        elif mem.shape == "rectangular":
                            widths = mem.sl[:, 0]
                            heights = mem.sl[:, 1]
                            pnl.meshRectangularMember(mem.stations, widths, heights, mem.rA, mem.rB, dz_max=dz, dw_max=da, dh_max=dh, savedNodes=nodes, savedPanels=panels) #
                            # for GDF output
                            vertices_i = pnl.meshRectangularMemberForGDF(mem.stations, widths, heights, mem.rA, mem.rB, dz_max=dz, dw_max=da, dh_max=dh)
                            vertices = np.vstack([vertices, vertices_i])  # append the member's vertices to the master list
                if len(panels) == 0:
                    print("WARNING: no panels to mesh.")
                pnl.writeMesh(nodes, panels, oDir=os.path.join(meshDir,'Input')) # generate a mesh file in the HAMS .pnl format

            elif self.design["platform"]["intersectMesh"] == 1:
                if pygmsh is None or meshmagick is None:
                    raise ImportError("The 'intersectMesh' option requires 'pygmsh' and 'meshmagick'. Please install them using 'pip install pygmsh==7.1.17' and 'pip install https://github.com/LHEEA/meshmagick/archive/refs/tags/3.4.zip'")
    
                import raft.IntersectionMesh as intersectMesh
                
                cylindrical_members = []
                rectangular_members = []

                platform = self.design.get("platform", {})
                for i, mem in enumerate(self.memberList):
                    if not mem.potMod:
                        continue
                          
                    stations = mem.stations
                    rA = mem.rA_original if hasattr(mem, "rA_original") else mem.rA
                    rB = mem.rB_original if hasattr(mem, "rB_original") else mem.rB
                    headings = mem.heading if hasattr(mem, "heading") else [0]
                    #print("name from raft:", mem.name)
                    #print("rA from raft: ",mem.rA)
                    #print("rB from raft: ", mem.rB)
                    #print(headings)
                    if mem.shape == "circular":
                        radius = mem.d / 2 if isinstance(mem.d, (int, float)) else None
                        diameters = mem.d
                        extensionA = getattr(mem, "extensionA", 0)
                        extensionB = getattr(mem, "extensionB", 0)
                        cylindrical_members.append({
                            "rA": rA,
                            "rB": rB,
                            "radius": radius,
                            "heading": headings,
                            "stations": stations,
                            "diameters": diameters,
                            "extensionA": extensionA,
                            "extensionB": extensionB
                        })

                    elif mem.shape == "rectangular":
                        widths = mem.sl[:, 0].tolist()
                        heights = mem.sl[:, 1].tolist()
                        extensionA = getattr(mem, "extensionA", 0)
                        extensionB = getattr(mem, "extensionB", 0)
                        #print(f"Extension for {mem.name}: {extensionA}")
                        rectangular_members.append({
                            "rA": rA,
                            "rB": rB,
                            "widths": widths,
                            "heights": heights,
                            "heading": headings,
                            "stations": stations,
                            "extensionA": extensionA,
                            "extensionB": extensionB
                        })
        

                intersectMesh.mesh(meshDir=os.path.join(meshDir,'Input'), cylindrical_members=cylindrical_members, rectangular_members=rectangular_members, dmin=self.characteristic_length_min, dmax=self.characteristic_length_max)


            ph.create_hams_dirs(meshDir)                #

            # HAMS needs a hydrostatics file, it's unused for .1 and .3,
            # but HAMS write the .hst file that OpenFAST uses
            ph.write_hydrostatic_file(meshDir, kHydro=self.C_hydro)

            # prepare frequency settings for HAMS
            dw_HAMS = self.dw_BEM if dw==0 else dw     # frequency increment - allow override if provided

            wMax_HAMS = max(wMax, max(self.w))         # make sure the HAMS runs includes both RAFT and export frequency extents

            nw_HAMS = int(np.ceil(wMax_HAMS/dw_HAMS))  # ensure the upper frequency of the HAMS analysis is large enough

            dw_HAMS = np.round(dw_HAMS, 15)             # ensure the frequency discretization is under 16 decimal places (to be a float)
            
            ph.write_control_file(meshDir, waterDepth=self.depth, incFLim=1, iFType=3, oFType=4,   # inputs are in rad/s, outputs in s
                                  numFreqs=-nw_HAMS, minFreq=dw_HAMS, dFreq=dw_HAMS,
                                  numHeadings=len(headings), headingList=headings) #, numThreads=self.numThreads) <<<
            
            # Note about zero/infinite frequencies from WAMIT-formatted output files (as per WAMIT v7 manual): 
            # The limiting values of the added-mass coefficients may be evaluated for zero or infinite
            # period by specifying the values PER= 0:0 and PER< 0:0, respectively.  These special values are always
            # associated with the wave period, irrespective of the value of IPERIN and the corresponding
            # interpretation of the positive elements of the array PER


            # execute the HAMS analysis
            ph.run_hams(meshDir)
            
            self.hydroPath = os.path.join(meshDir,'Output','Wamit_format','Buoy')
            
        elif self.potModMaster == 3: 
            pass

        else:
            return
                
        self.readHydro()

    def readHydro(self):
        '''Read preexisting WAMIT-style .1 and .3 files and use as the FOWT's
        added mass, damping, and excitation matrices. This is as an alternative 
        to PyHAMS or strip theory, and is done when potFirstOrder == 1/True.'''
        import pyhams.pyhams as ph
        
        # read the preexisting WAMIT-style output files
        addedMass, damping, w1 = ph.read_wamit1(self.hydroPath+'.1', TFlag=True)  # first two entries in frequency dimension are expected to be zero-frequency then infinite frequency
        M, P, R, I, w3, heads  = ph.read_wamit3(self.hydroPath+'.3', TFlag=True)
        # The Tflag means that the first column is in units of periods, not frequencies, and therefore the first set (-1) becomes zero-frequency and the second set is infinite

        # process and sort headings and sort frequencies
        R_unsorted = R.copy()
        I_unsorted = I.copy()
        self.BEM_headings = np.array(heads)%(360)  # save headings in range of 0-360 [deg]            # interpole to the frequencies RAFT is using        
        sorted_indices = np.argsort(self.BEM_headings)
        self.BEM_headings = self.BEM_headings[sorted_indices]
        M = M[sorted_indices,:,:]
        P = P[sorted_indices,:,:]
        R = R[sorted_indices,:,:]
        I = I[sorted_indices,:,:]

        # interpolate to RAFT model frequencies
        # zero frequency values are being stacked on to give smooth results if the requested frequency is below what's available from HAMS
        addedMassInterp = interp1d(np.hstack([w1[2:],  0.0]), np.dstack([addedMass[:,:,2:], addedMass[:,:,0]]), assume_sorted=False, axis=2)(self.w)
        dampingInterp   = interp1d(np.hstack([w1[2:],  0.0]), np.dstack([  damping[:,:,2:], np.zeros([6,6]) ]), assume_sorted=False, axis=2)(self.w)
        fExRealInterp   = interp1d(np.hstack([w3,      0.0]), np.dstack([       R, np.zeros([len(heads),6]) ]), assume_sorted=False, axis=2)(self.w)
        fExImagInterp   = interp1d(np.hstack([w3,      0.0]), np.dstack([       I, np.zeros([len(heads),6]) ]), assume_sorted=False, axis=2)(self.w)
        # note: fEx tensors are sized according to [nHeadings, 6 DOFs, frequencies]
        
        # copy results over to the FOWT's coefficient arrays
        self.A_BEM = self.rho_water * addedMassInterp
        self.B_BEM = self.w * self.rho_water * dampingInterp                                 
        X_BEM_temp = self.rho_water * self.g * (fExRealInterp + 1j*fExImagInterp)      
        
        # transform excitation coefficients so that DOFs are always relative to incident wave heading rather than global frame
        self.X_BEM = np.zeros_like(X_BEM_temp)     

        # Transform excitation coefficients so that DOFs are always 
        # relative to incident wave heading rather than global frame,
        # for accurate magnitudes when interpolating between directions.
        self.X_BEM = np.zeros_like(X_BEM_temp)
        
        for ih in range(len(self.BEM_headings)):
            sin_heading = np.sin(np.radians(self.BEM_headings[ih]))
            cos_heading = np.cos(np.radians(self.BEM_headings[ih]))
            
            self.X_BEM[ih,0,:] =  cos_heading * X_BEM_temp[ih,0,:] + sin_heading * X_BEM_temp[ih,1,:]
            self.X_BEM[ih,1,:] = -sin_heading * X_BEM_temp[ih,0,:] + cos_heading * X_BEM_temp[ih,1,:]
            self.X_BEM[ih,2,:] = X_BEM_temp[ih,2,:]
            self.X_BEM[ih,3,:] =  cos_heading * X_BEM_temp[ih,3,:] + sin_heading * X_BEM_temp[ih,4,:]
            self.X_BEM[ih,4,:] = -sin_heading * X_BEM_temp[ih,3,:] + cos_heading * X_BEM_temp[ih,4,:]
            self.X_BEM[ih,5,:] = X_BEM_temp[ih,5,:]
            
        # HAMS results error checks  >>> any more we should have? <<<
        if np.isnan(self.A_BEM).any():
            raise Exception("NaN values detected in HAMS calculations for added mass. Check the geometry.")
        if np.isnan(self.B_BEM).any():
            raise Exception("NaN values detected in HAMS calculations for damping. Check the geometry.")
        if np.isnan(self.X_BEM).any():
            raise Exception("NaN values detected in HAMS calculations for excitation. Check the geometry.")

        
        

    def calcTurbineConstants(self, case, ptfm_pitch=0):
        '''This computes turbine linear terms (excluding hydrodynamic added 
        mass and inertial excitation, which are handled by getHydroConstants).
        
        case
            dictionary of case information
        ptfm_pitch
            mean pitch angle of the platform [rad]

        '''
        turbine_heading = getFromDict(case, 'turbine_heading', shape=0, dtype = float, default=0.0)  # [deg]
        turbine_status  = getFromDict(case, 'turbine_status', shape=0, dtype=str, default='operating')

        # initialize arrays (can remain zero if aerodynamics are disabled)
        self.A_aero  = np.zeros([6,6,self.nw,self.nrotors])                      # frequency-dependent aero-servo added mass matrix
        self.B_aero  = np.zeros([6,6,self.nw,self.nrotors])                      # frequency-dependent aero-servo damping matrix
        self.f_aero  = np.zeros([6,  self.nw,self.nrotors], dtype=complex)       # dynamice excitation force and moment amplitude spectra
        self.f_aero0 = np.zeros([6,          self.nrotors])                      # mean aerodynamic forces and moments
        #todo: reorder above so that w is last index <<<
        
        self.B_gyro  = np.zeros([6,6,self.nrotors])  # rotor gyroscopic damping matrix
        self.cav = [0]
        
        if turbine_status == 'operating':
            
            for ir, rot in enumerate(self.rotorList):
                # only compute the aerodynamics if enabled and windspeed is nonzero

                if rot.r3[2] < 0:
                    current = True
                    speed = getFromDict(case, 'current_speed', shape=0, default=1.0)
                else:
                    current = False
                    speed = getFromDict(case, 'wind_speed', shape=0, default=10.0)
                
                if rot.aeroServoMod > 0 and speed > 0.0:
                
                    # Get mean aero forces and fore-aft coefficients 
                    # Note: these are about hub coordinate in global orientation.
                    f_aero0, f_aero, a_aero, b_aero = rot.calcAero(case, current=current)  # get values about hub
                    
                    # convert coefficients to platform reference frame and populate tensor slice for this rotor
                    for iw in range(self.nw):
                        self.A_aero[:,:,iw,ir] = translateMatrix6to6DOF(a_aero[:,:,iw], rot.r_hub_rel)
                        self.B_aero[:,:,iw,ir] = translateMatrix6to6DOF(b_aero[:,:,iw], rot.r_hub_rel)
                    
                    # convert forces to platform reference frame
                    self.f_aero0[:,ir] = transformForce(f_aero0, offset=rot.r_hub_rel) # mean forces and moments
                    
                    for iw in range(self.nw):
                        self.f_aero[:,iw,ir] = transformForce(f_aero[:,iw], offset=rot.r_hub_rel) # excitation
                    
                    # calculate cavitation of the rotor (platform motions should already be accounted for in the CCBlade object after running calcAero)
                    if rot.r3[2] < 0:  # if submerged
                        self.cav = rot.calcCavitation(case)  # TO-DO: wire this to be a result/output, then uncomment <<<
                    
                    # ----- calculate rotor gyroscopic effects -----
                    # rotor speed [rpm]
                    Omega_rpm = np.interp(speed, rot.Uhub, rot.Omega_rpm)
                    
                    Omega_rotor = rot.q * Omega_rpm*2*np.pi/60  # rotor angular velocity vector
                    
                    # rotating inertia vector [kg-m^2 * rad/s = N-m-s]
                    IO_rotor = rot.I_drivetrain * Omega_rotor
                    
                    GyroDampingMatrix = getH(IO_rotor)
                    
                    self.B_gyro[3:, 3:, ir] = GyroDampingMatrix  
                    
                    # note, gyroscopic effect is purely rotational so no translation adjustment needed

        else:
            print(f"Warning: turbine status is '{turbine_status}' so rotor fluid loads are neglected.")


    def calcHydroConstants(self):
        '''Compute FOWT hydrodynamic added mass matrix and member-level
        inertial excitation coefficients.'''


        rho = self.rho_water
        g   = self.g
        
        # --------------------- get constant hydrodynamic values along each member -----------------------------

        self.A_hydro_morison = np.zeros([6,6])                # hydrodynamic added mass matrix, from only Morison equation [kg, kg-m, kg-m^2]

        # loop through each member

        for i,mem in enumerate(self.memberList):
        
            # get member added mass matrix about PRP (also saves each member's inertial excitation coefficients)
            if mem.MCF:
                k_array = self.k
            else:
                k_array = None

            A_hydro_i = mem.calcHydroConstants(r_ref=self.r6[:3], rho=rho, g=g, k_array=k_array)                      
            self.A_hydro_morison += A_hydro_i  # add to FOWT added mass matrix
    
        # ----- Get hydrodynamic contributions from any underwater rotors ------
        for i, rot in enumerate(self.rotorList):
            
            # compute rotor hydro added mass/inertia properties
            A_hydro_i, I_hydro_i = rot.calcHydroConstants(rho=rho, g=g)
            
            # make relative to PRP and add to system added mass matrix
            self.A_hydro_morison += translateMatrix6to6DOF(A_hydro_i, rot.r3-self.r6[:3]) 
    
    
    def getStiffness(self):
        '''Sums up all the stiffness effects on a FOWT.'''
        
        C_tot = np.zeros([6,6])       # total stiffness matrix [N/m, N, N-m]
        
        # add in mooring stiffness from MoorPy system
        C_tot += self.C_moor
        # add any additional yaw stiffness that isn't included in the MoorPy 
        # model (e.g. if a bridle isn't modeled)
        C_tot[5,5] += self.yawstiff
        # add system-level stiffness effect if it exists...
        if self.body:
            C_tot += self.body.getStiffness()  # in future should make an analytic body function for this

        C_tot += self.C_struc + self.C_hydro   # stiffness
        
        return C_tot
    
    
    def solveEigen(self, display=0):
        '''Compute the natural frequencies and mode shapes of the FOWT.
        This considers the FOWT and any attached mooring lines, but it
        does not account for coupling with other FOWTs as could occur
        with an array-level MoorPy instance with shared lines or 
        suspended cables.
        
        Returns
        -------
        fns : array
            List of natural frequencies [Hz].
        modes : 2D array
            List of mode shapes (eigenvectors) corresponding to the natural 
            frequencies.
        '''

        # Total mass and added mass matrix [kg, kg-m, kg-m^2]
        M_tot = self.M_struc + self.A_hydro_morison   # mass  (BEM option not supported yet)
        C_tot = self.getStiffness()  # stiffness

        # check viability of matrices
        message=''
        for i in range(self.nDOF):
            if M_tot[i,i] < 1.0:
                message += f'Diagonal entry {i} of system mass matrix is less than 1 ({M_tot[i,i]}). '
            if C_tot[i,i] < 1.0:
                message += f'Diagonal entry {i} of system stiffness matrix is less than 1 ({C_tot[i,i]}). '
                
        if len(message) > 0:
            raise RuntimeError('System matrices computed by RAFT have one or more small or negative diagonals: '+message)

        # calculate natural frequencies (using eigen analysis to get proper values for pitch and roll - otherwise would need to base about CG if using diagonal entries only)
        eigenvals, eigenvectors = np.linalg.eig(np.linalg.solve(M_tot, C_tot))   # <<< need to sort this out so it gives desired modes, some are currently a bit messy

        if any(eigenvals <= 0.0):
            raise RuntimeError("Error: zero or negative system eigenvalues detected.")

        # sort to normal DOF order based on which DOF is largest in each eigenvector
        ind_list = []
        for i in range(5,-1, -1):
            vec = np.abs(eigenvectors[i,:])  # look at each row (DOF) at a time (use reverse order to pick out rotational DOFs first)

            for j in range(6):               # now do another loop in case the index was claimed previously

                ind = np.argmax(vec)         # find the index of the vector with the largest value of the current DOF

                if ind in ind_list:          # if a previous vector claimed this DOF, set it to zero in this vector so that we look at the other vectors
                    vec[ind] = 0.0
                else:
                    ind_list.append(ind)     # if it hasn't been claimed before, assign this vector to the DOF
                    break

        ind_list.reverse()   # reverse the index list since we made it in reverse order

        fns = np.sqrt(eigenvals[ind_list])/2.0/np.pi   # apply sorting to eigenvalues and convert to natural frequency in Hz
        modes = eigenvectors[:,ind_list]               # apply sorting to eigenvectors
        
        if display > 0:
            print("")
            print("--------- Natural frequencies and mode shapes -------------")
            print("Mode        1         2         3         4         5         6")
            print("Fn (Hz)"+"".join([f"{fn:10.4f}" for fn in fns]))
            print("")
            for i in range(6):
                print(f"DOF {i+1}  "+"".join([f"{modes[i,j]:10.4f}" for j in range(6)]))
            print("-----------------------------------------------------------")
    
        return fns, modes
    

    def calcHydroExcitation(self, case, memberList=[], dgamma=0):
        '''This computes the wave kinematics and linear excitation for a given case.
        It calculates and F_BEM and F_hydro_iner, each with dimensions [wave headings, DOFs, frequencies].
        '''
        
        # ----- set up sea state -----
        
        # JvS made support for a second set of case wave info for a different heading. Instead, to generalize,
        # we will allow the case wave entries to be lists, with size nHeadings.
        
        if np.isscalar(case['wave_heading']):  # deal with the typical case of just one set of waves specified
            self.nWaves = 1
        else:
            self.nWaves = len(case['wave_heading'])
        
        # ensure our inputs are all arrays and of size nWaves (any scalar inputs will be tiled)
        case['wave_heading'] = getFromDict(case, 'wave_heading'  , shape=self.nWaves, dtype=float, default=0)
        case['wave_spectrum']= getFromDict(case, 'wave_spectrum' , shape=self.nWaves, dtype=str, default='JONSWAP')
        case['wave_period']  = getFromDict(case, 'wave_period'   , shape=self.nWaves, dtype=float)
        case['wave_height']  = getFromDict(case, 'wave_height'   , shape=self.nWaves, dtype=float)
        case['wave_gamma']   = getFromDict(case, 'wave_gamma'    , shape=self.nWaves, dtype=float, default=0)
        
        
        self.beta = deg2rad(case['wave_heading'])   # array of wave headings. Input in [deg], but the code uses [rad]
        self.zeta = np.zeros([self.nWaves,self.nw], dtype=complex)

        # make wave spectrum for each heading
        self.S = np.zeros([self.nWaves,self.nw])
        for ih in range(self.nWaves):
            if case['wave_spectrum'][ih] == 'unit':
                self.S[ih,:] = np.tile(1, self.nw)
                self.zeta[ih,:] = np.sqrt(2*self.S[ih,:]*self.dw)    # wave elevation amplitudes (these are easiest to use)
            elif case['wave_spectrum'][ih] == 'constant':
                self.S[ih,:] = case['wave_height'][ih]
                self.zeta[ih,:] = np.sqrt(2*self.S*self.dw)
            elif case['wave_spectrum'][ih] == 'JONSWAP':
                self.S[ih,:] = JONSWAP(self.w, case['wave_height'][ih], case['wave_period'][ih], Gamma=case['wave_gamma'][ih])        
                self.zeta[ih,:] = np.sqrt(2*self.S[ih,:]*self.dw)    # wave elevation amplitudes (these are easiest to use)
            elif case['wave_spectrum'][ih] in ['none','still']:
                self.zeta[ih,:] = np.zeros(self.nw)        
                self.S[ih,:] = np.zeros(self.nw)        
            else:
                raise ValueError(f"Wave spectrum input '{case['wave_spectrum'][ih]}' not recognized.")                        

        #print(f"significant wave height:  {4*np.sqrt(np.sum(S)*self.dw):5.2f} = {4*getRMS(self.zeta, self.dw):5.2f}") # << temporary <<<

        # TODO: consider current and viscous drift <<<
        
        # resize members' wave kinematics arrays for this case's sea states
        for i,mem in enumerate(memberList):
            mem.u    = np.zeros([self.nWaves, mem.ns, 3, self.nw], dtype=complex)
            mem.ud   = np.zeros([self.nWaves, mem.ns, 3, self.nw], dtype=complex)
            mem.pDyn = np.zeros([self.nWaves, mem.ns,    self.nw], dtype=complex)
        
        # also set up wave kinematics arrays for the rotors
        for i,rot in enumerate(self.rotorList):
            rot.u    = np.zeros([self.nWaves, 3, self.nw], dtype=complex)
            rot.ud   = np.zeros([self.nWaves, 3, self.nw], dtype=complex)
            rot.pDyn = np.zeros([self.nWaves,    self.nw], dtype=complex)
            
        # ----- calculate potential-flow wave excitation force -----

        self.F_BEM = np.zeros([self.nWaves,6,self.nw], dtype=complex)
        self.F_hydro_iner = np.zeros([self.nWaves, 6, self.nw],dtype=complex) # inertia excitation force/moment complex amplitudes vector [N, N-m]

        # BEM-based wave excitation force on platform for each wave heading 
        # (will be zero if only using strip theory). Includes wave heading interpolation.
        if self.potMod or self.potModMaster in [2,3]:

            for ih in range(self.nWaves):
                
                # phase offset due to FOWT position in array
                phase_offset = np.exp(-1j*self.k* ( self.x_ref*np.cos(np.deg2rad(case['wave_heading'][ih])) 
                                                  + self.y_ref*np.sin(np.deg2rad(case['wave_heading'][ih])) ) )

                beta = (np.degrees(self.beta[ih]) - self.heading_adjust)%360   # heading in range of 0-360 [deg]
                
                headings = self.BEM_headings      # the headings of the available BEM data [deg]
                nhs = len(headings)
                # find interpolation indices and factors, of format y* = y[iout[0]]*fout[0] + y[iout[1]]*fout[1]
                
                # this code needs checking!
                if (beta <= headings[0]):         # when wave heading (beta) is before first BEM heading
                    hlast = headings[-1] - 360    # make the last BEM heading negative so that it's before beta
                    i1 = nhs-1
                    i2 = 0
                    f2 = (beta - hlast)/( headings[0] - hlast )
                
                elif (beta >= headings[nhs-1]):   # when wave heading (beta) is after last BEM heading
                    hfirst = headings[0] + 360    # make the fisrt BEM heading positive so that it's after beta
                    i1 = nhs-1  
                    i2 = 0
                    f2 = (beta - headings[-1])/( hfirst - headings[-1] )
                
                else:                             # normal case
                    for i in range(nhs-1):
                        if (headings[i+1] > beta):
                            i1 = i
                            i2 = i+1 
                            f2 = (beta - headings[i] )/( headings[i+1] - headings[i] )
                            break
                
                f1 = 1.0 - f2
                      
                # interpolate excitation coefficients (which are coming in relative to their wave directions)
                X_prime = self.X_BEM[i1,:,:]*f1 + self.X_BEM[i2,:,:]*f2        
                
                #self.beta[ih], self.BEM_headings, self.X_BEM[, period=360) * 

                # rotate back to global frame
                sin_beta = np.sin(self.beta[ih])   #  check that I'm right in assuming HAMS outputs degrees
                cos_beta = np.cos(self.beta[ih])
                X_BEM_ih = np.zeros([6, self.nw], dtype=complex)
                X_BEM_ih[0,:] = X_prime[0,:]*cos_beta - X_prime[1,:]*sin_beta
                X_BEM_ih[1,:] = X_prime[0,:]*sin_beta + X_prime[1,:]*cos_beta
                X_BEM_ih[2,:] = X_prime[2,:]
                X_BEM_ih[3,:] = X_prime[3,:]*cos_beta - X_prime[4,:]*sin_beta
                X_BEM_ih[4,:] = X_prime[3,:]*sin_beta + X_prime[4,:]*cos_beta
                X_BEM_ih[5,:] = X_prime[5,:]
                
                # multiply excitation coefficients by wave elevation to get excitation forces and moments for this wave heading
                self.F_BEM[ih,:,:] = X_BEM_ih * self.zeta[ih,:] * phase_offset

        # ----- strip-theory wave excitation force -----
        # loop through each member to compute strip-theory contributions
        # This also saves the save wave kinematics over each member.
        for i,mem in enumerate(memberList):
            
            # loop through each node of the member
            for il in range(mem.ns):

                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:

                    # get wave kinematics spectra given a certain wave spectrum and location
                    # for each wave direction, calculate this node's frequency-dependent wave velocity, acceleration, and dynamic presure
                    for ih in range(self.nWaves):
                        mem.u[ih,il,:,:], mem.ud[ih,il,:,:], mem.pDyn[ih,il,:] = getWaveKin(self.zeta[ih,:], self.beta[ih], 
                                                                                 self.w, self.k, self.depth, mem.r[il,:], self.nw)
                    # Note: the above wave kinematics account for phasing due to the FOWT's mean offset position in the array
                    
                    if mem.potMod == False:
                        # calculate the linear excitation forces on this node for each wave heading and frequency
                        for ih in range(self.nWaves):
                            for i in range(self.nw):
                                if mem.MCF:
                                    Imat = mem.Imat_MCF[il,:,:, i]
                                else:
                                    Imat = mem.Imat[il,:,:]
                                F_exc_iner_temp = np.matmul(Imat, mem.ud[ih,il,:,i]) + mem.pDyn[ih,il,i]*mem.a_i[il]*mem.q 
                                
                                # add the excitation complex amplitude for this heading and frequency to the global excitation vector
                                self.F_hydro_iner[ih,:,i] += translateForce3to6DOF(F_exc_iner_temp, mem.r[il,:] - self.r6[:3]) # (about PRP)

        
        # ----- inertial excitation on rotor(s) -----
        
        for i, rot in enumerate(self.rotorList):
            if rot.r3[2] < 0:  # if submerged

                # get wave kinematics spectra given a certain wave spectrum and location
                # for each wave direction, calculate the hub wave kinematics
                for ih in range(self.nWaves):
                    rot.u[ih,:,:], rot.ud[ih,:,:], rot.pDyn[ih,:] = getWaveKin(self.zeta[ih,:], self.beta[ih], 
                                                                    self.w, self.k, self.depth, rot.r3, self.nw)
                # Note: the above wave kinematics account for phasing due to the FOWT's mean offset position in the array
                
                # Rotate rotor inertial excitation matrix to align with global
                I_hydro = rotateMatrix6(rot.I_hydro, rot.R_q)  
                # Note: Only first 3 columns of I_hydro currently have meaning <<<

                # compute force across frequency range
                for i in range(self.nw):
                    f3 = np.matmul( I_hydro[:3,:3], rot.ud[ih,:,i] )  # Forces due to acceleration
                    f6 = translateForce3to6DOF(f3, rot.r3 - self.r6[:3])  # Translate to about PRP (induces some moments)
                    f6[3:] += np.matmul( I_hydro[3:,:3], rot.ud[ih,:,i] )  # Add moments due to acceleration

                    self.F_hydro_iner[ih,:,i] += f6
                

    def calcHydroLinearization(self, Xi):
        '''The FOWT's dynamics solve iteration method. This calculates the amplitude-dependent 
        linearized coefficients, including the system linearized drag damping matrix. For the 
        drag-based excitation, call calcDragExcitation after this method.
        
        Currently hard-coded to only consider the first seastate/heading.

        Xi : complex array
            system response (just for this FOWT) - displacement and rotation complex amplitudes [m, rad]

        '''

        rho = self.rho_water
        g   = self.g

        # The linearized coefficients to be calculated

        B_hydro_drag = np.zeros([6,6])             # hydrodynamic damping matrix (just linearized viscous drag for now) [N-s/m, N-s, N-s-m]

        F_hydro_drag = np.zeros([6,self.nw],dtype=complex) # excitation force/moment complex amplitudes vector [N, N-m]

        ih = 0  # we will only consider the first sea state in this linearization process

        # loop through each member
        for mem in self.memberList:

            circ = mem.shape=='circular'  # convenience boolian for circular vs. rectangular cross sections

            # loop through each node of the member
            for il in range(mem.ns):

                # get node complex velocity spectrum based on platform motion's and relative position from PRP
                # node displacement, velocity, and acceleration (each [3 x nw])
                drnode, vnode, anode = getKinematics(mem.r[il,:] - self.r6[:3], Xi, self.w)

                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:

                    # interpolate coefficients for the current strip
                    Cd_q   = np.interp( mem.ls[il], mem.stations, mem.Cd_q  )
                    Cd_p1  = np.interp( mem.ls[il], mem.stations, mem.Cd_p1 )
                    Cd_p2  = np.interp( mem.ls[il], mem.stations, mem.Cd_p2 )
                    Cd_End = np.interp( mem.ls[il], mem.stations, mem.Cd_End)


                    # ----- compute side effects ------------------------

                    # member acting area assigned to this node in each direction
                    a_i_q  = np.pi*mem.ds[il]*mem.dls[il]  if circ else  2*(mem.ds[il,0]+mem.ds[il,0])*mem.dls[il]
                    a_i_p1 =       mem.ds[il]*mem.dls[il]  if circ else             mem.ds[il,0]      *mem.dls[il]
                    a_i_p2 =       mem.ds[il]*mem.dls[il]  if circ else             mem.ds[il,1]      *mem.dls[il]

                    # water relative velocity over node (complex amplitude spectrum)  [3 x nw]
                    vrel = mem.u[ih,il,:] - vnode

                    # break out velocity components in each direction relative to member orientation [nw]
                    vrel_q  = np.sum(vrel*mem.q[ :,None], axis=0)*mem.q[ :,None]     # (the ,None is for broadcasting q across all frequencies in vrel)
                    vrel_p  = vrel - vrel_q
                    vrel_p1 = np.sum(vrel*mem.p1[:,None], axis=0)*mem.p1[:,None]
                    vrel_p2 = np.sum(vrel*mem.p2[:,None], axis=0)*mem.p2[:,None]
                    
                    # get RMS of relative velocity component magnitudes (real-valued)
                    vRMS_q  = getRMS(vrel_q)
                    if circ: # Use the total perpendicular relative velocity 
                        vRMS_p1 = getRMS(vrel_p)
                        vRMS_p2 = vRMS_p1
                    else: # Otherwise treat each direction separately
                        vRMS_p1 = getRMS(vrel_p1)
                        vRMS_p2 = getRMS(vrel_p2)
                    
                    # linearized damping coefficients in each direction relative to member orientation [not explicitly frequency dependent...] (this goes into damping matrix)
                    Bprime_q  = np.sqrt(8/np.pi) * vRMS_q  * 0.5*rho * a_i_q  * Cd_q
                    Bprime_p1 = np.sqrt(8/np.pi) * vRMS_p1 * 0.5*rho * a_i_p1 * Cd_p1
                    Bprime_p2 = np.sqrt(8/np.pi) * vRMS_p2 * 0.5*rho * a_i_p2 * Cd_p2

                    # form damping matrix for the node based on linearized drag coefficients
                    Bmat_sides = Bprime_q*mem.qMat + Bprime_p1*mem.p1Mat + Bprime_p2*mem.p2Mat 


                    # ----- add end/axial effects for added mass, drag, and excitation including dynamic pressure ------
                    # note : v_a and a_i work out to zero for non-tapered sections or non-end sections

                    # end/axial area (removing sign for use as drag)
                    if circ:
                        a_i = np.abs(np.pi*mem.ds[il]*mem.drs[il])
                    else:
                        a_i = np.abs((mem.ds[il,0]+mem.drs[il,0])*(mem.ds[il,1]+mem.drs[il,1]) - (mem.ds[il,0]-mem.drs[il,0])*(mem.ds[il,1]-mem.drs[il,1]))

                    Bprime_End = np.sqrt(8/np.pi)*vRMS_q*0.5*rho*a_i*Cd_End

                    # form damping matrix for the node based on linearized drag coefficients
                    Bmat_end = Bprime_End*mem.qMat                                       #


                    # ----- sum up side and end damping matrices ------
                    
                    mem.Bmat[il,:,:] = Bmat_sides + Bmat_end   # store in Member object to be called later to get drag excitation for each wave heading

                    B_hydro_drag += translateMatrix3to6DOF(mem.Bmat[il,:,:], mem.r[il,:] - self.r6[:3])   # add to global damping matrix for Morison members


                    # ----- calculate wave drag excitation (this may be recalculated later) -----

                    for i in range(self.nw):

                        mem.F_exc_drag[il,:,i] = np.matmul(mem.Bmat[il,:,:], mem.u[ih,il,:,i])   # get local 3d drag excitation force complex amplitude for each frequency [3 x nw]

                        F_hydro_drag[:,i] += translateForce3to6DOF(mem.F_exc_drag[il,:,i], mem.r[il,:] - self.r6[:3])   # add to global excitation vector (frequency dependent)

        # save the arrays internally in case there's ever a need for the FOWT to solve it's own latest dynamics or for visualization
        self.B_hydro_drag = B_hydro_drag
        self.F_hydro_drag = F_hydro_drag

        # return the linearized coefficients
        return B_hydro_drag
    
    
    
    def calcDragExcitation(self, ih):
        '''Calculated linearized viscous drag excitation for a given sea state (wave heading). calcLinearizedTerms should be called first.

        ih : int
            index of wave case being evaluated here

        '''

        F_hydro_drag = np.zeros([6,self.nw],dtype=complex) # excitation force/moment complex amplitudes vector [N, N-m]
     
        for mem in self.memberList:   # loop through each member
            for il in range(mem.ns):  # loop through each node of the member
                if mem.r[il,2] < 0:   # only process hydrodynamics if this node is submerged
                    for i in range(self.nw):
                        
                        # get local 3d drag excitation force complex amplitude for each frequency [3 x nw]
                        mem.F_exc_drag[il,:,i] = np.matmul(mem.Bmat[il,:,:], mem.u[ih,il,:,i])   
                        
                        # add to global excitation vector (frequency dependent)
                        F_hydro_drag[:,i] += translateForce3to6DOF(mem.F_exc_drag[il,:,i], mem.r[il,:] - self.r6[:3])   

        self.F_hydro_drag = F_hydro_drag

        return F_hydro_drag
    
    
    
    def calcCurrentLoads(self, case):
        '''method to calculate the "static" current loads on each member and save as a current force
        Uses a simple power law relationship to calculate the current velocity as a function of member node depth'''

        rho = self.rho_water
        g   = self.g

        D_hydro = np.zeros(6)      # create variable to hold the total drag force

        # extract current variables out of the case dictionary
        speed = getFromDict(case, 'current_speed', shape=0, default=0.0)
        heading = getFromDict(case, 'current_heading', shape=0, default=0)

        
        Zref = 0.0  # reference z elevation for current profile (at the sea surface by default) (reference height set to submerged rotor hub depth if rotor is submerged)
        for ti in range(self.nrotors):
            if self.rotorList[ti].r3[2] < 0:     # If there is a submerged rotor,
                Zref = self.rotorList[ti].r3[2]  # use it for the reference current height.

        # loop through each member
        for mem in self.memberList:

            circ = mem.shape=='circular'  # convenience boolian for circular vs. rectangular cross sections

            # loop through each node of the member
            for il in range(mem.ns):

                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:

                    # calculate current velocity as a function of node depth [x,y,z] (assumes no vertical current velocity)
                    #v = speed * (((self.depth) - abs(mem.r[il,2]))/(self.depth + Zref))**self.shearExp_water
                    v = np.nanmax([ speed * (((self.depth) - abs(mem.r[il,2]))/(self.depth + Zref))**self.shearExp_water,  0])
                    #v = speed
                    vcur = np.array([v*np.cos(np.deg2rad(heading)), v*np.sin(np.deg2rad(heading)), 0])

                    # interpolate coefficients for the current strip
                    Cd_q   = np.interp( mem.ls[il], mem.stations, mem.Cd_q  )
                    Cd_p1  = np.interp( mem.ls[il], mem.stations, mem.Cd_p1 )
                    Cd_p2  = np.interp( mem.ls[il], mem.stations, mem.Cd_p2 )
                    Cd_End = np.interp( mem.ls[il], mem.stations, mem.Cd_End)

                    # current (relative) velocity over node (no complex numbers bc not function of frequency)
                    vrel = np.array(vcur)
                    # break out velocity components in each direction relative to member orientation
                    vrel_q  = np.sum(vrel*mem.q[:] )*mem.q[:]
                    vrel_p  = vrel-vrel_q 
                    vrel_p1 = np.sum(vrel*mem.p1[:])*mem.p1[:]
                    vrel_p2 = np.sum(vrel*mem.p2[:])*mem.p2[:]
                    
                    # ----- compute side effects ------------------------

                    # member acting area assigned to this node in each direction
                    a_i_q  = np.pi*mem.ds[il]*mem.dls[il]  if circ else  2*(mem.ds[il,0]+mem.ds[il,0])*mem.dls[il]
                    a_i_p1 =       mem.ds[il]*mem.dls[il]  if circ else             mem.ds[il,0]      *mem.dls[il]
                    a_i_p2 =       mem.ds[il]*mem.dls[il]  if circ else             mem.ds[il,1]      *mem.dls[il]

                    # calculate drag force wrt to each orientation using simple Morison's drag equation
                    Dq = 0.5 * rho * a_i_q * Cd_q * np.linalg.norm(vrel_q) * vrel_q

                    if circ: # Use the norm of the total perpendicular relative velocity
                        normVrel_p1 = np.linalg.norm(vrel_p)
                        normVrel_p2 = normVrel_p1
                    else: # Otherwise treat each direction separately
                        normVrel_p1 = np.linalg.norm(vrel_p1)
                        normVrel_p2 = np.linalg.norm(vrel_p2)
                    Dp1 = 0.5 * rho * a_i_p1 * Cd_p1 * normVrel_p1 * vrel_p1
                    Dp2 = 0.5 * rho * a_i_p2 * Cd_p2 * normVrel_p2 * vrel_p2
                    
                    # ----- end/axial effects drag ------

                    # end/axial area (removing sign for use as drag)
                    if circ:
                        a_i_End = np.abs(np.pi*mem.ds[il]*mem.drs[il])
                    else:
                        a_i_End = np.abs((mem.ds[il,0]+mem.drs[il,0])*(mem.ds[il,1]+mem.drs[il,1]) - (mem.ds[il,0]-mem.drs[il,0])*(mem.ds[il,1]-mem.drs[il,1]))
                    
                    Dq_End = 0.5 * rho * a_i_End * Cd_End * np.linalg.norm(vrel_q) * vrel_q

                    # ----- sum forces and add to total mean drag load about PRP ------
                    D = Dq + Dp1 + Dp2 + Dq_End     # sum drag forces at node in member's local orientation frame

                    D_hydro += translateForce3to6DOF(D, mem.r[il,:] - self.r6[:3])  # sum as forces and moments about PRP
                    
        self.D_hydro = D_hydro  # save hydro drag forces/moments to FOWT for later access

        return D_hydro


    def calcQTF_slenderBody(self, waveHeadInd, Xi0=None, verbose=False, iCase=None, iWT=None):
        '''Calculate the Quadratic Transfer Function (QTF) of the body using the slender body approximation.
           Inputs:
           - waveHeadInd: Wave heading indice from the list of headings in self.beta
           - Xi0: motion RAOs, i.e body motions per unit wave amplitude [m/m, rad/m]. If Xi0=None, the body is considered to be fixed
           - verbose: If True, print some status messages and save the QTFs (WAMIT .12d file format) and RAOs (WAMIT .4 file format)
            - iCase: case number used to name the output file 
            - iWT: wind turbine identification number (for arrays) used to name the output file         

            The resulting QTF is saved in self.qtf, which has dimensions [frequencies, frequencies, headings, DOFs].
        '''
        # TODO: I think a lot of the calculations below should be moved to the member class
        # TODO: Add references to the equations in our paper once it's published
        # TODO: Using waveHeadInd is useful within the software, but perhaps it would be also useful to have an option to specify the wave heading?

        # In case the body is fixed
        if Xi0 is None:
            Xi0 = np.zeros([self.nDOF, len(self.w)], dtype=complex)
        
        rho = self.rho_water
        g   = self.g

        # Get wave heading beta
        beta = self.beta[waveHeadInd]
        self.heads_2nd = [beta] # Need this one to print the QTF
        whead = f"{np.degrees(beta)%360:.2f}".replace('.', 'p') # String with heading in range of 0-360 [deg] for output files

        # Resample Xi0, which is input in the same frequency as the first-order loads, to the frequencies of the 2nd order hydrodynamic forces.
        # We don't use the same frequencies as the 1st order loads, self.w, for the QTFs because it would be too expensive.
        # Note that the second-order wave loads are interpolated to self.w in fowt.calcHydroForce_2ndOrd()
        Xi = np.zeros([self.nDOF, len(self.w1_2nd)], dtype=complex)
        for iDoF in range(self.nDOF):
            Xi[iDoF,:] = np.interp(self.w1_2nd, self.w, Xi0[iDoF,:], left=0, right=0)

        # Print Xi (motion RAOs) in the same format as a WAMIT .4 file
        if self.outFolderQTF is not None and verbose:
            # If both are available, use case number and fowt identification number to name the output file
            if isinstance(iCase, int) and isinstance(iWT, int):
                outPath = os.path.join(self.outFolderQTF, f"raos-slender_body_Head{whead}_Case{iCase+1}_WT{iWT}.4")
            # Otherwise, use only the heading. Would it be useful to identify the file if we only have iCase or iWT?
            else:
                outPath = os.path.join(self.outFolderQTF, f"raos-slender_body_Head{whead}.4")

            # Write the .4 file. Columns are: frequency, heading, DoF number, amplitude, phase, real part, imaginary part
            with open(outPath, "w") as f:
                ULEN = 1
                for iDoF in range(self.nDOF):
                    for w1, x in zip(self.w1_2nd, Xi[iDoF,:]):
                        f.write(f"{2*np.pi/w1: 8.4e} {beta: 8.4e} {iDoF+1} {np.abs(x): 8.4e} {np.angle(x): 8.4e} {x.real: 8.4e} {x.imag: 8.4e}\n")

        # First-order forces, which are used to compute Pinkster's IV term
        # They are taken as F_1stOrder = Mass * Acceleration_1stOrder
        F1st = np.zeros([self.nDOF, len(self.w1_2nd)], dtype=complex)
        F1st[0:3,:] = self.M_struc[0,0] * (-self.w1_2nd**2 * Xi[0:3,:])
        F1st[3:6,:] = np.matmul(self.M_struc[3:,3:], (-self.w1_2nd**2 * Xi[3:,:]))
          
        # Initialize qtf that will actually be used by the solver
        self.qtf = np.zeros([len(self.w1_2nd), len(self.w2_2nd), 1, self.nDOF], dtype=complex)  # Need this fourth dimension for conformity with the case where the QTFs are read from a file

        if verbose:
            print(f" Computing QTF for heading {beta:.2f}")

        # We start with the force component due to rotation of first-order wave forces (force component IV from Pinkster (1979)).
        # This component depends on the forces on the whole body, hence it is outside of the member loop below.
        for i1 in range(len(self.w1_2nd)):
            for i2 in range(i1, len(self.w2_2nd)):
                if self.w2_2nd[i2] < self.w1_2nd[i1]: # We don't need to fill the whole matrix, only the upper triangle
                        continue
                F_rotN = np.zeros(6, dtype='complex')    
                F_rotN[0:3] = 0.25 * (np.cross(Xi[3:,i1], np.conj(F1st[0:3,i2])) + np.cross(np.conj(Xi[3:,i2]), F1st[0:3,i1]))
                F_rotN[3: ] = 0.25 * (np.cross(Xi[3:,i1], np.conj(F1st[3: ,i2])) + np.cross(np.conj(Xi[3:,i2]), F1st[3: ,i1]))
                self.qtf[i1,i2,waveHeadInd,:] = F_rotN

        # Loop each member to compute force terms along the member
        for i,mem in enumerate(self.memberList):
            # Don't do anything if the member is above the water
            if mem.rA[2] > 0 and mem.rB[2] > 0:
                continue

            # convenience boolian for circular vs. rectangular cross sections
            circ = mem.shape=='circular'

            # Get fluid and body kinematics at each node of the member
            nodeV           = np.zeros([3, len(self.w1_2nd), mem.ns], dtype=complex)    # Node velocity
            dr              = np.zeros([3, len(self.w1_2nd), mem.ns], dtype=complex)    # Node displacement
            u               = np.zeros([3, len(self.w1_2nd), mem.ns], dtype=complex)    # Incident fluid velocity at node
            grad_u          = np.zeros([3, 3, len(self.w1_2nd), mem.ns], dtype=complex) # Gradient matrix of first-order velocity
            grad_dudt       = np.zeros([3, 3, len(self.w1_2nd), mem.ns], dtype=complex) # Gradient matrix of first-order acceleration
            nodeV_axial_rel = np.zeros([len(self.w1_2nd), mem.ns], dtype=complex)       # Node relative axial velocity
            grad_pres1st    = np.zeros([3, len(self.w1_2nd), mem.ns], dtype=complex)    # Gradient of first order pressure at each node
            for iNode, r in enumerate(mem.r):
                dr[:,:, iNode], nodeV[:,:, iNode], _ = getKinematics(r, Xi, self.w1_2nd)
                u[:,:, iNode], _, _ = getWaveKin(np.ones(self.w1_2nd.shape), beta, self.w1_2nd, self.k1_2nd, self.depth, r, len(self.w1_2nd), rho=rho, g=g)

                for iw in range(len(self.w1_2nd)):
                    grad_u[:, :, iw, iNode]    = getWaveKin_grad_u1(self.w1_2nd[iw], self.k1_2nd[iw], beta, self.depth, r)
                    grad_dudt[:, :, iw, iNode] = getWaveKin_grad_dudt(self.w1_2nd[iw], self.k1_2nd[iw], beta, self.depth, r)
                    nodeV_axial_rel[iw, iNode] = np.dot(u[:, iw, iNode]-nodeV[:,iw, iNode], mem.q)
                    grad_pres1st[:, iw, iNode] = getWaveKin_grad_pres1st(self.k1_2nd[iw], beta, self.depth, r, rho=rho, g=g)
                                                
            # Get fluid and body kinematics at the intersection of the member intersection with the water line
            eta   = np.zeros([len(self.w1_2nd)], dtype=complex)    # Incoming wave elevation
            eta_r = np.zeros([len(self.w1_2nd)], dtype=complex)    # Relative incoming wave elevation (i.e. wave elevation - vertical body motion)
            ud_wl = np.zeros([3, len(self.w1_2nd)], dtype=complex) # Incoming wave acceleration
            dr_wl = np.zeros([3, len(self.w1_2nd)], dtype=complex) # Body displacement at intersection with water line
            a_wl  = np.zeros([3, len(self.w1_2nd)], dtype=complex) # Body acceleration at intersection with water line
            if mem.r[-1,2] * mem.r[0,2] < 0: # Only if the member intersects the water line
                r_int = mem.r[0,:] + (mem.r[-1,:] - mem.r[0,:]) * (0. - mem.r[0,2]) / (mem.r[-1,2] - mem.r[0,2]) # Intersection point
                _, ud_wl, eta  = getWaveKin(np.ones(self.w1_2nd.shape), beta, self.w1_2nd, self.k1_2nd, self.depth, r_int, len(self.w1_2nd), rho=1, g=1) # getWaveKin actually returns dynamic pressure, so we use unitary rho and g to get the wave elevation
                dr_wl, _, a_wl = getKinematics(r_int, Xi, self.w1_2nd)
            
            # Vector with the acceleration of gravity projected along the p1 and p2 axes of the member (which are perpendicular to the member's axis)
            g_e1 = np.zeros([3, len(self.w1_2nd)], dtype=complex)
            for iw in range(len(self.w1_2nd)):
                g_e1[:, iw] = -g * (np.cross(Xi[3:, iw], mem.p1)[2] * mem.p1 + np.cross(Xi[3:, iw], mem.p2)[2] * mem.p2)

            # Relative wave elevation
            eta_r = eta-dr_wl[2,:]

            # Loop through each pair of frequency
            for i1, (w1, k1) in enumerate(zip(self.w1_2nd, self.k1_2nd)):
                if verbose:
                     print(f" Element {i+1} of {len(self.memberList)} - Row {i1+1:02d} of {len(self.w1_2nd):02d}", end='\r')            
                for i2, (w2, k2) in enumerate(zip(self.w2_2nd, self.k2_2nd)):
                    F_2ndPot = np.zeros(6, dtype='complex') # Force component due to second-order wave potential
                    F_conv   = np.zeros(6, dtype='complex') # Force component due to convective acceleration
                    F_axdv   = np.zeros(6, dtype='complex') # Force component due to Rainey's axial-divergence acceleration
                    F_eta    = np.zeros(6, dtype='complex') # Force component due to the relative wave elevation
                    F_nabla  = np.zeros(6, dtype='complex') # Force component due to body motions within the first-order wave field
                    F_rslb   = np.zeros(6, dtype='complex') # Force component due to Rainey's body rotation term

                    # Need to loop only half of the matrix due to symmetry (QTFs are Hermitian matrices)
                    if w2 < w1:
                        continue

                    # loop through each node of the member
                    for il in range(mem.ns):
                        # only process hydrodynamics if this node is submerged
                        if mem.r[il,2] >= 0:
                            continue

                        # interpolate coefficients for the current strip
                        Ca_p1  = np.interp( mem.ls[il], mem.stations, mem.Ca_p1 )
                        Ca_p2  = np.interp( mem.ls[il], mem.stations, mem.Ca_p2 )
                        Ca_End = np.interp( mem.ls[il], mem.stations, mem.Ca_End)

                        # ----- compute side effects ---------------------------------------------------------
                        if circ:
                            v_i = 0.25*np.pi*mem.ds[il]**2*mem.dls[il]
                        else:
                            v_i = mem.ds[il,0]*mem.ds[il,1]*mem.dls[il]  # member volume assigned to this node
                            
                        if mem.r[il,2] + 0.5*mem.dls[il] > 0:    # if member extends out of water              # <<< may want a better appraoch for this...
                            v_i = v_i * (0.5*mem.dls[il] - mem.r[il,2]) / mem.dls[il]  # scale volume by the portion that is under water

                        # Force component due to second-order wave potential
                        acc_2ndPot, p_2nd = getWaveKin_pot2ndOrd(w1, w2, k1, k2, beta, beta, self.depth, mem.r[il,:], g=g, rho=rho) # Second-order pressure will be used later
                        f_2ndPot = rho*v_i * np.matmul((1.+Ca_p1)*mem.p1Mat + (1.+Ca_p2)*mem.p2Mat, acc_2ndPot)

                        # Force component due to convective acceleration
                        conv_acc = 0.25 * ( np.matmul(grad_u[:, :, i1, il], np.conj(u[:, i2, il])) + np.matmul(np.conj(grad_u[:, :, i2, il]), u[:, i1, il]) )
                        f_conv   = rho*v_i * np.matmul((1.+Ca_p1)*mem.p1Mat + (1.+Ca_p2)*mem.p2Mat, conv_acc)

                        # Force component due to Rainey's axial-divergence acceleration
                        f_axdv   = rho*v_i * np.matmul(Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat, getWaveKin_axdivAcc(w1, w2, k1, k2, beta, beta, self.depth, mem.r[il,:], nodeV[:, i1, il], nodeV[:, i2, il], mem.q, g=g))
                        
                        # Force component due to body motions within the first-order wave field
                        acc_nabla = 0.25*np.matmul(np.squeeze(grad_dudt[:, :, i1, il]), np.squeeze(np.conj(dr[:, i2, il]))) + 0.25*np.matmul(np.squeeze(np.conj(grad_dudt[:, :, i2, il])), np.squeeze(dr[:, i1, il]))
                        f_nabla  = rho*v_i * np.matmul((1.+Ca_p1)*mem.p1Mat + (1.+Ca_p2)*mem.p2Mat, acc_nabla)

                        # Force component due to Rainey's body rotation term
                        OMEGA1  = -getH(1j*w1*Xi[3:,i1]) # The alternator matrix is the opposite of what we need, that is why we have a minus sign
                        OMEGA2  = -getH(1j*w2*Xi[3:,i2])
                        f_rslb  = -0.25*2*np.matmul(Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat, np.matmul(OMEGA1, np.conj(nodeV_axial_rel[i2, il]*mem.q)) + np.matmul(np.conj(OMEGA2), nodeV_axial_rel[i1, il]*mem.q))
                        f_rslb *= rho*v_i                                                
                        
                        # Rainey load that is only non zero for non-circular cylinders
                        # TODO: Now that it's working, should be a separate variable and not inside f_rslb
                        u1_aux  = u[:,i1, il]-nodeV[:,i1, il]
                        u2_aux  = u[:,i2, il]-nodeV[:,i2, il]
                        Vmatrix1 = getWaveKin_grad_u1(w1, k1, beta, self.depth, mem.r[il,:]) + OMEGA1
                        Vmatrix2 = getWaveKin_grad_u1(w2, k2, beta, self.depth, mem.r[il,:]) + OMEGA2
                        aux      = 0.25*(np.matmul(Vmatrix1, np.conj(np.matmul(Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat, u2_aux))) + np.matmul(np.conj(Vmatrix2), np.matmul(Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat, u1_aux)))
                        aux     -= np.matmul(mem.qMat, aux) # remove axial component                            
                        f_rslb  += rho*v_i * aux

                        # Similar to the one right above, but note that the order of multiplications with matmul is different
                        u1_aux  -= np.matmul(mem.qMat, u1_aux) # remove axial component
                        u2_aux  -= np.matmul(mem.qMat, u2_aux)
                        aux      = 0.25*(np.matmul(Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat, np.matmul(Vmatrix1, np.conj(u2_aux))) + np.matmul(Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat, np.matmul(np.conj(Vmatrix2), u1_aux)))
                        f_rslb  += - rho*v_i * aux

                                                    
                        # ----- end effects  ------                               
                        # note : v_i and a_i work out to zero for non-tapered sections or non-end sections
                        # TODO: Should probably skip this part if along the length of the element, i.e. if a_i and v_i are different from zero due to tapering
                        if circ:
                            v_i = np.pi/12.0 * abs((mem.ds[il]+mem.drs[il])**3 - (mem.ds[il]-mem.drs[il])**3)  # volume assigned to this end surface
                        else:
                            v_i = np.pi/12.0 * ((np.mean(mem.ds[il]+mem.drs[il]))**3 - (np.mean(mem.ds[il]-mem.drs[il]))**3)    # so far just using sphere eqn and taking mean of side lengths as d
                        
                        f_2ndPot += mem.a_i[il]*p_2nd*mem.q # 2nd order pressure
                        f_2ndPot += rho*v_i*Ca_End*np.matmul(mem.qMat, acc_2ndPot) # 2nd order axial acceleration
                        f_conv   += rho*v_i*Ca_End*np.matmul(mem.qMat, conv_acc)   # convective acceleration
                        f_nabla  += rho*v_i*Ca_End*np.matmul(mem.qMat, acc_nabla)  # due to body motions within the first-order wave field - acceleration part
                        p_nabla   = 0.25*np.dot(grad_pres1st[:, i1, il], np.conj(dr[:, i2, il])) + 0.25*np.dot(np.conj(grad_pres1st[:, i2, il]), dr[:, i1, il])
                        f_nabla  += mem.a_i[il]*p_nabla*mem.q # due to body motions within the first-order wave field - pressure part
                        p_drop    = -2*0.25*0.5*rho*np.dot(np.matmul(mem.p1Mat + mem.p2Mat, u[:,i1, il]-nodeV[:, i1, il]), np.conj(np.matmul(Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat, u[:,i2, il]-nodeV[:, i2, il])))
                        f_conv   += mem.a_i[il]*p_drop*mem.q

                        u1_aux    = np.matmul(Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat, u1_aux)
                        u2_aux    = np.matmul(Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat, u2_aux)
                        f_transv  = 0.25*mem.a_i[il]*rho*(np.conj(u1_aux)*nodeV_axial_rel[i2, il] + u2_aux*np.conj(nodeV_axial_rel[i1, il]))
                        f_conv   += f_transv

                        F_2ndPot += translateForce3to6DOF(f_2ndPot, mem.r[il,:])
                        F_conv   += translateForce3to6DOF(f_conv, mem.r[il,:])                   
                        F_axdv   += translateForce3to6DOF(f_axdv, mem.r[il,:])
                        F_nabla  += translateForce3to6DOF(f_nabla, mem.r[il,:])
                        F_rslb   += translateForce3to6DOF(f_rslb, mem.r[il,:])
                                                        
                    # Force acting at the intersection of the member with the mean waterline
                    f_eta = np.zeros(3, dtype=complex)
                    F_eta = np.zeros(6, dtype=complex)
                    if mem.r[-1,2] * mem.r[0,2] < 0:
                        # Need just the cross section area, as the length along the cylinder is the relative wave elevation
                        # Get the area of the cross section at the mean waterline
                        i_wl = np.where(mem.r[:,2] < 0)[0][-1] # find index of mem.r[:,2] that is right before crossing 0
                        if circ:
                            if i_wl != len(mem.ds)-1:
                                d_wl = 0.5*(mem.ds[i_wl]+mem.ds[i_wl+1])
                            else:
                                d_wl = mem.ds[i_wl]
                            a_i = 0.25*np.pi*d_wl**2
                        else:
                            if i_wl != len(mem.ds)-1:
                                d1_wl = 0.5*(mem.ds[i_wl,0]+mem.ds[i_wl+1,0])
                                d2_wl = 0.5*(mem.ds[i_wl,1]+mem.ds[i_wl+1,1])
                            else:
                                d1_wl = mem.ds[i_wl, 0]
                                d2_wl = mem.ds[i_wl, 1]
                            a_i = d1_wl*d2_wl

                        f_eta = 0.25*((ud_wl[:,i1])*np.conj(eta_r[i2])+np.conj((ud_wl[:,i2]))*eta_r[i1]) # Product between incoming wave acceleration and relative wave elevation
                        f_eta = rho*a_i*np.matmul((1.+Ca_p1)*mem.p1Mat + (1.+Ca_p2)*mem.p2Mat, f_eta) # Project in the correct directions (and scale by the area and density)
                        a_eta = 0.25*((a_wl[:,i1])*np.conj(eta_r[i2])+np.conj((a_wl[:,i2]))*eta_r[i1])  # Product between body acceleration and relative wave elevation
                        f_eta -= rho*a_i*np.matmul(Ca_p1*mem.p1Mat + Ca_p2*mem.p2Mat, a_eta) # Project in the correct directions (and scale by the area and density)
                        f_eta -= 0.25*rho*a_i * (g_e1[:,i1]*np.conj(eta_r[i2])+np.conj(g_e1[:,i2])*eta_r[i1]) # Add hydrostatic term

                        F_eta = translateForce3to6DOF(f_eta, r_int) # Load vector in 6 DOF 
                    
                    # Total contribution to this frequency pair of the QTF due to the current member
                    self.qtf[i1,i2,waveHeadInd,:] += F_2ndPot + F_axdv + F_conv + F_nabla + F_eta + F_rslb

                    # Add Kim and Yue correction                                                     
                    self.qtf[i1,i2,waveHeadInd,:] += mem.correction_KAY(self.depth, w1, w2, beta, rho=rho, g=g, k1=k1, k2=k2, Nm=10)

        # We just filled the upper triangle of the QTF matrices above. We exploit the hermitian symmetry of the QTFs to fill the lower triangle
        for i in range(self.nDOF):
            self.qtf[:,:, waveHeadInd, i] = self.qtf[:,:, waveHeadInd,i] + np.conj(self.qtf[:,:, waveHeadInd,i]).T - np.diag(np.diag(np.conj(self.qtf[:,:, waveHeadInd,i])))

        # Save the QTF to a .12d file
        if self.outFolderQTF is not None and verbose:
            if isinstance(iCase, int) and isinstance(iWT, int):
                outPath = os.path.join(self.outFolderQTF, f"qtf-slender_body-total_Head{whead}_Case{iCase+1}_WT{iWT}.12d")
            else:
                outPath = os.path.join(self.outFolderQTF, f"qtf-slender_body-total_Head{whead}.12d")
            self.writeQTF(self.qtf, outPath)


    def readQTF(self, flPath, ULEN=1):
        '''Read a complex QTF matrix from a text file following WAMIT .12d file format
        Inputs:
        - flPath: path to the .12d file with the QTFs. Assumed to be written as a function of wave periods.
        - ULEN: length used to dimensionalize the QTFs. Default is 1.

        Outputs:
        Doesn't return anything, but stores data in
        - self.heads_2nd: wave headings [deg] - only unidirectional QTFs are supported for now.
        - self.w1_2nd: first frequency vector [rad/s]
        - self.w2_2nd: second frequency vector [rad/s]. Both vectors have to be the same. TODO: do we need to keep both? I don't think we will have different w1 and w2 in any case of interest.
        - self.qtf: complex QTF matrix with dimensions [number of frequencies, number of frequencies, number of headings, number of dofs]
        '''
        data = np.loadtxt(flPath) 
        data[:,0:2] = 2.*np.pi/data[:,0:2] # Input is assumed to be as a function of wave period

        # Consider only unidirectional QTFs
        if not (data[:,2] == data[:,3]).all():
            raise ValueError("Only unidirectional QTFs are supported for now.")
        self.heads_2nd = deg2rad(np.sort(np.unique(data[:,2])))
        nheads = len(self.heads_2nd)

        # Both frequency vectors should contain the same frequencies,
        # but they are not necessarily the same as self.w 
        self.w1_2nd = np.unique(data[:,0])  
        self.w2_2nd = np.unique(data[:,1])
        nw1 = len(self.w1_2nd)
        nw2 = len(self.w2_2nd)
        if not (self.w1_2nd==self.w2_2nd).all():
            raise ValueError("Both frequency columns in the input QTF must contain the same values.")
        
        self.qtf = np.zeros([nw1, nw2, nheads, self.nDOF], dtype=complex)                
        for row in data:
            indw1, = np.where(self.w1_2nd==row[0]) # index for first frequency
            indw2, = np.where(self.w2_2nd==row[1]) # index for second frequency
            indhead, = np.where(self.heads_2nd==row[2]) # index for heading
            indDOF = round(row[4]-1) # index for degree of freedom. Needs to be an int. -1 is due to being from 1 to 6 in the input file

            # Factor for dimensionalization (except for wave amplitudes, which are assumed unitary for QTFs)
            factor = self.rho_water * self.g * ULEN
            if indDOF >= 3:
                factor *= ULEN # Need an extra ULEN for moments
            self.qtf[indw1[0], indw2[0], indhead[0], indDOF] = factor*(row[7]+1j*row[8])

            # The data in a .12d only fills one half of the matrix (upper or lower triangle)
            # Fill the other half of the matrix (which is equal to the conjugate of its symmetric element - QTFs are hermitian matrices)
            if not indw1[0] == indw2[0]:
                self.qtf[indw2[0], indw1[0], indhead[0], indDOF] = factor*(row[7]-1j*row[8])


    def writeQTF(self, qtfIn, outPath, w=None):
        '''Writes a qtf matrix to a text file following WAMIT .12d file format
        Inputs:
        - qtfIn: complex QTF matrix with dimensions [number of frequencies, number of frequencies, number of headings, number of dofs] (same as self.qtf)
        - outPath: path to the output file
        - w: frequency vector with size equal to the number of frequencies. If not specified, self.w1_2nd and self.w2_2nd are used.
        '''
        if w is None:
            w1 = self.w1_2nd
            w2 = self.w2_2nd
        else:
            w1 = w
            w2 = w

        with open(outPath, "w") as f:
            ULEN = 1 # ULEN is the length used to dimensionalize the QTFs. Taken as 1.
            for ih in range(len(self.heads_2nd)):
                for iDoF in range(self.nDOF):
                    qtf = qtfIn[:,:,ih,iDoF]
                    for i1 in range(len(w1)):
                        for i2 in range(i1, len(w2)):
                            F = qtf[i1,i2]/(self.rho_water*self.g*ULEN)

                            # .12d format, following columns: w1, w2, heading 1, heading 1, DoF, abs(F), phase(F), real(F), imag(F)
                            f.write(f"{2*np.pi/w1[i1]: 8.4e} {2*np.pi/w2[i2]: 8.4e} {rad2deg(self.heads_2nd[ih]): 8.4e} {rad2deg(self.heads_2nd[ih]): 8.4e} {iDoF+1} {np.abs(F): 8.4e} {np.angle(F): 8.4e} {F.real: 8.4e} {F.imag: 8.4e}\n")
                        

    def calcHydroForce_2ndOrd(self, beta, S0, iCase=None, iWT=None, interpMode='qtf'):
        ''' Compute force due to 2nd order hydrodynamic loads based on the second-order load spectrum.
        See Pinkster (1980), Section IV.3.
        With this approach, we lose the phases between force components. In the future, we should at least
        correct the phase of the incoming waves to account for different positions in arrays.
        Inputs:
        - beta: wave heading [rad], assumed the same for all wave components (long-crested waves)
        - S0: wave spectrum [m^2.s/rad] at the frequencies self.w
        - iCase: case number used to name the output file 
        - iWT: wind turbine identification number (for arrays) used to name the output file
        - interpMode: interpolation mode for the QTF. Options are 'spectrum' and 'qtf', see comments below for details.

        Outputs:
        - f_mean: mean force amplitude, numpy array with size equal [self.nDOF] (real values)
        - f: difference-frequency force amplitude at different frequencies, numpy array with size equal [self.nDOF, self.nw] (complex values)
        '''
        f = np.zeros([self.nDOF, self.nw], dtype=complex) # Force amplitude at different frequencies
        f_mean = np.zeros([self.nDOF]) # Mean force amplitude

        # Interpolate for the wave incidence        
        if beta < self.heads_2nd[0]: 
            print(f"Warning in calcHydroForce_2ndOrd: angle {beta} is less than the minimum incidence angle in the QTF. An incidence of {self.heads_2nd[0]} will be considered for 2nd order loads.")
        if beta > self.heads_2nd[-1]:
            print(f"Warning in calcHydroForce_2ndOrd: angle {beta} is more than the maximum incidence angle in the QTF. An incidence of {self.heads_2nd[-1]} will be considered for 2nd order loads.")
        if len(self.heads_2nd)==1: # If there is only one heading, no need to interpolate. The warnings above already tell the user if the required heading is out of range.
            qtf_interpBeta = self.qtf[:,:,0,:]
        else:
            qtf_interpBetaRe = interp1d(self.heads_2nd, self.qtf, assume_sorted=True, axis=2, bounds_error=False, fill_value=(self.qtf[:,:,0,:], self.qtf[:,:,-1,:].real))(beta)
            qtf_interpBetaIm = interp1d(self.heads_2nd, self.qtf, assume_sorted=True, axis=2, bounds_error=False, fill_value=(self.qtf[:,:,0,:], self.qtf[:,:,-1,:].imag))(beta)
            qtf_interpBeta   = qtf_interpBetaRe + 1j*qtf_interpBetaIm

        # Compute force spectrum with QTF resolution and then interpolate to the frequency vector of the input wave spectrum
        if interpMode == 'spectrum':
            # Resample wave spectrum (the input is assumed to be in rad/s)
            nw1 = len(self.w1_2nd)
            S = np.interp(self.w1_2nd, self.w, S0, left=0, right=0) 

            mu = self.w1_2nd - self.w1_2nd[0] # The number of difference frequencies is the same as the number of frequencies, but starting from frequency mu=0.
            Sf = np.zeros([self.nDOF, nw1]) # Second-order force spectrum
            Sf_interp = np.zeros([self.nDOF, self.nw]) # Second-order force spectrum interpolated to the same resolution used for the wave spectrum (and body dynamics)
            for idof in range(0,self.nDOF):                    
                for imu in range(1, nw1): # Loop the difference frequencies (this is why we start from 1)
                    Saux = np.zeros(nw1)
                    Saux[0:nw1-imu] = S[imu:] # Auxiliar wave spectrum that is dislocated in frequency. See the definition of second-order force spectrum
                    Qaux = np.zeros(nw1, dtype=complex) # Auxiliar variable to get the part the QTF diagonal that we need (one different diagonal for each difference frequency) 

                    Qaux[0:nw1-imu] = np.diag(np.squeeze(qtf_interpBeta[:,:,idof]), imu) # We use only the upper half of the QTF (exploiting hermitian symmetry)
                    Sf[idof, imu] = 8 * np.sum(S*Saux*np.abs(Qaux)**2) * (self.w1_2nd[1]-self.w1_2nd[0])
                            
                # Mean drift uses a simpler expression because you have just the product of the same wave
                f_mean[idof] = 2*np.sum(S*np.diag(np.squeeze(qtf_interpBeta[:,:,idof].real), 0)) * (self.w1_2nd[1]-self.w1_2nd[0])

                # Interpolate the force spectrum to the same resolution as the original wave spectrum
                Sf_interp[idof, :] = np.interp(self.w - self.w[0], mu, Sf[idof,:], left=0, right=0)

                # Compute the force amplitude from the force spectrum. Losing phases between force components!
                f[idof, :] = np.sqrt(2*Sf_interp[idof, :]*self.dw)
                
        # Otherwise, interpolate the QTF first and then compute the force spectrum and amplitude.
        # We got better results with this approach for the tests we have done so far, so this is why it is the default
        else:
            f = np.zeros([self.nDOF, self.nw]) # Force amplitude
            for idof in range(0,self.nDOF):
                qtf_interp_Re_interpolator = RegularGridInterpolator((self.w1_2nd, self.w1_2nd), qtf_interpBeta[:, :, idof].real, bounds_error=False, fill_value=0)
                qtf_interp_Im_interpolator = RegularGridInterpolator((self.w1_2nd, self.w1_2nd), qtf_interpBeta[:, :, idof].imag, bounds_error=False, fill_value=0)

                w_mesh = np.meshgrid(self.w, self.w, indexing='ij')
                points = np.array([w_mesh[0].ravel(), w_mesh[1].ravel()]).T

                qtf_interp_Re = qtf_interp_Re_interpolator(points).reshape(len(self.w), len(self.w))
                qtf_interp_Im = qtf_interp_Im_interpolator(points).reshape(len(self.w), len(self.w))
                qtf_interp = qtf_interp_Re + 1j * qtf_interp_Im

                for imu in range(1, self.nw): # Loop the difference frequencies
                    Saux = np.zeros(self.nw)
                    Saux[0:self.nw-imu] = S0[imu:] # Auxiliar wave spectrum that is dislocated in frequency. See the definition of second-order force spectrum
                    Qaux = np.zeros(self.nw, dtype=complex) # Auxiliar variable to get the part the QTF diagonal that we need (one different diagonal for each difference frequency)
                    Qaux[0:self.nw-imu] = np.diag(np.squeeze(qtf_interp), imu) # Sum only the upper half of the QTF
                    f[idof, imu] = 4 *np.sqrt( np.sum(S0*Saux*np.abs(Qaux)**2) ) * self.dw # We use only the upper half of the QTF (exploiting hermitian symmetry)

                # Mean drift uses a simpler expression because you have just the product of the same wave
                f_mean[idof] = 2*np.sum(S0*np.diag(np.squeeze(qtf_interp.real), 0)) * self.dw

        # Displace f by one frequency so that it aligns with the frequency vector that is used to solve body dynamics.
        # Need to do that because the difference frequencies start at 0rad/s and end at w[-2], while the wave spectrum
        # and body dynamics start at w[0]=dw and end at w[-1].
        f[:, 0:-1] = f[:, 1:]
        f[:, -1] = 0 # We don't have loads at the last frequency because the difference frequencies end at w[-2]

        # Write force amplitudes to an output file 
        if self.outFolderQTF is not None:             
            with open(os.path.join(self.outFolderQTF, f'f_2nd-_Case{ iCase+1 }_WT{ iWT }.txt'), 'w') as file:
                for w, frow in zip(self.w, f.T):
                    file.write(f'{w:.5f} {frow[0]:.5f} {frow[1]:.5f} {frow[2]:.5f} {frow[3]:.5f} {frow[4]:.5f} {frow[5]:.5f}\n')

        return f_mean, f


    def saveTurbineOutputs(self, results, case):
        '''Calculate and store output metrics of the FOWT response at the current load case.
        Note that the FOWT offset, motions, load case info, etc. are taken from what is stored
        in the FOWT object. I.e. >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   
        Load cases may have multiple sources of excitation (such as wind, wave, and another wave
        at a different heading). Results are computed by RMS summing across these excitation sources.
        '''
        
        self.Xi0 = self.r6 - np.array([self.x_ref, self.y_ref,0,0,0,0])  # FOWT's mean offset vector [m, rad]

        # platform motions
        results['surge_avg'] = self.Xi0[0]
        results['surge_std'] = getRMS(self.Xi[:,0,:]) 
        results['surge_max'] = self.Xi0[0] + 3*results['surge_std']
        results['surge_min'] = self.Xi0[0] - 3*results['surge_std']
        results['surge_PSD'] = getPSD(self.Xi[:,0,:], self.dw)
        results['surge_RA' ] = self.Xi[:,0,:]
        
        results['sway_avg'] = self.Xi0[1]
        results['sway_std'] = getRMS(self.Xi[:,1,:])
        results['sway_max'] = self.Xi0[1] + 3*results['sway_std']
        results['sway_min'] = self.Xi0[1] - 3*results['sway_std']
        results['sway_PSD'] = getPSD(self.Xi[:,1,:], self.dw)
        results['sway_RA' ] = self.Xi[:,1,:]
        
        results['heave_avg'] = self.Xi0[2]
        results['heave_std'] = getRMS(self.Xi[:,2,:])
        results['heave_max'] = self.Xi0[2] + 3*results['heave_std']
        results['heave_min'] = self.Xi0[2] - 3*results['heave_std']
        results['heave_PSD'] = getPSD(self.Xi[:,2,:], self.dw)
        results['heave_RA' ] = self.Xi[:,2,:]
        
        roll_deg = rad2deg(self.Xi[:,3,:])
        results['roll_avg'] = rad2deg(self.Xi0[3])
        results['roll_std'] = getRMS(roll_deg)
        results['roll_max'] = rad2deg(self.Xi0[3]) + 3*results['roll_std']
        results['roll_min'] = rad2deg(self.Xi0[3]) - 3*results['roll_std']
        results['roll_PSD'] = getPSD(roll_deg, self.dw)
        results['roll_RA' ] = rad2deg(self.Xi[:,3,:])
        
        pitch_deg = rad2deg(self.Xi[:,4,:])
        results['pitch_avg'] = rad2deg(self.Xi0[4])
        results['pitch_std'] = getRMS(pitch_deg)
        results['pitch_max'] = rad2deg(self.Xi0[4]) + 3*results['pitch_std']
        results['pitch_min'] = rad2deg(self.Xi0[4]) - 3*results['pitch_std']
        results['pitch_PSD'] = getPSD(pitch_deg, self.dw)
        results['pitch_RA' ] = rad2deg(self.Xi[:,4,:])
        
        yaw_deg = rad2deg(self.Xi[:,5,:])
        results['yaw_avg'] = rad2deg(self.Xi0[5])
        results['yaw_std'] = getRMS(yaw_deg)
        results['yaw_max'] = rad2deg(self.Xi0[5]) + 3*results['yaw_std']
        results['yaw_min'] = rad2deg(self.Xi0[5]) - 3*results['yaw_std']
        results['yaw_PSD'] = getPSD(yaw_deg, self.dw)
        results['yaw_RA' ] = rad2deg(self.Xi[:,5,:])
        
        # ----- turbine-level mooring outputs (similar code as array-level) -----
        if self.ms:
            nLines = len(self.ms.lineList)
            T_moor_amps = np.zeros([self.nWaves+1, 2*nLines, self.nw], dtype=complex)  # mooring tension amplitudes for each excitation source and line end
            C_moor, J_moor = self.ms.getCoupledStiffness(lines_only=True, tensions=True) # get stiffness matrix and tension jacobian matrix
            T_moor = self.ms.getTensions()  # get line end mean tensions
            
            for ih in range(self.nWaves+1):
                for iw in range(self.nw):
                    T_moor_amps[ih,:,iw] = np.matmul(J_moor, self.Xi[ih,:,iw])   # FFT of mooring tensions
        
            results['Tmoor_avg'] = T_moor
            results['Tmoor_std'] = np.zeros(2*nLines)
            results['Tmoor_max'] = np.zeros(2*nLines)
            results['Tmoor_min'] = np.zeros(2*nLines)
            results['Tmoor_PSD'] = np.zeros([ 2*nLines, self.nw])
            for iT in range(2*nLines):
                TRMS = getRMS(T_moor_amps[:,iT,:]) # estimated mooring line RMS tension [N]
                results['Tmoor_std'][iT] = TRMS
                results['Tmoor_max'][iT] =  T_moor[iT] + 3*TRMS
                results['Tmoor_min'][iT] =  T_moor[iT] - 3*TRMS
                results['Tmoor_PSD'][iT, :] = (getPSD(T_moor_amps[:,iT,:], self.w[0])) # PSD in N^2/(rad/s)
        
        # hub fore-aft displacement amplitude and acceleration (used as an approximation in a number of outputs)
        XiHub = np.zeros([self.Xi.shape[0], self.nrotors, self.nw], dtype=complex)
        results['AxRNA_std'] = np.zeros(self.nrotors) 
        results['AxRNA_PSD'] = np.zeros([self.nw, self.nrotors]) 
        results['AxRNA_avg'] = np.zeros(self.nrotors)
        results['AxRNA_max'] = np.zeros(self.nrotors)
        results['AxRNA_min'] = np.zeros(self.nrotors)
        
        for ir, rotor in enumerate(self.rotorList):
            XiHub[:,ir,:] = self.Xi[:,0,:] + rotor.r_rel[2]*self.Xi[:,4,:]  # planar approximation; to improve <<<
        
            # nacelle acceleration
            results['AxRNA_std'][ir] = getRMS(XiHub[:,ir,:]*self.w**2)
            results['AxRNA_PSD'][:,ir] = (getPSD(XiHub[:,ir,:]*self.w**2, self.dw))
            results['AxRNA_avg'][ir] = abs(np.sin(self.Xi0[4])*9.81) # @Matt check this! 
            results['AxRNA_max'][ir] = results['AxRNA_avg'][ir]+3*results['AxRNA_std'][ir]
            results['AxRNA_min'][ir] = results['AxRNA_avg'][ir]-3*results['AxRNA_std'][ir]
            
        # tower base bending moment  >>> should three-dimensionalize this <<<
        m_turbine = np.zeros(len(self.mtower))
        zCG_turbine = np.zeros_like(m_turbine)
        zBase = np.zeros_like(m_turbine)
        hArm = np.zeros_like(m_turbine)
        
        aCG_turbine = np.zeros_like(XiHub, dtype=complex)
        ICG_turbine = np.zeros_like(m_turbine)
        M_I            = np.zeros_like(XiHub)
        M_w            = np.zeros_like(XiHub)
        M_X_aero       = np.zeros_like(XiHub)
        dynamic_moment = np.zeros_like(XiHub)
        dynamic_moment_RMS = np.zeros(self.nrotors)
        
        
        results['Mbase_avg'] = np.zeros(self.nrotors)
        results['Mbase_std'] = np.zeros(self.nrotors)
        results['Mbase_PSD'] = np.zeros([self.nw, self.nrotors])
        results['Mbase_max'] = np.zeros(self.nrotors)
        results['Mbase_min'] = np.zeros(self.nrotors)
        
        for ir, rotor in enumerate(self.rotorList):
            # mass and moment arm >>> should three-dimensionalize <<<
            m_turbine[ir] = self.mtower[ir] + rotor.mRNA  # total masses of each turbine
            zCG_turbine[ir] = (self.rCG_tow[ir][2]*self.mtower[ir]                 # CoG of each turbine
                                + rotor.r_rel[2]*rotor.mRNA)/m_turbine[ir]
            zBase[ir] = self.memberList[self.nplatmems + ir].rA[2]                  # tower base elevation [m]
            hArm[ir] = zCG_turbine[ir] - zBase[ir]                                 # vertical distance from tower base to turbine CG [m]

            aCG_turbine[:,ir,:] = -self.w**2 *( self.Xi[:,0,:] + zCG_turbine[ir]*self.Xi[:,4,:] )  # fore-aft acceleration of turbine CG

            # turbine pitch moment of inertia about CG [kg-m^2]
            ICG_turbine[ir] = (translateMatrix6to6DOF(self.memberList[self.nplatmems+ir].M_struc, [0,0,-zCG_turbine[ir]])[4,4] # tower MOI about turbine CG
                        + rotor.mRNA*(rotor.r_rel[2]-zCG_turbine[ir])**2 + rotor.IrRNA)  # RNA MOI with parallel axis theorem
            # moment components and summation (all complex amplitudes)
            M_I[:,ir,:] = -m_turbine[ir]*aCG_turbine[:,ir,:]*hArm[ir] - ICG_turbine[ir]*(-self.w**2 *self.Xi[:,4,:] ) # tower base inertial reaction moment
            M_w[:,ir,:] =  m_turbine[ir]*self.g * hArm[ir]*self.Xi[:,4]                                    # tower base weight moment
            
            M_F_aero = 0.0 # <<<<self.f_aero[0,:]*(self.hHub - zBase)  # tower base moment from turbulent wind excitation  <<<<<<<<<<<<<
            
            M_X_aero[:,ir,:] = -(-self.w**2 *self.A_aero[0,0,:,ir]                                 # tower base aero reaction moment
                        + 1j*self.w *self.B_aero[0,0,:,ir] )*(rotor.r_rel[2] - zBase[ir])**2 *self.Xi[:,4,:]        
            dynamic_moment[:,ir,:] = M_I[:,ir,:] + M_w[:,ir,:] + M_F_aero + M_X_aero[:,ir,:]       # total tower base fore-aft bending moment [N-m]
            dynamic_moment_RMS[ir] = getRMS(dynamic_moment[:,ir,:])

            # fill in metrics
            # mean moment from weight and thrust
            results['Mbase_avg'][ir] = (m_turbine[ir]*self.g * hArm[ir]*np.sin(self.Xi0[4]) 
                          + transformForce(self.f_aero0[:,ir], offset=[0,0,-hArm[ir]])[4] )
            results['Mbase_std'][ir] = dynamic_moment_RMS[ir]
            results['Mbase_PSD'][:,ir] = (getPSD(dynamic_moment[:,ir,:], self.dw))
            results['Mbase_max'][ir] = results['Mbase_avg'][ir]+3*results['Mbase_std'][ir]
            results['Mbase_min'][ir] = results['Mbase_avg'][ir]-3*results['Mbase_std'][ir]
            #results['Mbase_DEL'][iCase]
        
        # wave PSD for reference
        results['wave_PSD'] = getPSD(self.zeta, self.dw)        # wave elevation spectrum

        # initialize complex amplitudes for rotor response
        phi_w    = np.zeros([self.nWaves+1, self.nrotors, self.nw], dtype=complex)  # rotor azimuth deviation 
        omega_w  = np.zeros([self.nWaves+1, self.nrotors, self.nw], dtype=complex)  # rotor speed deviation
        torque_w = np.zeros([self.nWaves+1, self.nrotors, self.nw], dtype=complex)  # generator torque
        bPitch_w = np.zeros([self.nWaves+1, self.nrotors, self.nw], dtype=complex)  # blade pitch


        results['omega_avg']  = np.zeros(self.nrotors)
        results['omega_std']  = np.zeros(self.nrotors)
        results['omega_max']  = np.zeros(self.nrotors)
        results['omega_min']  = np.zeros(self.nrotors)
        results['omega_PSD']  = np.zeros([self.nw, self.nrotors])
        results['torque_avg'] = np.zeros(self.nrotors)
        results['torque_std'] = np.zeros(self.nrotors)
        results['torque_PSD'] = np.zeros([self.nw, self.nrotors])
        results['power_avg']  = np.zeros(self.nrotors)
        results['bPitch_avg'] = np.zeros(self.nrotors)
        results['bPitch_std'] = np.zeros(self.nrotors)
        results['bPitch_PSD'] = np.zeros([self.nw, self.nrotors])
        
        
        for ir, rot in enumerate(self.rotorList):
        
            # get inflow speed for wind or current turbine
            if rot.r3[2] < 0:
                speed = getFromDict(case, 'current_speed', shape=0, default=1.0)
            else:
                speed = getFromDict(case, 'wind_speed', shape=0, default=10.0)
        
            # rotor-related outputs are only available if aerodynamics modeling is enabled
            if rot.aeroServoMod > 1 and speed > 0.0:
            
                # compute spectra of rotor azimuth variation, rotor speed, generator torque, and blade pitch
                for ih in range(self.nWaves):
                    phi_w[ih,ir,:] = rot.C * XiHub[ih,ir,:]
                
                phi_w[-1,ir,:] = rot.C * (XiHub[-1,ir,:] - rot.V_w / (1j *self.w))
                
                # TODO
                omega_w[ :,ir,:] =  1j*self.w * phi_w[:,ir,:]
                torque_w[:,ir,:] = (1j*self.w * rot.kp_tau  + rot.ki_tau ) * phi_w[:,ir,:]
                bPitch_w[:,ir,:] = (1j*self.w * rot.kp_beta + rot.ki_beta) * phi_w[:,ir,:]
                
                # rotor speed (rpm) 
                results['omega_avg'][ir] = rot.Omega_case
                results['omega_std'][ir] = radps2rpm(getRMS(omega_w[:,ir,:]))
                # note: _max values are (avg + 2 or 3 * std)   (95% or 99% max)
                results['omega_max'][ir] = results['omega_avg'][ir] + 2 * results['omega_std'][ir]
                results['omega_min'][ir] = results['omega_avg'][ir] - 2 * results['omega_std'][ir]
                results['omega_PSD'][:,ir] = radps2rpm(1)**2 * getPSD(omega_w[:,ir,:], self.dw)
                
                # generator torque (Nm)
                results['torque_avg'][ir] = rot.aero_torque / rot.Ng
                results['torque_std'][ir] = getRMS(torque_w[:,ir,:])
                results['torque_PSD'][:,ir] = getPSD(torque_w[:,ir,:], self.dw)
                # results['torque_max'][iCase]    # skip, nonlinear
                
                # rotor power (W)
                results['power_avg'][ir] = rot.aero_power  # compute from cc-blade coeffs
                # results['power_std'][iCase]     # nonlinear near rated, covered by torque_ and omega_std
                # results['power_max'][iCase]     # skip, nonlinear
                
                # collective blade pitch (deg)
                results['bPitch_avg'][ir] = rot.pitch_case
                results['bPitch_std'][ir] = rad2deg(getRMS(bPitch_w[:,ir,:]))
                results['bPitch_PSD'][:,ir] = rad2deg(1)**2 *getPSD(bPitch_w[:,ir,:], self.dw)
                # results['bPitch_max'][iCase]    # skip, not something we'd consider in design
                
                # wind PSD for reference
                results['wind_PSD'] = getPSD(rot.V_w, self.dw)   # <<< need to confirm

            if rot.r3[2] < 0:
                if len(self.cav) > 0:
                    results['cavitation'] = self.cav

        '''
        Outputs from OpenFAST to consider covering:

        # Rotor power outputs
        self.add_output('V_out', val=np.zeros(n_ws_dlc11), units='m/s', desc='wind speed vector from the OF simulations')
        self.add_output('P_out', val=np.zeros(n_ws_dlc11), units='W', desc='rotor electrical power')
        self.add_output('Cp_out', val=np.zeros(n_ws_dlc11), desc='rotor aero power coefficient')
        self.add_output('Omega_out', val=np.zeros(n_ws_dlc11), units='rpm', desc='rotation speeds to run')
        self.add_output('pitch_out', val=np.zeros(n_ws_dlc11), units='deg', desc='pitch angles to run')
        self.add_output('AEP', val=0.0, units='kW*h', desc='annual energy production reconstructed from the openfast simulations')

        self.add_output('My_std',      val=0.0,            units='N*m',  desc='standard deviation of blade root flap bending moment in out-of-plane direction')
        self.add_output('flp1_std',    val=0.0,            units='deg',  desc='standard deviation of trailing-edge flap angle')

        self.add_output('rated_V',     val=0.0,            units='m/s',  desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0,            units='rpm',  desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0,            units='deg',  desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0,            units='N',    desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0,            units='N*m',  desc='rotor aerodynamic torque at rated')

        self.add_output('loads_r',      val=np.zeros(n_span), units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads_Px',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads_Py',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads_Pz',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_output('loads_Omega',  val=0.0, units='rpm', desc='rotor rotation speed')
        self.add_output('loads_pitch',  val=0.0, units='deg', desc='pitch angle')
        self.add_output('loads_azimuth', val=0.0, units='deg', desc='azimuthal angle')

        # Control outputs
        self.add_output('rotor_overspeed', val=0.0, desc='Maximum percent overspeed of the rotor during an OpenFAST simulation')  # is this over a set of sims?

        # Blade outputs
        self.add_output('max_TipDxc', val=0.0, units='m', desc='Maximum of channel TipDxc, i.e. out of plane tip deflection. For upwind rotors, the max value is tower the tower')
        self.add_output('max_RootMyb', val=0.0, units='kN*m', desc='Maximum of the signals RootMyb1, RootMyb2, ... across all n blades representing the maximum blade root flapwise moment')
        self.add_output('max_RootMyc', val=0.0, units='kN*m', desc='Maximum of the signals RootMyb1, RootMyb2, ... across all n blades representing the maximum blade root out of plane moment')
        self.add_output('max_RootMzb', val=0.0, units='kN*m', desc='Maximum of the signals RootMzb1, RootMzb2, ... across all n blades representing the maximum blade root torsional moment')
        self.add_output('DEL_RootMyb', val=0.0, units='kN*m', desc='damage equivalent load of blade root flap bending moment in out-of-plane direction')
        self.add_output('max_aoa', val=np.zeros(n_span), units='deg', desc='maxima of the angles of attack distributed along blade span')
        self.add_output('std_aoa', val=np.zeros(n_span), units='deg', desc='standard deviation of the angles of attack distributed along blade span')
        self.add_output('mean_aoa', val=np.zeros(n_span), units='deg', desc='mean of the angles of attack distributed along blade span')
        # Blade loads corresponding to maximum blade tip deflection
        self.add_output('blade_maxTD_Mx', val=np.zeros(n_span), units='kN*m', desc='distributed moment around blade-aligned x-axis corresponding to maximum blade tip deflection')
        self.add_output('blade_maxTD_My', val=np.zeros(n_span), units='kN*m', desc='distributed moment around blade-aligned y-axis corresponding to maximum blade tip deflection')
        self.add_output('blade_maxTD_Fz', val=np.zeros(n_span), units='kN', desc='distributed force in blade-aligned z-direction corresponding to maximum blade tip deflection')

        # Hub outputs
        self.add_output('hub_Fxyz', val=np.zeros(3), units='kN', desc = 'Maximum hub forces in the non rotating frame')
        self.add_output('hub_Mxyz', val=np.zeros(3), units='kN*m', desc = 'Maximum hub moments in the non rotating frame')

        # Tower outputs
        self.add_output('max_TwrBsMyt',val=0.0, units='kN*m', desc='maximum of tower base bending moment in fore-aft direction')
        self.add_output('DEL_TwrBsMyt',val=0.0, units='kN*m', desc='damage equivalent load of tower base bending moment in fore-aft direction')
        self.add_output('tower_maxMy_Fx', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned x-direction corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Fy', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned y-direction corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Fz', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned z-direction corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Mx', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_My', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
        self.add_output('tower_maxMy_Mz', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
        '''

    def plot(self, ax, color=None, nodes=0, plot_rotor=True, station_plot=[], 
             airfoils=False, zorder=2, plot_fowt=True, plot_ms=True, 
             shadow=True, mp_args={}):
        '''plots the FOWT...'''

        R = rotationMatrix(self.r6[3], self.r6[4], self.r6[5])  # note: eventually Rotor could handle orientation internally <<<

        if plot_ms:
            if self.ms:
                self.ms.plot(ax=ax, color=color, shadow=shadow)
        
        if color==None:
            color='k'
        else:
            mp_args.update(dict(color=color))
        
        #if self.ms:
            #self.ms.plot(ax=ax, **mp_args)
        
        if plot_fowt:
            if plot_rotor:
                for rotor in self.rotorList:
                    rotor.plot(ax, color=color, airfoils=airfoils, zorder=zorder)

            # loop through each member and plot it
            for mem in self.memberList:

                mem.setPosition()  # offsets/rotations could be done in this function rather than in mem.plot <<<

                mem.plot(ax, r_ptfm=self.r6[:3], R_ptfm=R, color=color, 
                        nodes=nodes, station_plot=station_plot, zorder=zorder)

        # in future should consider ability to animate mode shapes and also to animate response at each frequency
        # including hydro excitation vectors stored in each member


    def plot2d(self, ax, color=None, plot_rotor=1, 
        Xuvec=[1,0,0], Yuvec=[0,0,1]):
        '''Plot the FOWT in 2D based on specified axes.
        
        Parameters
        ----------
        plot_rotor : int
            0 : don't plot; 1: plot 3D view; 2: plot a simple line
        '''

        R = rotationMatrix(self.r6[3], self.r6[4], self.r6[5])  # note: eventually Rotor could handle orientation internally <<<
        
        Xuvec = np.array(Xuvec)
        Yuvec = np.array(Yuvec)
        
        if self.ms:
            self.ms.plot2d(ax=ax, color=color, Xuvec=Xuvec, Yuvec=Yuvec)

        if color==None:
            color='k'
            
        # loop through each member and plot it
        for mem in self.memberList:
            mem.setPosition()
            mem.plot(ax, r_ptfm=self.r6[:3], R_ptfm=R, color=color, plot2d=True, Xuvec=Xuvec, Yuvec=Yuvec)

        # Plot the rotor(s)
        if plot_rotor == 1:
            for rotor in self.rotorList:
                rotor.plot(ax, color=color, plot2d=True, Xuvec=Xuvec, Yuvec=Yuvec)
                
        elif plot_rotor == 2:
            # simple circle/line plot of rotor, ignoring precone or tilt
            for rotor in self.rotorList:
                r = rotor.ccblade.r[-1]  # rotor tip radius [m]
                
                X=[]  # lists to be filled with coordinates for plotting
                Y=[]
                Z=[]
                
                n = 24  # number of sides for a circle
                for i in range(n+1):
                    y = np.cos(float(i)/float(n)*2.0*np.pi)    # x coordinates of a unit circle
                    z = np.sin(float(i)/float(n)*2.0*np.pi)    # y

                    X.append(0)
                    Y.append(r*y)
                    Z.append(r*z)
                
                P2 = np.vstack([X, Y, Z])  # combine into a matrix of coordinates
                  
                R_heading = rotationMatrix(0, 0, rotor.turbine_heading)  # rotation matrix for rotor heading
                
                P2 = np.matmul(R_heading, P2) + rotor.r3[:,None]  # with only heading
                
                # apply any 3D to 2D transformation here to provide desired viewing angle
                Xs2d = np.matmul(Xuvec, P2)
                Ys2d = np.matmul(Yuvec, P2)
                ax.plot(Xs2d, Ys2d, color=color, lw=1.0)
