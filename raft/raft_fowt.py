# RAFT's floating wind turbine class

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator, griddata

import raft.member2pnl as pnl
from raft.helpers import *
from raft.raft_member import Member
from raft.raft_rotor import Rotor
from raft.raft_node import Node
import moorpy as mp
from moorpy.helpers import lines2ss, transformPosition

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
        self.nw = len(w)                     # number of frequencies
        self.heading_adjust = heading_adjust # rotation to the heading of the platform and mooring system to be applied [deg]

        # Lists of components used to describe the FOWT
        self.memberList    = [] # list of member objects
        self.rotorList     = [] # list of rotor objects
        self.nodeList      = [] # list of node objects
        self.jointList     = [] # list of joint dictionaries
        self.rigidLinkList = {'id': [], 'node1': [], 'node2': []} # Dict of lists instead of list of dicts because this does not come from the input file, so easier to check what the keys are
        
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
        self.r6 = np.zeros(6)   # rigid body position/orientation in absolute/array coordinates [m,rad] TODO: replace this by the rigid-body node

        # Check if we have repeated member names
        # TODO: Only if we have joints? Or always? And do we even need this? 
        # Check later after structural capabilities are working and we have tests
        if 'platform' in design:
            member_names = [m['name'] for m in design['platform']['members']]
            if len(member_names) != len(set(member_names)):
                raise Exception("Member names must be unique. Please check the input data.")

        # Platform lumped inertias
        self.pointInertias = []
        self.pointLoads    = []
        if 'additional_effects' in design['platform']:
            for ie, additional_effect in enumerate(design['platform']['additional_effects']):
                if 'type' not in additional_effect:
                    raise Exception(f"Additional effect {ie} in platform design must have a 'type' field.")

                if additional_effect['type'] == 'point_inertia':
                    self.pointInertias.append({'m': 0, 'inertia': np.zeros([6, 6]), 'r': np.zeros([3,])}) # mass, inertia, and position of each point mass to be added to the platform
                    pointMass = getFromDict(additional_effect, 'mass', shape=0, default=0) # mass of the substructure [kg]
                    auxInertia = getFromDict(additional_effect, 'moments_of_inertia', shape=6, default=[0,0,0]) # moments of inertia of the substructure [kg*m^2] - specified in the order Jxx, Jyy, Jzz, Jxy, Jxz, Jyz

                    self.pointInertias[-1]['m'] = pointMass                    
                    self.pointInertias[-1]['inertia'] = np.array([[pointMass, 0, 0, 0, 0, 0],
                                                                [0, pointMass, 0, 0, 0, 0],
                                                                [0, 0, pointMass, 0, 0, 0],
                                                                [0, 0, 0, auxInertia[0], auxInertia[3], auxInertia[4]],
                                                                [0, 0, 0, auxInertia[3], auxInertia[1], auxInertia[5]],
                                                                [0, 0, 0, auxInertia[4], auxInertia[5], auxInertia[2]]])
                    self.pointInertias[-1]['r'] = getFromDict(additional_effect, 'location', shape=3, default=[0,0,0])
                elif additional_effect['type'] == 'mean_load':
                    self.pointLoads.append({'f': np.zeros(6), 'r': np.zeros(3)}) # load and position of each point load to be added to the platform
                    self.pointLoads[-1]['f'] = getFromDict(additional_effect, 'load', shape=6, default=np.zeros(6)) # load vector [N, N-m]
                    self.pointLoads[-1]['r'] = getFromDict(additional_effect, 'location', shape=3, default=[0,0,0]) # position where the load is applied [m]
                                            
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

        self.depth = depth
        self.w = np.array(w)
        self.dw = w[1]-w[0] # frequency increment [rad/s]
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

        # Create list of joint DATA from the input data.
        # Actual joints are created when creating the members.
        #
        # If no joint data is specified, we create a single joint entry at the origin to rigidly connect all members to it.
        # We can only do this if all platform members are rigid.
        self.joint_data = []
        if 'joints' in design:
            self.joint_data = design['joints']

            # Check if we have repeated joint names in the data dictionary.
            # We can have repeated joint names in the joints list due to headings but not in the joint_data dictionary
            joint_names = [j['name'] for j in self.joint_data]
            if len(joint_names) != len(set(joint_names)):
                raise Exception("Joint names must be unique. Please check the input data.")        
        else:
            j_member_names = [] # To store the names of the members which will be connected to the virtual joint
            j_location = [0, 0, 0] # Place the joint at the origin

            # Add platform members to the list of members connected to the tower + check if any of the platform members is flexible, which would require joints to be specified
            if 'platform' in design:
                for m in design['platform']['members']:
                    if m['type' ] == 'beam':
                        raise Exception("To model platforms with flexible members, RAFT needs joints in the input data.")
                    j_member_names.append(m['name'])
                
            # Add towers to the list of members connected to the joint
            if 'turbine' in design:
                if 'tower' in design['turbine']:                    
                    j_member_names += [m['name'] for m in design['turbine']['tower']] # Add all towers to the list of members connected to the joint
            self.joint_data.append({'name': 'origin_joint', 'type': 'cantilever', 'location': j_location, 'members': j_member_names})


        # Check consistency between joints and members
        # In particular, check if headings in joints are consistent with the headings in the connected members                
        for j_data in self.joint_data:
            for m_name in j_data['members']: # Loop all the members that are connected to this joint
                # Check if the member is in the list of tower members, in which case they don't have headings
                if 'turbine' in design and 'tower' in design['turbine']:
                    if m_name in [m['name'] for m in design['turbine']['tower']]:
                        continue

                # Check if the member is in the list of platform members
                m = [m for m in design['platform']['members'] if m['name'] == m_name]
                if len(m) == 0:
                    raise Exception(f"Member {m_name} not found in the list of members. Please check the input data.")
                else:
                    m = m[0]

                # A single joint/member can be connected to multiple members/joints
                # But if the user provides a list of headings for both, they must have the same length
                if 'heading' in j_data.keys() and 'heading' in m.keys():
                    if len(j_data['heading']) != 1 and len(m['heading']) != 1:
                        if len(j_data['heading']) != len(m['heading']):
                            raise Exception(f"Joint '{j_data['name']}' has {len(j_data['heading'])} headings, but connected member '{m_name}' has {len(m['heading'])} headings. Please check the input data.")

        # member-based platform description
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
                headings = [headings] # if only one heading is specified, then just create one member

            for heading in headings:
                self.memberList.append(Member(mi, self.nw, heading=heading+heading_adjust, part_of='platform', first_node_id=len(self.nodeList)))
                self.nodeList += self.memberList[-1].nodeList                
                self.memberList[-1].headings = headings # Storing a copy of headings at each member to use it in model.adjustBallast
        
        # add tower(s) and nacelle(s) to member list if applicable
        if 'turbine' in design:
            if 'tower' in design['turbine']:
                for mem in design['turbine']['tower']:
                    self.memberList.append(Member(mem, self.nw, part_of='tower', first_node_id=len(self.nodeList)))
                    self.nodeList += self.memberList[-1].nodeList
            if 'nacelle' in design['turbine']:
                for mem in design['turbine']['nacelle']:
                    self.memberList.append(Member(mem, self.nw, part_of='nacelle', first_node_id=len(self.nodeList)))
                    self.nodeList += self.memberList[-1].nodeList
        #TODO: consider putting the tower somewhere else rather than in end of memberList <<<
        
        # Create joints
        for j_data in self.joint_data:
            j_headings = getFromDict(j_data, 'heading', shape=-1, default=0.)
            if np.isscalar(j_headings):
                j_headings = [j_headings]

            for count_heading, j_heading in enumerate(j_headings):
                this_joint = self.addJoint(j_data, heading=j_heading)

                for member_name in this_joint['members']:
                    # Find the members with this name. We already know this is a non-empty list because we checked above
                    members = [m for m in self.memberList if m.name == member_name]

                    # If a single member with this name (i.e., a single heading), attach it to the joint
                    if len(members) == 1:
                        self.attachMemberToJoint(members[0], this_joint)
                        continue
                    
                    # If multiple members but a single joint heading, attach all members to the joint
                    if len(j_headings) == 1:
                        for m in members:
                            self.attachMemberToJoint(m, this_joint)
                    
                    # If multiple members and multiple joint headings, we attach the member corresponding to the heading of this joint
                    # We already checked above if the number of headings of the joint is the same as the number of headings of the connected members (when both are lists)
                    else:
                        this_member = members[count_heading]
                        self.attachMemberToJoint(this_member, this_joint)

        # build Rotor list
        # Each rotor is connected to the top of its corresponding tower by a cantilever joint
        towerList = [m for m in self.memberList if m.part_of == 'tower']
        for ir in range(self.nrotors):
            self.rotorList.append(Rotor(design['turbine'], self.w, ir, node_id=len(self.nodeList)))
            self.nodeList += self.rotorList[-1].nodeList  # add the rotor nodes to the node list
            towerTopJoint = self.addJoint({'name': 'tower2rotor', 'type': 'cantilever', 'location': self.rotorList[ir].r_RRP, 'members': []}) # Do not need the members list in the joint data because we will provide the members directly
            
            self.attachMemberToJoint(towerList[ir], towerTopJoint)       # attach the tower to the joint
            self.attachMemberToJoint(self.rotorList[-1], towerTopJoint)  # attach the rotor to the tower top joint

        
        # Define a node to be used as a reference for rigid body motions
        # TODO: For now, using the node that is the closest to (0,0,0).
        #       In the future, let the user pick a node
        self.rigidBodyNode = min(self.nodeList, key=lambda n: np.linalg.norm(n.r0[0:3]))
        # print(f"Rigid body motions correspond to node located at {self.rigidBodyNode.r0[0:3]} wrt the PRP.")

        # Move this node to be the first node in the list.
        # TODO: Only doing this now for compatibility with previous code. Remove lines later.
        # if len(self.joint_data) == 1 and self.joint_data[0]['name'] == 'origin_joint':
        if self.rigidBodyNode.member is None or self.rigidBodyNode.member.type == 'rigid':
            self.nodeList.remove(self.rigidBodyNode)
            self.nodeList.insert(0, self.rigidBodyNode)
            for i, n in enumerate(self.nodeList): # Reset the id of all nodes to be their index in the list. TODO: maybe just assign the ids here instead at node creation. I think I just need to set them before reduceDOF() and computeTransformationMatrix(), but not sure
                n.id = i

        # Store the list of nodes of the FOWT in each node for reference
        # And initialize the inertia and flexibility stiffness matrices of each node
        self.nFullDOF = sum([n.nDOF for n in self.nodeList])       # total number of DOFs of the structure (including virtual nodes created by rigid links)        
        for n in self.nodeList:
            n.nodeList = self.nodeList

        self.reduceDOF()                                           # evaluate the set of reduced dofs
        self.Xi0  = np.zeros( self.nDOF)                           # mean offsets of platform from its reference point [m, rad]
        self.Xi   = np.zeros([self.nDOF, self.nw], dtype=complex)  # complex response amplitudes as a function of frequency  [m, rad]        
        self.rReducedDOF = np.zeros(self.nDOF)                     # position of the platform in the reduced dofs         

        # array-level mooring system connection
        # TODO: Will need to be connected to nodes instead of the body
        self.body = mpb                                              # reference to Body in mooring system corresponding to this turbine

        # this FOWT's own MoorPy system (may not be used)
        if 'mooring' in design and isinstance(design['mooring'], dict):

            self.ms = mp.System()
            self.ms.parseYAML(design['mooring'])
            self.moorMod = getFromDict(design['mooring'], 'moorMod', default=0, dtype=int)
            self.ms_tol = getFromDict(design['mooring'], 'tol', dtype=float, default=0.05)  # mooring system tolerance for solving equilibrium
            
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
            self.moorMod = 0
        
        self.F_moor0 = np.zeros(self.nDOF)     # mean mooring forces in a given scenario
        self.C_moor = np.zeros([self.nDOF,self.nDOF])  # mooring stiffness matrix in a given scenario

        if 'yaw_stiffness' in design['platform']:
            self.yawstiff = design['platform']['yaw_stiffness']       # If you're modeling OC3 spar, for example, import the manual yaw stiffness needed by the bridle config
        else:
            self.yawstiff = 0
        
        # initialize mean force arrays to zero, so the model can work before we calculate excitation
        self.f_aero0 = np.zeros([6, self.nrotors])
        # mean weight and hydro force arrays are set elsewhere. In future hydro could include current.
        self.D_hydro = np.zeros(6)      # initialize mean drag force from current - acts as a "static" force, like f_aero0

        # flag to signal whether any members will be modeled with BEM
        self.potMod = any([member['potMod']==True for member in design['platform']['members'] ])


        # initialize BEM arrays, whether or not a BEM solver is used
        # __, __, nWaveHeadings = getUniqueCaseHeadings(design['cases']['keys'], design['cases']['data'])  # identify how many wave headings need to be preprocessed
        self.A_BEM = np.zeros([self.nDOF,self.nDOF,self.nw], dtype=float)                 # hydrodynamic added mass matrix [kg, kg-m, kg-m^2]
        self.B_BEM = np.zeros([self.nDOF,self.nDOF,self.nw], dtype=float)                 # wave radiation drag matrix [kg, kg-m, kg-m^2]
        #self.X_BEM = np.zeros([self.nWaves, self.nDOF, self.nw], dtype=complex)               # linaer wave excitation force/moment coefficients vector [N, N-m]
        #self.F_BEM = np.zeros([self.nWaves, self.nDOF, self.nw], dtype=complex)               # linaer wave excitation force/moment complex amplitudes vector [N, N-m]
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
    
    
    def addJoint(self, this_joint_data, heading=0., tol=1e-3):
        ''' Add a joint to the list of joints.        
        Also returns the joint data as a dictionary with the following fields:
        [id, r, type, name, heading, members]
        
        PARAMETERS
        ----------
        this_joint_data: Dictionary with fields 'name', 'type', and 'location'
            Data corresponding to this joint, as provided in the `joint` section of the yaml file.        
        eps: float
            olerance for comparing joint position
        heading: float
            Z rotation around (x,y)=(0,0) in the the local frame of the FOWT.
        '''
        r = np.array(this_joint_data['location'])
        r = applyHeadingToPoint(r, heading=heading)

        # Check if an equivalent joint already exists
        for joint in self.jointList:
            if joint['name'] == this_joint_data['name']:
                # Compare position           
                existing_position = np.array(joint['r'])
                if np.linalg.norm(existing_position - r) <= tol:
                    # Joint already exists, return it instead of adding a new one
                    return joint

        j_id = len(self.jointList)
        joint = {}
        joint['id'] = j_id
        joint['r']  = r.copy()
        joint['name'] = this_joint_data['name']
        joint['type'] = this_joint_data['type']
        joint['heading'] = heading
        joint['members'] = this_joint_data['members']
        
        self.jointList.append(joint)
        return joint

    def attachMemberToJoint(self, member, joint, tol=1e-3):
        ''' 
        Attach the member to the joint by assigning joint_id and joint_type to a member's node.
        
        For rigid elements, which correspond to a single node, this is done by
        1. Assigning the joint to the node if the joint and node are within the tolerance
        2. Creating two extra nodes connected by a rigid link. One of the rigid-link nodes is attached to the joint, while the other is cantilevered to the member node.
                 
        Same thing for flexible elements, but we use the end node that is the closest to the joint

        PARAMETERS
        ----------
        member: Member object
            Member to be attached to the joint
        joint: dict
            Joint dictionary, as returned by addJoint() and stored in self.jointList
        tol: float
            Tolerance for comparing joint position and node position [m]
        ''' 
        # print(f"Attaching member {member.name} to joint {joint['name']}")

        # Get the node that will be connected to the joint
        # - Rigid members have only one node, so we use that node
        # - Flexible members have several nodes. We use the end one that is closest to the joint        
        if member.type == 'rigid' or member.type == 'rotor':
            node = member.nodeList[0]
            dist = np.linalg.norm(node.r0[0:3] - joint['r'])            
        elif member.type == 'beam':
            nA = member.nodeList[0]
            dA = np.linalg.norm(nA.r0[0:3] - joint['r'])

            nB = member.nodeList[-1]
            dB = np.linalg.norm(nB.r0[0:3] - joint['r'])

            node, dist = (nA, dA) if dA < dB else (nB,dB)
        else:
            raise Exception(f"Member type {member.type} not supported.")
        
        # If the joint and node are close enough, just connect them by the joint
        if dist <= tol:
            node.joint_id   = joint['id']
            node.joint_type = joint['type']
        
        # Otherwise, we need to create a rigid link between the joint and the node
        # TODO: Each rigid links create two dummy nodes that increase the size of the system.
        #       This does not impact computation time, as expensive computations are performed in the reduced dofs,
        #       but it would be nice to find a different way to join nodes with a geommetric offset.
        else:
            # Add two new nodes: one at the joint location, and one at the member's node location
            newNodeA = Node(len(self.nodeList)  , joint['r']  , self.nw, member=None, end_node=True)
            newNodeB = Node(len(self.nodeList)+1, node.r0[0:3], self.nw, member=None, end_node=True)

            # Add a rigid link between the two nodes
            self.rigidLinkList['id'].append(len(self.rigidLinkList['id'])) # Don't think we need an id, could use the index in the list instead
            self.rigidLinkList['node1'].append(newNodeA)
            self.rigidLinkList['node2'].append(newNodeB)

            # newNodeA is connected to the joint
            newNodeA.joint_id      = joint['id']
            newNodeA.joint_type    = joint['type']
            newNodeA.rigid_link_id = self.rigidLinkList['id'][-1]
            self.nodeList.append(newNodeA)

            # Node B is attached to the other node. Use the same joint as node if it already has one, otherwise create a new cantilever joint for the connection
            if node.joint_id is not None:
                jointB = [j for j in self.jointList if j['id'] == node.joint_id][0]
            else:
                j_data = {'name': f"joint4rigidLink", 'type': 'cantilever', 'location': newNodeB.r0[0:3].tolist(), 'members': []} # We won't use members for rigid-link joints
                jointB = self.addJoint(j_data)
            newNodeB.joint_id      = jointB['id']
            newNodeB.joint_type    = jointB['type']
            newNodeB.rigid_link_id = self.rigidLinkList['id'][-1]
            node.joint_id          = jointB['id']
            node.joint_type        = jointB['type']
            self.nodeList.append(newNodeB)

    def reduceDOF(self):
        '''
        Find reduced set of dofs of the structure.
        They are stored in self.reducedDOF, a list where each element is a sublist of the form [node_id, dof_index].
        Where dof_index can be (0,1,2,3,4,5) corresponding to (x,y,z,roll,pitch,yaw) of a node.
        '''
        if len(self.nodeList) < 1:
            raise Exception("No nodes in the structure. Cannot compute reduced dofs.")

        # Reset dofs of all nodes
        for n in self.nodeList:
            n.reducedDOF = None
            n.T = None
            n.T_aux = None
            n.parentNode_id = None

        # We will loop all nodes based on the connected nodes        
        queue = []      # Store the nodes that still need to be processed
        visited = set() # Store the nodes that have been processed (to avoid repeating nodes)
        queue.append(self.nodeList[0]) # We will start with the first node of the list        

        while queue:
            node = queue.pop(0)

            # If this node does not have reducedDOF, assign its own dofs
            if node.reducedDOF is None:
                node.reducedDOF = []
                for i in range(node.nDOF):
                    node.reducedDOF.append([node.id, i])
                node.T_aux = np.eye(node.nDOF)
                node.parentNode_id = node.id
                visited.add(node.id)

            # Attach the node connected by a rigid link (if it exists)
            rigidConnectedNode = node.getRigidConnectedNode()
            if (rigidConnectedNode is not None) and (rigidConnectedNode.id not in visited):
                rigidConnectedNode.attachToNode(node, rigid_link=True)
                queue.append(rigidConnectedNode)
                visited.add(rigidConnectedNode.id)

            # Attach the nodes connected by the joint
            connectedNodes = node.getNodesConnectedByJoint()
            for n in connectedNodes:
                if n.id not in visited:
                    n.attachToNode(node)
                    queue.append(n)
                    visited.add(n.id)

            # If this is the end of a flexible member, we need to add the other nodes to the queue
            if node.member and node.member.type == 'beam' and node.end_node:
                for n in node.member.nodeList:
                    if (n.id != node.id) and (n.id not in visited):
                        queue.append(n)

        # Check if we visited all nodes
        if len(visited) != len(self.nodeList):
            # Find nodes that were not visited
            unvisited_nodes = [n.id for n in self.nodeList if n.id not in visited]
            raise Exception(f"Could not reach all nodes from the first node. Please check the connectivity of the structure. Unvisited node ids: {unvisited_nodes}")
            
        # Get unique dofs of the whole structure
        reducedDOF = []
        for n in self.nodeList:
            if n.reducedDOF is not None:
                for dof in n.reducedDOF:
                    if dof not in reducedDOF:
                        reducedDOF.append(dof)
        self.reducedDOF = reducedDOF
        self.nDOF = len(self.reducedDOF) # number of reduced dofs needed to describe the structure
        self.computeTransformationMatrix()  

    def computeTransformationMatrix(self):
        '''
        Compute the transformation matrix T that transforms from the reduced degrees of freedom 
        to the full degrees of freedom, fullDOF = T @ reducedDOF.

        The transformation matrix is stored in self.T, which is a (nFullDOF, nReducedDOF) matrix.
        '''
        self.T = np.zeros((self.nFullDOF, self.nDOF))

        row0 = 0
        for n in self.nodeList:
            n.setT(self.reducedDOF)
            self.T[row0:row0+n.nDOF, :] = n.T
            row0 += n.nDOF
        return self.T

    def computeDerivativeTransformationMatrix(self):
        '''Compute the derivative of the transformation matrix T with respect to the reduced degrees of freedom.
        
        The derivative is stored in self.dT, which is a (nFullDOF, nReducedDOF, nReducedDOF) matrix.
        '''
        import copy
        
        self.dT = np.zeros((self.nFullDOF, self.nDOF, self.nDOF))

        # Create a deep copy to avoid modifying the original object
        fowt_temp = copy.deepcopy(self)

        # Only need to do the derivative for rotational dofs and end nodes
        # Translations and motions of internal nodes do not change the transformation matrix
        for i, dof in enumerate(self.reducedDOF):
            if dof[1] > 2 and self.nodeList[dof[0]].end_node:
                reducedDisp = np.zeros(self.nDOF)
                reducedDisp[i] = 1.0
                
                # Use the temporary copy for computations
                fowt_temp.setNodesPosition(fowt_temp.rReducedDOF + reducedDisp, linear=True)
                fowt_temp.reduceDOF()
                self.dT[:, :, i] = fowt_temp.T - self.T  # Compute the derivative
                
                # Reset the temporary copy to original state for next iteration
                fowt_temp = copy.deepcopy(self)

        return self.dT

    def setNodesPosition(self, reducedXi0, linear=False):
        '''Set the mean displacements, node.Xi0, and positions, node.r, of all the nodes 
        of the structure based on the displacements in the reduced degrees of freedom.

        PARAMETERS
        ----------
        reducedXi0 : (nReducedDoF,) float array 
            Mean displacements of the FOWT in the reduced degrees of freedom [m, rad]
        linear: bool, optional
            If True, the displacements are obtained using linear relations. This option
            does not preserve distance between nodes connected by rigid links.
        '''
        # If setting the displacement using linear relations, we can simply use the T matrix stored in each node
        # This does not preserve the lengths of rigid links but should be good enough for small rotations
        if linear:            
            for n in self.nodeList:
                n.setDisplacementLinear(reducedXi0)
        else:
            # Otherwise, set nodes' displacement using nonlinear relations. More work but preserves the lenghts of rigid links.
            # The nonlinearities arise from the rotations of rigid links, which include sin and cos.        
            self.nodeList[0].setDisplacementLinear(reducedXi0) # The first node is always the first of the reduced dofs, so we simply use the linear relation

            # The remaining nodes depend on how they are connected to the previous node
            # We will loop all nodes in the same way as in reduceDOF()
            queue = []      # Store the nodes that still need to be processed
            visited = set() # Store the nodes that have been processed (to avoid repeating nodes)
            queue.append(self.nodeList[0]) # We will start with the first node of the list
            visited.add(self.nodeList[0].id)

            while queue:
                node = queue.pop(0)

                # We just loop through the end nodes. Internal nodes are taken care based on their end nodes
                if not node.end_node:
                    continue

                # Deal with the node connected by a rigid link (if it exists)
                rigidConnectedNode = node.getRigidConnectedNode()
                if (rigidConnectedNode is not None) and (rigidConnectedNode.id != node.parentNode_id) and (rigidConnectedNode.id not in visited):
                    dx = rigidConnectedNode.r0[0] - node.r0[0]
                    dy = rigidConnectedNode.r0[1] - node.r0[1]
                    dz = rigidConnectedNode.r0[2] - node.r0[2]

                    # Get rotation (displacement part, not including initial value) around node. x, y, and z components
                    rotation = (node.T @ reducedXi0)[-3:]
                    rotMat   = rotationMatrix(*rotation) - np.eye(3) # Remove identity matrix to get the displacement only
                    
                    # RigidConnectedNode has the same rotation but the translation accounts for the distance between both nodes
                    rigidConnectedNode.Xi0 = node.Xi0.copy()
                    rigidConnectedNode.Xi0[0:rigidConnectedNode.nTransDOF] += rotMat @ np.array([dx, dy, dz])

                    queue.append(rigidConnectedNode)
                    visited.add(rigidConnectedNode.id)                    

                # Deal with the nodes connected by the joint
                connectedNodes = node.getNodesConnectedByJoint()
                for n in connectedNodes:
                    if (n.id != node.parentNode_id) and (n.id not in visited):
                        n.Xi0 = node.Xi0.copy() # Assign the same displacements to the connected nodes
                        if n.joint_type == "ball": # But if ball joint, overried rotation with the node's own rotation
                            n.Xi0[-3:] = (n.T @ reducedXi0)[-3:]
                        queue.append(n)
                        visited.add(n.id)

                # If this node is part of a flexible element, we need to set the displacement of the other nodes (remember that we are looping through the end nodes)
                # We do that by adding the difference between the nonlinear and linear displacements of the end node to the linear displacement of the other nodes
                if node.member and node.member.type == 'beam':
                    # We need to add the other end to the queue to visit the nodes that are connected to it.
                    # Only do that if the other end wasn't visited yet - it could have been visited by nodes connected to the other side of the flexible member.
                    for n in [node.member.nodeList[0], node.member.nodeList[-1]]:
                        if (n.id != node.id) and (n.id not in visited):
                            queue.append(n)

                    disp     = node.Xi0            # Nonlinear displacement of the end node
                    disp_lin = node.T @ reducedXi0 # Linear displacement of the end node
                    dR       = disp - disp_lin     # Difference between nonlinear and linear displacements
                    for n in node.member.nodeList:
                        if n.id != node.id and (n.id not in visited):
                            n.setDisplacementLinear(reducedXi0) # Start by setting the displacement linearly
                            n.Xi0 += dR # then we add the difference between nonlinear and linear displacements of the first node
                            visited.add(n.id)

        for n in self.nodeList:
            n.r = n.r0 + n.Xi0

    def setPosition(self, rReducedDOF):
        '''Updates the FOWT's (mean) position including all components.

        PARAMETERS
        ----------
        rReducedDOF : (nDOF, ) float array 
            Mean positions along the reduced dofs of the FOWT [m, rad].
            For a fully rigid FOWT, this is a 6-dof array with the position and attitude of the PRP.
        '''
        # if offset provided, set things according to those positions, otherwise zero it
        self.rReducedDOF = rReducedDOF
        initial_displacement = np.zeros(self.nDOF)  # Initialize the displacement vector in the reduced dofs
        for i, dof in enumerate(self.reducedDOF):
            initial_displacement[i] += self.x_ref if dof[1] == 0 else 0 # If this is a translation in the X direction (Global dof 0)
            initial_displacement[i] += self.y_ref if dof[1] == 1 else 0 # If this is a translation in the Y direction (Global dof 1)
        self.Xi0 = self.rReducedDOF - initial_displacement

        # Set the position of all nodes
        # rReducedDOF can be interpreted as a displacement in the reduced degrees of freedom, as the r0's of the nodes are wrt the platform
        self.setNodesPosition(rReducedDOF)
        self.reduceDOF()  # Recompute the transformation matrix
        self.computeDerivativeTransformationMatrix() # Also compute the derivative of the transformation matrix, which is used to compute the stiffness matrices
        
        # Compute motions of the PRP (the intersection of the tower centerline and the mean waterline)
        # as a rigid body transformation from the rigidBodyNode to the PRP
        rotMat   = rotationMatrix(*self.rigidBodyNode.Xi0[-3:])
        self.r6 = self.rigidBodyNode.r.copy()
        self.r6[:3] += rotMat @ (-self.rigidBodyNode.r0[:3])

        # calculate and save a rotation/orientation matrix
        self.Rmat = rotationMatrix(*self.r6[3:])  # rotation matrix for fowt orientation
        
        # set the positions of the FOWT's members, rotors, and MoorPy system
        if self.ms:
            self.ms.bodyList[0].setPosition(self.r6)
        
        for rot in self.rotorList:            
            rot.setPosition()
            
        for mem in self.memberList:
            mem.setPosition()
            mem.computeStiffnessMatrix_FE() # Recompute stiffness matrix at the updated position
        
        # solve the mooring system equilibrium of this FOWT's own MoorPy system
        if self.ms:
            self.ms.solveEquilibrium(tol=self.ms_tol)
            if self.moorMod == 0 or self.moorMod == 1:
                C_moor = self.ms.getCoupledStiffnessA(lines_only=True)
            elif self.moorMod == 2:
                self.ms.updateSystemDynamicMatrices()
                _, _, _, C_moor = self.ms.getCoupledDynamicMatrices(lines_only=True)

            # For now, we will just lump the mooring forces and stiffness at the first 6 dofs
            self.C_moor[:6, :6]  = translateMatrix6to6DOF(C_moor, self.ms.bodyList[0].r6[:3] - self.nodeList[self.reducedDOF[0][0]].r[:3]) # This isn't right because this matrix translation doesn't account for the geometric stiffness
            self.F_moor0[:6] = transformForce(self.ms.bodyList[0].getForces(lines_only=True), offset=self.ms.bodyList[0].r6[:3] - self.nodeList[self.reducedDOF[0][0]].r[:3])


    def calcStatics(self):
        '''Computes the static properties of the FOWT in terms of mass and hydrostatic-related
        matrices and mean force vectors about the FOWT PRP in unrotated directions, based on
        the mean offsets of the FOWT, Xi0.
        '''

        rho = self.rho_water
        g   = self.g

        # structure-related arrays - in the reduced set of dofs
        self.M_struc = np.zeros([self.nDOF,self.nDOF])        # structure/static mass/inertia matrix [kg, kg-m, kg-m^2]
        self.B_struc = np.zeros([self.nDOF,self.nDOF])        # structure damping matrix [N-s/m, N-s, N-s-m] (may not be used)
        self.C_struc = np.zeros([self.nDOF,self.nDOF])        # structure effective stiffness matrix [N/m, N, N-m]        
        self.W_struc = np.zeros([self.nDOF])                  # static weight vector [N, N-m]
        self.C_elast = np.zeros([self.nDOF,self.nDOF])        # structure elastic stiffness matrix [N/m, N, N-m]

        # structure-related arrays - in the full set of dofs
        # Transformed into the reduced set of dofs using the transformation matrix
        M_struc_fullDOF = np.zeros([self.nFullDOF,self.nFullDOF])
        B_struc_fullDOF = np.zeros([self.nFullDOF,self.nFullDOF])
        C_struc_fullDOF = np.zeros([self.nFullDOF,self.nFullDOF])
        W_struc_fullDOF = np.zeros([self.nFullDOF])
        C_elast_fullDOF = np.zeros([self.nFullDOF,self.nFullDOF])
        
        # hydrostatic arrays - in the reduced set of dofs
        self.C_hydro = np.zeros([self.nDOF,self.nDOF])        # hydrostatic stiffness matrix [N/m, N, N-m]
        self.W_hydro = np.zeros(self.nDOF)                    # buoyancy force/moment vector [N, N-m]  <<<<< not used yet

        C_hydro_fullDOF = np.zeros([self.nFullDOF,self.nFullDOF])
        W_hydro_fullDOF = np.zeros(self.nFullDOF)

        # Additional user-defined constant force
        # self.f0_additional = np.zeros([6], dtype=float)
        self.f0_additional = np.zeros(self.nFullDOF)  # Initialize the additional force vector in full dofs
        f0_additional_fullDOF = np.zeros(self.nFullDOF)  # Initialize the additional force vector in full dofs

        # --------------- add in linear hydrodynamic coefficients here if applicable --------------------
        #[as in load them] <<<<<<<<<<<<<<<<<<<<<

        # --------------- Get general geometry properties including hydrostatics ------------------------
        # initialize some variables for running totals
        VTOT = 0.                   # Total underwater volume of all members combined
        m_all = 0.                  # Total mass of all members [kg]
        AWP_TOT = 0.                # Total waterplane area of all members [m^2]
        IWPx_TOT = 0                # Total waterplane moment of inertia of all members about x axis [m^4]
        IWPy_TOT = 0                # Total waterplane moment of inertia of all members about y axis [m^4]
        Sum_V_rCB = np.zeros(3)     # product of each member's buoyancy multiplied by center of buoyancy [m^4]
        Sum_AWP_rWP = np.zeros(2)   # product of each member's waterplane area multiplied by the area's center point [m^3]
        m_center_sum = np.zeros(3)  # product of each member's mass multiplied by its center of mass [kg-m] (Only considers the shell mass right now)

        self.m_sub = 0              # total mass of just the members that make up the substructure [kg]
        self.C_struc_sub = np.zeros([self.nDOF,self.nDOF])  # substructure effective stiffness matrix [N/m, N, N-m]
        self.M_struc_sub = np.zeros([self.nDOF,self.nDOF])  # total mass matrix of just the substructure about the PRP
        self.W_struc_sub = np.zeros(self.nDOF)              # weight vector of just the substructure [N, N-m]
        m_sub_sum = 0               # product of each substructure member's mass and CG, to be used to find the total substructure CG [kg-m]
        self.m_shell = 0            # total mass of the shells/steel of the members in the substructure [kg]
        mballast = []               # list to store the mass of the ballast in each of the substructure members [kg]
        pballast = []               # list to store the density of ballast in each of the substructure members [kg]
        self.mtower = np.zeros(self.ntowers)    # assume that the whole tower will always be one member
        self.rCG_tow = []

        C_struc_sub_fullDOF = np.zeros([self.nFullDOF,self.nFullDOF])
        M_struc_sub_fullDOF = np.zeros([self.nFullDOF,self.nFullDOF])
        W_struc_sub_fullDOF = np.zeros(self.nFullDOF)

        # loop through each member that's not part of the nacelle
        memberList = [mem for mem in self.memberList if mem.part_of != 'nacelle']
        for i,mem in enumerate(memberList):
            # Find the indices of the first and last nodes of the member to fill the _fullDOF matrices
            iFirst =  mem.nodeList[ 0].id      * mem.nodeList[0].nDOF
            iLast  = (mem.nodeList[-1].id + 1) * mem.nodeList[0].nDOF # Assuming all nodes have the same dofs, namely 6

            # calculate member's orientation information (stored in the member and used in later steps)
            mem.setPosition()  # <<< is this redundant, assume fowt.setPosition has been called?

            C_elast_fullDOF[iFirst:iLast, iFirst:iLast] += mem.computeStiffnessMatrix_FE()
            
            # note: quantities in the following section are relative to the PRP (but with global direcions)

            # ---------------------- get member's mass and inertia properties ------------------------------
            # get member mass and inertia info (including mem.M_struc) <<< still split between converting to PRP in or out of these functions
            mass, center, m_shell, mfill, pfill = mem.getInertia()
            W = mem.getWeight(g=g) # Using zeros because mem.cog is about the PRP

            # Calculate the mass matrix of the FOWT about the PRP
            W_struc_fullDOF[iFirst:iLast]               += W               # weight vector
            M_struc_fullDOF[iFirst:iLast, iFirst:iLast] += mem.M_struc     # mass/inertia matrix
            C_struc_fullDOF[iFirst:iLast, iFirst:iLast] += mem.C_struc     # part of the hydrostatic stiffness that is due to weight           

            center += mem.nodeList[0].r0[:3] # Update center position to be wrt the PRP
            m_center_sum += center*mass     # product sum of the mass and center of mass to find the total center of mass [kg-m]

            # Tower calculations
            if mem.part_of == 'tower':   # <<<<<<<<<<<< maybe find a better way to do the if condition
                self.mtower[i-self.nplatmems] = mass                  # mass of the tower [kg]
                self.rCG_tow.append(center)               # center of mass of the tower from the PRP [m]
            # Substructure calculations
            else:
                M_struc_sub_fullDOF[iFirst:iLast, iFirst:iLast] += mem.M_struc     # mass matrix of the substructure about the PRP
                C_struc_sub_fullDOF[iFirst:iLast, iFirst:iLast] += mem.C_struc
                W_struc_sub_fullDOF[iFirst:iLast]               += W               # weight vector of the substructure [N, N-m]

                self.m_sub += mass              # mass of the substructure
                m_sub_sum += center*mass        # product sum of the substructure members and their centers of mass [kg-m]
                self.m_shell += m_shell         # mass of the substructure shell material [kg]
                mballast.extend(mfill)          # list of ballast masses in each substructure member (list of lists) [kg]
                pballast.extend(pfill)          # list of ballast densities in each substructure member (list of lists) [kg/m^3]

            # -------------------- get each member's buoyancy/hydrostatic properties -----------------------
            Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = mem.getHydrostatics(rho=self.rho_water, g=self.g)
            
            # add to fowt's mean force vector and stiffness matrix
            W_hydro_fullDOF[iFirst:iLast] += Fvec # translateForce3to6DOF( np.array([0,0, Fz]), mem.rA )  # buoyancy vector
            C_hydro_fullDOF[iFirst:iLast, iFirst:iLast] += Cmat # translateMatrix6to6DOF(Cmat, mem.rA)                       # hydrostatic stiffness matrix
       
            # convert metrics to be about the PRP (platform reference point)
            r_CB += mem.nodeList[0].r0[:3]
            xWP += mem.nodeList[0].r0[0]
            yWP += mem.nodeList[0].r0[1]
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
                        # Find the indices of the first and last nodes of the member to fill the _fullDOF matrices
                        iFirst =  afmem.nodeList[ 0].id      * afmem.nodeList[0].nDOF
                        iLast  = (afmem.nodeList[-1].id + 1) * afmem.nodeList[0].nDOF # Assuming all nodes have the same dofs, namely 6                            

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
                        Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = afmem.getHydrostatics(rho=self.rho_water, g=self.g)
                        
                        # outputs of getHydrostatics should already be about the PRP
                        W_hydro_fullDOF[iFirst:iLast] += Fvec # buoyancy vector
                        C_hydro_fullDOF[iFirst:iLast, iFirst:iLast] += Cmat # hydrostatic stiffness matrix

                        # convert metrics to be about the PRP (platform reference point)
                        r_CB += mem.nodeList[0].r0[:3]
                        xWP += mem.nodeList[0].r0[0]
                        yWP += mem.nodeList[0].r0[1]
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
            # Find the indices of the first and last nodes of the member to fill the _fullDOF matrices
            iFirst =  mem.nodeList[ 0].id      * mem.nodeList[0].nDOF
            iLast  = (mem.nodeList[-1].id + 1) * mem.nodeList[0].nDOF # Assuming all nodes have the same dofs, namely 6            

            # call getHydroStatics for nacelles
            Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = mem.getHydrostatics(rho=self.rho_water, g=self.g)
            
            # add to fowt's mean force vector and stiffness matrix
            W_hydro_fullDOF[iFirst:iLast] += Fvec # translateForce3to6DOF( np.array([0,0, Fz]), mem.rA )  # buoyancy vector
            C_hydro_fullDOF[iFirst:iLast, iFirst:iLast] += Cmat # translateMatrix6to6DOF(Cmat, mem.rA)                       # hydrostatic stiffness matrix
       
            # convert metrics to be about the PRP (platform reference point)
            r_CB += mem.nodeList[0].r0[:3]
            xWP += mem.nodeList[0].r0[0]
            yWP += mem.nodeList[0].r0[1]
            VTOT    += V_UW    # add to total underwater volume of all members combined
            AWP_TOT += AWP
            IWPx_TOT += IWP + AWP*yWP**2
            IWPy_TOT += IWP + AWP*xWP**2
            Sum_V_rCB   += r_CB*V_UW
            Sum_AWP_rWP += np.array([xWP, yWP])*AWP
        
        
        # ------------------------- include RNA properties -----------------------------
        for i, rotor in enumerate(self.rotorList):
            # Find the indices of the first and last nodes of the member to fill the _fullDOF matrices
            iFirst =  rotor.nodeList[ 0].id      * rotor.nodeList[0].nDOF
            iLast  = (rotor.nodeList[-1].id + 1) * rotor.nodeList[0].nDOF # Assuming all nodes have the same dofs, namely 6
            
            # create mass/inertia matrix
            Mmat = np.diag([rotor.mRNA, rotor.mRNA, rotor.mRNA, 
                            rotor.IxRNA, rotor.IrRNA, rotor.IrRNA])
            
            # Rotate RNA mass matrix into the global orientation
            Mmat = rotateMatrix6(Mmat, rotor.R_q)
            
            # now convert everything to be about the RNA reference point and add to global vectors/matrices
            rotor_W, rotor_C_struc = getWeightOfPointMass(rotor.mRNA, rotor.r_CG_rel - rotor.r_RRP_rel, g=g)  # get weight vector and effective stiffness matrix of the rotor
            W_struc_fullDOF[iFirst:iLast] += rotor_W   # weight vector
            M_struc_fullDOF[iFirst:iLast, iFirst:iLast] += translateMatrix6to6DOF(Mmat, rotor.r_CG_rel -  rotor.r_RRP_rel)      # mass/inertia matrix
            C_struc_fullDOF[iFirst:iLast, iFirst:iLast] += rotor_C_struc
            
            m_center_sum += rotor.r_CG_rel*rotor.mRNA

        # ------------------------- include point inertia properties -----------------------------
        for ip, pointInertia in enumerate(self.pointInertias):
            # Find node that is nearest to pointInertia['r']        
            closest_node = min(self.nodeList, key=lambda n: np.linalg.norm(n.r0[:3] - pointInertia['r'])) # using r0 because pointInertia['r'] is in the platform reference frame
            iFirst =  closest_node.id      * closest_node.nDOF
            iLast  = (closest_node.id + 1) * closest_node.nDOF

            point_W, point_C_struc = getWeightOfPointMass(pointInertia['m'], pointInertia['r']-closest_node.r0[:3], g=g)  # get weight vector and effective stiffness matrix of the point inertia.
            W_struc_fullDOF[iFirst:iLast] += point_W
            M_struc_fullDOF[iFirst:iLast, iFirst:iLast] += translateMatrix6to6DOF(pointInertia['inertia'], pointInertia['r']-closest_node.r0[:3])            
            C_struc_fullDOF[iFirst:iLast, iFirst:iLast] += point_C_struc

            m_center_sum += pointInertia['r']*pointInertia['m']

            self.m_sub += pointInertia['m']
            M_struc_sub_fullDOF[iFirst:iLast, iFirst:iLast] += translateMatrix6to6DOF(pointInertia['inertia'], pointInertia['r']-closest_node.r0[:3])     # mass matrix of the substructure about the PRP
            C_struc_sub_fullDOF[iFirst:iLast, iFirst:iLast] += point_C_struc
            W_struc_sub_fullDOF[iFirst:iLast]               += point_W               # weight vector of the substructure [N, N-m]
            m_sub_sum += pointInertia['r']*pointInertia['m']        # product sum of the substructure members and their centers of mass [kg-m]

        # ------------------------- include user-specified mean loads -----------------------------
        for ip, pointLoad in enumerate(self.pointLoads):
            # Find node that is nearest to pointLoad['r']        
            closest_node = min(self.nodeList, key=lambda n: np.linalg.norm(n.r0[:3] - pointLoad['r']))
            iFirst =  closest_node.id      * closest_node.nDOF
            iLast  = (closest_node.id + 1) * closest_node.nDOF
            f0_additional_fullDOF[iFirst:iLast] += transformForce(pointLoad['f'], offset=pointLoad['r']-closest_node.r0[:3])  # add the user-specified mean load to the additional force vector

        # ----- internal loads at end nodes of flexible members
        # Used for the geometric stiffness of flexible members.
        # I think the proper way would be to first compute the internal loads and then compute the geometric stiffness. But this hackish approach seems to work.
        #
        # Check if the end nodes of the member are not in the reducedDOF list, in which
        # case we need to add the contribution of their internal forces to the geometric stiffness        
        W_internal_endNodes_struc_fullDOF, W_internal_endNodes_hydro_fullDOF = np.zeros(self.nFullDOF), np.zeros(self.nFullDOF)
        for mem in memberList:            
            if mem.type == 'beam':
                includeEndA = (mem.nodeList[0].id not in [dof[0] for dof in self.reducedDOF])
                includeEndB = (mem.nodeList[-1].id not in [dof[0] for dof in self.reducedDOF])
                FweightA, FweightB = np.zeros(6), np.zeros(6)
                FbuoyancyA, FbuoyancyB = np.zeros(6), np.zeros(6)
                if includeEndA and includeEndB: # In this case, we will just split the loads between nodes (assuming cantilevered both ends)
                    FweightA, _ = getWeightOfPointMass(mem.mass/2, mem.rCoG-mem.nodeList[ 0].r[:3], g=g)
                    FweightB, _ = getWeightOfPointMass(mem.mass/2, mem.rCoG-mem.nodeList[-1].r[:3], g=g)
                    FbuoyancyA  = transformForce(np.array([0,0, self.rho_water*g*mem.V/2]), offset=mem.rCB-mem.nodeList[ 0].r[:3])
                    FbuoyancyB  = transformForce(np.array([0,0, self.rho_water*g*mem.V/2]), offset=mem.rCB-mem.nodeList[-1].r[:3])

                elif includeEndA:
                    FweightA, _ = getWeightOfPointMass(mem.mass, mem.rCoG-mem.nodeList[0].r[:3], g=g)
                    FbuoyancyA  = transformForce(np.array([0,0, self.rho_water*g*mem.V]), offset=mem.rCB-mem.nodeList[0].r[:3])

                elif includeEndB:
                    FweightB, _ = getWeightOfPointMass(mem.mass, mem.rCoG-mem.nodeList[-1].r[:3], g=g)
                    FbuoyancyB  = transformForce(np.array([0,0, self.rho_water*g*mem.V]), offset=mem.rCB-mem.nodeList[-1].r[:3])

                iFirstEndA, iLastEndA = mem.nodeList[ 0].id * mem.nodeList[ 0].nDOF, (mem.nodeList[ 0].id + 1) * mem.nodeList[ 0].nDOF
                iFirstEndB, iLastEndB = mem.nodeList[-1].id * mem.nodeList[-1].nDOF, (mem.nodeList[-1].id + 1) * mem.nodeList[-1].nDOF

                W_internal_endNodes_struc_fullDOF[iFirstEndA:iLastEndA] += FweightA
                W_internal_endNodes_struc_fullDOF[iFirstEndB:iLastEndB] += FweightB
                W_internal_endNodes_hydro_fullDOF[iFirstEndA:iLastEndA] += FbuoyancyA
                W_internal_endNodes_hydro_fullDOF[iFirstEndB:iLastEndB] += FbuoyancyB

        # ------------------------- Transform quantities above to the reduced set of dofs -----------------------------
        self.M_struc       = self.T.T @ M_struc_fullDOF @ self.T
        self.M_struc_sub   = self.T.T @ M_struc_sub_fullDOF  @ self.T
        self.C_hydro       = self.T.T @ C_hydro_fullDOF @ self.T
        self.C_struc       = self.T.T @ C_struc_fullDOF @ self.T
        self.C_struc_sub   = self.T.T @ C_struc_sub_fullDOF  @ self.T
        self.C_elast       = self.T.T @ C_elast_fullDOF @ self.T
        self.W_struc       = self.T.T @ W_struc_fullDOF
        self.W_hydro       = self.T.T @ W_hydro_fullDOF        
        self.f0_additional = self.T.T @ f0_additional_fullDOF
        self.W_struc_internal = self.T.T @ W_internal_endNodes_struc_fullDOF
        self.W_hydro_internal = self.T.T @ W_internal_endNodes_hydro_fullDOF


        # Geometric stiffness of flexible members (due to internal loads)
        # Following the approach from Lee et al, 2024, 'On the correction of hydrostatic stiffness for discrete-module-based hydroelasticity analysis of vertically arrayed modules'
        # doi.org/10.1016/j.engstruct.2024.118710        
        def _compute_geom_stiffness(member, force):
            """
            Internal helper to compute geometric stiffness for flexible members
            """
            # Get force acting on each node of the flexible member
            Wnodes = np.zeros([len(member.nodeList), 6])
            for inode, node in enumerate(member.nodeList):
                Wnodes[inode, :] = node.T @ force # I'm not sure if we can do this

            # Compute geometric stiffness by assuming that the body is in equilibrium to get the internal forces
            K_geom = np.zeros([member.nDOF, member.nDOF])  # Initialize geometric stiffness matrix for the member
            for inode, node in enumerate(member.nodeList):
                W_after  = np.sum(Wnodes[inode+1:, :], axis=0)
                # W_before = np.sum(Wnodes[:inode, :], axis=0)
                W_before = -W_after - Wnodes[inode, :]

                r_before, r_after = np.zeros(3), np.zeros(3)
                if inode != 0:
                    r_before = (mem.nodeList[inode].r[:3] + mem.nodeList[inode-1].r[:3])/2 - mem.nodeList[inode].r[:3]
                if inode != len(mem.nodeList)-1:
                    r_after  = (mem.nodeList[inode].r[:3] + mem.nodeList[inode+1].r[:3])/2 - mem.nodeList[inode].r[:3]

                K_node = np.zeros([6, 6])
                K_node[3, 3] = (W_after[2] * r_after[2] + W_before[2] * r_before[2]) + (W_after[1] * r_after[1] + W_before[1] * r_before[1])
                K_node[4, 4] = (W_after[2] * r_after[2] + W_before[2] * r_before[2]) + (W_after[0] * r_after[0] + W_before[0] * r_before[0])
                K_node[5, 5] = (W_after[1] * r_after[1] + W_before[1] * r_before[1]) + (W_after[0] * r_after[0] + W_before[0] * r_before[0])
                K_node[3, 4] = -W_after[1] * r_after[0] - W_before[1] * r_before[0]
                K_node[3, 5] = -W_after[2] * r_after[0] - W_before[2] * r_before[0]
                K_node[4, 5] = -W_after[2] * r_after[1] - W_before[2] * r_before[1]
                K_node[4, 3] = -W_after[0] * r_after[1] - W_before[0] * r_before[1]
                K_node[5, 4] = -W_after[0] * r_after[2] - W_before[0] * r_before[2]
                K_node[5, 3] = -W_after[1] * r_after[2] - W_before[1] * r_before[2]

                K_geom[inode*node.nDOF:(inode+1)*node.nDOF, inode*node.nDOF:(inode+1)*node.nDOF] = K_node
                
            return K_geom
        
        K_geom_struc_fullDOF, K_geom_hydro_fullDOF = np.zeros([self.nFullDOF,self.nFullDOF]), np.zeros([self.nFullDOF,self.nFullDOF])
        for mem in memberList:            
            if mem.type == 'beam':
                iFirst =  mem.nodeList[ 0].id      * mem.nodeList[0].nDOF
                iLast  = (mem.nodeList[-1].id + 1) * mem.nodeList[0].nDOF
                K_geom_struc_fullDOF[iFirst:iLast, iFirst:iLast] = _compute_geom_stiffness(mem, self.W_struc+self.W_struc_internal)
                K_geom_hydro_fullDOF[iFirst:iLast, iFirst:iLast] = _compute_geom_stiffness(mem, self.W_hydro+self.W_hydro_internal)

        self.C_struc += self.T.T @ K_geom_struc_fullDOF @ self.T
        self.C_hydro += self.T.T @ K_geom_hydro_fullDOF @ self.T

        #--- Geommetric stiffness due to the variation of the transformation matrix
        C_hydro_geom = np.zeros([self.nDOF,self.nDOF])
        C_struc_geom = np.zeros([self.nDOF,self.nDOF])
        C_struc_sub_geom = np.zeros([self.nDOF,self.nDOF])
        for i in range(self.nDOF):
            for j in range(self.nDOF):
                dT = self.dT[:,:,j].T
                C_hydro_geom[i,j] = -dT[i,:] @ (W_hydro_fullDOF + W_internal_endNodes_hydro_fullDOF)
                C_struc_geom[i,j] = -dT[i,:] @ (W_struc_fullDOF + W_internal_endNodes_struc_fullDOF)
                C_struc_sub_geom[i,j] = -dT[i,:] @ W_struc_sub_fullDOF
        self.C_hydro += C_hydro_geom
        self.C_struc += C_struc_geom
        self.C_struc_sub += C_struc_sub_geom


        # Make the matrices symmetric. They should be, as the _fullDOF matrices are symmetric, but the matrix products
        # above can introduce numerical errors because some elements are usually many orders of magnitude larger than others.
        self.M_struc     = (self.M_struc + self.M_struc.T) / 2
        self.M_struc_sub = (self.M_struc_sub + self.M_struc_sub.T) / 2
        self.C_hydro     = (self.C_hydro + self.C_hydro.T) / 2
        self.C_struc     = (self.C_struc + self.C_struc.T) / 2
        self.C_struc_sub = (self.C_struc_sub + self.C_struc_sub.T) / 2
        self.C_elast     = (self.C_elast + self.C_elast.T) / 2

        # ----------- process inertia-related totals ----------------
        # To get the mass, we apply an unit translational acceleration in the X direction. This makes the whole FOWT translate as a single rigid body
        Xhelper = np.array([1 if dof[1] == 0 else 0 for dof in self.reducedDOF])
        m_all = sum((self.M_struc @ Xhelper) * Xhelper)  # Do an element-wise multiplication by Xhelper to remove couplings with other dofs
                
        # Should we compute this in a similar way as the helper function getMassAndCenterOfBeam()?
        rCG_all = m_center_sum/m_all          # total CG of all the members        
        self.rCG = rCG_all
        self.rCG_sub = m_sub_sum/self.m_sub   # solve for just the substructure mass and CG                
  
        # get principal moments of inertia (about CG) 
        # note: these are likely only useful in the unrotated frame, i.e.
        # the first time calcStatics is called.
        # TODO: Not sure about the best way to deal with these when nDOF>6. Maybe Craig-Bampton keeping the first 6 dofs?
        #       For now, just using the first 6 dofs but knowing that it is wrong for nDOF>6
        
        # the mass matrix of the substructure about the substruc's CM
        M_sub = translateMatrix6to6DOF(self.M_struc_sub[:6,:6], -self.rCG_sub)  # -rCG_sub due to convention of the function
        
        # overall structure mass matrix about its CM
        M_all = translateMatrix6to6DOF(self.M_struc[:6,:6], -self.rCG)

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
                            pnl.meshMember(mem.stations, mem.d, mem.rA0, mem.rB0, dz_max=dz, da_max=da, savedNodes=nodes, savedPanels=panels)
                            # for GDF output
                            vertices_i = pnl.meshMemberForGDF(mem.stations, mem.d, mem.rA0, mem.rB0, dz_max=dz, da_max=da)
                            vertices = np.vstack([vertices, vertices_i])  # append the member's vertices to the master list
                        elif mem.shape == "rectangular":
                            widths = mem.sl[:, 0]
                            heights = mem.sl[:, 1]
                            pnl.meshRectangularMember(mem.stations, widths, heights, mem.rA0, mem.rB0, dz_max=dz, dw_max=da, dh_max=dh, savedNodes=nodes, savedPanels=panels) #
                            # for GDF output
                            vertices_i = pnl.meshRectangularMemberForGDF(mem.stations, widths, heights, mem.rA0, mem.rB0, dz_max=dz, dw_max=da, dh_max=dh)
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
                    heading = mem.heading if hasattr(mem, "heading") else [0]
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
                            "heading": heading,
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
                            "heading": heading,
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
        to PyHAMS or strip theory, and is done when potFirstOrder == 1/True.
        
        We still need to work on multibody BEM excitation in the future
        For now, just lumping it at the first 6 reduced dofs of the structure
        '''
        import pyhams.pyhams as ph
               
        # read the preexisting WAMIT-style output files
        addedMass, damping, w1 = ph.read_wamit1(self.hydroPath+'.1', TFlag=True)  # first two entries in frequency dimension are expected to be zero-frequency then infinite frequency
        M, P, R, I, w3, heads  = ph.read_wamit3(self.hydroPath+'.3', TFlag=True)
        # The Tflag means that the first column is in units of periods, not frequencies, and therefore the first set (-1) becomes zero-frequency and the second set is infinite

        # process and sort headings and sort frequencies
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
        # For now, we will just lump at the first 6 dofs. Assuming coefficients computed wrt (0,0,0)
        for iw in range(self.nw):
            self.A_BEM[:6, :6, iw] = translateMatrix6to6DOF(             self.rho_water * addedMassInterp[:,:,iw], -self.nodeList[self.reducedDOF[0][0]].r0[:3])
            self.B_BEM[:6, :6, iw] = translateMatrix6to6DOF(self.w[iw] * self.rho_water *   dampingInterp[:,:,iw], -self.nodeList[self.reducedDOF[0][0]].r0[:3])

        # Transform excitation coefficients so that DOFs are always 
        # relative to incident wave heading rather than global frame,
        # for accurate magnitudes when interpolating between directions.
        X_BEM_temp = self.rho_water * self.g * (fExRealInterp + 1j*fExImagInterp)
        X_BEM      = np.zeros_like(X_BEM_temp)
        self.X_BEM = np.zeros((X_BEM_temp.shape[0], self.nDOF, X_BEM_temp.shape[2]), dtype=complex)

        for ih in range(len(self.BEM_headings)):
            sin_heading = np.sin(np.radians(self.BEM_headings[ih]))
            cos_heading = np.cos(np.radians(self.BEM_headings[ih]))
            
            X_BEM[ih,0,:] =  cos_heading * X_BEM_temp[ih,0,:] + sin_heading * X_BEM_temp[ih,1,:]
            X_BEM[ih,1,:] = -sin_heading * X_BEM_temp[ih,0,:] + cos_heading * X_BEM_temp[ih,1,:]
            X_BEM[ih,2,:] = X_BEM_temp[ih,2,:]
            X_BEM[ih,3,:] =  cos_heading * X_BEM_temp[ih,3,:] + sin_heading * X_BEM_temp[ih,4,:]
            X_BEM[ih,4,:] = -sin_heading * X_BEM_temp[ih,3,:] + cos_heading * X_BEM_temp[ih,4,:]
            X_BEM[ih,5,:] = X_BEM_temp[ih,5,:]
            
            for iw in range(self.nw):
                self.X_BEM[ih, :6, iw] = transformForce(X_BEM[ih,:,iw], offset= -self.nodeList[self.reducedDOF[0][0]].r0[:3])

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
        Loads and matrices are wrt to the rotor node location.

        
        case
            dictionary of case information
        ptfm_pitch
            mean pitch angle of the platform [rad]

        '''
        turbine_heading = getFromDict(case, 'turbine_heading', shape=0, dtype = float, default=0.0)  # [deg]
        turbine_status  = getFromDict(case, 'turbine_status', shape=0, dtype=str, default='operating')

        # initialize arrays (can remain zero if aerodynamics are disabled)
        self.A_aero  = np.zeros([self.nDOF,self.nDOF,self.nw,self.nrotors])                      # frequency-dependent aero-servo added mass matrix
        self.B_aero  = np.zeros([self.nDOF,self.nDOF,self.nw,self.nrotors])                      # frequency-dependent aero-servo damping matrix
        self.f_aero  = np.zeros([self.nDOF,  self.nw,self.nrotors], dtype=complex)       # dynamice excitation force and moment amplitude spectra
        self.f_aero0 = np.zeros([self.nDOF,          self.nrotors])                      # mean aerodynamic forces and moments
        #todo: reorder above so that w is last index <<<
        
        self.B_gyro  = np.zeros([self.nDOF,self.nDOF,self.nrotors])  # rotor gyroscopic damping matrix
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
                    # Note: these are about the rotor's node in global orientation.
                    f_aero0, f_aero, a_aero, b_aero = rot.calcAero(case, current=current)  # get values about hub
                    
                    # Transform quantities to the reduced set of dofs
                    # TODO: This is different from other parts of the code where I used _fullDOF matrix and then transformed to reduced DOFs.
                    #       The idea was to use the T.T matrix as a sort of a locator matrix, but not sure how correct this is. Change that later
                    T = rot.nodeList[0].T # transformation matrix from the reduced set of dofs of the FOWT to the 6 dofs of the rotor node
                    self.f_aero0[:,ir] = T.T @ f_aero0  # mean forces and moments
                    for iw in range(self.nw):
                        self.A_aero[:,:,iw,ir] = T.T @ a_aero[:,:,iw] @ T
                        self.B_aero[:,:,iw,ir] = T.T @ b_aero[:,:,iw] @ T
                        self.f_aero[:,iw,ir]   = T.T @ f_aero[:,iw]  # convert excitation forces to the reduced set of dofs

                    # calculate cavitation of the rotor (platform motions should already be accounted for in the CCBlade object after running calcAero)
                    if rot.r3[2] < 0:  # if submerged
                        self.cav = rot.calcCavitation(case)  # TO-DO: wire this to be a result/output, then uncomment <<<
                    
                    # ----- calculate rotor gyroscopic effects -----
                    # rotor speed [rpm]
                    Omega_rpm = np.interp(speed, rot.Uhub, rot.Omega_rpm)
                    
                    Omega_rotor = rot.q * Omega_rpm*2*np.pi/60  # rotor angular velocity vector
                    
                    # rotating inertia vector [kg-m^2 * rad/s = N-m-s]
                    IO_rotor = rot.I_drivetrain * Omega_rotor
                    
                    GyroDampingMatrix = np.zeros((6, 6))
                    GyroDampingMatrix[3:, 3:] = getH(IO_rotor)
                    
                    self.B_gyro[:, :, ir] = T.T @ GyroDampingMatrix @ T
                    
                    # note, gyroscopic effect is purely rotational so no translation adjustment needed

        else:
            print(f"Warning: turbine status is '{turbine_status}' so rotor fluid loads are neglected.")


    def calcHydroConstants(self):
        '''Compute FOWT hydrodynamic added mass matrix and member-level
        inertial excitation coefficients.'''


        rho = self.rho_water
        g   = self.g
        
        # --------------------- get constant hydrodynamic values along each member -----------------------------        
        self.A_hydro_morison = np.zeros([self.nDOF, self.nDOF])                # hydrodynamic added mass matrix, from only Morison equation [kg, kg-m, kg-m^2]
        A_hydro_morison_fullDOF = np.zeros([self.nFullDOF, self.nFullDOF])   # same but in the full set of dofs

        # loop through each member

        for i,mem in enumerate(self.memberList):
            # Find the indices of the first and last nodes of the member to fill the _fullDOF matrices
            iFirst =  mem.nodeList[ 0].id      * mem.nodeList[0].nDOF
            iLast  = (mem.nodeList[-1].id + 1) * mem.nodeList[0].nDOF # Assuming all nodes have the same dofs, namely 6
        
            # get member added mass matrix about PRP (also saves each member's inertial excitation coefficients)
            if mem.MCF:
                k_array = self.k
            else:
                k_array = None

            A_hydro_morison_fullDOF[iFirst:iLast, iFirst:iLast] += mem.calcHydroConstants(rho=rho, g=g, k_array=k_array)
    
        # ----- Get hydrodynamic contributions from any underwater rotors ------
        for i, rot in enumerate(self.rotorList):
            iFirst =  rot.nodeList[ 0].id      * rot.nodeList[0].nDOF
            iLast  = (rot.nodeList[-1].id + 1) * rot.nodeList[0].nDOF
            
            # compute rotor hydro added mass/inertia properties
            A_hydro_i, I_hydro_i = rot.calcHydroConstants(rho=rho, g=g)
            A_hydro_morison_fullDOF[iFirst:iLast, iFirst:iLast] += A_hydro_i
            
        self.A_hydro_morison = self.T.T @ A_hydro_morison_fullDOF @ self.T  # transform to the reduced set of dofs
    
    
    def getStiffness(self):
        '''Sums up all the stiffness effects on a FOWT.'''
        
        C_tot = np.zeros([self.nDOF, self.nDOF])       # total stiffness matrix [N/m, N, N-m]
        
        # add in mooring stiffness from MoorPy system
        C_tot += self.C_moor
        # add any additional yaw stiffness that isn't included in the MoorPy 
        # model (e.g. if a bridle isn't modeled)
        C_tot[5,5] += self.yawstiff # TODO: Remove this now that we have additional_effects?
        # add system-level stiffness effect if it exists...
        if self.body:
            C_tot += self.body.getStiffness()  # in future should make an analytic body function for this

        C_tot += self.C_struc + self.C_hydro + self.C_elast  # stiffness
        
        return C_tot
    
    def solveEigen(self, display=0, outPath=None):
        '''Compute the natural frequencies and mode shapes of the FOWT.
        This considers the FOWT and any attached mooring lines, but it
        does not account for coupling with other FOWTs as could occur
        with an array-level MoorPy instance with shared lines or 
        suspended cables.
        
        Returns
        -------
        wns : array
            List of natural frequencies [rad/s].
        modes : 2D array
            List of mode shapes (eigenvectors) corresponding to the natural 
            frequencies.
        '''

        # Total mass and added mass matrix [kg, kg-m, kg-m^2]
        M_tot = self.M_struc + self.A_hydro_morison + self.A_BEM[:,:,0]   # Mass. Using BEM added mass at w=0 because it is closer to the expected natural frequencies than w=inf
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

        # For rigid FOWTs, using the old method for now
        if self.nDOF == 6:
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
        
        else:
            eigenvals, eigenvectors = np.linalg.eig(np.linalg.solve(M_tot, C_tot))

            # Sort in ascending order of eigenvals
            sorted_indices = np.argsort(eigenvals)
            eigenvals = eigenvals[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]

            fns = np.sqrt(eigenvals)/2.0/np.pi
            modes = eigenvectors
        
        if isinstance(outPath, str):
            self.write_modes_json(outPath, fns, modes)

        if display > 0:
            print("")
            print("--------- Natural frequencies and mode shapes -------------")
            print("Mode        1         2         3         4         5         6")
            print("Fn (Hz)"+"".join([f"{fn:10.4f}" for fn in fns[:6]]))
            print("")
            for i in range(6):
                print(f"DOF {i+1}  "+"".join([f"{modes[i,j]:10.4f}" for j in range(6)]))
            print("-----------------------------------------------------------")
    
        return fns, modes
    

    def calcHydroExcitation(self, case, memberList=[]):
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
        self.zeta = np.zeros([self.nWaves,self.nw])

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
        
        # Set up wave kinematics arrays for the rotors
        for i,rot in enumerate(self.rotorList):
            rot.u    = np.zeros([self.nWaves, 3, self.nw], dtype=complex)
            rot.ud   = np.zeros([self.nWaves, 3, self.nw], dtype=complex)
            rot.pDyn = np.zeros([self.nWaves,    self.nw], dtype=complex)
            
        # ----- calculate potential-flow wave excitation force -----

        self.F_BEM = np.zeros([self.nWaves,self.nDOF,self.nw], dtype=complex)
        self.F_BEM_fullDOF = np.zeros([self.nWaves,self.nFullDOF,self.nw], dtype=complex)
        self.F_hydro_iner = np.zeros([self.nWaves, self.nDOF, self.nw],dtype=complex) # inertia excitation force/moment complex amplitudes vector [N, N-m]
        self.F_hydro_iner_fullDOF = np.zeros([self.nWaves, self.nFullDOF, self.nw],dtype=complex) # Same but in full DOFs        

        # BEM-based wave excitation force on platform for each wave heading 
        # (will be zero if only using strip theory). Includes wave heading interpolation.
        # TODO: currently lumping at the first 6dofs. Should implement multi-body solution in the future
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
                self.F_BEM_fullDOF[ih, :6, :] = X_BEM_ih * self.zeta[ih,:] * phase_offset

        # ----- strip-theory wave excitation force -----
        # loop through each member to compute strip-theory contributions
        # This also saves the save wave kinematics over each member.
        for i,mem in enumerate(memberList):
            iFirst =  mem.nodeList[ 0].id      * mem.nodeList[0].nDOF
            iLast  = (mem.nodeList[-1].id + 1) * mem.nodeList[-1].nDOF
            self.F_hydro_iner_fullDOF[:, iFirst:iLast, :] += mem.calcHydroExcitation(self.zeta, self.beta, self.w, self.depth, k=self.k)
        
        # ----- inertial excitation on rotor(s) -----
        
        for i, rot in enumerate(self.rotorList):
            if rot.r3[2] < 0:  # if submerged
                # Find the indices of the first and last nodes of the member to fill the _fullDOF matrices
                iFirst =  rot.nodeList[ 0].id      * rot.nodeList[0].nDOF
                iLast  = (rot.nodeList[-1].id + 1) * rot.nodeList[0].nDOF # Assuming all nodes have the same dofs, namely 6

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
                    self.F_hydro_iner_fullDOF[ih, iFirst:iLast, i] += f6  # add to the full DOF matrix

        # transform to the reduced set of dofs
        for ih in range(self.nWaves):
            self.F_BEM[ih, :, :] = self.T.T @ self.F_BEM_fullDOF[ih, :, :] 
            self.F_hydro_iner[ih, :, :] = self.T.T @ self.F_hydro_iner_fullDOF[ih, :, :]            
                

    def calcHydroLinearization(self, Xi):
        '''The FOWT's dynamics solve iteration method. This calculates the amplitude-dependent 
        linearized coefficients, including the system linearized drag damping matrix. For the 
        drag-based excitation, call calcDragExcitation after this method.
        
        Currently hard-coded to only consider the first seastate/heading.

        Xi : complex array
            system response (just for this FOWT) - displacement and rotation complex amplitudes 
            in the reduced dofs of the FOWT [m, rad]

        '''
        # The linearized coefficients to be calculated

        B_hydro_drag = np.zeros([self.nDOF,self.nDOF])             # hydrodynamic damping matrix (just linearized viscous drag for now) [N-s/m, N-s, N-s-m]
        F_hydro_drag = np.zeros([self.nDOF,self.nw],dtype=complex) # excitation force/moment complex amplitudes vector [N, N-m]
        B_hydro_drag_fullDOF = np.zeros([self.nFullDOF,self.nFullDOF])  # same but in full dofs
        F_hydro_drag_fullDOF = np.zeros([self.nFullDOF,self.nw],dtype=complex)

        ih = 0  # we will only consider the first sea state in this linearization process

        # loop through each member
        for mem in self.memberList:
            # Find the indices of the first and last nodes of the member to fill the _fullDOF matrices
            iFirst =  mem.nodeList[ 0].id      * mem.nodeList[0].nDOF
            iLast  = (mem.nodeList[-1].id + 1) * mem.nodeList[0].nDOF # Assuming all nodes have the same dofs, namely 6

            # Compute the displacements of each structural node
            Xi_nodes = np.zeros((mem.nDOF, len(self.w)), dtype=complex)
            for inode, node in enumerate(mem.nodeList):
                Xi_nodes[inode*node.nDOF:inode*node.nDOF+node.nDOF, :] = node.T @ Xi

            B_hydro_drag_member, F_hydro_drag_member = mem.calcHydroLinearization(self.w, ih=ih, Xi_nodes=Xi_nodes, rho=self.rho_water)
            B_hydro_drag_fullDOF[iFirst:iLast, iFirst:iLast] = B_hydro_drag_member
            F_hydro_drag_fullDOF[iFirst:iLast, :] = F_hydro_drag_member

        B_hydro_drag += self.T.T @ B_hydro_drag_fullDOF @ self.T  # add to damping matrix for Morison members in the reduced set of dofs
        for iw in range(self.nw):
            F_hydro_drag[:,iw] += self.T.T @ F_hydro_drag_fullDOF[:,iw]  # add to global excitation vector (frequency dependent) in the reduced set of dofs

        # save the arrays internally in case there's ever a need for the FOWT to solve it's own latest dynamics or for visualization
        self.B_hydro_drag = B_hydro_drag
        self.F_hydro_drag = F_hydro_drag

        # return the linearized coefficients
        return B_hydro_drag
    
    
    
    def calcDragExcitation(self, ih):
        '''Calculated linearized viscous drag excitation for a given sea state (wave heading). calcHydroLinearization should be called first.

        ih : int
            index of wave case being evaluated here

        '''
        self.F_hydro_drag = np.zeros([self.nDOF,self.nw],dtype=complex) # excitation force/moment complex amplitudes vector [N, N-m]
        self.F_hydro_drag_fullDOF = np.zeros([self.nFullDOF,self.nw],dtype=complex)
        for mem in self.memberList:   # loop through each member
            iFirst =  mem.nodeList[ 0].id      * mem.nodeList[0].nDOF
            iLast  = (mem.nodeList[-1].id + 1) * mem.nodeList[0].nDOF
            self.F_hydro_drag_fullDOF[iFirst:iLast, :] += mem.calcDragExcitation(ih)

        for iw in range(self.nw):
            self.F_hydro_drag[:,iw] += self.T.T @ self.F_hydro_drag_fullDOF[:,iw]

        return self.F_hydro_drag
    
    
    
    def calcCurrentLoads(self, case):
        '''method to calculate the "static" current loads on each member and save as a current force
        Uses a simple power law relationship to calculate the current velocity as a function of member node depth'''
        D_hydro_fullDOF = np.zeros(self.nFullDOF)  # same but in full dofs

        # extract current variables out of the case dictionary
        speed = getFromDict(case, 'current_speed', shape=0, default=0.0)
        heading = getFromDict(case, 'current_heading', shape=0, default=0)

        
        Zref = 0.0  # reference z elevation for current profile (at the sea surface by default) (reference height set to submerged rotor hub depth if rotor is submerged)
        for ti in range(self.nrotors):
            if self.rotorList[ti].r3[2] < 0:     # If there is a submerged rotor,
                Zref = self.rotorList[ti].r3[2]  # use it for the reference current height.

        # loop through each member
        for mem in self.memberList:
            iFirst =  mem.nodeList[ 0].id      * mem.nodeList[0].nDOF
            iLast  = (mem.nodeList[-1].id + 1) * mem.nodeList[0].nDOF
            D_hydro_fullDOF[iFirst:iLast] += mem.calcCurrentLoads(self.depth, speed=speed, heading=heading, Zref=Zref, shearExp_water=self.shearExp_water, rho=self.rho_water, g=self.g)
            
        # transform to the reduced set of dofs
        self.D_hydro = self.T.T @ D_hydro_fullDOF

        return self.D_hydro


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
        # Big TODO: This doesn't work with flexible/multibody FOWTs yet

        # In case the body is fixed
        if Xi0 is None:
            Xi0 = np.zeros([self.nDOF, len(self.w)], dtype=complex)

        # Get wave heading beta
        beta = self.beta[waveHeadInd]
        self.heads_2nd = [beta] # Need this one to print the QTF
        whead = f"{np.degrees(beta)%360:.2f}".replace('.', 'p') # String with heading in range of 0-360 [deg] for output files
        
        # Initialize qtf that will be used by the solver
        self.qtf = np.zeros([len(self.w1_2nd), len(self.w2_2nd), 1, self.nDOF], dtype=complex)  # Need this fourth dimension for conformity with the case where the QTFs are read from a file
        if self.nDOF > 6:
            print("Function calcQTF_slenderBody() is not implemented for flexible/multibody FOWTs yet. Considering null qtf matrices for now.")
            return

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
        F1st = np.matmul(self.M_struc, (-self.w1_2nd**2 * Xi))

        if verbose:
            print(f" Computing QTF for heading {beta:.2f}")

        # We start with the force component due to rotation of first-order wave forces (force component IV from Pinkster (1979)).
        # This component depends on the forces on the whole body, hence it is outside of the member loop below.
        # TODO: Need to change this for multibody/flexible FOWTs
        for i1 in range(len(self.w1_2nd)):
            for i2 in range(i1, len(self.w2_2nd)):
                if self.w2_2nd[i2] < self.w1_2nd[i1]: # We don't need to fill the whole matrix, only the upper triangle
                        continue
                F_rotN = np.zeros(self.nDOF, dtype='complex')
                F_rotN[0:3] = 0.25 * (np.cross(Xi[3:,i1], np.conj(F1st[0:3,i2])) + np.cross(np.conj(Xi[3:,i2]), F1st[0:3,i1]))
                F_rotN[3: ] = 0.25 * (np.cross(Xi[3:,i1], np.conj(F1st[3: ,i2])) + np.cross(np.conj(Xi[3:,i2]), F1st[3: ,i1]))
                self.qtf[i1,i2,waveHeadInd,:] = F_rotN

        # Loop each member to compute force terms along the member
        for i,mem in enumerate(self.memberList):
            if verbose:
                print(f"    Element {i+1} of {len(self.memberList)}       ")
            self.qtf[:, :, waveHeadInd, :] += mem.calcQTF_slenderBody(Xi, beta, self.w1_2nd, self.k1_2nd, self.depth, rho=self.rho_water, g=self.g, verbose=verbose)

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
            indhead, = np.where(self.heads_2nd==deg2rad(row[2])) # index for heading
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
            qtf_interpBetaRe = interp1d(self.heads_2nd, self.qtf.real, assume_sorted=True, axis=2, bounds_error=False, fill_value=(self.qtf[:,:,0,:].real, self.qtf[:,:,-1,:].real))(beta)
            qtf_interpBetaIm = interp1d(self.heads_2nd, self.qtf.imag, assume_sorted=True, axis=2, bounds_error=False, fill_value=(self.qtf[:,:,0,:].imag, self.qtf[:,:,-1,:].imag))(beta)
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

    def evaluateStiffnessMatrix(self):        
        if len(self.nodeList) == 0:
            raise Exception("No nodes in the structure. Cannot compute stiffness matrix.")
        nDOF = self.nodeList[0].nDOF # Number of dofs of a single node

        # Stiffness matrix of the whole 6*Nnodes system of equations
        self.Kfull = np.zeros([len(self.nodeList)*nDOF, len(self.nodeList)*nDOF])

        # Node's external stiffness - will be hydrostatics, mooring, etc in the future
        for i, node in enumerate(self.nodeList):            
            self.Kfull[i*nDOF:(i+1)*nDOF, i*nDOF:(i+1)*nDOF] = node.K
        
        # Internal stiffness for flexible members
        # TODO: Make this directly from node? Store the rows of the matrix in the node
        for m in self.memberList:
            m.computeStiffnessMatrix() # Make sure the stiffness matrix is computed
            if m.type == 'beam':
                col_first = m.nodeList[0].id*nDOF # ID of the first node of the member
                col_last  = (m.nodeList[-1].id+1)*nDOF # ID of the last node of the member
                for n in m.nodeList:
                    # Assign the rows of m.K that correspond to this node to matrix self.Kfull
                    i   = n.id # Node of the current member - Index in the matrix with all nodes
                    i_m = i - m.nodeList[0].id # Index of the node in the member list of nodes
                    self.Kfull[i*nDOF:(i+1)*nDOF, col_first:col_last] += m.K[i_m*nDOF:(i_m+1)*nDOF, :]
        return self.Kfull        

    def updateMooringDynamicMatrices(self, Xi, S):
        '''Update matrices from mooring dynamics
        Inputs
        Xi: 6 x Nfreq array with the motion amplitudes of the fowt
        S:  1 x Nfreq array with the wave spectrum
        '''
        for line in self.ms.lineList:
            RAO_A, RAO_B = getLineEndsRAO(line, self.ms, self.w, [Xi], S, [self.r6[:3]]) # Need Xi and r6 within a list
            line.updateLumpedMass(self.w, S, self.depth, kbot=0, cbot=0, RAO_A=RAO_A.T, RAO_B=RAO_B.T) # Need to transpose RAO_fl so that it is in the right shape (nFreq x 3)

    def saveTurbineOutputs(self, results, case):
        '''Calculate and store output metrics of the FOWT response at the current load case.
        Note that the FOWT offset, motions, load case info, etc. are taken from what is stored
        in the FOWT object. I.e. >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   
        Load cases may have multiple sources of excitation (such as wind, wave, and another wave
        at a different heading). Results are computed by RMS summing across these excitation sources.
        '''
        
        # Get motions of the PRP based on the motions of the rigidBodyNode
        Xi0_PRP = self.r6 - np.array([self.x_ref, self.y_ref,0,0,0,0])  # FOWT's mean offset vector [m, rad]
        
        Xi_rigidBodyNode  = self.Xi_fullDOF[:, self.rigidBodyNode.id:self.rigidBodyNode.id+6, :]        
        Xi_PRP  = np.zeros_like(Xi_rigidBodyNode)
        for ih in range(Xi_PRP.shape[0]):
            for iw in range(Xi_PRP.shape[2]):
                Xi_PRP[ih,:3, iw] = Xi_rigidBodyNode[ih,:3,iw] + SmallRotate(-self.rigidBodyNode.r0, Xi_rigidBodyNode[ih,3:,iw])
                Xi_PRP[ih,3:, iw] = Xi_rigidBodyNode[ih,3:,iw]

        # platform motions
        results['surge_avg'] = Xi0_PRP[0]
        results['surge_std'] = getRMS(Xi_PRP[:,0,:]) 
        results['surge_max'] = Xi0_PRP[0] + 3*results['surge_std']
        results['surge_min'] = Xi0_PRP[0] - 3*results['surge_std']
        results['surge_PSD'] = getPSD(Xi_PRP[:,0,:], self.dw)
        results['surge_RA' ] = Xi_PRP[:,0,:]

        results['sway_avg'] = Xi0_PRP[1]
        results['sway_std'] = getRMS(Xi_PRP[:,1,:])
        results['sway_max'] = Xi0_PRP[1] + 3*results['sway_std']
        results['sway_min'] = Xi0_PRP[1] - 3*results['sway_std']
        results['sway_PSD'] = getPSD(Xi_PRP[:,1,:], self.dw)
        results['sway_RA' ] = Xi_PRP[:,1,:]

        results['heave_avg'] = Xi0_PRP[2]
        results['heave_std'] = getRMS(Xi_PRP[:,2,:])
        results['heave_max'] = Xi0_PRP[2] + 3*results['heave_std']
        results['heave_min'] = Xi0_PRP[2] - 3*results['heave_std']
        results['heave_PSD'] = getPSD(Xi_PRP[:,2,:], self.dw)
        results['heave_RA' ] = Xi_PRP[:,2,:]
        
        roll_deg = rad2deg(Xi_PRP[:,3,:])
        results['roll_avg'] = rad2deg(Xi0_PRP[3])
        results['roll_std'] = getRMS(roll_deg)
        results['roll_max'] = rad2deg(Xi0_PRP[3]) + 3*results['roll_std']
        results['roll_min'] = rad2deg(Xi0_PRP[3]) - 3*results['roll_std']
        results['roll_PSD'] = getPSD(roll_deg, self.dw)
        results['roll_RA' ] = rad2deg(Xi_PRP[:,3,:])
        
        pitch_deg = rad2deg(Xi_PRP[:,4,:])
        results['pitch_avg'] = rad2deg(Xi0_PRP[4])
        results['pitch_std'] = getRMS(pitch_deg)
        results['pitch_max'] = rad2deg(Xi0_PRP[4]) + 3*results['pitch_std']
        results['pitch_min'] = rad2deg(Xi0_PRP[4]) - 3*results['pitch_std']
        results['pitch_PSD'] = getPSD(pitch_deg, self.dw)
        results['pitch_RA' ] = rad2deg(Xi_PRP[:,4,:])
        
        yaw_deg = rad2deg(Xi_PRP[:,5,:])
        results['yaw_avg'] = rad2deg(Xi0_PRP[5])
        results['yaw_std'] = getRMS(yaw_deg)
        results['yaw_max'] = rad2deg(Xi0_PRP[5]) + 3*results['yaw_std']
        results['yaw_min'] = rad2deg(Xi0_PRP[5]) - 3*results['yaw_std']
        results['yaw_PSD'] = getPSD(yaw_deg, self.dw)
        results['yaw_RA' ] = rad2deg(Xi_PRP[:,5,:])
        
        # ----- turbine-level mooring outputs (similar code as array-level) -----
        if self.ms:            
            self.ms = lines2ss(self.ms) # convert composite lines to subsystem
            nLines = len(self.ms.lineList)
            T_moor_amps = np.zeros([self.nWaves+1, 2*nLines, self.nw], dtype=complex)  # mooring tension amplitudes for each wave component and each line end
            T_moor_psd  = np.zeros([2*nLines, self.nw], dtype=float)
            T_moor_std  = np.zeros([2*nLines], dtype=float)
            C_moor, J_moor = self.ms.getCoupledStiffness(lines_only=True, tensions=True) # get stiffness matrix and tension jacobian matrix
            T_moor = self.ms.getTensions()  # get line end mean tensions                        
            if self.moorMod == 0:
                for ih in range(self.nWaves+1):
                    for iw in range(self.nw):
                        T_moor_amps[ih,:,iw] = np.matmul(J_moor, Xi_PRP[ih,:,iw])   # FFT of mooring tensions # TODO: account for nDOF > 6

                for iT in range(2*nLines):
                    T_moor_psd[iT,:]  = getPSD(T_moor_amps[:,iT,:], self.w[0]) # PSD in N^2/(rad/s)
                    T_moor_std[iT] = getRMS(T_moor_amps[:,iT,:])
            
            else:
                for il, line in enumerate(self.ms.lineList):                    
                    for ih in range(self.nWaves):
                        RAO_A, RAO_B = getLineEndsRAO(line, self.ms, self.w, [Xi_PRP[ih,:,:]], self.S[ih,:], [self.r6[:3]]) # Need Xi and r6 within a list
                        T_nodes_amp, _, _, _, _, _, _, _ = line.dynamicSolve(self.w, self.S[ih,:], RAO_A=RAO_A.T, RAO_B=RAO_B.T, depth=self.depth, kbot=0,cbot=0, tol = 0.01, conv_time=False)

                        # Tension at the end nodes of the line
                        T_moor_amps[ih, il, :] += T_nodes_amp[:,0]
                        T_moor_amps[ih, il+nLines,:] += T_nodes_amp[:,-1]

                # Compute PSD and std from the different wave component
                for iT in range(2*nLines):
                    T_moor_psd[iT,:] = getPSD(T_moor_amps[:,iT,:], self.w[1]-self.w[0])       
                    T_moor_std[iT] = getRMS(T_moor_amps[:,iT,:])

                
            results['Tmoor_avg'] = T_moor
            results['Tmoor_std'] = np.zeros(2*nLines)
            results['Tmoor_max'] = np.zeros(2*nLines)
            results['Tmoor_min'] = np.zeros(2*nLines)
            results['Tmoor_PSD'] = np.zeros([ 2*nLines, self.nw])
            for iT in range(2*nLines):
                TRMS = T_moor_std[iT]
                results['Tmoor_std'][iT] = TRMS
                results['Tmoor_max'][iT] =  T_moor[iT] + 3*TRMS
                results['Tmoor_min'][iT] =  T_moor[iT] - 3*TRMS
                results['Tmoor_PSD'][iT, :] = (getPSD(T_moor_amps[:,iT,:], self.w[0])) # PSD in N^2/(rad/s)
        
        # hub displacement amplitude and acceleration
        XiHub = np.zeros([self.Xi.shape[0], 6*self.nrotors, self.nw], dtype=complex)

        results['AxRNA_std'] = np.zeros(self.nrotors) 
        results['AxRNA_PSD'] = np.zeros([self.nw, self.nrotors]) 
        results['AxRNA_avg'] = np.zeros(self.nrotors)
        results['AxRNA_max'] = np.zeros(self.nrotors)
        results['AxRNA_min'] = np.zeros(self.nrotors)

        results['AyRNA_std'] = np.zeros(self.nrotors) 
        results['AyRNA_PSD'] = np.zeros([self.nw, self.nrotors]) 
        results['AyRNA_avg'] = np.zeros(self.nrotors)
        results['AyRNA_max'] = np.zeros(self.nrotors)
        results['AyRNA_min'] = np.zeros(self.nrotors)

        results['AzRNA_std'] = np.zeros(self.nrotors) 
        results['AzRNA_PSD'] = np.zeros([self.nw, self.nrotors]) 
        results['AzRNA_avg'] = np.zeros(self.nrotors)
        results['AzRNA_max'] = np.zeros(self.nrotors)
        results['AzRNA_min'] = np.zeros(self.nrotors)
        
        for ir, rotor in enumerate(self.rotorList):
            XiHub[:,ir*6:(ir+1)*6,:] = self.Xi_fullDOF[:, rotor.nodeList[0].id*6:(rotor.nodeList[0].id+1)*6, :] # get the hub motion

            # nacelle acceleration in the global X direction
            results['AxRNA_std'][ir]   = getRMS(XiHub[:,ir*6,:]*self.w**2)
            results['AxRNA_PSD'][:,ir] = getPSD(XiHub[:,ir*6,:]*self.w**2, self.dw)
            results['AxRNA_avg'][ir]   = abs(np.sin(rotor.nodeList[0].r[4])*self.g) # @Matt check this! 
            results['AxRNA_max'][ir]   = results['AxRNA_avg'][ir]+3*results['AxRNA_std'][ir]
            results['AxRNA_min'][ir]   = results['AxRNA_avg'][ir]-3*results['AxRNA_std'][ir]

            # nacelle acceleration in the global Y direction
            results['AyRNA_std'][ir]   = getRMS(XiHub[:,ir*6+1,:]*self.w**2)
            results['AyRNA_PSD'][:,ir] = getPSD(XiHub[:,ir*6+1,:]*self.w**2, self.dw)
            results['AyRNA_avg'][ir]   = abs(np.sin(rotor.nodeList[0].r[3])*self.g)
            results['AyRNA_max'][ir]   = results['AyRNA_avg'][ir]+3*results['AyRNA_std'][ir]
            results['AyRNA_min'][ir]   = results['AyRNA_avg'][ir]-3*results['AyRNA_std'][ir]

            # nacelle acceleration in the global Z direction
            results['AzRNA_std'][ir]   = getRMS(XiHub[:,ir*6+2,:]*self.w**2)
            results['AzRNA_PSD'][:,ir] = getPSD(XiHub[:,ir*6+2,:]*self.w**2, self.dw)
            results['AzRNA_avg'][ir]   = abs(self.g)
            results['AzRNA_max'][ir]   = results['AzRNA_avg'][ir]+3*results['AzRNA_std'][ir]
            results['AzRNA_min'][ir]   = results['AzRNA_avg'][ir]-3*results['AzRNA_std'][ir]


        # tower base bending moment
        # TODO: should we compute the loads on all joints in solveStatics() and solveDynamics()?
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

        results['FbaseX_avg'] = np.zeros(self.nrotors)
        results['FbaseX_std'] = np.zeros(self.nrotors)
        results['FbaseX_PSD'] = np.zeros([self.nw, self.nrotors])
        results['FbaseX_max'] = np.zeros(self.nrotors)
        results['FbaseX_min'] = np.zeros(self.nrotors)

        results['FbaseY_avg'] = np.zeros(self.nrotors)
        results['FbaseY_std'] = np.zeros(self.nrotors)
        results['FbaseY_PSD'] = np.zeros([self.nw, self.nrotors])
        results['FbaseY_max'] = np.zeros(self.nrotors)
        results['FbaseY_min'] = np.zeros(self.nrotors)

        results['FbaseZ_avg'] = np.zeros(self.nrotors)
        results['FbaseZ_std'] = np.zeros(self.nrotors)
        results['FbaseZ_PSD'] = np.zeros([self.nw, self.nrotors])
        results['FbaseZ_max'] = np.zeros(self.nrotors)
        results['FbaseZ_min'] = np.zeros(self.nrotors)

        results['MbaseX_avg'] = np.zeros(self.nrotors)
        results['MbaseX_std'] = np.zeros(self.nrotors)
        results['MbaseX_PSD'] = np.zeros([self.nw, self.nrotors])
        results['MbaseX_max'] = np.zeros(self.nrotors)
        results['MbaseX_min'] = np.zeros(self.nrotors)

        results['MbaseY_avg'] = np.zeros(self.nrotors)
        results['MbaseY_std'] = np.zeros(self.nrotors)
        results['MbaseY_PSD'] = np.zeros([self.nw, self.nrotors])
        results['MbaseY_max'] = np.zeros(self.nrotors)
        results['MbaseY_min'] = np.zeros(self.nrotors)

        results['MbaseZ_avg'] = np.zeros(self.nrotors)
        results['MbaseZ_std'] = np.zeros(self.nrotors)
        results['MbaseZ_PSD'] = np.zeros([self.nw, self.nrotors])
        results['MbaseZ_max'] = np.zeros(self.nrotors)
        results['MbaseZ_min'] = np.zeros(self.nrotors)                                        

        for ir, rotor in enumerate(self.rotorList):
            mem_tower = self.memberList[self.nplatmems+ir]

            # For now, using same method as before for rigid towers
            # TODO: remove this and compute tower base loads based on the reaction loads at the tower-bottom joint
            if mem_tower.type == 'rigid':
                m_turbine[ir] = self.mtower[ir] + rotor.mRNA  # total masses of each turbine
                zCG_turbine[ir] = (self.rCG_tow[ir][2]*self.mtower[ir]                 # CoG of each turbine
                                    + rotor.r_rel[2]*rotor.mRNA)/m_turbine[ir]
                zBase[ir] = mem_tower.rA[2]                  # tower base elevation [m]
                hArm[ir] = zCG_turbine[ir] - zBase[ir]                                 # vertical distance from tower base to turbine CG [m]

                aCG_turbine[:,ir,:] = -self.w**2 *( self.Xi[:,0,:] + zCG_turbine[ir]*self.Xi[:,4,:] )  # fore-aft acceleration of turbine CG

                # turbine pitch moment of inertia about CG [kg-m^2]
                ICG_turbine[ir] = (translateMatrix6to6DOF(mem_tower.M_struc, mem_tower.nodeList[0].r0[:3] - [0,0, zCG_turbine[ir]])[4,4] # tower MOI about turbine CG
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
                results['Mbase_avg'][ir] = (m_turbine[ir]*self.g * hArm[ir]*np.sin(self.Xi0[4]) + transformForce(self.rotorList[ir].f0, offset=[0,0,self.rotorList[ir].nodeList[0].r0[2]-hArm[ir]])[4] )
                results['Mbase_std'][ir] = dynamic_moment_RMS[ir]
                results['Mbase_PSD'][:,ir] = (getPSD(dynamic_moment[:,ir,:], self.dw))
                results['Mbase_max'][ir] = results['Mbase_avg'][ir]+3*results['Mbase_std'][ir]
                results['Mbase_min'][ir] = results['Mbase_avg'][ir]-3*results['Mbase_std'][ir]
                #results['Mbase_DEL'][iCase]                
            
            # For a flexible tower, we compute the internal loads using the finite-element stiffness matrix
            else:
                # Get displacements of the internal nodes
                Xi0_internal = np.concatenate([n.Xi0 for n in mem_tower.nodeList])   # mean displacements of the internal nodes. Stored at node level
                iFirst, iLast = mem_tower.nodeList[0].id, mem_tower.nodeList[-1].id  # index range of the nodes of the tower
                Xi_internal = self.Xi_fullDOF[:, iFirst*6:(iLast+1)*6, :]            # dynamic displacements of the internal nodes

                # Internal loads acting on each node
                Fi0_internal = -mem_tower.Kf @ Xi0_internal  # static (mean) loads. K @ X gives the static load acting on the nodes, whereas -K @ X gives the internal reaction
                Fi_internal  = np.zeros_like(Xi_internal)    # dynamic loads
                for ih in range(self.nWaves+1):
                    Fi_internal[ih, :, :] = -mem_tower.Kf @ Xi_internal[ih, :, :]

                # Find which tower end is the base node
                if mem_tower.nodeList[0].r0[2] <= mem_tower.nodeList[-1].r0[2]:
                    Fi0_base = Fi0_internal[0:6]      # base node is the first node
                    Fi_base  = Fi_internal[:, 0:6, :] # base node is the first
                else:
                    Fi0_base = Fi0_internal[-6:]      # base node is the last node
                    Fi_base  = Fi_internal[:, -6:, :] # base node is the last

                # Fill in metrics
                results['FbaseX_avg'][ir]    = Fi0_base[0]
                results['FbaseX_std'][ir]    = getRMS(Fi_base[:, 0, :])
                results['FbaseX_PSD'][:, ir] = getPSD(Fi_base[:, 0, :], self.dw)
                results['FbaseX_max'][ir]    = results['FbaseX_avg'][ir] + 3 * results['FbaseX_std'][ir]
                results['FbaseX_min'][ir]    = results['FbaseX_avg'][ir] - 3 * results['FbaseX_std'][ir]

                results['FbaseY_avg'][ir]    = Fi0_base[1]
                results['FbaseY_std'][ir]    = getRMS(Fi_base[:, 1, :])
                results['FbaseY_PSD'][:, ir] = getPSD(Fi_base[:, 1, :], self.dw)
                results['FbaseY_max'][ir]    = results['FbaseY_avg'][ir] + 3 * results['FbaseY_std'][ir]
                results['FbaseY_min'][ir]    = results['FbaseY_avg'][ir] - 3 * results['FbaseY_std'][ir]

                results['FbaseZ_avg'][ir]    = Fi0_base[2]
                results['FbaseZ_std'][ir]    = getRMS(Fi_base[:, 2, :])
                results['FbaseZ_PSD'][:, ir] = getPSD(Fi_base[:, 2, :], self.dw)
                results['FbaseZ_max'][ir]    = results['FbaseZ_avg'][ir] + 3 * results['FbaseZ_std'][ir]
                results['FbaseZ_min'][ir]    = results['FbaseZ_avg'][ir] - 3 * results['FbaseZ_std'][ir]

                results['MbaseX_avg'][ir]    = Fi0_base[3]
                results['MbaseX_std'][ir]    = getRMS(Fi_base[:, 3, :])
                results['MbaseX_PSD'][:, ir] = getPSD(Fi_base[:, 3, :], self.dw)
                results['MbaseX_max'][ir]    = results['MbaseX_avg'][ir] + 3 * results['MbaseX_std'][ir]
                results['MbaseX_min'][ir]    = results['MbaseX_avg'][ir] - 3 * results['MbaseX_std'][ir]

                results['MbaseY_avg'][ir]    = Fi0_base[4]
                results['MbaseY_std'][ir]    = getRMS(Fi_base[:, 4, :])
                results['MbaseY_PSD'][:, ir] = getPSD(Fi_base[:, 4, :], self.dw)
                results['MbaseY_max'][ir]    = results['MbaseY_avg'][ir] + 3 * results['MbaseY_std'][ir]
                results['MbaseY_min'][ir]    = results['MbaseY_avg'][ir] - 3 * results['MbaseY_std'][ir]

                results['MbaseZ_avg'][ir]    = Fi0_base[5]
                results['MbaseZ_std'][ir]    = getRMS(Fi_base[:, 5, :])
                results['MbaseZ_PSD'][:, ir] = getPSD(Fi_base[:, 5, :], self.dw)
                results['MbaseZ_max'][ir]    = results['MbaseZ_avg'][ir] + 3 * results['MbaseZ_std'][ir]
                results['MbaseZ_min'][ir]    = results['MbaseZ_avg'][ir] - 3 * results['MbaseZ_std'][ir]

                # For backwards capability, also save MBase_ for now
                results['Mbase_avg'][ir]   = results['MbaseY_avg'][ir]  # tower base fore-aft bending moment
                results['Mbase_std'][ir]   = results['MbaseY_std'][ir]
                results['Mbase_PSD'][:,ir] = results['MbaseY_PSD'][:,ir]
                results['Mbase_max'][ir]   = results['MbaseY_max'][ir]
                results['Mbase_min'][ir]   = results['MbaseY_min'][ir]


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
             shadow=True, plot_frame=False, mp_args={}, frame_opts={}, plot_joints=False, axes_around_fowt=None):
        '''plots the FOWT...'''

        # Assign values to axes_around_fowt if not specified by the user
        if axes_around_fowt is None:
            if plot_ms:
                axes_around_fowt=False # Do not zoom in on the FOWT
            else:
                axes_around_fowt=True  # Zoom in on the FOWT

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

                mem.plot(ax, color=color, nodes=nodes, station_plot=station_plot, zorder=zorder, plot_frame=plot_frame, frame_opts=frame_opts)

        if plot_frame:
            for i_rl in range(len(self.rigidLinkList['id'])):
                nodeA = self.rigidLinkList['node1'][i_rl]
                nodeB = self.rigidLinkList['node2'][i_rl]
                ax.plot([nodeA.r[0], nodeB.r[0]], [nodeA.r[1], nodeB.r[1]], [nodeA.r[2], nodeB.r[2]], color='k')
                ax.scatter(nodeA.r[0], nodeA.r[1], nodeA.r[2], color='k', facecolors='None')
                ax.scatter(nodeB.r[0], nodeB.r[1], nodeB.r[2], color='k', facecolors='None')

        # TODO: joint position is currently not updated with the rest of the model, so this is only useful to when plotting the model at its undisplaced position
        if plot_joints:
            for joint in self.jointList:
                color = 'green' if joint['type'] == 'ball' else 'red'
                ax.scatter(joint['r'][0], joint['r'][1], joint['r'][2], color=color, marker='o', facecolors='None')

        # The code below makes the plot zoom into the FOWT. Useful when plotting the FOWT without its mooring system
        if axes_around_fowt:
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

            # --- Ensure equal axis limits ---
            # Gather all points
            xs, ys, zs = [], [], []
            for member in self.memberList:
                for n in member.nodeList:
                    xs.append(n.r[0])
                    ys.append(n.r[1])
                    zs.append(n.r[2])
            for rotor in self.rotorList:
                for n in rotor.nodeList:
                    xs.append(n.r[0])
                    ys.append(n.r[1])
                    zs.append(n.r[2])
            for i_rl in range(len(self.rigidLinkList['id'])):
                nodeA = self.rigidLinkList['node1'][i_rl]
                nodeB = self.rigidLinkList['node2'][i_rl]
                xs += [nodeA.r[0], nodeB.r[0]]
                ys += [nodeA.r[1], nodeB.r[1]]
                zs += [nodeA.r[2], nodeB.r[2]]

            # Compute limits
            xlim = [np.min(xs), np.max(xs)]
            ylim = [np.min(ys), np.max(ys)]
            zlim = [np.min(zs), np.max(zs)]
            max_range = max(np.ptp(xlim), np.ptp(ylim), np.ptp(zlim))
            mid_x = np.mean(xlim)
            mid_y = np.mean(ylim)
            mid_z = np.mean(zlim)

            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        

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
            mem.plot(ax, color=color, plot2d=True, Xuvec=Xuvec, Yuvec=Yuvec)

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

    def write_modes_json(self, filename, fns, modes):
        """
        Write eigenmodes and frequencies to a JSON file for visualization in viz3Danim (https://github.com/ebranlard/viz3Danim).

        Parameters
        ----------
        filename : str
            Output JSON file path.
        fns : (nDOF,) array
            Natural frequencies [Hz].
        modes : (nDOF, nDOF) array
            Mode shapes (eigenvectors)
        """

        import numpy as np

        # Helper formatting functions
        def fmt(x, width=12, prec=6):
            return f"{x: {width}.{prec}E}"

        def fmt_list(lst, width=12, prec=6):
            return "[" + ",".join(fmt(x, width, prec) for x in lst) + "]"


        # Nodes: list of [x, y, z] for each node
        # Connectivity: list of [start_node_idx, end_node_idx] for each element
        # elem_props: list of dicts with properties for each element
        # TODO: viz3Danim only supports cylinders for now. Need to implement rectangular member later
        nodes = [list(np.array(n.r[:3], dtype=float)) for n in self.nodeList]
        connectivity = []
        elem_props = []
        virtual_node_map = {}  # using this to plot rigid members. Could I use setPosition() instead? But would need to reset to previous position later
        real_node_count = len(self.nodeList)

        for mem in self.memberList:
            if mem.type == 'rigid':
                # Loop stations and append virtual nodes for plotting only
                for i in range(mem.r.shape[0] - 1):
                    rA = mem.r[i, :]
                    rB = mem.r[i + 1, :]

                    if i == 0:
                        n1 = mem.nodeList[0].id
                    else:
                        nodes.append(rA[:3])
                        n1 = len(nodes) - 1
                        # Map virtual node to real node and offset
                        virtual_node_map[n1] = (mem.nodeList[0].id, rA[:3] - mem.nodeList[0].r[:3])

                    # Same but for the other end, which is never part of the original node list
                    nodes.append(rB[:3])
                    n2 = len(nodes) - 1
                    virtual_node_map[n2] = (mem.nodeList[0].id, rB[:3] - mem.nodeList[0].r[:3])

                    connectivity.append([n1, n2])

                    if mem.shape == 'circular':
                        diameter = 0.5 * (mem.dorsl_node_ext[i] + mem.dorsl_node_ext[i + 1])
                    else:
                        side0 = 0.5 * (mem.dorsl_node_ext[i][0] + mem.dorsl_node_ext[i + 1][0])
                        side1 = 0.5 * (mem.dorsl_node_ext[i][1] + mem.dorsl_node_ext[i + 1][1])
                        diameter = max(side0, side1)
                    elem_props.append({
                        "shape": 'cylinder',
                        "type": 1,
                        "Diam": diameter
                    })

            if mem.type == 'beam':
                for i in range(len(mem.nodeList) - 1):
                    n1 = mem.nodeList[i].id
                    n2 = mem.nodeList[i + 1].id
                    connectivity.append([n1, n2])

                    if mem.shape == 'circular':
                        diameter = 0.5 * (mem.dorsl_node_ext[i] + mem.dorsl_node_ext[i + 1])
                    else:
                        side0 = 0.5 * (mem.dorsl_node_ext[i][0] + mem.dorsl_node_ext[i + 1][0])
                        side1 = 0.5 * (mem.dorsl_node_ext[i][1] + mem.dorsl_node_ext[i + 1][1])
                        diameter = max(side0, side1)
                    elem_props.append({
                        "shape": 'cylinder',
                        "type": 1,
                        "Diam": diameter
                    })

        # # Rigid links
        # for i_rl in range(len(self.rigidLinkList['id'])):
        #     n1 = self.rigidLinkList['node1'][i_rl]
        #     n2 = self.rigidLinkList['node2'][i_rl]
        #     connectivity.append([n1.id, n2.id])
        #     elem_props.append({
        #         "shape": 'cylinder',
        #         "type": 3,
        #         "Diam": 1 # Dummy diameter, as it seems that rigid links do not use the diameter that is provided
        #     })

        # Modes: list of dicts with name, frequency, omega, Displ
        modes_list = []
        n_modes = modes.shape[1]
        for i in range(n_modes):
            mode_full = self.T @ modes[:, i]  # Get displacements in full dofs

            displ = []
            for idx, node_xyz in enumerate(nodes):
                if idx < real_node_count:
                    # Real node: use mode_full
                    n = self.nodeList[idx]
                    base = n.id * n.nDOF
                    displ.append([
                        float(mode_full[base + 0]),
                        float(mode_full[base + 1]),
                        float(mode_full[base + 2])
                    ])
                else:
                    # Virtual node: compute from associated real node
                    real_node_id, offset = virtual_node_map[idx]
                    n = self.nodeList[real_node_id]
                    base = n.id * n.nDOF
                    t = np.array([
                        float(mode_full[base + 0]),
                        float(mode_full[base + 1]),
                        float(mode_full[base + 2])
                    ])
                    rot = np.array([
                        float(mode_full[base + 3]),
                        float(mode_full[base + 4]),
                        float(mode_full[base + 5])
                    ])
                    disp_virtual = t + np.cross(rot, offset)
                    displ.append(list(disp_virtual))
            modes_list.append({
                "name": f"FEM{i+1}",
                "frequency": float(fns[i]),
                "omega": float(fns[i] * 2 * np.pi),
                "Displ": displ
            })

        # Write to file
        with open(filename, "w") as f:
            f.write('{\n')
            f.write(f'"writer": "RAFT",\n')
            f.write(f'"fileKind": "Modes",\n')
            f.write(f'"groundLevel": {fmt(self.depth)},\n')

            # Connectivity
            f.write('"Connectivity": [')
            for i, conn in enumerate(connectivity):
                f.write(f"[{conn[0]},{conn[1]}]")
                if i < len(connectivity) - 1:
                    f.write(",")
            f.write("],\n")

            # Nodes
            f.write('"Nodes": [')
            for i, node in enumerate(nodes):
                f.write(fmt_list(node))
                if i < len(nodes) - 1:
                    f.write(",")
            f.write("],\n")

            # ElemProps
            f.write('"ElemProps": [\n')
            for i, ep in enumerate(elem_props):
                f.write(f'  {{"shape": "cylinder", "type": {ep["type"]}, "Diam": {fmt(ep["Diam"], width=8, prec=4)}}}')
                if i < len(elem_props) - 1:
                    f.write(",\n")
            f.write("],\n")

            # Modes
            f.write('"Modes": [\n')
            for i, mode in enumerate(modes_list):
                f.write(f'  {{"name": "{mode["name"]}", "frequency": {fmt(mode["frequency"])}, "omega": {fmt(mode["omega"])}, "Displ": [')
                for j, disp in enumerate(mode["Displ"]):
                    f.write(fmt_list(disp))
                    if j < len(mode["Displ"]) - 1:
                        f.write(",")
                f.write("]}")
                if i < len(modes_list) - 1:
                    f.write(",\n")
            f.write("]\n")
            f.write('}\n')

