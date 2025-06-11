# RAFT's support structure member class

import numpy as np

from raft.helpers import *

from moorpy.helpers import transformPosition
from scipy.special import jn, yn, jv, kn, hankel1

## This class represents linear (for now cylindrical and rectangular) components in the substructure.
#  It is meant to correspond to Member objects in the WEIS substructure ontology, but containing only
#  the properties relevant for the Level 1 frequency-domain model, as well as additional data strucutres
#  used during the model's operation.
class Member:

    def __init__(self, mi, nw, BEM=[], heading=0, part_of='platform'):
        '''Initialize a Member. For now, this function accepts a space-delimited string with all member properties.

        PARAMETERS
        ----------

        mi : dict
            Dictionary containing the member description data structure
        nw : int
            Number of frequencies in the analysis - used for initializing.
        heading : float, optional
            Heading rotation to apply to the coordinates when setting up the 
            member. Used for member arrangements or FOWT heading offsets [deg].
        part_of : str, optional
            String identifying the subcomponent of the FOWT that this member is part of (platform, tower, nacelle, ...)
        '''

        # overall member parameters
        self.id       = int(1)                                       # set the ID value of the member
        self.name     = str(mi['name'])
        self.type     = str(mi['type'])                              # set the type of the member (will be used to identify rigid and flexible members)
        self.part_of  = part_of.lower()                              # String identifying the subcomponent of the FOWT that this member is part of

        self.rA0 = np.array(mi['rA'], dtype=np.double)               # [x,y,z] coordinates of lower node relative to PRP [m]
        self.rB0 = np.array(mi['rB'], dtype=np.double)               # [x,y,z] coordinates of upper node relative to PRP [m]
        if (self.rA0[2] == 0 or self.rB0[2] == 0) and self.type != 3:
            raise ValueError("RAFT Members cannot start or end on the waterplane")
        if self.rB0[2] < self.rA0[2]:
            pass
            #print(f"The z position of rA is {self.rA0[2]}, and the z position of rB is {self.rB0[2]}. RAFT Members can have trouble when their rA position is below the rB position")
            #self.rA0 = np.array(mi['rB'], dtype=np.double)
            #self.rB0 = np.array(mi['rA'], dtype=np.double)

        shape      = str(mi['shape'])                                # the shape of the cross section of the member as a string (the first letter should be c or r)

        self.potMod = getFromDict(mi, 'potMod', dtype=bool, default=False)     # hard coding BEM analysis enabled for now <<<< need to move this to the member YAML input instead <<<
        self.MCF    = getFromDict(mi, 'MCF', dtype=bool, default=False) # Flag to use MacCamy-Fuchs correction or not
        self.extensionA = getFromDict(mi, 'extensionA', default=0)
        self.extensionB = getFromDict(mi, 'extensionB', default=0)
        self.gamma = getFromDict(mi, 'gamma', default=0.)  # twist angle about the member's z-axis [degrees] (if gamma=90, then the side lengths are flipped)
        rAB = self.rB0-self.rA0       # The relative coordinates of upper node from lower node [m]
        self.l = np.linalg.norm(rAB)  # member length [m]
    
        # heading feature for rotation members about the z axis (used for rotated patterns)
        if heading != 0.0:
            self.rA0 = applyHeadingToPoint(self.rA0, heading)
            self.rB0 = applyHeadingToPoint(self.rB0, heading)
            
            if rAB[0] == 0.0 and rAB[1] == 0:  # special case of vertical member
                self.gamma += heading  # heading must be applied as twist about z


        # ----- process station positions and other distributed inputs -----
        
        # Station inputs are mapped linearly over the length of the member from
        # end A to B. The inputted station list must be in increasing order.
        st = np.array(mi['stations'], dtype=float)      # station input values (abritrary units/scale)
        n = len(st)                                    # number of stations
        
        if n < 2:
            raise ValueError("At least two stations entries must be provided")
            
        # ensure the stations are in ascending order (assumed to be from end A to B)
        if not sorted(st) == st.tolist():
            raise ValueError(f"Member {self.name}: the station list is not in ascending order.")
        
        # calculate station positions along the member axis from 0 to l [m]
        self.stations = (st - st[0])/(st[-1] - st[0])*self.l

        # shapes
        if shape[0].lower() == 'c':
            self.shape = 'circular'
            self.d     = getFromDict(mi, 'd', shape=n)               # diameter of member nodes [m]  <<< should maybe check length of all of these

            self.gamma = 0  # zero any twist angle about axis (irrelevant for circular)

        elif shape[0].lower() == 'r':   # <<< this case not checked yet since update <<<
            self.shape = 'rectangular'

            self.sl    = getFromDict(mi, 'd', shape=[n,2])           # array of side lengths of nodes along member [m]

        else:
            raise ValueError('The only allowable shape strings are circular and rectangular')

        # We disable the MCF correction if the element is not circular
        if self.MCF:
            if not self.shape=='circular':
                print(f'MacCamy-Fuchs correction not applicable to member {self.name}. Member needs to be circular. Disabling MCF.')
                self.MCF = False


        self.t         = getFromDict(mi, 't', shape=n, default=0)  # shell thickness at each station [m]
        self.rho_shell = getFromDict(mi, 'rho_shell', shape=0, default=8500.) # shell mass density [kg/m^3]
        
        
        # ----- ballast inputs (for each section between stations) -----
        
        # read the ballast fill level of each section (same units/scale as stations list)
        st_fill = getFromDict(mi, 'l_fill'  , shape=n-1, default=0)   
        
        #ensure each ballast entry is valid (fill level doesn't exceed section length)
        for i in range(n-1):
            if st_fill[i] < 0: 
                raise Exception(f"Member {self.name}: ballast level in section {i+1} is negative.")
            if st_fill[i] > st[i+1] - st[i]: 
                raise Exception(f"Member {self.name}: ballast level in section {i+1} exceeds section length."
                                +f" ({st_fill[i]} > {st[i+1] - st[i]}).")
        
        # convert ballast fill levels into length units [m]
        self.l_fill = st_fill/(st[-1] - st[0])*self.l  

        # density of ballast in member [kg/m^3]
        rho_fill = getFromDict(mi, 'rho_fill', shape=-1, default=1025)
        
        if np.isscalar(rho_fill):
            self.rho_fill = np.zeros(n-1) + rho_fill
        else:
            if len(rho_fill) == n-1:
                self.rho_fill = np.array(rho_fill)
            else:
                raise Exception(f"Member {self.name}: the number of provided ballast densities (rho_fill) must be 1 less than the number of stations.")
        
        # initialize member orientation variables
        self.q = rAB/self.l                                         # member axial unit vector
        self.p1 = np.zeros(3)                                       # member transverse unit vectors (to be filled in later)
        self.p2 = np.zeros(3)                                       # member transverse unit vectors
        self.R = np.eye(3)                                          # rotation matrix from global x,y,z to member q,p1,p2


        # ----- end cap and bulkhead info -----

        cap_stations = getFromDict(mi, 'cap_stations', shape=-1, default=[])   # station location inputs before scaling
        if len(cap_stations) == 0:
            self.cap_t        = []
            self.cap_d_in     = []
            self.cap_stations = []
        else:
            self.cap_t        = getFromDict(mi, 'cap_t'   , shape=cap_stations.shape[0])   # thicknesses [m]
            if self.shape == 'circular':
                self.cap_d_in = getFromDict(mi, 'cap_d_in', shape=cap_stations.shape[0])   # inner diameter (if it isn't a solid plate) [m]
            elif self.shape == 'rectangular':
                self.cap_d_in = getFromDict(mi, 'cap_d_in', shape=[cap_stations.shape[0],2])   # inner diameter (if it isn't a solid plate) [m]

            self.cap_stations = (cap_stations - st[0])/(st[-1] - st[0])*self.l             # calculate station positions along the member axis from 0 to l [m]
            

        # Drag coefficients
        self.Cd_q   = getFromDict(mi, 'Cd_q' , shape=n, default=0.0 )     # axial drag coefficient
        self.Cd_p1  = getFromDict(mi, 'Cd'   , shape=n, default=0.6, index=0)      # transverse1 drag coefficient
        self.Cd_p2  = getFromDict(mi, 'Cd'   , shape=n, default=0.6, index=1)      # transverse2 drag coefficient
        self.Cd_End = getFromDict(mi, 'CdEnd', shape=n, default=0.6 )     # end drag coefficient

        # Added mass coefficients
        self.Ca_q   = getFromDict(mi, 'Ca_q' , shape=n, default=0.0 )     # axial added mass coefficient
        self.Ca_p1  = getFromDict(mi, 'Ca'   , shape=n, default=0.97, index=0)     # transverse1 added mass coefficient
        self.Ca_p2  = getFromDict(mi, 'Ca'   , shape=n, default=0.97, index=1)     # transverse2 added mass coefficient
        self.Ca_End = getFromDict(mi, 'CaEnd', shape=n, default=0.6 )     # end added mass coefficient

        # ----- Strip theory discretization -----

        # discretize into strips with a node at the midpoint of each strip (flat surfaces have dl=0)
        dorsl  = list(self.d) if self.shape=='circular' else list(self.sl)   # get a variable that is either diameter or side length pair
        dlsMax = getFromDict(mi, 'dlsMax', shape=0, default=5)

        
        # start things off with the strip for end A
        ls     = [0.0]                 # list of lengths along member axis where a node is located <<< should these be midpoints instead of ends???
        dls    = [0.0]                 # lumped node lengths (end nodes have half the segment length)
        ds     = [0.5*dorsl[0]]       # mean diameter or side length pair of each strip
        drs    = [0.5*dorsl[0]]       # change in radius (or side half-length pair) over each strip (from node i-1 to node i)

        for i in range(1,n):

            lstrip = self.stations[i]-self.stations[i-1]             # the axial length of the strip

            if lstrip > 0.0:
                ns= int(np.ceil( (lstrip) / dlsMax ))             # number of strips to split this segment into
                dlstrip = lstrip/ns
                m   = 0.5*(dorsl[i] - dorsl[i-1])/lstrip          # taper ratio
                ls  += [self.stations[i-1] + dlstrip*(0.5+j) for j in range(ns)] # add node locations
                dls += [dlstrip]*ns
                ds  += [dorsl[i-1] + dlstrip*2*m*(0.5+j) for j in range(ns)]
                drs += [dlstrip*m]*ns
                
            elif lstrip == 0.0:                                      # flat plate case (ends, and any flat transitions), a single strip for this section
                dlstrip = 0
                ls  += [self.stations[i-1]]                          # add node location
                dls += [dlstrip]
                ds  += [0.5*(dorsl[i-1] + dorsl[i])]               # set diameter as midpoint diameter
                drs += [0.5*(dorsl[i] - dorsl[i-1])]

        # finish things off with the strip for end B
        dlstrip = 0
        ls  += [self.stations[-1]]         
        dls += [0.0]
        ds  += [0.5*dorsl[-1]]
        drs += [-0.5*dorsl[-1]]
        
        # >>> may want to have a way to not have an end strip for members that intersect things <<<
        
        self.ns  = len(ls)                                           # number of hydrodynamic strip theory nodes per member
        self.ls  = np.array(ls, dtype=float)                          # node locations along member axis
        self.dls = np.array(dls)
        self.ds  = np.array(ds)
        self.drs = np.array(drs)
        self.mh  = np.array(m)

        self.r   = np.zeros([self.ns,3])                             # undisplaced node positions along member  [m]
        for i in range(self.ns):
            self.r[i,:] = self.rA0 + (ls[i]/self.l)*rAB              # locations of hydrodynamics nodes (will later be displaced) [m]
        
        # ----- initialize arrays used later for hydro calculations -----
        self.a_i       = np.zeros([self.ns])            # signed axial area vector that dynamic pressure will act on [m2]
        # complex frequency-dependent amplitudes of quantities at each node along member (to be filled in later)
        self.dr        = np.zeros([self.ns,3,nw], dtype=complex)            # displacement
        self.v         = np.zeros([self.ns,3,nw], dtype=complex)            # velocity
        self.a         = np.zeros([self.ns,3,nw], dtype=complex)            # acceleration
        # note: wave quantities below will be the summed values if there is more than one sea state heading
        self.u         = np.zeros([self.ns,3,nw], dtype=complex)            # wave velocity
        self.ud        = np.zeros([self.ns,3,nw], dtype=complex)            # wave acceleration
        self.pDyn      = np.zeros([self.ns,  nw], dtype=complex)            # dynamic pressure
        self.F_exc_iner= np.zeros([self.ns,3,nw], dtype=complex)            # wave excitation from inertia (Froude-Krylov)
        self.F_exc_a   = np.zeros([self.ns,3,nw], dtype=complex)            #  component due to wave acceleration
        self.F_exc_p   = np.zeros([self.ns,3,nw], dtype=complex)            #  component due to dynamic pressure
        self.F_exc_drag= np.zeros([self.ns,3,nw], dtype=complex)            # wave excitation from linearized drag
        
        
        # new hydro matrices that are now used?
        self.Amat = np.zeros([self.ns,3,3])
        self.Bmat = np.zeros([self.ns,3,3])
        self.Imat = np.zeros([self.ns,3,3])
        self.Imat_MCF = np.zeros([self.ns,3,3, nw], dtype=complex)


    def setPosition(self, r6=np.zeros(6)):
        '''Calculates member pose -- node positions and vectors q, p1, and p2 
        as well as member orientation matrix R based on the end positions and 
        twist angle gamma along with any mean displacements and rotations.
        
        Parameters
        ----------
        r6 : array, optional
            Absolute position/orientation of FOWT to which member is attached.
            
        '''
        # formerly calcOrientation

        rAB = self.rB0-self.rA0                                     # vector from end A to end B, undisplaced [m]
        q = rAB/np.linalg.norm(rAB)                                 # member axial unit vector

        beta = np.arctan2(q[1],q[0])                                # member incline heading from x axis
        phi  = np.arctan2(np.sqrt(q[0]**2 + q[1]**2), q[2])         # member incline angle from vertical

        # trig terms for Euler angles rotation based on beta, phi, and gamma
        s1 = np.sin(beta)
        c1 = np.cos(beta)
        s2 = np.sin(phi)
        c2 = np.cos(phi)
        s3 = np.sin(np.deg2rad(self.gamma))
        c3 = np.cos(np.deg2rad(self.gamma))

        R = np.array([[ c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3,  c1*s2],
                      [ c1*s3+c2*c3*s1,  c1*c3-c2*s1*s3,  s1*s2],
                      [   -c3*s2      ,      s2*s3     ,    c2 ]])  #Z1Y2Z3 from https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

        p1 = np.matmul( R, [1,0,0] )               # unit vector that is in the 'beta' plane if gamma is zero
        p2 = np.cross( q, p1 )                     # unit vector orthogonal to both p1 and q
        
        # apply any platform offset and rotation to the values already obtained
        R_platform = rotationMatrix(*r6[3:])  # rotation matrix for the platform roll, pitch, yaw
    
        R  = np.matmul(R_platform, R)
        q  = np.matmul(R_platform, q)
        p1 = np.matmul(R_platform, p1)
        p2 = np.matmul(R_platform, p2)
        
        self.rA = transformPosition(self.rA0, r6)
        self.rB = transformPosition(self.rB0, r6)
               
        # update node positions
        rAB = self.rB - self.rA
        for i in range(self.ns):
            self.r[i,:] = self.rA + (self.ls[i]/self.l)*rAB              # locations of hydrodynamics nodes (will later be displaced) [m]

        # save direction vectors and matrices
        self.R  = R
        self.q  = q
        self.p1 = p1
        self.p2 = p2

        # matrices of vector multiplied by vector transposed, used in computing force components
        self.qMat  = VecVecTrans(self.q)
        self.p1Mat = VecVecTrans(self.p1)
        self.p2Mat = VecVecTrans(self.p2)                 


    def getInertia(self, rPRP=np.zeros(3)):
        '''Calculates member inertia properties: mass, center of mass, moments of inertia.
        Properties are calculated relative to the platform reference point (PRP) in the
        global orientation directions.
        
        Parameters
        ----------
        rPRP : float array
            Coordinates of the platform reference point (the first three entries of fowt.Xi0),
            which the moment of inertia matrix will be calculated relative to. [m]
        '''

        # Moment of Inertia Helper Functions (to move to helper file) <<<
        
        def FrustumMOI(dA, dB, H, p):
            '''returns the radial and axial moments of inertia of a potentially tapered circular member about the end node.
            Previously used equations found in a HydroDyn paper, now it uses newly derived ones. Ask Stein for reference if needed'''
            if H==0:        # if there's no height, mainly refering to no ballast, there shouldn't be any extra MoI
                I_rad = 0                                                   # radial MoI about end node [kg-m^2]
                I_ax = 0                                                    # axial MoI about axial axis [kg-m^2]
            else:
                if dA==dB:  # if it's a cylinder
                    r1 = dA/2                                               # bottom radius [m]
                    r2 = dB/2                                               # top radius [m]
                    I_rad = (1/12)*(p*H*np.pi*r1**2)*(3*r1**2 + 4*H**2)     # radial MoI about end node [kg-m^2]
                    I_ax = (1/2)*p*np.pi*H*r1**4                            # axial MoI about axial axis [kg-m^2]
                else:       # if it's a tapered cylinder (frustum)
                    r1 = dA/2                                               # bottom radius [m]
                    r2 = dB/2                                               # top radius [m]
                    I_rad = (1/20)*p*np.pi*H*(r2**5 - r1**5)/(r2 - r1) + (1/30)*p*np.pi*H**3*(r1**2 + 3*r1*r2 + 6*r2**2) # radial MoI about end node [kg-m^2]
                    I_ax = (1/10)*p*np.pi*H*(r2**5-r1**5)/(r2-r1)           # axial MoI about axial axis [kg-m^2]

            return I_rad, I_ax

        def RectangularFrustumMOI(La, Wa, Lb, Wb, H, p):
            '''returns the moments of inertia about the end node of a cuboid that can be tapered.
            - Inputs the lengths and widths at the top and bottom of the cuboid, as well as the height and material density.
            - L is the side length along the local x-direction, W is the side length along the local y-direction.
            - Does not work for members that are not symmetrical about the axial axis.
            - Works for cases when it is a perfect cuboid, a truncated pyramid, and a truncated triangular prism
            - Equations derived by hand, ask Stein for reference if needed'''

            if H==0: # if there's no height, mainly refering to no ballast, there shouldn't be any extra MoI
                Ixx = 0                                         # MoI around the local x-axis about the end node [kg-m^2]
                Iyy = 0                                         # MoI around the local y-axis about the end node [kg-m^2]
                Izz = 0                                         # MoI around the local z-axis about the axial axis [kg-m^2]
            else:
                if La==Lb and Wa==Wb: # if it's a cuboid
                    L = La                                      # length of the cuboid (La=Lb) [m]
                    W = Wa                                      # width of the cuboid (Wa=Wb) [m]
                    M = p*L*W*H                                 # mass of the cuboid [kg]

                    Ixx = (1/12)*M*(W**2 + 4*H**2)              # MoI around the local x-axis about the end node [kg-m^2]
                    Iyy = (1/12)*M*(L**2 + 4*H**2)              # MoI around the local y-axis about the end node [kg-m^2]
                    Izz = (1/12)*M*(L**2 + W**2)                # MoI around the local z-axis about the axial axis [kg-m^2]

                elif La!=Lb and Wa!=Wb: # if it's a truncated pyramid for both side lengths

                    x2 = (1/12)*p* ( (Lb-La)**3*H*(Wb/5 + Wa/20) + (Lb-La)**2*La*H*(3*Wb/4 + Wa/4) + \
                                     (Lb-La)*La**2*H*(Wb + Wa/2) + La**3*H*(Wb/2 + Wa/2) )

                    y2 = (1/12)*p* ( (Wb-Wa)**3*H*(Lb/5 + La/20) + (Wb-Wa)**2*Wa*H*(3*Lb/4 + La/4) + \
                                     (Wb-Wa)*Wa**2*H*(Lb + La/2) + Wa**3*H*(Lb/2 + La/2) )

                    z2 = p*( Wb*Lb/5 + Wa*Lb/20 + La*Wb/20 + Wa*La*(1/30) ) * H**3

                    Ixx = y2+z2                                 # MoI around the local x-axis about the end node [kg-m^2]
                    Iyy = x2+z2                                 # MoI around the local y-axis about the end node [kg-m^2]
                    Izz = x2+y2                                 # MoI around the local z-axis about the axial axis [kg-m^2]

                elif La==Lb and Wa!=Wb: # if it's a truncated triangular prism where only the lengths are the same on top and bottom
                    L = La                                      # length of the truncated triangular prism [m]

                    x2 = (1/24)*p*(L**3)*H*(Wb+Wa)
                    y2 = (1/48)*p*L*H*( Wb**3 + Wa*Wb**2 + Wa**2*Wb + Wa**3 )
                    z2 = (1/12)*p*L*(H**3)*( 3*Wb + Wa )

                    Ixx = y2+z2                                 # MoI around the local x-axis about the end node [kg-m^2]
                    Iyy = x2+z2                                 # MoI around the local y-axis about the end node [kg-m^2]
                    Izz = x2+y2                                 # MoI around the local z-axis about the axial axis [kg-m^2]

                elif La!=Lb and Wa==Wb: # if it's a truncated triangular prism where only the widths are the same on top and bottom
                    W = Wa                                      # width of the truncated triangular prism [m]

                    x2 = (1/48)*p*W*H*( Lb**3 + La*Lb**2 + La**2*Lb + La**3 )
                    y2 = (1/24)*p*(W**3)*H*(Lb+La)
                    z2 = (1/12)*p*W*(H**3)*( 3*Lb + La )

                    Ixx = y2+z2                                 # MoI around the local x-axis about the end node [kg-m^2]
                    Iyy = x2+z2                                 # MoI around the local y-axis about the end node [kg-m^2]
                    Izz = x2+y2                                 # MoI around the local z-axis about the axial axis [kg-m^2]

                else:
                    raise ValueError('You either have inconsistent inputs, or you are trying to calculate the MoI of a member that is not supported')

            return Ixx, Iyy, Izz



        # ------- member inertial calculations ---------
        
        mass_center = 0                                 # total sum of mass the center of mass of the member [kg-m]
        mshell = 0                                      # total mass of the shell material only of the member [kg]
        self.vfill = []                                 # list of ballast volumes in each submember [m^3] - stored in the object for later access
        mfill = []                                      # list of ballast masses in each submember [kg]
        pfill = []                                      # list of ballast densities in each submember [kg]
        self.M_struc = np.zeros([6,6])                  # member mass/inertia matrix [kg, kg-m, kg-m^2]

        # loop through each sub-member
        for i in range(1,len(self.stations)):                            # start at 1 rather than 0 because we're looking at the sections (from station i-1 to i)

            # initialize common variables
            l = self.stations[i]-self.stations[i-1]     # length of the submember [m]
            if l==0.0:
                mass = 0
                center = np.zeros(3)
                m_shell = 0
                v_fill = 0
                m_fill = 0
                rho_fill = 0
            else:
                # if the following variables are input as scalars, keep them that way, if they're vectors, take the [i-1]th value
                rho_shell = self.rho_shell              # density of the shell material [kg/m^3]
                if np.isscalar(self.l_fill):            # set up l_fill and rho_fill based on whether it's scalar or not
                    l_fill = self.l_fill
                else:
                    l_fill = self.l_fill[i-1]
                if np.isscalar(self.rho_fill):
                    rho_fill = self.rho_fill
                else:
                    rho_fill = self.rho_fill[i-1]
    
                
                if self.shape=='circular':
                    # MASS AND CENTER OF GRAVITY
                    dA = self.d[i-1]                        # outer diameter of the lower node [m]
                    dB = self.d[i]                          # outer diameter of the upper node [m]
                    dAi = self.d[i-1] - 2*self.t[i-1]       # inner diameter of the lower node [m]
                    dBi = self.d[i] - 2*self.t[i]           # inner diameter of the upper node [m]
                    
                    V_outer, hco = FrustumVCV(dA, dB, l)    # volume and center of volume of solid frustum with outer diameters [m^3] [m]
                    V_inner, hci = FrustumVCV(dAi, dBi, l)  # volume and center of volume of solid frustum with inner diameters [m^3] [m] 
                    v_shell = V_outer-V_inner               # volume of hollow frustum with shell thickness [m^3]
                    m_shell = v_shell*rho_shell             # mass of hollow frustum [kg]
                    
                    hc_shell = ((hco*V_outer)-(hci*V_inner))/(V_outer-V_inner) if V_outer-V_inner!=0 else 0.0  # center of volume of hollow frustum with shell thickness [m]
                         
                    dBi_fill = (dBi-dAi)*(l_fill/l) + dAi   # interpolated inner diameter of frustum that ballast is filled to [m] 
                    v_fill, hc_fill = FrustumVCV(dAi, dBi_fill, l_fill)         # volume and center of volume of solid inner frustum that ballast occupies [m^3] [m]
                    m_fill = v_fill*rho_fill                # mass of the ballast in the submember [kg]
                    
                    # <<< The ballast is calculated as if it starts at the same end as the shell, however, if the end of the sub-member has an end cap,
                    # then the ballast sits on top of the end cap. Depending on the thickness of the end cap, this can affect m_fill, hc_fill, and MoI_fill >>>>>
                    
                    mass = m_shell + m_fill                 # total mass of the submember [kg]
                    hc = ((hc_fill*m_fill) + (hc_shell*m_shell))/mass if mass!=0 else 0.0      # total center of mass of the submember from the submember's rA location [m]
                    
                    
                    # MOMENT OF INERTIA
                    I_rad_end_outer, I_ax_outer = FrustumMOI(dA, dB, l, rho_shell)          # radial and axial MoI about the end of the solid outer frustum [kg-m^2]
                    I_rad_end_inner, I_ax_inner = FrustumMOI(dAi, dBi, l, rho_shell)        # radial and axial MoI about the end of the imaginary solid inner frustum [kg-m^2]
                    I_rad_end_shell = I_rad_end_outer-I_rad_end_inner                       # radial MoI about the end of the frustum shell through superposition [kg-m^2]
                    I_ax_shell = I_ax_outer - I_ax_inner                                    # axial MoI of the shell through superposition [kg-m^2]
                    
                    I_rad_end_fill, I_ax_fill = FrustumMOI(dAi, dBi_fill, l_fill, rho_fill) # radial and axial MoI about the end of the solid inner ballast frustum [kg-m^2]
                    
                    I_rad_end = I_rad_end_shell + I_rad_end_fill                            # radial MoI about the end of the submember [kg-m^2]
                    I_rad = I_rad_end - mass*hc**2                                          # radial MoI about the CoG of the submember through the parallel axis theorem [kg-m^2]
                    
                    I_ax = I_ax_shell + I_ax_fill                                           # axial MoI of the submember about the total CoG (= about the end also bc axial)
    
                    Ixx = I_rad                             # circular, so the radial MoI is about the x and y axes
                    Iyy = I_rad                             # circular, so the radial MoI is about the x and y axes
                    Izz = I_ax                              # circular, so the axial MoI is about the z axis
                    
                
                elif self.shape=='rectangular':
                    # MASS AND CENTER OF GRAVITY
                    slA = self.sl[i-1]                          # outer side lengths of the lower node, of length 2 [m]
                    slB = self.sl[i]                            # outer side lengths of the upper node, of length 2 [m]
                    slAi = self.sl[i-1] - 2*self.t[i-1]         # inner side lengths of the lower node, of length 2 [m]
                    slBi = self.sl[i] - 2*self.t[i]             # inner side lengths of the upper node, of length 2 [m]
                    
                    V_outer, hco = FrustumVCV(slA, slB, l)      # volume and center of volume of solid frustum with outer side lengths [m^3] [m]
                    V_inner, hci = FrustumVCV(slAi, slBi, l)    # volume and center of volume of solid frustum with inner side lengths [m^3] [m]
                    v_shell = V_outer-V_inner                   # volume of hollow frustum with shell thickness [m^3]
                    m_shell = v_shell*rho_shell                 # mass of hollow frustum [kg]
                    
                    hc_shell = ((hco*V_outer)-(hci*V_inner))/(V_outer-V_inner) if V_outer-V_inner!=0 else 0.0  # center of volume of the hollow frustum with shell thickness [m]
                                        
                    slBi_fill = (slBi-slAi)*(l_fill/l) + slAi   # interpolated side lengths of frustum that ballast is filled to [m]
                    v_fill, hc_fill = FrustumVCV(slAi, slBi_fill, l_fill)   # volume and center of volume of inner frustum that ballast occupies [m^3]
                    m_fill = v_fill*rho_fill                    # mass of ballast in the submember [kg]
                    
                    mass = m_shell + m_fill                     # total mass of the submember [kg]
                    hc = ((hc_fill*m_fill) + (hc_shell*m_shell))/mass if mass !=0 else 0.0     # total center of mass of the submember from the submember's rA location [m]
                    
                    
                    # MOMENT OF INERTIA
                    # MoI about each axis at the bottom end node of the solid outer truncated pyramid [kg-m^2]
                    Ixx_end_outer, Iyy_end_outer, Izz_end_outer = RectangularFrustumMOI(slA[0], slA[1], slB[0], slB[1], l, rho_shell)
                    # MoI about each axis at the bottom end node of the solid imaginary inner truncated pyramid [kg-m^2]
                    Ixx_end_inner, Iyy_end_inner, Izz_end_inner = RectangularFrustumMOI(slAi[0], slAi[1], slBi[0], slBi[1], l, rho_shell)
                    # MoI about each axis at the bottom end node of the shell using superposition [kg-m^2]
                    Ixx_end_shell = Ixx_end_outer - Ixx_end_inner
                    Iyy_end_shell = Iyy_end_outer - Iyy_end_inner
                    Izz_end_shell = Izz_end_outer - Izz_end_inner
                    
                    # MoI about each axis at the bottom end node of the solid inner ballast truncated pyramid [kg-m^2]
                    Ixx_end_fill, Iyy_end_fill, Izz_end_fill = RectangularFrustumMOI(slAi[0], slAi[1], slBi_fill[0], slBi_fill[1], l_fill, rho_fill)
                    
                    # total MoI of each axis at the center of gravity of the member using the parallel axis theorem [kg-m^2]
                    Ixx_end = Ixx_end_shell + Ixx_end_fill
                    Ixx = Ixx_end - mass*hc**2
                    Iyy_end = Iyy_end_shell + Iyy_end_fill
                    Iyy = Iyy_end - mass*hc**2
                    
                    Izz_end = Izz_end_shell + Izz_end_fill
                    Izz = Izz_end       # the total MoI of the member about the z-axis is the same at any point along the z-axis

                # center of mass of the submember from the PRP in global orientation (note: some of above could streamlined out of the if/else)
                center = self.rA + self.q*(self.stations[i-1] + hc) - rPRP      # center of mass of the submember relative to the PRP [m]

            # add/append terms
            mass_center += mass*center                  # total sum of mass the center of mass of the member [kg-m]
            mshell += m_shell                           # total mass of the shell material only of the member [kg]
            self.vfill.append(v_fill)                        # list of ballast volumes in each submember [m^3]
            mfill.append(m_fill)                        # list of ballast masses in each submember [kg]
            pfill.append(rho_fill)                     # list of ballast densities in each submember [kg]
            
            # create a local submember mass matrix
            Mmat = np.diag([mass, mass, mass, 0, 0, 0]) # submember's mass matrix without MoI tensor
            # create the local submember MoI tensor in the correct directions
            I = np.diag([Ixx, Iyy, Izz])                # MoI matrix about the member's local CG. 0's on off diagonals because of symmetry
            T = self.R.T                                # transformation matrix to unrotate the member's local axes. Transposed because rotating axes.
            I_rot = np.matmul(T.T, np.matmul(I,T))      # MoI about the member's local CG with axes in same direction as global axes. [I'] = [T][I][T]^T -> [T]^T[I'][T] = [I]

            Mmat[3:,3:] = I_rot     # mass and inertia matrix about the submember's CG in unrotated, but translated local frame

            # translate this submember's local inertia matrix to the PRP and add it to the total member's M_struc matrix
            self.M_struc += translateMatrix6to6DOF(Mmat, center) # mass matrix of the member about the PRP


            # end of submember for loop

        
        # END CAPS/BULKHEADS
        # --------- Add the inertia properties of any end caps ---------
        self.m_cap_list = []
        # Loop through each cap or bulkhead
        for i in range(len(self.cap_stations)):

            L = self.cap_stations[i]        # The station position along the member where there is a cap or bulkhead
            h = self.cap_t[i]               # The thickness, or height of the cap or bulkhead [m]
            rho_cap = self.rho_shell        # set the cap density to the density of the member for now [kg/m^3]

            if self.shape=='circular':
                d_hole = self.cap_d_in[i]   # The diameter of the missing hole in the middle, if any [m]
                d = self.d-2*self.t         # The list of inner diameters along the member [m]

                if L==self.stations[0]:  # if the cap is on the bottom end of the member
                    dA = d[0]
                    dB = np.interp(L+h, self.stations, d)
                    dAi = d_hole
                    dBi = dB*(dAi/dA)       # keep the same proportion in d_hole from bottom to top
                elif L==self.stations[-1]:    # if the cap is on the top end of the member
                    dA = np.interp(L-h, self.stations, d)
                    dB = d[-1]
                    dBi = d_hole
                    dAi = dA*(dBi/dB)
                elif (L > self.stations[0] and L < self.stations[0] + h) or (L < self.stations[-1] and L > self.stations[-1] - h):
                    # there could be another case where 0 < L < h or self.l-h < L < self.l
                    # this would cause the inner member to stick out beyond the end point based on the following else calcs
                    # not including this for now since the modeler should be aware to not do this
                    raise ValueError('This setup cannot be handled by getIneria yet')
                elif i < len(self.cap_stations)-1 and L==self.cap_stations[i+1]: # if there's a discontinuity in the member and l=0
                    dA = np.interp(L-h, self.stations, d)   # make an end cap going down from the lower member
                    dB = d[i]
                    dBi = d_hole
                    dAi = dA*(dBi/dB)
                elif i > 0 and L==self.cap_stations[i-1]:   # and make an end cap going up from the upper member
                    dA = d[i]
                    dB = np.interp(L+h, self.stations, d)
                    dAi = d_hole
                    dBi = dB*(dAi/dA)
                else:
                    dA = np.interp(L-h/2, self.stations, d)
                    dB = np.interp(L+h/2, self.stations, d)
                    dM = np.interp(L, self.stations, d)         # find the diameter at the middle, where L is referencing
                    dMi = d_hole
                    dAi = dA*(dMi/dM)
                    dBi = dB*(dMi/dM)

                # run inertial calculations for circular caps/bulkheads
                V_outer, hco = FrustumVCV(dA, dB, h)
                V_inner, hci = FrustumVCV(dAi, dBi, h)
                v_cap = V_outer-V_inner
                m_cap = v_cap*rho_cap    # assume it's made out of the same material as the shell for now (can add in cap density input later if needed)
                hc_cap = ((hco*V_outer)-(hci*V_inner))/(V_outer-V_inner) if V_outer-V_inner!=0 else 0.0
                
                I_rad_end_outer, I_ax_outer = FrustumMOI(dA, dB, h, rho_cap)
                I_rad_end_inner, I_ax_inner = FrustumMOI(dAi, dBi, h, rho_cap)
                I_rad_end = I_rad_end_outer-I_rad_end_inner
                I_rad = I_rad_end - m_cap*hc_cap**2
                I_ax = I_ax_outer - I_ax_inner

                Ixx = I_rad
                Iyy = I_rad
                Izz = I_ax



            elif self.shape=='rectangular':
                sl_hole = self.cap_d_in[i,:]
                sl = self.sl - 2*self.t

                if L==self.stations[0]:  # if the cap is on the bottom end of the member
                    slA = sl[0,:]
                    slB = np.zeros(slA.shape)
                    slB = np.array([np.interp(L+h, self.stations, sl[:,0]), np.interp(L+h, self.stations, sl[:,1])])
                    slAi = sl_hole
                    slBi = slB*(slAi/slA)       # keep the same proportion in d_hole from bottom to top
                elif L==self.stations[-1]:    # if the cap is on the top end of the member
                    slA = np.array([np.interp(L-h, self.stations, sl[:,0]), np.interp(L-h, self.stations, sl[:,1])])
                    slB = sl[-1,:]
                    slAi = slA*(slBi/slB)
                    slBi = sl_hole
                elif (L > self.stations[0] and L < self.stations[0] + h) or (L < self.stations[-1] and L > self.stations[-1] - h):
                    # there could be another case where 0 < L < h or self.l-h < L < self.l
                    # this would cause the inner member to stick out beyond the end point based on the following else calcs
                    # not including this for now since the modeler should be aware to not do this
                    raise ValueError('This setup cannot be handled by getIneria yet')
                elif i < len(self.cap_stations)-1 and L==self.cap_stations[i+1]:
                    slA = np.interp(L-h, self.stations, sl)
                    slB = sl[i]
                    slBi = sl_hole
                    slAi = slA*(slBi/slB)
                elif i > 0 and L==self.cap_stations[i-1]:
                    slA = sl[i]
                    slB = np.interp(L+h, self.stations, sl)
                    slAi = sl_hole
                    slBi = slB*(slAi/slA)
                else:
                    slA = np.interp(L-h/2, self.stations, sl)
                    slB = np.interp(L+h/2, self.stations, sl)
                    slM = np.interp(L, self.stations, sl)
                    slMi = sl_hole
                    slAi = slA*(slMi/slM)
                    slBi = slB*(slMi/slM)


                # run inertial calculations for rectangular caps/bulkheads
                V_outer, hco = FrustumVCV(slA, slB, h)
                V_inner, hci = FrustumVCV(slAi, slBi, h)
                v_cap = V_outer-V_inner
                m_cap = v_cap*rho_cap    # assume it's made out of the same material as the shell for now (can add in cap density input later if needed)
                hc_cap = ((hco*V_outer)-(hci*V_inner))/(V_outer-V_inner) if V_outer-V_inner!=0 else 0.0

                Ixx_end_outer, Iyy_end_outer, Izz_end_outer = RectangularFrustumMOI(slA[0], slA[1], slB[0], slB[1], h, rho_cap)
                Ixx_end_inner, Iyy_end_inner, Izz_end_inner = RectangularFrustumMOI(slAi[0], slAi[1], slBi[0], slBi[1], h, rho_cap)
                Ixx_end = Ixx_end_outer-Ixx_end_inner
                Iyy_end = Iyy_end_outer-Iyy_end_inner
                Izz_end = Izz_end_outer-Izz_end_inner
                Ixx = Ixx_end - m_cap*hc_cap**2
                Iyy = Iyy_end - m_cap*hc_cap**2
                Izz = Izz_end


            # get centerpoint of cap relative to PRP
            pos_cap = self.rA + self.q*L - rPRP                 # position of the referenced cap station from the PRP
            if L==self.stations[0]:         # if it's a bottom end cap, the position is at the bottom of the end cap
                center_cap = pos_cap + self.q*hc_cap            # and the CG of the cap is at hc from the bottom, so this is the simple case
            elif L==self.stations[-1]:      # if it's a top end cap, the position is at the top of the end cap
                center_cap = pos_cap - self.q*(h - hc_cap)      # and the CG of the cap goes from the top, to h below the top, to hc above h below the top (wording...sorry)
            else:                           # if it's a middle bulkhead, the position is at the middle of the bulkhead
                center_cap = pos_cap - self.q*((h/2) - hc_cap)  # so the CG goes from the middle of the bulkhead, down h/2, then up hc

            
            # ----- add properties to relevant variables -----
            
            mass_center += m_cap*center_cap
            mshell += m_cap                # include end caps and bulkheads in the mass of the shell
            self.m_cap_list.append(m_cap)

            # create a local submember mass matrix
            Mmat = np.diag([m_cap, m_cap, m_cap, 0, 0, 0]) # submember's mass matrix without MoI tensor
            # create the local submember MoI tensor in the correct directions
            I = np.diag([Ixx, Iyy, Izz])                # MoI matrix about the member's local CG. 0's on off diagonals because of symmetry
            T = self.R.T                                # transformation matrix to unrotate the member's local axes. Transposed because rotating axes.
            I_rot = np.matmul(T.T, np.matmul(I,T))      # MoI about the member's local CG with axes in same direction as global axes. [I'] = [T][I][T]^T -> [T]^T[I'][T] = [I]

            Mmat[3:,3:] = I_rot     # mass and inertia matrix about the submember's CG in unrotated, but translated local frame

            # translate this submember's local inertia matrix to the PRP and add it to the total member's M_struc matrix
            self.M_struc += translateMatrix6to6DOF(Mmat, center_cap) # mass matrix of the member about the PRP

        self.mshell = mshell
        self.mfill  = mfill
        mass = self.M_struc[0,0]        # total mass of the entire member [kg]
        center = mass_center/mass if mass!=0 else np.zeros(3)       # total center of mass of the entire member from the PRP [m]


        return mass, center, mshell, mfill, pfill




    def getHydrostatics(self, rPRP=np.zeros(3), rho=1025, g=9.81):
        '''Calculates member hydrostatic properties, namely buoyancy and stiffness matrix.
        Properties are calculated relative to the platform reference point (PRP) in the
        global orientation directions.
        
        Parameters
        ----------
        rPRP : float array
            Coordinates of the platform reference point (the first three entries of fowt.Xi0),
            which the moment of inertia matrix will be calculated relative to. [m]
        '''
    
        pi = np.pi

        # initialize some values that will be returned
        Fvec = np.zeros(6)              # this will get added to by each segment of the member
        Cmat = np.zeros([6,6])          # this will get added to by each segment of the member
        V_UW = 0                        # this will get added to by each segment of the member
        r_centerV = np.zeros(3)         # center of buoyancy times volumen total - will get added to by each segment
        # these will only get changed once, if there is a portion crossing the water plane
        AWP = 0
        IWP = 0
        xWP = 0
        yWP = 0


        # loop through each member segment, and treat each segment like how we used to treat each member
        n = len(self.stations)

        for i in range(1,n):     # starting at 1 rather than 0 because we're looking at the sections (from station i-1 to i)

            # calculate end locations for this segment relative to the point on 
            # the waterplane directly above the PRP in unrotated directions (rHS_ref)
            rHS_ref = np.array([rPRP[0], rPRP[1], 0])
            rA = self.rA + self.q*self.stations[i-1] - rHS_ref
            rB = self.rA + self.q*self.stations[i  ] - rHS_ref

            # partially submerged case
            if rA[2]*rB[2] <= 0:    # if member crosses (or touches) water plane

                # angles
                beta = np.arctan2(self.q[1],self.q[0])  # member incline heading from x axis
                phi  = np.arctan2(np.sqrt(self.q[0]**2 + self.q[1]**2), self.q[2])  # member incline angle from vertical

                # precalculate trig functions
                cosPhi=np.cos(phi)
                sinPhi=np.sin(phi)
                tanPhi=np.tan(phi)
                cosBeta=np.cos(beta)
                sinBeta=np.sin(beta)
                tanBeta=sinBeta/cosBeta

                # -------------------- buoyancy and waterplane area properties ------------------------

                xWP = intrp(0, rA[2], rB[2], rA[0], rB[0])                     # x coordinate where member axis cross the waterplane [m]
                yWP = intrp(0, rA[2], rB[2], rA[1], rB[1])                     # y coordinate where member axis cross the waterplane [m]
                if self.shape=='circular':
                    dWP = intrp(0, rA[2], rB[2], self.d[i], self.d[i-1])       # diameter of member where its axis crosses the waterplane [m]
                    AWP = (np.pi/4)*dWP**2                                     # waterplane area of member [m^2]
                    IWP = (np.pi/64)*dWP**4                                    # waterplane moment of inertia [m^4] approximates as a circle
                    IxWP = IWP                                                 # MoI of circular waterplane is the same all around
                    IyWP = IWP                                                 # MoI of circular waterplane is the same all around
                elif self.shape=='rectangular':
                    slWP = intrp(0, rA[2], rB[2], self.sl[i], self.sl[i-1])    # side lengths of member where its axis crosses the waterplane [m]
                    AWP = slWP[0]*slWP[1]                                      # waterplane area of rectangular member [m^2]
                    IxWP = (1/12)*slWP[0]*slWP[1]**3                           # waterplane MoI [m^4] about the member's LOCAL x-axis, not the global x-axis
                    IyWP = (1/12)*slWP[0]**3*slWP[1]                           # waterplane MoI [m^4] about the member's LOCAL y-axis, not the global y-axis
                    I = np.diag([IxWP, IyWP, 0])                               # area moment of inertia tensor
                    T = self.R.T                                               # the transformation matrix to unrotate the member's local axes
                    I_rot = np.matmul(T.T, np.matmul(I,T))                     # area moment of inertia tensor where MoI axes are now in the same direction as PRP
                    IxWP = I_rot[0,0]
                    IyWP = I_rot[1,1]

                LWP = abs(rA[2]/cosPhi)                   # get length of segment along member axis that is underwater [m]

                # Assumption: the areas and MoI of the waterplane are as if the member were completely vertical, i.e. it doesn't account for phi
                # This can be fixed later on if needed. We're using this assumption since the fix wouldn't significantly affect the outputs

                # Total enclosed underwater volume [m^3] and distance along axis from end A to center of buoyancy of member [m]
                if self.shape=='circular':
                    V_UWi, hc = FrustumVCV(self.d[i-1], dWP, LWP)
                elif self.shape=='rectangular':
                    V_UWi, hc = FrustumVCV(self.sl[i-1], slWP, LWP)

                r_center = rA + self.q*hc          # absolute coordinates of center of volume of this segment [m]


                # >>>> question: should this function be able to use displaced/rotated values? <<<<

                # ------------- get hydrostatic derivatives ----------------

                # derivatives from global to local
                dPhi_dThx  = -sinBeta                     # \frac{d\phi}{d\theta_x} = \sin\beta
                dPhi_dThy  =  cosBeta
                dFz_dz   = -rho*g*AWP /cosPhi

                # note: below calculations are based on untapered case, but
                # temporarily approximated for taper by using dWP (diameter at water plane crossing) <<< this is rough

                # buoyancy force and moment about end A
                Fz = rho*g* V_UWi
                M = 0
                if self.shape=='circular': # Need to find the equivalent of this for the rectangular case
                    M  = -rho*g*pi*( dWP**2/32*(2.0 + tanPhi**2) + 0.5*(rA[2]/cosPhi)**2)*sinPhi  # moment about axis of incline
                Mx = M*dPhi_dThx
                My = M*dPhi_dThy

                Fvec[2] += Fz                           # vertical buoyancy force [N]
                Fvec[3] += Mx + Fz*rA[1]                # moment about x axis [N-m]
                Fvec[4] += My - Fz*rA[0]                # moment about y axis [N-m]


                # normal approach to hydrostatic stiffness, using this temporarily until above fancier approach is verified
                Cmat[2,2] += -dFz_dz
                Cmat[2,3] += rho*g*(     -AWP*yWP    )
                Cmat[2,4] += rho*g*(      AWP*xWP    )
                Cmat[3,2] += rho*g*(     -AWP*yWP    )
                Cmat[3,3] += rho*g*(IxWP + AWP*yWP**2 )
                Cmat[3,4] += rho*g*(      AWP*xWP*yWP)
                Cmat[4,2] += rho*g*(      AWP*xWP    )
                Cmat[4,3] += rho*g*(      AWP*xWP*yWP)
                Cmat[4,4] += rho*g*(IyWP + AWP*xWP**2 )

                Cmat[3,3] += rho*g*V_UWi * r_center[2]
                Cmat[4,4] += rho*g*V_UWi * r_center[2]

                V_UW += V_UWi
                r_centerV += r_center*V_UWi


            # fully submerged case
            elif rA[2] <= 0 and rB[2] <= 0:

                # displaced volume [m^3] and distance along axis from end A to center of buoyancy of member [m]
                if self.shape=='circular':
                    V_UWi, hc = FrustumVCV(self.d[i-1], self.d[i], self.stations[i]-self.stations[i-1])
                elif self.shape=='rectangular':
                    V_UWi, hc = FrustumVCV(self.sl[i-1], self.sl[i], self.stations[i]-self.stations[i-1])

                r_center = rA + self.q*hc             # center of volume of this segment relative to PRP [m]

                # buoyancy force (and moment) vector
                Fvec += translateForce3to6DOF(np.array([0, 0, rho*g*V_UWi]), r_center)

                # hydrostatic stiffness matrix (about end A)
                Cmat[3,3] += rho*g*V_UWi * r_center[2]
                Cmat[4,4] += rho*g*V_UWi * r_center[2]

                V_UW += V_UWi
                r_centerV += r_center*V_UWi

            else: # if the members are fully above the surface

                pass

        if V_UW > 0:
            r_center = r_centerV/V_UW    # calculate overall member center of buoyancy
        else:
            r_center = np.zeros(3)       # temporary fix for out-of-water members
        
        self.V = V_UW  # store submerged volume
        
        return Fvec, Cmat, V_UW, r_center, AWP, IWP, xWP, yWP


    def calcHydroConstants(self, r_ref=np.zeros(3), sum_inertia=False, rho=1025, g=9.81, k_array=None):
        '''Compute the Member's linear strip-theory-hydrodynamics terms, 
        related to drag and added mass, which are also a precursor to 
        excitation. All computed quantities are in global orientations.
        
        Parameters
        ----------
        r_ref : size-3 vector
            Reference point coordinates to compute matrices about [m].
        sum_inertia : boolean, optional
            Flag to calculate and return an overall inertial excitation matrix
            (default False).
        
        Returns
        -------
        A_hydro, I_hydro : 6x6 matrices
            Hydrodynamic added mass and inertial excitation matrices.
        '''
        
        # hydrodynamic added mass and excitation matrices from strip theory [kg, kg-m, kg-m^2]
        A_hydro = np.zeros([6,6])
        I_hydro = np.zeros([6,6])

        circ = self.shape=='circular'  # boolean for circular vs. rectangular
        
        # Local inertial excitation matrix - Froude-Krylov.
        # Calculated in a separate function to allow for cases with MacCamy-Fuchs correction.
        self.calcImat(rho=rho, g=g, k_array=k_array)

        # loop through each node of the member
        for il in range(self.ns):

            # only process hydrodynamics if this node is submerged
            if self.r[il,2] < 0:
                
                # only compute inertial loads and added mass for members that aren't modeled with potential flow
                if self.potMod==False:

                    # interpolate coefficients for the current strip
                    Ca_q   = np.interp( self.ls[il], self.stations, self.Ca_q  )
                    Ca_p1  = np.interp( self.ls[il], self.stations, self.Ca_p1 )
                    Ca_p2  = np.interp( self.ls[il], self.stations, self.Ca_p2 )
                    Ca_End = np.interp( self.ls[il], self.stations, self.Ca_End)


                    # ----- compute side effects (transverse only) -----

                    # volume assigned to this node
                    if circ:
                        v_i = 0.25*np.pi*self.ds[il]**2*self.dls[il]
                    else:
                        v_i = self.ds[il,0]*self.ds[il,1]*self.dls[il]  
                        
                    if self.r[il,2] + 0.5*self.dls[il] > 0:    # if member extends out of water 
                        v_i = v_i * (0.5*self.dls[il] - self.r[il,2]) / self.dls[il]  # scale volume by the portion that is under water
                    
                    # Local added mass matrix (axial term explicitly excluded here - we aren't dealing with chains)
                    Amat_sides = rho*v_i *( Ca_p1*self.p1Mat + Ca_p2*self.p2Mat )
                                       
                    # ----- add axial/end effects for added mass, and excitation including dynamic pressure ------
                    # Note : v_i and a_i work out to zero for non-tapered sections or non-end sections

                    # compute volume assigned to this end surface, and
                    # signed end area (positive facing down) = mean diameter of strip * radius change of strip
                    if circ:
                        v_i = np.pi/12.0 * abs(  (self.ds[il]+self.drs[il])**3 
                                               - (self.ds[il]-self.drs[il])**3 )  
                        a_i = np.pi*self.ds[il] * self.drs[il]   
                    else:
                        v_i = np.pi/12.0 * (  (np.mean(self.ds[il]+self.drs[il]))**3 
                                            - (np.mean(self.ds[il]-self.drs[il]))**3 )    # so far just using sphere eqn and taking mean of side lengths as d
                        a_i = (  (self.ds[il,0]+self.drs[il,0])*(self.ds[il,1]+self.drs[il,1]) 
                               - (self.ds[il,0]-self.drs[il,0])*(self.ds[il,1]-self.drs[il,1]))
                        # >>> should support different coefficients or reference volumes for rectangular cross sections <<<

                    # Local added mass matrix
                    Amat_end = rho*v_i * Ca_End*self.qMat
                    

                    # ----- sum up side and end added mass and inertial excitation coefficient matrices ------
                    self.Amat[il,:,:] = Amat_sides + Amat_end
                    self.a_i[il] = a_i  # signed axial reference area for use in dynamic pressure force
                    
                    
                    # add to global added mass and inertial excitation matrices
                    # which consider the mean offsets and are relative to the ref in global orientation
                    A_hydro += translateMatrix3to6DOF(self.Amat[il,:,:], self.r[il,:] - r_ref[:3])    
                    if sum_inertia:
                        I_hydro += translateMatrix3to6DOF(self.Imat[il,:,:], self.r[il,:] - r_ref[:3])   

        if sum_inertia:
            return A_hydro, I_hydro
        else:
            return A_hydro

    def calcImat(self, rho=1025, g=9.81, k_array=None):
        '''Compute the Member's linear strip-theory-hydrodynamics excitation 
        matrix, Imat, which is the term Cm=(1+Ca) from Morison's equation.
        Optionally, Cm can be computed using the MacCamy-Fuchs correction for
        a circular cylinder if the wave number k is specified.
        All computed quantities are in global orientation.
        '''
        
        circ = self.shape=='circular'  # boolean for circular vs. rectangular
        
        MCF = False
        if k_array is not None:
            MCF = self.MCF
            if len(k_array) != self.Imat_MCF.shape[3] and MCF:
                raise ValueError(f'Number of elements in wave number vector ({len(k_array)}) must match the number of frequencies previously specified for member ({self.Imat.shape[3]}).')

        # loop through each node of the member
        for il in range(self.ns):

            # only process hydrodynamics if this node is submerged
            if self.r[il,2] < 0:
                
                # only compute inertial loads and added mass for members that aren't modeled with potential flow
                if self.potMod==False:
                    
                    # interpolate coefficients for the current strip
                    Ca_p1  = np.interp( self.ls[il], self.stations, self.Ca_p1 )
                    Ca_p2  = np.interp( self.ls[il], self.stations, self.Ca_p2 )
                    Ca_End = np.interp( self.ls[il], self.stations, self.Ca_End)


                    # ----- compute side effects (transverse only) -----

                    # volume assigned to this node
                    if circ:
                        v_i = 0.25*np.pi*self.ds[il]**2*self.dls[il]
                    else:
                        v_i = self.ds[il,0]*self.ds[il,1]*self.dls[il]  
                        
                    if self.r[il,2] + 0.5*self.dls[il] > 0:    # if member extends out of water 
                        v_i = v_i * (0.5*self.dls[il] - self.r[il,2]) / self.dls[il]  # scale volume by the portion that is under water
                                       
                    # Local inertial excitation matrix - Froude-Krylov  
                    # (axial term explicitly excluded here - we aren't dealing with chains)
                    # Note, the 1 is the Cp, dynamic pressure, term.
                    if MCF:
                        Imat_sides = np.zeros([3, 3, len(k_array)], dtype=complex)
                        for ik, k in enumerate(k_array):
                            # Apply MCF correction to the nodes of sections that cross the water surface
                            Cm_p1, Cm_p2 = self.getCmSides(il, k=k)
                            Imat_sides[:, :, ik] = rho*v_i * (Cm_p1*self.p1Mat + Cm_p2*self.p2Mat)
                    else:
                        Cm_p1, Cm_p2 = self.getCmSides(il, k=None)
                        Imat_sides = rho*v_i *(Cm_p1*self.p1Mat + Cm_p2*self.p2Mat)


                    # ----- add axial/end effects for excitation ------
                    # Note : v_i work out to zero for non-tapered sections or non-end sections

                    # compute volume assigned to this end surface, and
                    # signed end area (positive facing down) = mean diameter of strip * radius change of strip
                    if circ:
                        v_i = np.pi/12.0 * abs(  (self.ds[il]+self.drs[il])**3 
                                               - (self.ds[il]-self.drs[il])**3 )  
                    else:
                        v_i = np.pi/12.0 * (  (np.mean(self.ds[il]+self.drs[il]))**3 
                                            - (np.mean(self.ds[il]-self.drs[il]))**3 )    # so far just using sphere eqn and taking mean of side lengths as d
                        # >>> should support different coefficients or reference volumes for rectangular cross sections <<<

                    
                    # Local inertial excitation matrix 
                    # Note, there is no 1 added to Ca_End because dynamic pressure is handled separately
                    Imat_end = rho*v_i * Ca_End*self.qMat  
                    
                    # ----- sum up side and end added mass and inertial excitation coefficient matrices ------
                    if MCF != 0:
                        self.Imat_MCF[il,:,:, :] = Imat_sides[:,:, :] + Imat_end[:,:, None]
                    else:
                        self.Imat[il,:,:] = Imat_sides[:,:] + Imat_end[:,:]
    

    def getCmSides(self, il, k=None):
        if il < 0 or il >= self.ns:
            raise Exception(f"Member {self.name}: node outside range in getCm.")

        MCF = False
        if k is not None:
            MCF = self.MCF

        Ca_p1  = np.interp( self.ls[il], self.stations, self.Ca_p1 )
        Ca_p2  = np.interp( self.ls[il], self.stations, self.Ca_p2 )
        Cm_p1_0, Cm_p2_0 = (1.+Ca_p1), (1.+Ca_p2)

        if not MCF:
            Cm_p1 = Cm_p1_0
            Cm_p2 = Cm_p2_0

        else:            
            R = self.ds[il]/2
            Hp1 = 0.5 * (hankel1(0, k*R) - hankel1(2, k*R)) # Derivative of the Hankel function of first kind and order one
            Cm_p1 = 4j  / (np.pi * (k*R)**2 * Hp1)
            Cm_p2 = Cm_p1

            # The MCF correction only makes sense for short waves.
            # We use the threshold lamda/D < 5 commonly adopted for the Morison equation
            # but changing smoothly from Cm_p1_0 and Cm_p2_0 using a ramp function.
            # This is particularly useful for cases where the tuned value of Ca was too 
            # different from the theoretical value of 1 (e.g. the OC4 Platform, with Ca = 0.63)
            # and the user doesn't want to tune the model again
            Tr = np.pi/5/R # Threshold value
            T0 = 0 # Value to start the transition
            ramp = 0.5*(1-np.cos(np.pi*(k-T0)/Tr)) if k<Tr else 1
            ramp = 0 if k<=T0 else ramp 
            Cm_p1 = Cm_p1 * ramp + Cm_p1_0 * (1-ramp)
            Cm_p2 = Cm_p2 * ramp + Cm_p2_0 * (1-ramp)

        return Cm_p1, Cm_p2

    def calcQTF_slenderBody(self, Xi, beta, w, k, depth, rho=1025, g=9.8, verbose=False):
        nw = len(w)
        qtf = np.zeros([nw, nw, 6], dtype=complex)

        # Don't do anything if the member is above the water        
        if self.rA[2] > 0 and self.rB[2] > 0:
            return qtf            

        # convenience boolian for circular vs. rectangular cross sections
        circ = self.shape=='circular'

        # Get fluid and body kinematics at each node of the member
        nodeV           = np.zeros([3, nw, self.ns], dtype=complex)    # Node velocity
        dr              = np.zeros([3, nw, self.ns], dtype=complex)    # Node displacement
        u               = np.zeros([3, nw, self.ns], dtype=complex)    # Incident fluid velocity at node
        grad_u          = np.zeros([3, 3, nw, self.ns], dtype=complex) # Gradient matrix of first-order velocity
        grad_dudt       = np.zeros([3, 3, nw, self.ns], dtype=complex) # Gradient matrix of first-order acceleration
        nodeV_axial_rel = np.zeros([nw, self.ns], dtype=complex)       # Node relative axial velocity
        grad_pres1st    = np.zeros([3, nw, self.ns], dtype=complex)    # Gradient of first order pressure at each node
        for iNode, r in enumerate(self.r):
            dr[:,:, iNode], nodeV[:,:, iNode], _ = getKinematics(r, Xi, w)
            u[:,:, iNode], _, _ = getWaveKin(np.ones(w.shape), beta, w, k, depth, r, nw, rho=rho, g=g)

            for iw in range(nw):
                grad_u[:, :, iw, iNode]    = getWaveKin_grad_u1(w[iw], k[iw], beta, depth, r)
                grad_dudt[:, :, iw, iNode] = getWaveKin_grad_dudt(w[iw], k[iw], beta, depth, r)
                nodeV_axial_rel[iw, iNode] = np.dot(u[:, iw, iNode]-nodeV[:,iw, iNode], self.q)
                grad_pres1st[:, iw, iNode] = getWaveKin_grad_pres1st(k[iw], beta, depth, r, rho=rho, g=g)
                                            
        # Get fluid and body kinematics at the intersection of the member intersection with the water line
        eta   = np.zeros([nw], dtype=complex)    # Incoming wave elevation
        eta_r = np.zeros([nw], dtype=complex)    # Relative incoming wave elevation (i.e. wave elevation - vertical body motion)
        ud_wl = np.zeros([3, nw], dtype=complex) # Incoming wave acceleration
        dr_wl = np.zeros([3, nw], dtype=complex) # Body displacement at intersection with water line
        a_wl  = np.zeros([3, nw], dtype=complex) # Body acceleration at intersection with water line
        if self.r[-1,2] * self.r[0,2] < 0: # Only if the member intersects the water line
            r_int = self.r[0,:] + (self.r[-1,:] - self.r[0,:]) * (0. - self.r[0,2]) / (self.r[-1,2] - self.r[0,2]) # Intersection point
            _, ud_wl, eta  = getWaveKin(np.ones(w.shape), beta, w, k, depth, r_int, nw, rho=1, g=1) # getWaveKin actually returns dynamic pressure, so we use unitary rho and g to get the wave elevation
            dr_wl, _, a_wl = getKinematics(r_int, Xi, w)
        
        # Vector with the acceleration of gravity projected along the p1 and p2 axes of the member (which are perpendicular to the member's axis)
        g_e1 = np.zeros([3, nw], dtype=complex)
        for iw in range(nw):
            g_e1[:, iw] = -g * (np.cross(Xi[3:, iw], self.p1)[2] * self.p1 + np.cross(Xi[3:, iw], self.p2)[2] * self.p2)

        # Relative wave elevation
        eta_r = eta-dr_wl[2,:]

        # Loop through each pair of frequency
        for i1, (w1, k1) in enumerate(zip(w, k)):
            if verbose:
                    print(f"     - Row {i1+1:02d} of {nw:02d}", end='\r')
            for i2, (w2, k2) in enumerate(zip(w, k)):
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
                for il in range(self.ns):
                    # only process hydrodynamics if this node is submerged
                    if self.r[il,2] >= 0:
                        continue

                    # interpolate coefficients for the current strip
                    Ca_p1  = np.interp( self.ls[il], self.stations, self.Ca_p1 )
                    Ca_p2  = np.interp( self.ls[il], self.stations, self.Ca_p2 )
                    Ca_End = np.interp( self.ls[il], self.stations, self.Ca_End)

                    # ----- compute side effects ---------------------------------------------------------
                    if circ:
                        v_i = 0.25*np.pi*self.ds[il]**2*self.dls[il]
                    else:
                        v_i = self.ds[il,0]*self.ds[il,1]*self.dls[il]  # member volume assigned to this node
                        
                    if self.r[il,2] + 0.5*self.dls[il] > 0:    # if member extends out of water              # <<< may want a better appraoch for this...
                        v_i = v_i * (0.5*self.dls[il] - self.r[il,2]) / self.dls[il]  # scale volume by the portion that is under water

                    # Force component due to second-order wave potential
                    acc_2ndPot, p_2nd = getWaveKin_pot2ndOrd(w1, w2, k1, k2, beta, beta, depth, self.r[il,:], g=g, rho=rho) # Second-order pressure will be used later
                    f_2ndPot = rho*v_i * np.matmul((1.+Ca_p1)*self.p1Mat + (1.+Ca_p2)*self.p2Mat, acc_2ndPot)

                    # Force component due to convective acceleration
                    conv_acc = 0.25 * ( np.matmul(grad_u[:, :, i1, il], np.conj(u[:, i2, il])) + np.matmul(np.conj(grad_u[:, :, i2, il]), u[:, i1, il]) )
                    f_conv   = rho*v_i * np.matmul((1.+Ca_p1)*self.p1Mat + (1.+Ca_p2)*self.p2Mat, conv_acc)

                    # Force component due to Rainey's axial-divergence acceleration
                    f_axdv   = rho*v_i * np.matmul(Ca_p1*self.p1Mat + Ca_p2*self.p2Mat, getWaveKin_axdivAcc(w1, w2, k1, k2, beta, beta, depth, self.r[il,:], nodeV[:, i1, il], nodeV[:, i2, il], self.q, g=g))
                    
                    # Force component due to body motions within the first-order wave field
                    acc_nabla = 0.25*np.matmul(np.squeeze(grad_dudt[:, :, i1, il]), np.squeeze(np.conj(dr[:, i2, il]))) + 0.25*np.matmul(np.squeeze(np.conj(grad_dudt[:, :, i2, il])), np.squeeze(dr[:, i1, il]))
                    f_nabla  = rho*v_i * np.matmul((1.+Ca_p1)*self.p1Mat + (1.+Ca_p2)*self.p2Mat, acc_nabla)

                    # Force component due to Rainey's body rotation term
                    OMEGA1  = -getH(1j*w1*Xi[3:,i1]) # The alternator matrix is the opposite of what we need, that is why we have a minus sign
                    OMEGA2  = -getH(1j*w2*Xi[3:,i2])
                    f_rslb  = -0.25*2*np.matmul(Ca_p1*self.p1Mat + Ca_p2*self.p2Mat, np.matmul(OMEGA1, np.conj(nodeV_axial_rel[i2, il]*self.q)) + np.matmul(np.conj(OMEGA2), nodeV_axial_rel[i1, il]*self.q))
                    f_rslb *= rho*v_i                                                
                    
                    # Rainey load that is only non zero for non-circular cylinders
                    # TODO: Now that it's working, should be a separate variable and not inside f_rslb
                    u1_aux  = u[:,i1, il]-nodeV[:,i1, il]
                    u2_aux  = u[:,i2, il]-nodeV[:,i2, il]
                    Vmatrix1 = getWaveKin_grad_u1(w1, k1, beta, depth, self.r[il,:]) + OMEGA1
                    Vmatrix2 = getWaveKin_grad_u1(w2, k2, beta, depth, self.r[il,:]) + OMEGA2
                    aux      = 0.25*(np.matmul(Vmatrix1, np.conj(np.matmul(Ca_p1*self.p1Mat + Ca_p2*self.p2Mat, u2_aux))) + np.matmul(np.conj(Vmatrix2), np.matmul(Ca_p1*self.p1Mat + Ca_p2*self.p2Mat, u1_aux)))
                    aux     -= np.matmul(self.qMat, aux) # remove axial component                            
                    f_rslb  += rho*v_i * aux

                    # Similar to the one right above, but note that the order of multiplications with matmul is different
                    u1_aux  -= np.matmul(self.qMat, u1_aux) # remove axial component
                    u2_aux  -= np.matmul(self.qMat, u2_aux)
                    aux      = 0.25*(np.matmul(Ca_p1*self.p1Mat + Ca_p2*self.p2Mat, np.matmul(Vmatrix1, np.conj(u2_aux))) + np.matmul(Ca_p1*self.p1Mat + Ca_p2*self.p2Mat, np.matmul(np.conj(Vmatrix2), u1_aux)))
                    f_rslb  += - rho*v_i * aux

                                                
                    # ----- axial/end effects  ------
                    # note : v_i and a_i work out to zero for non-tapered sections or non-end sections
                    if circ:
                        v_i = np.pi/12.0 * abs((self.ds[il]+self.drs[il])**3 - (self.ds[il]-self.drs[il])**3)  # volume assigned to this end surface
                        a_i = np.pi*self.ds[il] * self.drs[il]   # signed end area (positive facing down) = mean diameter of strip * radius change of strip
                    else:
                        v_i = np.pi/12.0 * ((np.mean(self.ds[il]+self.drs[il]))**3 - (np.mean(self.ds[il]-self.drs[il]))**3)    # so far just using sphere eqn and taking mean of side lengths as d
                        a_i = (self.ds[il,0]+self.drs[il,0])*(self.ds[il,1]+self.drs[il,1]) - (self.ds[il,0]-self.drs[il,0])*(self.ds[il,1]-self.drs[il,1])
                    
                    f_2ndPot += self.a_i[il]*p_2nd*self.q # 2nd order pressure
                    f_2ndPot += rho*v_i*Ca_End*np.matmul(self.qMat, acc_2ndPot) # 2nd order axial acceleration
                    f_conv   += rho*v_i*Ca_End*np.matmul(self.qMat, conv_acc)   # convective acceleration
                    f_nabla  += rho*v_i*Ca_End*np.matmul(self.qMat, acc_nabla)  # due to body motions within the first-order wave field - acceleration part
                    p_nabla   = 0.25*np.dot(grad_pres1st[:, i1, il], np.conj(dr[:, i2, il])) + 0.25*np.dot(np.conj(grad_pres1st[:, i2, il]), dr[:, i1, il])
                    f_nabla  += self.a_i[il]*p_nabla*self.q # due to body motions within the first-order wave field - pressure part
                    p_drop    = -2*0.25*0.5*rho*np.dot(np.matmul(self.p1Mat + self.p2Mat, u[:,i1, il]-nodeV[:, i1, il]), np.conj(np.matmul(Ca_p1*self.p1Mat + Ca_p2*self.p2Mat, u[:,i2, il]-nodeV[:, i2, il])))
                    f_conv   += self.a_i[il]*p_drop*self.q

                    u1_aux    = np.matmul(Ca_p1*self.p1Mat + Ca_p2*self.p2Mat, u1_aux)
                    u2_aux    = np.matmul(Ca_p1*self.p1Mat + Ca_p2*self.p2Mat, u2_aux)
                    f_transv  = 0.25*self.a_i[il]*rho*(np.conj(u1_aux)*nodeV_axial_rel[i2, il] + u2_aux*np.conj(nodeV_axial_rel[i1, il]))
                    f_conv   += f_transv

                    F_2ndPot += translateForce3to6DOF(f_2ndPot, self.r[il,:])
                    F_conv   += translateForce3to6DOF(f_conv, self.r[il,:])                   
                    F_axdv   += translateForce3to6DOF(f_axdv, self.r[il,:])
                    F_nabla  += translateForce3to6DOF(f_nabla, self.r[il,:])
                    F_rslb   += translateForce3to6DOF(f_rslb, self.r[il,:])
                                                    
                # Force acting at the intersection of the member with the mean waterline
                f_eta = np.zeros(3, dtype=complex)
                F_eta = np.zeros(6, dtype=complex)
                if self.r[-1,2] * self.r[0,2] < 0:
                    # Need just the cross section area, as the length along the cylinder is the relative wave elevation
                    # Get the area of the cross section at the mean waterline
                    i_wl = np.where(self.r[:,2] < 0)[0][-1] # find index of self.r[:,2] that is right before crossing 0
                    if circ:
                        if i_wl != len(self.ds)-1:
                            d_wl = 0.5*(self.ds[i_wl]+self.ds[i_wl+1])
                        else:
                            d_wl = self.ds[i_wl]
                        a_i = 0.25*np.pi*d_wl**2
                    else:
                        if i_wl != len(self.ds)-1:
                            d1_wl = 0.5*(self.ds[i_wl,0]+self.ds[i_wl+1,0])
                            d2_wl = 0.5*(self.ds[i_wl,1]+self.ds[i_wl+1,1])
                        else:
                            d1_wl = self.ds[i_wl, 0]
                            d2_wl = self.ds[i_wl, 1]
                        a_i = d1_wl*d2_wl

                    f_eta = 0.25*((ud_wl[:,i1])*np.conj(eta_r[i2])+np.conj((ud_wl[:,i2]))*eta_r[i1]) # Product between incoming wave acceleration and relative wave elevation
                    f_eta = rho*a_i*np.matmul((1.+Ca_p1)*self.p1Mat + (1.+Ca_p2)*self.p2Mat, f_eta) # Project in the correct directions (and scale by the area and density)
                    a_eta = 0.25*((a_wl[:,i1])*np.conj(eta_r[i2])+np.conj((a_wl[:,i2]))*eta_r[i1])  # Product between body acceleration and relative wave elevation
                    f_eta -= rho*a_i*np.matmul(Ca_p1*self.p1Mat + Ca_p2*self.p2Mat, a_eta) # Project in the correct directions (and scale by the area and density)
                    f_eta -= 0.25*rho*a_i * (g_e1[:,i1]*np.conj(eta_r[i2])+np.conj(g_e1[:,i2])*eta_r[i1]) # Add hydrostatic term

                    F_eta = translateForce3to6DOF(f_eta, r_int) # Load vector in 6 DOF 
                
                # Total contribution to this frequency pair of the QTF due to the current member
                qtf[i1, i2, :] += F_2ndPot + F_axdv + F_conv + F_nabla + F_eta + F_rslb

                # Add Kim and Yue correction                                                     
                qtf[i1, i2, :] += self.correction_KAY(depth, w1, w2, beta, rho=rho, g=g, k1=k1, k2=k2, Nm=10)        
        return qtf

    def correction_KAY(self, h, w1, w2, beta, rho=1025, g=9.81, k1=None, k2=None, Nm=10):
        '''For surface-piercing vertical cylinders, we can partially account for second-order diffraction loads
           by using the analytical solution for a bottom-mounted, surface-piercing, vertical cylinder. 
           For the mean loads: Kim and Yue (1989) The complete second-order diffraction solution for an axisymmetric body - Part 1. Monochromatic incident waves
           For the difference-frequency loads: Kim and Yue (1990) The complete second-order diffraction solution for an axisymmetric body - Part 2. Bichromatic incident waves and body motions
    
           Output = F / (Aj * Al), where Aj and Al are the complex amplitudes of the wave pair
           The output force is in the same direction as the incoming waves. 
           For now, only long crested seas are considered (both wave components along the same direction). 
        '''

        # Omega is a convenience variable used in the analytical solution
        def omega(k1R, k2R, n):
            # Derivatives of the Hankel function
            H_N_ii = 0.5 * (hankel1(n - 1, k1R) - hankel1(n + 1, k1R))
            H_N_jj = 0.5 * np.conj(hankel1(n - 1, k2R) - hankel1(n + 1, k2R))
            H_Nm1_ii = 0.5 * (hankel1(n, k1R) - hankel1(n + 2, k1R))
            H_Nm1_jj = 0.5 * np.conj(hankel1(n, k2R) - hankel1(n + 2, k2R))

            return 1 / (H_Nm1_ii * H_N_jj) - 1 / (H_N_ii * H_Nm1_jj)

        F = np.zeros(6, dtype=complex)
        if not self.MCF:
            return F
                
        # Compute k1 and k2 if not provided
        if k1 is None:
            k1 = waveNumber(w1, h)
        if k2 is None:
            k2 = waveNumber(w2, h)

        # In this whole function, we assume that the wave direction is the same for both components, but I will leave this below
        # in case we want to expand to short-crested seas in the future    
        cosB1, sinB1 = np.cos(beta), np.sin(beta) 
        cosB2, sinB2 = cosB1, sinB1    
        k1_k2 = np.array([k1 * cosB1 - k2 * cosB2, k1 * sinB1 - k2 * sinB2, 0]) # For the phase    

        # The force is derived for a vertical cylinder and, in the original solution, it is aligned with the waves.
        # We will say that it is perpendicular to the cylinder axis but aligned with the wave direction
        beta_vec = np.array([cosB1, sinB1, 0])
        pforce = np.dot(beta_vec, self.p1)*self.p1 + np.dot(beta_vec, self.p2)*self.p2
        pforce = pforce / np.linalg.norm(pforce) # Normalize the force
                
        #==== Component due to the relative wave elevation. Lumped at the intersection with the mean waterline
        if self.rA[2]*self.rB[2] < 0:           
            # Find intersection with the mean waterline (z=0) and its radius
            rwl = self.rA + (self.rB - self.rA) * (0 - self.rA[2]) / (self.rB[2] - self.rA[2])
            radii = 0.5*np.array(self.ds)
            R = np.interp(0, self.r[:,2], radii)

            # Compute force lumped at the intersection with the mean waterline
            k1R, k2R = k1*R, k2*R
            Fwl = 0+0j
            # if k1R >= np.pi/10 or k2R >= np.pi/10:
            for nn in range(Nm + 1):
                Fwl += -rho*g*R*2j/np.pi/(k1R*k2R) * omega(k1R, k2R, nn)
            
            Fwl = np.real(Fwl) # Get only the part related to diffraction effects to avoid double counting with Rainey's equation
            Fwl *= np.exp(-1j*np.dot(k1_k2, rwl)) # Solution considers cylinder at (0,0). Displace it to actual location
            F += translateForce3to6DOF(Fwl*pforce, rwl)


        #==== Component due to the quadratic velocity in Bernoullis equation
        # The integration along the member is very sensitive. So, we perform the 
        # integration analytically around each node.
        if self.rA[2]*self.rB[2] < 0:
            for il, r1 in enumerate(self.r[:-1]): # The integration is based on the bottom node, so we skip the last one
                # No point in computing for nodes above the mean water line
                z1 = r1[2]
                if z1 > 0:
                    continue

                r2 = self.r[il+1]
                z1, z2 = r1[2], r2[2]
                z2 = 0 if z2 > 0 else z2

                # Get values used in the analytical solution
                R1 = self.ds[il]/2
                if self.dls[il] == 0: # If this is an end node, the diameter was divided by two
                    R1 = self.ds[il]

                R2 = self.ds[il+1]/2
                if self.dls[il+1] == 0:
                    R2 = self.ds[il]
                
                R = 0.5*(R1 + R2) # Consider the mean radius
                k1R, k2R = k1*R, k2*R # Nondimensional wave numbers
                H = h/R # Nondimensional water depth
                wm = (w1-w2)/np.sqrt(g/h) # Nondimensional difference frequency
                k1h, k2h = k1R*H, k2R*H

                # For mean loads
                dF = 0+0j  # Force per unit length. Assumed to be aligned with the wave propagation direction
                if w1 == w2:
                    Im = 0.5 * (np.sinh((k1 + k2)*(z2+h)) / (k1h + k2h) - (z2+h)/h - np.sinh((k1 + k2)*(z1+h)) / (k1h + k2h) + (z1+h)/h)
                    Ip = 0.5 * (np.sinh((k1 + k2)*(z2+h)) / (k1h + k2h) + (z2+h)/h - np.sinh((k1 + k2)*(z1+h)) / (k1h + k2h) - (z1+h)/h)                    

                else:
                    Im = 0.5 * (np.sinh((k1 + k2)*(z2+h)) / (k1h + k2h) - np.sinh((k1 - k2)*(z2+h)) / (k1h - k2h) - np.sinh((k1 + k2)*(z1+h)) / (k1h + k2h) + np.sinh((k1 - k2)*(z1+h)) / (k1h - k2h))
                    Ip = 0.5 * (np.sinh((k1 + k2)*(z2+h)) / (k1h + k2h) + np.sinh((k1 - k2)*(z2+h)) / (k1h - k2h) - np.sinh((k1 + k2)*(z1+h)) / (k1h + k2h) - np.sinh((k1 - k2)*(z1+h)) / (k1h - k2h))                    

                coshk1h, coshk2h = np.cosh(k1h), np.cosh(k2h)                    
                for nn in range(Nm + 1):
                    dF += rho*g*R*2j/np.pi/(k1R*k2R) * omega(k1R, k2R, nn) * (k1h*k2h/np.sqrt(k1h*np.tanh(k1h))/np.sqrt(k2h*np.tanh(k2h)) * (Im + Ip*nn*(nn+1)/k1R/k2R)/coshk1h/coshk2h)

                        
                # The force calculated above considers a cylinder at (x,y)=(0,0), so we need to account for the wave phase
                r = 0.5*(r1 + r2)
                dF = np.real(dF) # Get only the part related to diffraction effects to avoid double counting with Rainey's equation
                dF *= np.exp(-1j*np.dot(k1_k2, rwl)) # Solution considers cylinder at (0,0). Displace it to actual location
                F += translateForce3to6DOF(dF*pforce, r)

        if k1 < k2:
            F = np.conj(F)
        
        return F

    def calcCurrentLoads(self, depth, speed=0, heading=0, Zref=0, shearExp_water=0.12, rho=1025, g=9.81, r_ref=np.zeros(3)):
        '''method to calculate the "static" current loads on each member and save as a current force
        Uses a simple power law relationship to calculate the current velocity as a function of member node depth
        
        Parameters
        ----------
        depth: float
            Water depth [m]
        speed: float
            Current speed [m/s]
        heading: float
            Current heading from global X (rotates about global Z) [deg]
        Zref: float
            Reference z elevation for current profile (at the sea surface by default) [m]
        shearExp_water: float
            Exponent for current profile [-]
        rho: float
            Water density [kg/m^3]
        g: float
            Gravitational acceleration [m/s^2]
        r_ref: size-3 vector
            Reference point coordinates to compute current loads [m].

        Returns:
        D_hydro: 6x1 array
            Mean current force and moment on the member in global coordinates wrp to r_ref [N, N-m]
        '''
        D_hydro = np.zeros(6)

        circ = self.shape=='circular'  # convenience boolian for circular vs. rectangular cross sections

        # loop through each node of the member
        for il in range(self.ns):

            # only process hydrodynamics if this node is submerged
            if self.r[il,2] < 0:

                # calculate current velocity as a function of node depth [x,y,z] (assumes no vertical current velocity)
                v = speed * ((depth - abs(self.r[il,2]))/(depth + Zref))**shearExp_water
                #v = speed
                vcur = np.array([v*np.cos(np.deg2rad(heading)), v*np.sin(np.deg2rad(heading)), 0])

                # interpolate coefficients for the current strip
                Cd_q   = np.interp( self.ls[il], self.stations, self.Cd_q  )
                Cd_p1  = np.interp( self.ls[il], self.stations, self.Cd_p1 )
                Cd_p2  = np.interp( self.ls[il], self.stations, self.Cd_p2 )
                Cd_End = np.interp( self.ls[il], self.stations, self.Cd_End)

                # current (relative) velocity over node (no complex numbers bc not function of frequency)
                vrel = np.array(vcur)
                # break out velocity components in each direction relative to member orientation
                vrel_q  = np.sum(vrel*self.q[:] )*self.q[:]
                vrel_p  = vrel-vrel_q 
                vrel_p1 = np.sum(vrel*self.p1[:])*self.p1[:]
                vrel_p2 = np.sum(vrel*self.p2[:])*self.p2[:]
                
                # ----- compute side effects ------------------------

                # member acting area assigned to this node in each direction
                a_i_q  = np.pi*self.ds[il]*self.dls[il]  if circ else  2*(self.ds[il,0]+self.ds[il,0])*self.dls[il]
                a_i_p1 =       self.ds[il]*self.dls[il]  if circ else             self.ds[il,0]      *self.dls[il]
                a_i_p2 =       self.ds[il]*self.dls[il]  if circ else             self.ds[il,1]      *self.dls[il]

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
                    a_i_End = np.abs(np.pi*self.ds[il]*self.drs[il])
                else:
                    a_i_End = np.abs((self.ds[il,0]+self.drs[il,0])*(self.ds[il,1]+self.drs[il,1]) - (self.ds[il,0]-self.drs[il,0])*(self.ds[il,1]-self.drs[il,1]))
                
                Dq_End = 0.5 * rho * a_i_End * Cd_End * np.linalg.norm(vrel_q) * vrel_q

                # ----- sum forces and add to total mean drag load about PRP ------
                D = Dq + Dp1 + Dp2 + Dq_End     # sum drag forces at node in member's local orientation frame

                D_hydro += translateForce3to6DOF(D, self.r[il,:] - r_ref)  # sum as forces and moments about PRP
        return D_hydro

    def computeWaveKinematics(self, zeta, beta, w, depth, k=None, rho=1025, g=9.81):
        '''
        Get wave kinematics (velocity, acceleration, and dynamic pressure) at member nodes given a certain wave condition
        Because it uses the position of the member nodes, this function accounts for phasing due to the FOWT's mean offset position in the array.      
        Parameters
        ----------
        zeta: [nWaves x nw] float array - nWaves: number of wave headings; nw: number of frequencies
            Amplitude of each wave component [m]
        beta: nWaves float array
            Wave headings [rad]
        w: nw float array
            Wave frequencies [rad/s]
        k: nw float array
            Wave numbers [1/m] (related to w by the dispersion relation, but faster to pass it instead of recalculating)
        depth: float
            Water depth [m]
        rho: float
            Water density [kg/m^3]
        g: float
            Gravitational acceleration [m/s^2]        

        Returns
        ----------
        Does not return anything. This function updates self.u, self.ud, and self.pDyn for each node of the member.
        Updated these internal functions is required by other functions.
        '''
        nWaves    = zeta.shape[0]
        nw        = len(w)
        self.u    = np.zeros([nWaves, self.ns, 3, nw], dtype=complex)
        self.ud   = np.zeros([nWaves, self.ns, 3, nw], dtype=complex)
        self.pDyn = np.zeros([nWaves, self.ns,    nw], dtype=complex)

        if k is None:
            k = np.array([waveNumber(w, self.depth) for w in self.w])

        for il in range(self.ns):
            if self.r[il,2] < 0:
                for ih in range(nWaves):
                    self.u[ih,il,:,:], self.ud[ih,il,:,:], self.pDyn[ih,il,:] = getWaveKin(zeta[ih,:], beta[ih], w, k, depth, self.r[il,:], nw, rho=rho, g=g)


    def calcHydroExcitation(self, zeta, beta, w, depth, k=None, rho=1025, g=9.81, r_ref=np.zeros(3)):
        '''
        Compute strip-theory wave excitation force (inertial part of Morison's equation).
        Need to call computeWaveKinematics() first to get the wave kinematics at each node.        
        Because computeWaveKinematics() uses the position of the member nodes, this function accounts for phasing due to the FOWT's mean offset position in the array.

        Parameters
        ----------
        r_ref: size-3 vector
            Reference point coordinates to compute hydrodynamic loads [m].
        
        Returns
        ----------
        self.F_hydro_iner: [nWaves x 6 x nw] complex array. nWaves: number of wave headings; 6: number of dofs; nw: number of frequencies
        '''
        self.computeWaveKinematics(zeta, beta, w, depth, k=k, rho=rho, g=g)

        nWaves, _, _, nw = self.ud.shape
        self.F_hydro_iner = np.zeros([nWaves, 6, nw],dtype=complex) # inertia excitation force/moment complex amplitudes vector [N, N-m]        

        # loop through each node of the member
        for il in range(self.ns):

            # only process hydrodynamics if this node is submerged
            if self.r[il,2] < 0:
                if self.potMod == False:
                    # calculate the linear excitation forces on this node for each wave heading and frequency
                    for ih in range(nWaves):
                        for i in range(nw):
                            if self.MCF:
                                Imat = self.Imat_MCF[il,:,:, i]
                            else:
                                Imat = self.Imat[il,:,:]
                            F_exc_iner_temp = np.matmul(Imat, self.ud[ih,il,:,i]) + self.pDyn[ih,il,i]*self.a_i[il]*self.q 
                            
                            # add the excitation complex amplitude for this heading and frequency to the global excitation vector
                            self.F_hydro_iner[ih,:,i] += translateForce3to6DOF(F_exc_iner_temp, self.r[il,:] - r_ref) # (about PRP)
        return self.F_hydro_iner


    def calcHydroLinearization(self, w, ih=0, Xi=np.zeros(6), rho=1025, r_ref=np.zeros(3)):
        '''To be used within the FOWT's dynamics solve iteration method. This calculates the
        amplitude-dependent linearized coefficients, including the system linearized drag damping matrix, 
        of this member.
        
        Considers only one sea state which is specified by the index ih.        
        
        Parameters
        ----------
        w: nw float array
            Wave frequencies [rad/s]
        ih: int
            Index of the wave heading to consider for the drag linearization
        Xi : size-6 complex array
            response of the FOWT that this member is attached to - displacement and rotation complex amplitudes [m, rad]
            TODO: Change this to be the response of the member itself
        rho: float
            Water density [kg/m^3]
        r_ref: size-3 vector
            Reference point coordinates for outputs [m].
        
        Returns
        ----------
        B_hydro_drag: 6x6 array
            Hydrodynamic damping matrix from linearized viscous drag [N-s/m, N-s, N-s-m]
        F_hydro_drag: 6 x nw complex array
            Excitation force/moment complex amplitude vector [N, N-m]

        This function also updates self.Bmat (size nNodes x 3 x 3) and self.F_exc_drag (size nNodes x 3 x nw)
        '''

        circ = self.shape=='circular'  # convenience boolian for circular vs. rectangular cross sections
        nw = len(w)
        B_hydro_drag = np.zeros([6,6])             # hydrodynamic damping matrix (just linearized viscous drag for now) [N-s/m, N-s, N-s-m]
        F_hydro_drag = np.zeros([6, nw],dtype=complex) # excitation force/moment complex amplitudes vector [N, N-m]

        # loop through each node of the member
        for il in range(self.ns):

            # get node complex velocity amplitude based on platform motion's and relative position from PRP
            # node displacement, velocity, and acceleration (each [3 x nw])
            drnode, vnode, anode = getKinematics(self.r[il,:] - r_ref, Xi, w)

            # only process hydrodynamics if this node is submerged
            if self.r[il,2] < 0:

                # interpolate coefficients for the current strip
                Cd_q   = np.interp( self.ls[il], self.stations, self.Cd_q  )
                Cd_p1  = np.interp( self.ls[il], self.stations, self.Cd_p1 )
                Cd_p2  = np.interp( self.ls[il], self.stations, self.Cd_p2 )
                Cd_End = np.interp( self.ls[il], self.stations, self.Cd_End)


                # ----- compute side effects ------------------------

                # member acting area assigned to this node in each direction
                a_i_q  = np.pi*self.ds[il]*self.dls[il]  if circ else  2*(self.ds[il,0]+self.ds[il,0])*self.dls[il]
                a_i_p1 =       self.ds[il]*self.dls[il]  if circ else             self.ds[il,0]       *self.dls[il]
                a_i_p2 =       self.ds[il]*self.dls[il]  if circ else             self.ds[il,1]       *self.dls[il]

                # water relative velocity over node (complex amplitude spectrum)  [3 x nw]
                vrel = self.u[ih,il,:] - vnode

                # break out velocity components in each direction relative to member orientation [nw]
                vrel_q  = np.sum(vrel*self.q[ :,None], axis=0)*self.q[ :,None]     # (the ,None is for broadcasting q across all frequencies in vrel)
                vrel_p  = vrel - vrel_q
                vrel_p1 = np.sum(vrel*self.p1[:,None], axis=0)*self.p1[:,None]
                vrel_p2 = np.sum(vrel*self.p2[:,None], axis=0)*self.p2[:,None]
                
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
                Bmat_sides = Bprime_q*self.qMat + Bprime_p1*self.p1Mat + Bprime_p2*self.p2Mat 


                # ----- add end/axial effects for added mass, drag, and excitation including dynamic pressure ------
                # note : v_a and a_i work out to zero for non-tapered sections or non-end sections

                # end/axial area (removing sign for use as drag)
                if circ:
                    a_i = np.abs(np.pi*self.ds[il]*self.drs[il])
                else:
                    a_i = np.abs((self.ds[il,0]+self.drs[il,0])*(self.ds[il,1]+self.drs[il,1]) - (self.ds[il,0]-self.drs[il,0])*(self.ds[il,1]-self.drs[il,1]))

                Bprime_End = np.sqrt(8/np.pi)*vRMS_q*0.5*rho*a_i*Cd_End

                # form damping matrix for the node based on linearized drag coefficients
                Bmat_end = Bprime_End*self.qMat                                       #


                # ----- sum up side and end damping matrices ------
                
                self.Bmat[il,:,:] = Bmat_sides + Bmat_end   # store in Member object to be called later to get drag excitation for each wave heading

                B_hydro_drag += translateMatrix3to6DOF(self.Bmat[il,:,:], self.r[il,:] - r_ref)   # add to global damping matrix for Morison members


                # ----- calculate wave drag excitation (this may be recalculated later) -----

                for i in range(nw):

                    self.F_exc_drag[il,:,i] = np.matmul(self.Bmat[il,:,:], self.u[ih,il,:,i])   # get local 3d drag excitation force complex amplitude for each frequency [3 x nw]

                    F_hydro_drag[:,i] += translateForce3to6DOF(self.F_exc_drag[il,:,i], self.r[il,:] - r_ref)   # add to global excitation vector (frequency dependent)
        
        return B_hydro_drag, F_hydro_drag

    def calcDragExcitation(self, ih, r_ref=np.zeros(3)):
        nw = self.u.shape[3]
        F_hydro_drag = np.zeros([6, nw], dtype=complex) # excitation force/moment complex amplitudes vector [N, N-m]
        for il in range(self.ns):  # loop through each node of the member
            if self.r[il,2] < 0:   # only process hydrodynamics if this node is submerged
                for i in range(nw):                    
                    # get local 3d drag excitation force complex amplitude for each frequency [3 x nw]
                    self.F_exc_drag[il,:,i] = np.matmul(self.Bmat[il,:,:], self.u[ih,il,:,i])   
                    
                    # add to global excitation vector (frequency dependent)
                    F_hydro_drag[:,i] += translateForce3to6DOF(self.F_exc_drag[il,:,i], self.r[il,:] - r_ref)
        return F_hydro_drag


    def getSectionProperties(self, station):
        '''Get member cross sectional area and moments of inertia at a user-
        specified location along the member.'''
        
        A = 0
        I = 0        
        
        return A, I

    def plot(self, ax, r_ptfm=[0,0,0], R_ptfm=[], color='k', nodes=0, 
             station_plot=[], plot2d=False, Xuvec=[1,0,0], Yuvec=[0,0,1], zorder=2):
        '''Draws the member on the passed axes, and optional platform offset and rotation matrix
        
        Parameters
        ----------
        
        plot2d: bool
            If true, produces a 2d plot on the axes defined by Xuvec and Yuvec. 
            Otherwise produces a 3d plot (default).
        
        '''
        
        # support self color option
        if color == 'self':
            color = self.color  # attempt to allow custom colors
        
        
        # --- get coordinates of member edges in member reference frame -------------------

        if not station_plot:
            m = np.arange(0, len(self.stations), 1)
        else:
            m = station_plot

        nm = len(m)

        # lists to be filled with coordinates for plotting
        X = []
        Y = []
        Z = []

        if self.shape=="circular":   # circular member cross section
            n = 12                                          # number of sides for a circle
            for i in range(n+1):
                x = np.cos(float(i)/float(n)*2.0*np.pi)    # x coordinates of a unit circle
                y = np.sin(float(i)/float(n)*2.0*np.pi)    # y

                for j in m:
                    X.append(0.5*self.d[j]*x)
                    Y.append(0.5*self.d[j]*y)
                    Z.append(self.stations[j])

            coords = np.vstack([X, Y, Z])

        elif self.shape=="rectangular":    # rectangular member cross section
            n=4
            for x,y in zip([1,-1,-1,1,1], [1,1,-1,-1,1]):

                for j in m:
                    X.append(0.5*self.sl[j,1]*x)
                    Y.append(0.5*self.sl[j,0]*y)
                    Z.append(self.stations[j])

            coords = np.vstack([X, Y, Z])


        # ----- move to global frame ------------------------------
        
        # Note: the below transformations can probably be replaced by using the new member.setPosition function before calling this.
        
        newcoords = np.matmul(self.R, coords)          # relative orientation in platform

        newcoords = newcoords + self.rA[:,None]        # shift to end A location, still relative to platform
        
        if len(R_ptfm) > 0:
            newcoords = np.matmul(R_ptfm, newcoords)   # account for offset platform orientation

        # apply platform translational offset
        Xs = newcoords[0,:] + r_ptfm[0]
        Ys = newcoords[1,:] + r_ptfm[1]
        Zs = newcoords[2,:] + r_ptfm[2]
        
        # plot on the provided axes
        linebit = []  # make empty list to hold plotted lines, however many there are
        
        if plot2d:  # new 2d plotting option
                
            # apply any 3D to 2D transformation here to provide desired viewing angle
            Xs2d = Xs*Xuvec[0] + Ys*Xuvec[1] + Zs*Xuvec[2] 
            Ys2d = Xs*Yuvec[0] + Ys*Yuvec[1] + Zs*Yuvec[2] 
            
            for i in range(n): 
                linebit.append(ax.plot(Xs2d[nm*i:nm*i+nm],Ys2d[nm*i:nm*i+nm], color=color, lw=0.5, zorder=zorder))  # side edges
            
            for j in range(nm):
                linebit.append(ax.plot(Xs2d[j::nm], Ys2d[j::nm], color=color, lw=0.5, zorder=zorder))  # station rings
        
        else:  # normal 3d case
            
            for i in range(n): 
                linebit.append(ax.plot(Xs[nm*i:nm*i+nm],Ys[nm*i:nm*i+nm],Zs[nm*i:nm*i+nm], color=color, lw=0.5, zorder=zorder))  # side edges
            
            for j in range(nm):
                linebit.append(ax.plot(Xs[j::nm], Ys[j::nm], Zs[j::nm], color=color, lw=0.5, zorder=zorder))  # station rings
            
            # plot nodes if asked
            if nodes > 0:
                ax.scatter(self.r[:,0], self.r[:,1], self.r[:,2])
        
        return linebit


