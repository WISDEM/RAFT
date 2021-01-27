# 2020-05-03: This is a start at a frequency-domain floating support structure model for WEIS-WISDEM.
#             Based heavily on the GA-based optimization framework from Hall 2013

# 2020-05-23: Starting as a script that will evaluate a design based on properties/data provided in separate files.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from meshBEMforRAFT import memberMesh

import sys
sys.path.insert(1, '../../MoorPy')
import MoorPy as mp
#import F6T1RNA as structural    # import turbine structural model functions

# reload the libraries each time in case we make any changes
import importlib
mp   = importlib.reload(mp)



class Env:
    '''This could be a simple environmental parameters class <<< needed most immediately to pass rho and g info.'''
    def __init__(self):
        self.rho = 1025.0
        self.g = 9.81
        self.Hs = 1.0
        self.Tp = 10.0
        self.V = 10.0
        self.beta = 0.0


## This class represents linear (for now cylindrical and rectangular) components in the substructure. 
#  It is meant to correspond to Member objects in the WEIS substructure ontology, but containing only 
#  the properties relevant for the Level 1 frequency-domain model, as well as additional data strucutres
#  used during the model's operation.
class Member:

    def __init__(self, mi, nw, BEM=[]):
        '''Initialize a Member. For now, this function accepts a space-delimited string with all member properties.
        
        PARAMETERS
        ----------
        
        min : dict
            Dictionary containing the member description data structure
        nw : int
            Number of frequencies in the analysis - used for initializing.
        heading : float, optional
            Rotation to apply to the coordinates when setting up the member - used for circular patterns of members.
        
        '''
    
        # note: haven't decided yet how to lump masses and handle segments
        
        self.id    = int(1)                                          # set the ID value of the member
        self.name  = str(mi['name'])
        self.type  = int(mi['type'])                                 # set the type of the member (for now, just arbitrary numbers: 0,1,2, etc.)
        
        self.rA = np.array(mi['rA'], dtype=np.double)                # [x,y,z] coordinates of lower node [m]
        self.rB = np.array(mi['rB'], dtype=np.double)                # [x,y,z] coordinates of upper node [m]
        
        # put heading rotation capability here <<<<
        
        shape      = str(mi['shape'])                                # the shape of the cross section of the member as a string (the first letter should be c or r)
        
        rAB = self.rB-self.rA                                        # The relative coordinates of upper node from lower node [m]
        self.l = np.linalg.norm(rAB)                                 # member length [m]
        
        
        self.potMod = 1  # hard coding BEM analysis enabled for now <<<< need to move this to the member YAML input instead <<<
        
        
        # station positions
        n = len(mi['stations'])                                     # number of stations
        if n < 2:  
            raise ValueError("At least two stations entries must be provided")
            
        A = np.array(mi['stations'], dtype=float)    
        self.stations = (A - A[0])/(A[-1] - A[0])*self.l             # calculate relative station positions from 0 to 1
        
        
        # helper function to deal with scalar vs list inputs that should be size n
        def handleScalars(input):
            if np.isscalar(input): 
                return np.zeros(n) + float(input)    # if a scalar is provided, tile it out to the length of n
            else:
                return np.array(input, dtype=float)
            # at some point should upgrade this to take dict and field name to provide default values if it's not provided in the dict
        
        
        # shapes
        if shape[0].lower() == 'c':
            self.shape = 'circular'
            self.d = handleScalars(mi['d'])                          # diameter of member nodes [m]  <<< should maybe check length of all of these
            
            self.gamma = 0                                           # twist angle about the member's z-axis [degrees] (don't need this for a circular member)
        
        elif shape[0].lower() == 'r':   # <<< this case not checked yet since update <<<
            self.shape = 'rectangular'
            
            if np.isscalar(mi['d']): 
                self.sl = np.zeros([n,2]) + float(mi['d'])          # handle if a scaler is provided
            
            elif np.isscalar(mi['d'][0]) and len(mi['d'])==2: 
                self.sl = np.tile(np.array(mi['d'], dtype=float), [n,1])  # handle if a single set of side lengths is provided (assume constant cross section)
            
            else:
                self.sl = np.array(mi['d'], dtype=float)             # array of side lengths of nodes along member [m]
            
            self.gamma = float(mi['gamma'])                          # twist angle about the member's z-axis [degrees] (if gamma=90, then the side lengths are flipped)
        
        else:
            raise ValueError('The only allowable shape strings are circular and rectangular')
                    
        
        self.t         = handleScalars(mi['t'])                     # shell thickness of upper node [m]
        
        self.l_fill    = handleScalars(mi['l_fill'])                # length of member (from end A to B) filled with ballast [m]
        self.rho_fill  = handleScalars(mi['rho_fill'])              # density of ballast in member [kg/m^3]
                                             
        self.rho_shell = handleScalars(mi['rho_shell'])             # shell mass density [kg/m^3]
        
        
        # initialize member orientation variables
        self.q = rAB/self.l                                         # member axial unit vector
        self.p1 = np.zeros(3)                                       # member transverse unit vectors (to be filled in later)
        self.p2 = np.zeros(3)                                       # member transverse unit vectors
        self.R = np.eye(3)                                          # rotation matrix from global x,y,z to member q,p1,p2
        
        
        # store end cap and bulkhead info
        self.cap_stations = np.array(mi['cap_stations'], dtype=float) 
        self.cap_t        = np.array(mi['cap_t'       ], dtype=float)
        self.cap_d_in     = np.array(mi['cap_d_in'    ], dtype=float)
        
        
        # Drag coefficients
        self.Cd_q   = handleScalars(0          )                    # axial drag coefficient
        self.Cd_p1  = handleScalars(mi['Cd'   ])                    # transverse1 drag coefficient
        self.Cd_p2  = handleScalars(mi['Cd'   ])                    # transverse2 drag coefficient
        self.Cd_End = handleScalars(mi['CdEnd'])                    # end drag coefficient        
        # Added mass coefficients
        self.Ca_q   = handleScalars(0          )                    # axial added mass coefficient
        self.Ca_p1  = handleScalars(mi['Ca'   ])                    # transverse1 added mass coefficient
        self.Ca_p2  = handleScalars(mi['Ca'   ])                    # transverse2 added mass coefficient
        self.Ca_End = handleScalars(mi['CaEnd'])                    # end added mass coefficient
        
        
        # discretize into strips with a node at the midpoint of each strip (flat surfaces have dl=0)
        dlsMax = 10.0                  # maximum node spacing <<< this should be an optional input at some point <<<
        ls     = [0.0]                 # list of lengths along member axis where a node is located <<< should these be midpoints instead of ends???
        dls    = [0.0]                 # lumped node lengths (end nodes have half the segment length)
        ds     = [0.5*self.d[0]]       # mean diameter of each strip
        drs    = [0.5*self.d[0]]       # change in radius over each strip (from node i-1 to node i)
        
        # below is for circular shapes only - rectangular case is not handled yet
        if self.shape != 'circular': raise ValueError("Only circular shapes implemented here so far!")
        
        for i in range(1,n):
            
            lstrip = self.stations[i]-self.stations[i-1]             # the axial length of the strip
            
            if lstrip > 0.0:
                ns= int(np.ceil( (lstrip) / dlsMax ))
                dlstrip = lstrip/ns
                m   = 0.5*(self.d[i] - self.d[i-1])/dlstrip          # taper ratio
                ls  += [self.stations[i-1] + dlstrip*(0.5+j) for j in range(ns)] # add node locations
                dls += [dlstrip]*ns                
                ds  += [self.d[i-1] + dlstrip*m*(0.5+j) for j in range(ns)]                
                drs += [dlstrip*m]*ns
                
            elif lstrip == 0.0:                                      # flat plate case (ends, and any flat transitions)
                ns = 1
                dlstrip = 0
                ls  += [self.stations[i-1]]                          # add node location
                dls += [dlstrip]
                ds  += [0.5*(self.d[i-1] + self.d[i])]               # set diameter as midpoint diameter
                drs += [0.5*(self.d[i] - self.d[i-1])]
            
        self.ns  = len(ls)                                           # number of hydrodynamic strip theory nodes per member
        self.ls  = np.array(ls, dtype=float)                          # node locations along member axis
        #self.dl = 0.5*(np.diff([0.]+lh) + np.diff(lh+[lh[-1]]))      
        self.dls = np.array(dls)
        self.ds  = np.array(ds)
        self.drs = np.array(drs)
        self.mh  = np.array(m)
        
        self.r   = np.zeros([self.ns,3])                             # undisplaced node positions along member  [m]
        
        for i in range(self.ns):
            self.r[i,:] = self.rA + (ls[i]/self.l)*rAB               # locations of hydrodynamics nodes
        
        #self.slh[i,0] = np.interp(lh[i], self.stations, self.sl1)
        #self.slh[i,1] = np.interp(lh[i], self.stations, self.sl2)
                
        
        # complex frequency-dependent amplitudes of quantities at each node along member (to be filled in later)
        self.dr        = np.zeros([self.ns,3,nw], dtype=complex)            # displacement
        self.v         = np.zeros([self.ns,3,nw], dtype=complex)            # velocity
        self.a         = np.zeros([self.ns,3,nw], dtype=complex)            # acceleration
        self.u         = np.zeros([self.ns,3,nw], dtype=complex)            # wave velocity
        self.ud        = np.zeros([self.ns,3,nw], dtype=complex)            # wave acceleration
        self.pDyn      = np.zeros([self.ns,  nw], dtype=complex)            # dynamic pressure
        self.F_exc_iner= np.zeros([self.ns,3,nw], dtype=complex)            # wave excitation from inertia (Froude-Krylov)
        self.F_exc_drag= np.zeros([self.ns,3,nw], dtype=complex)            # wave excitation from linearized drag
    
    
        
    def calcOrientation(self):
        '''Calculates member direction vectors q, p1, and p2 as well as member orientation matrix R 
        based on the end positions and twist angle gamma.'''
        
        rAB = self.rB-self.rA                                       # displacement vector from end A to end B [m]
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
        
        
        p1 = np.matmul( R, [1,0,0] )               # unit vector that is perpendicular to the 'beta' plane if gamma is zero
        p2 = np.cross( q, p1 )                     # unit vector orthogonal to both p1 and q
        
        self.R  = R
        self.q  = q
        self.p1 = p1
        self.p2 = p2
        
        # matrices of vector multiplied by vector transposed, used in computing force components
        self.qMat  = VecVecTrans(self.q)
        self.p1Mat = VecVecTrans(self.p1)
        self.p2Mat = VecVecTrans(self.p2)

        
        return q, p1, p2  # also return the unit vectors for convenience
    
    
    
    def getInertia(self):
        '''Calculates member inertia properties: mass, center of mass, moments of inertia.
        Assumes that the members are continuous and symmetrical (i.e. no weird shapes)'''
        
        # Moment of Inertia Helper Functions -----------------------
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
                
                    x2 = (1/12)*p* ( (Lb-La)**3*H*(Wb/5 + Wa/20) + (Lb-La)**2*La*H(3*Wb/4 + Wa/4) + \
                                     (Lb-La)*La**2*H*(Wb + Wa/2) + La**3*H*(Wb/2 + Wa/2) )
                        
                    y2 = (1/12)*p* ( (Wb-Wa)**3*H*(Lb/5 + La/20) + (Wb-Wa)**2*Wa*H(3*Lb/4 + La/4) + \
                                     (Wb-Wa)*Wa**2*H*(Lb + La/2) + Wa**3*H*(Lb/2 + La/2) )
                    
                    z2 = p*( Wb*Lb/5 + Wa*Lb/20 + La*Wb/20 + Wa*La*(8/15) )
                    
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
        
        
        
        # ------- get inertial calculations ---------
        n = len(self.stations)                          # set n as the number of stations = number of sub-members minus 1
        
        mass_center = 0                                 # total sum of mass the center of mass of the member [kg-m]
        mshell = 0                                      # total mass of the shell material only of the member [kg]
        mfill = []                                      # list of ballast masses in each submember [kg]
        pfill = []                                      # list of ballast densities in each submember [kg]
        self.M_struc = np.zeros([6,6])                  # member mass/inertia matrix [kg, kg-m, kg-m^2]
        
        # loop through each sub-member
        for i in range(1,n):                            # start at 1 rather than 0 because we're looking at the sections (from station i-1 to i)
            
            # initialize common variables
            rA = self.rA + self.q*self.stations[i-1]    # lower node position of the submember [m]
            l = self.stations[i]-self.stations[i-1]     # length of the submember [m]
            # if the following variables are input as scalars, keep them that way, if they're vectors, take the [i-1]th value
            if np.isscalar(self.rho_shell):     # <<<<<< not sure if this is the best way to handle this
                rho_shell = self.rho_shell              # density of the shell material [kg/m^3]
                l_fill = self.l_fill                    # height of the ballast in the submember [m]
                rho_fill = self.rho_fill                # density of the ballast material [kg/m^3]
            else:
                rho_shell = self.rho_shell[i-1]
                l_fill = self.l_fill[i-1]
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
                
                hc_shell = ((hco*V_outer)-(hci*V_inner))/(V_outer-V_inner)  # center of volume of hollow frustum with shell thickness [m]
                
                dBi_fill = (dBi-dAi)*(l_fill/l) + dAi   # interpolated inner diameter of frustum that ballast is filled to [m] <<<<<<<<<<<< can maybe use the intrp function in getHydrostatics
                v_fill, hc_fill = FrustumVCV(dAi, dBi_fill, l_fill)         # volume and center of volume of solid inner frustum that ballast occupies [m^3] [m]
                m_fill = v_fill*rho_fill                # mass of the ballast in the submember [kg]
                
                mass = m_shell + m_fill                 # total mass of the submember [kg]
                hc = ((hc_fill*m_fill) + (hc_shell*m_shell))/mass       # total center of mass of the submember from the submember's rA location [m]
                
                center = rA + (self.q*hc)               # total center of mass of the member from the PRP [m]
                
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
                
                hc_shell = ((hco*V_outer)-(hci*V_inner))/(V_outer-V_inner)  # center of volume of the hollow frustum with shell thickness [m]
                
                slBi_fill = (slBi-slAi)*(l_fill/l) + slAi   # interpolated side lengths of frustum that ballast is filled to [m]
                v_fill, hc_fill = FrustumVCV(slAi, slBi_fill, l_fill)   # volume and center of volume of inner frustum that ballast occupies [m^3]
                m_fill = v_fill*rho_fill                    # mass of ballast in the submember [kg]
                
                mass = m_shell + m_fill                     # total mass of the submember [kg]
                hc = ((hc_fill*m_fill) + (hc_shell*m_shell))/mass       # total center of mass of the submember from the submember's rA location [m]
                
                center = rA + (self.q*hc)                   # total center of mass of the member from the PRP [m]
                
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

            
            
            # add/append terms 
            mass_center += mass*center                  # total sum of mass the center of mass of the member [kg-m]
            mshell += m_shell                           # total mass of the shell material only of the member [kg]
            mfill.append(m_fill)                        # list of ballast masses in each submember [kg]
            pfill.append(rho_shell)                     # list of ballast densities in each submember [kg]
            
            # create a local submember mass matrix
            Mmat = np.diag([mass, mass, mass, 0, 0, 0]) # submember's mass matrix without MoI tensor
            # create the local submember MoI tensor in the correct directions
            I = np.diag([Ixx, Iyy, Izz])                # MoI matrix about the member's local CG. 0's on off diagonals because of symmetry
            T = self.R.T                                # transformation matrix to unrotate the member's local axes. Transposed because rotating axes.
            I_rot = np.matmul(T.T, np.matmul(I,T))      # MoI about the member's local CG with axes in same direction as global axes. [I'] = [T][I][T]^T -> [T]^T[I'][T] = [I]

            Mmat[3:,3:] = I_rot     # mass and inertia matrix about the submember's CG in unrotated, but translated local frame
            
            # translate this submember's local inertia matrix to the PRP and add it to the total member's M_struc matrix
            self.M_struc += translateMatrix6to6DOF(center, Mmat) # mass matrix of the member about the PRP
            
            
            # end of submember for loop
            
        mass = self.M_struc[0,0]        # total mass of the entire member [kg]
        center = mass_center/mass       # total center of mass of the entire member from the PRP [m]
        
      
        return mass, center, m_shell, mfill, pfill
        
        
    
    
    def getHydrostatics(self, env):
        '''Calculates member hydrostatic properties, namely buoyancy and stiffness matrix'''
        
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
        
            # calculate end locations for this segment only
            rA = self.rA + self.q*self.stations[i-1]
            rB = self.rA + self.q*self.stations[i  ]
                
            # partially submerged case
            if self.rA[2]*self.rB[2] <= 0:    # if member crosses (or touches) water plane
                
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
                
                def intrp(x, xA, xB, yA, yB):  # little basic interpolation function for 2 values rather than a vector
                    return yA + (x-xA)*(yB-yA)/(xB-xA)
                    
                # -------------------- buoyancy and waterplane area properties ------------------------
                
                xWP = intrp(0, rA[2], rB[2], rA[0], rB[0])                     # x coordinate where member axis cross the waterplane [m]
                xWP = intrp(0, rA[2], rB[2], rA[1], rB[1])                     # y coordinate where member axis cross the waterplane [m]
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
                    IyWP = (1/12)*slWP[0]**3*slWP[0]                           # waterplane MoI [m^4] about the member's LOCAL y-axis, not the global y-axis
                    I = np.diag([IxWP, IyWP, 0])                               # area moment of inertia tensor
                    T = self.R.T                                               # the transformation matrix to unrotate the member's local axes
                    I_rot = np.matmul(T.T, np.matmul(I,T))                     # area moment of inertia tensor where MoI axes are now in the same direction as PRP
                    IxWP = I_rot[0,0]
                    IyWP = I_rot[1,1]
                
                LWP = abs(self.rA[2])/cosPhi                   # get length of segment along member axis that is underwater [m]
                
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
                dFz_dz   = -env.rho*env.g*AWP /cosPhi
                
                # note: below calculations are based on untapered case, but 
                # temporarily approximated for taper by using dWP (diameter at water plane crossing) <<< this is rough
                
                # buoyancy force and moment about end A
                Fz = env.rho*env.g* V_UWi
                M  = -env.rho*env.g*pi*( dWP**2/32*(2.0 + tanPhi**2) + 0.5*(rA[2]/cosPhi)**2)*sinPhi  # moment about axis of incline
                Mx = M*dPhi_dThx
                My = M*dPhi_dThy
                
                Fvec[2] += Fz                           # vertical buoyancy force [N]
                Fvec[3] += Mx + Fz*rA[1]                # moment about x axis [N-m]
                Fvec[4] += My - Fz*rA[0]                # moment about y axis [N-m]
                

                # normal approach to hydrostatic stiffness, using this temporarily until above fancier approach is verified
                Cmat[2,2] += -dFz_dz
                Cmat[2,3] += env.rho*env.g*(     -AWP*yWP    )
                Cmat[2,4] += env.rho*env.g*(      AWP*xWP    )
                Cmat[3,2] += env.rho*env.g*(     -AWP*yWP    )
                Cmat[3,3] += env.rho*env.g*(IxWP + AWP*yWP**2 )
                Cmat[3,4] += env.rho*env.g*(      AWP*xWP*yWP)
                Cmat[4,2] += env.rho*env.g*(      AWP*xWP    )
                Cmat[4,3] += env.rho*env.g*(      AWP*xWP*yWP)
                Cmat[4,4] += env.rho*env.g*(IyWP + AWP*xWP**2 )
                
                Cmat[3,3] += env.rho*env.g*V_UWi * r_center[2]
                Cmat[4,4] += env.rho*env.g*V_UWi * r_center[2]
                
                V_UW += V_UWi
                r_centerV += r_center*V_UWi
                
            
            # fully submerged case 
            elif self.r[0,2] <= 0 and self.r[-1,2] <= 0:
                                
                # displaced volume [m^3] and distance along axis from end A to center of buoyancy of member [m]
                if self.shape=='circular':
                    V_UWi, hc = FrustumVCV(self.d[i-1], self.d[i], self.station[i]-self.station[i-1])
                elif self.shape=='rectangular':
                    V_UWi, hc = FrustumVCV(self.sl[i-1], self.sl[i], self.station[i]-self.station[i-1])
                
                r_center = rA + self.q*hc             # absolute coordinates of center of volume of this segment[m]
            
                # buoyancy force (and moment) vector
                Fvec += translateForce3to6DOF( r_center, np.array([0, 0, env.rho*env.g*V_UWi]) ) 
      
                # hydrostatic stiffness matrix (about end A)
                Cmat[3,3] += env.rho*env.g*V_UWi * r_center[2]
                Cmat[4,4] += env.rho*env.g*V_UWi * r_center[2]
                
                V_UW += V_UWi
                r_centerV += r_center*V_UWi
                
            else: # if the members are fully above the surface
                
                pass
            
        
        r_center = r_centerV/V_UW    # calculate overall member center of buoyancy
        
        return Fvec, Cmat, V_UW, r_center, AWP, IWP, xWP, yWP


    def plot(self, ax):
        '''Draws the member on the passed axes'''
        
        # get coordinates of lines along sides relative to end A in member reference frame
        
        if self.shape=="circular":   # circular member cross section
            n = 8                                                     # number of sides for a circle
            X = []
            Y = []
            Z = []
            for i in range(n+1):
                x = np.cos(float(i)/float(n)*2.0*np.pi)    # x coordinates of a unit circle
                y = np.sin(float(i)/float(n)*2.0*np.pi)    # y
                
                X.append(0.5*self.dA*x)   # point on end A
                Y.append(0.5*self.dA*y)
                Z.append(0.0)            
                X.append(0.5*self.dB*x)   # point on end B
                Y.append(0.5*self.dB*y)
                Z.append(self.l)       
                
            coords = np.vstack([X, Y, Z])     
                
        elif self.shape=="rectangular":    # rectangular member cross section
            n=4
            coords = np.array([[ 0.5*self.slA[1], 0.5*self.slA[0], 0.0],      # point on end A
                               [ 0.5*self.slB[1], 0.5*self.slB[0], self.l],   # point on end B           
                               [-0.5*self.slA[1], 0.5*self.slA[0], 0.0],
                               [-0.5*self.slB[1], 0.5*self.slB[0], self.l],
                               [-0.5*self.slA[1],-0.5*self.slA[0], 0.0],
                               [-0.5*self.slB[1],-0.5*self.slB[0], self.l],
                               [ 0.5*self.slA[1],-0.5*self.slA[0], 0.0],
                               [ 0.5*self.slB[1],-0.5*self.slB[0], self.l],
                               [ 0.5*self.slA[1], 0.5*self.slA[0], 0.0],       # (we go full circle here, so five points for the rectangle rather than 4)
                               [ 0.5*self.slB[1], 0.5*self.slB[0], self.l]]).T  # need transposed
        
        
        # rotate into global frame
        newcoords = np.matmul(self.R, coords)
        
        # shift to end A location
        Xs = newcoords[0,:] + self.rA[0]
        Ys = newcoords[1,:] + self.rA[1]
        Zs = newcoords[2,:] + self.rA[2]
        
        # plot on the provided axes
        linebit = []  # make empty list to hold plotted lines, however many there are
        for i in range(n):  #range(int(len(Xs)/2-1)):
            linebit.append(ax.plot(Xs[2*i:2*i+2],Ys[2*i:2*i+2],Zs[2*i:2*i+2]            , color='k'))  # side edges
            linebit.append(ax.plot(Xs[[2*i,2*i+2]],Ys[[2*i,2*i+2]],Zs[[2*i,2*i+2]]      , color='k'))  # end A edges
            linebit.append(ax.plot(Xs[[2*i+1,2*i+3]],Ys[[2*i+1,2*i+3]],Zs[[2*i+1,2*i+3]], color='k'))  # end B edges
    
        return linebit


       
""" FrustumVCV function can calculate volume and CV of both circular and rectangular members, making these old frustum
functions (separate volume and CV calcs for only circular members) close to obsolete. Keeping here just in case, since
I'm still unsure of the best way to organize getInertia

def FrustumV(dA, dB, l):
    '''returns the volume of a frustum, which can be a cylinder, cone, or anything in between'''
    return (np.pi/4)*(1/3)*(dA**2+dB**2+dA*dB)*l

def FrustumCV(dA, dB, l):
    '''returns the height of the center of volume from the lower node of a frustum member'''
    return l*((dA**2 + 2*dA*dB + 3*dB**2)/(4*(dA**2 + dA*dB + dB**2)))  
"""

def FrustumVCV(dA, dB, H, rtn=0):
    '''returns the volume and center of volume of a frustum, which can be a cylinder (box), cone (pyramid), or anything in between
    Source: https://mathworld.wolfram.com/PyramidalFrustum.html '''
    
    if np.sum(dA)==0 and np.sum(dB)==0:
        V = 0
        hc = 0
    else:
        if np.isscalar(dA) and np.isscalar(dB): # if the inputs are scalar, meaning that it's just a diameter
            A1 = (np.pi/4)*dA**2
            A2 = (np.pi/4)*dB**2
            Amid = (np.pi/4)*dA*dB
        elif len(dA)==2 and len(dB)==2: # if the inputs are of length 2, meaning if it's two side lengths per node
            A1 = dA[0]*dA[1]
            A2 = dB[0]*dB[1]
            Amid = np.sqrt(A1*A2)
        else:
            raise ValueError('Input types not accepted')
        
        V = (A1 + A2 + Amid) * H/3
        hc = ((A1 + 2*Amid + 3*A2)/(A1 + Amid + A2)) * H/4
    
    if rtn==0:
        return V, hc
    elif rtn==1:
        return V
    elif rtn==2:
        return hc
    

def getVelocity(r, Xi, ws):
    '''Get node complex velocity spectrum based on platform motion's and relative position from PRP'''
    
    nw = len(ws)
        
    dr = np.zeros([3,nw], dtype=complex) # node displacement complex amplitudes
    v  = np.zeros([3,nw], dtype=complex) # velocity
    a  = np.zeros([3,nw], dtype=complex) # acceleration
        
    
    for i in range(nw):
        dr[:,i] = Xi[:3,i] + SmallRotate(r, Xi[3:,i])
        v[ :,i] = 1j*ws[i]*dr[:,i]
        a[ :,i] = 1j*ws[i]*v[ :,i]
    

    return dr, v, a # node dispalcement, velocity, acceleration (Each  [3 x nw])
        
    
## Get wave velocity and acceleration complex amplitudes based on wave spectrum at a given location
def getWaveKin(zeta0, w, k, h, r, nw, rho=1025.0, g=9.91):

    # inputs: wave elevation fft, wave freqs, wave numbers, depth, point position

    beta = 0  # no wave heading for now

    zeta = np.zeros(nw , dtype=complex ) # local wave elevation
    u  = np.zeros([3,nw], dtype=complex) # local wave kinematics velocity
    ud = np.zeros([3,nw], dtype=complex) # local wave kinematics acceleration
    pDyn = np.zeros(nw , dtype=complex ) # local dynamic pressure
    
    for i in range(nw):
    
        # .............................. wave elevation ...............................
        zeta[i] = zeta0[i]* np.exp( -1j*(k[i]*(np.cos(beta)*r[0] + np.sin(beta)*r[1])))  # shift each zetaC to account for location
        
        
        # ...................... wave velocities and accelerations ............................        
        z = r[2]
        
        # only process wave kinematics if this node is submerged
        if z < 0:
            
            # Calculate SINH( k*( z + h ) )/SINH( k*h ) and COSH( k*( z + h ) )/SINH( k*h ) and COSH( k*( z + h ) )/COSH( k*h )
            # given the wave number, k, water depth, h, and elevation z, as inputs.
            if (    k[i]   == 0.0  ):                   # When .TRUE., the shallow water formulation is ill-conditioned; thus, the known value of unity is returned.
                SINHNumOvrSIHNDen = 1.0
                breakpoint()
                COSHNumOvrSIHNDen = 99999.0
                COSHNumOvrCOSHDen = 99999.0   # <<< check
            elif ( k[i]*h >  89.4 ):                # When .TRUE., the shallow water formulation will trigger a floating point overflow error; however, with h > 14.23*wavelength (since k = 2*Pi/wavelength) we can use the numerically-stable deep water formulation instead.
                SINHNumOvrSIHNDen = np.exp( k[i]*z )
                COSHNumOvrSIHNDen = np.exp( k[i]*z )
                COSHNumOvrCOSHDen = np.exp( k[i]*z ) + np.exp(-k[i]*(z + 2.0*h))
            else:                                    # 0 < k*h <= 89.4; use the shallow water formulation.
                SINHNumOvrSIHNDen = np.sinh( k[i]*( z + h ) )/np.sinh( k[i]*h );
                COSHNumOvrSIHNDen = np.cosh( k[i]*( z + h ) )/np.sinh( k[i]*h );
                COSHNumOvrCOSHDen = np.real( np.cosh(k[i]*(z+h)) )/np.cosh(k[i]*h)   # <<< check
            
            # Fourier transform of wave velocities 
            u[0,i] =    w[i]* zeta[i]*COSHNumOvrSIHNDen *np.cos(beta) 
            u[1,i] =    w[i]* zeta[i]*COSHNumOvrSIHNDen *np.sin(beta)
            u[2,i] = 1j*w[i]* zeta[i]*SINHNumOvrSIHNDen

            # Fourier transform of wave accelerations                   
            ud[:,i] = 1j*w[i]*u[:,i]
            
            # Fourier transform of dynamic pressure
            pDyn[i] = rho*g* zeta[i] * COSHNumOvrCOSHDen 
            
        
    return u, ud, pDyn



# calculate wave number based on wave frequency in rad/s and depth
def waveNumber(omega, h, e=0.001):
    
    g = 9.81
    # omega - angular frequency of waves
    
    k = 0                           #initialize  k outside of loop
                                    #error tolerance
    k1 = omega*omega/g                  #deep water approx of wave number
    k2 = omega*omega/(np.tanh(k1*h)*g)      #general formula for wave number
    while np.abs(k2 - k1)/k1 > e:           #repeate until converges
        k1 = k2
        k2 = omega*omega/(np.tanh(k1*h)*g)
    
    k = k2
    
    return k
    

# translate point at location r based on three small angles in th
def SmallRotate(r, th):
    
    rt = np.zeros(3, dtype=complex)    # translated point
    
    rt[0] =              th[2]*r[1] - th[1]*r[2]
    rt[0] = th[2]*r[0]              - th[0]*r[2]
    rt[0] = th[1]*r[0] - th[0]*r[1]
    # shousner: @matthall, should these be rt[0], rt[1], rt[2] ?
    return rt
    
    
# given a size-3 vector, vec, return the matrix from the multiplication vec * vec.transpose
def VecVecTrans(vec):

    vvt = np.zeros([3,3])
    
    for i in range(3):
        for j in range(3):
            vvt[i,j]= vec[i]*vec[j]
            
    return vvt
    

# produce alternator matrix
def getH(r):

    H = np.zeros([3,3])
    H[0,1] =  r[2];
    H[1,0] = -r[2];
    H[0,2] = -r[1];
    H[2,0] =  r[1];
    H[1,2] =  r[0];
    H[2,1] = -r[0];
    
    return H



def translateForce3to6DOF(r, Fin):
    '''Takes in a position vector and a force vector (applied at the positon), and calculates 
    the resulting 6-DOF force and moment vector.    
    
    :param array r: x,y,z coordinates at which force is acting [m]
    :param array Fin: x,y,z components of force [N]
    :return: the resulting force and moment vector
    :rtype: array
    '''
    Fout = np.zeros(6, dtype=Fin.dtype) # initialize output vector as same dtype as input vector (to support both real and complex inputs)
    
    Fout[:3] = Fin
    
    Fout[3:] = np.cross(r, Fin)
    
    return Fout

    
    
# translate mass matrix to make 6DOF mass-inertia matrix
def translateMatrix3to6DOF(r, Min):
    '''Transforms a 3x3 matrix to be about a translated reference point, resulting in a 6x6 matrix.'''
    
    # sub-matrix definitions are accordint to  | m    J |
    #                                          | J^T  I |

    # note that the J term and I terms are zero in this case because the input is just a mass matrix (assumed to be about CG)

    H = getH(r)     # "anti-symmetric tensor components" from Sadeghi and Incecik
    
    Mout = np.zeros([6,6]) #, dtype=complex)
    
    # mass matrix  [m'] = [m]
    Mout[:3,:3] = Min
        
    # product of inertia matrix  [J'] = [m][H] + [J]  
    Mout[:3,3:] = np.matmul(Min, H)
    Mout[3:,:3] = Mout[:3,3:].T
    
    # moment of inertia matrix  [I'] = [H][m][H]^T + [J]^T [H] + [H]^T [J] + [I]
    
    Mout[3:,3:] = np.matmul(np.matmul(H,Min), H.T) 
    
    return Mout


def translateMatrix6to6DOF(r, Min):
    '''Transforms a 6x6 matrix to be about a translated reference point.'''
    
    # sub-matrix definitions are accordint to  | m    J |
    #                                          | J^T  I |

    H = getH(r)     # "anti-symmetric tensor components" from Sadeghi and Incecik
    
    Mout = np.zeros([6,6]) #, dtype=complex)
    
    # mass matrix  [m'] = [m]
    Mout[:3,:3] = Min[:3,:3]
        
    # product of inertia matrix  [J'] = [m][H] + [J]
    Mout[:3,3:] = np.matmul(Min[:3,:3], H) + Min[:3,3:]
    Mout[3:,:3] = Mout[:3,3:].T
    
    # moment of inertia matrix  [I'] = [H][m][H]^T + [J]^T [H] + [H]^T [J] + [I]
    Mout[3:,3:] = np.matmul(np.matmul(H,Min[:3,:3]), H.T) + np.matmul(Min[3:,:3], H) + np.matmul(H.T, Min[:3,3:]) + Min[3:,3:]
    
    return Mout
    

def JONSWAP(ws, Hs, Tp, Gamma=1.0):
    '''Returns the JONSWAP wave spectrum for the given frequencies and parameters.
    
    Parameters
    ----------    
    ws : float | array
        wave frequencies to compute spectrum at (scalar or 1-D list/array) [rad/s]
    Hs : float
        significant wave height of spectrum [m]
    Tp : float
        peak spectral period [s]
    Gamma : float
        wave peak shape parameter []. The default value of 1.0 gives the Pierson-Moskowitz spectrum.
        
    Returns
    -------
    S : array
        wave power spectral density array corresponding to frequencies in ws [m^2/Hz]
        
    
    This function calculates and returns the one-sided power spectral density spectrum 
    at the frequency/frequencies (ws) based on the provided significant wave height,
    peak period, and (optionally) shape parameter gamma.
    
    This formula for the JONSWAP spectrum is adapted from FAST v7 and based
    on what's documented in IEC 61400-3.
    '''
    
    # handle both scalar and array inputs
    if isinstance(ws, (list, tuple, np.ndarray)):
        ws = np.array(ws)
    else:
        ws = np.array([ws])

    # initialize output array
    S = np.zeros(len(ws))    
 
            
    # the calculations
    f        = 0.5/np.pi * ws                         # wave frequencies in Hz
    fpOvrf4  = pow((Tp*f), -4.0)                      # a common term, (fp/f)^4 = (Tp*f)^(-4)
    C        = 1.0 - ( 0.287*np.log(Gamma) )          # normalizing factor
    Sigma = 0.07*(f <= 1.0/Tp) + 0.09*(f > 1.0/Tp)    # scaling factor
    
    Alpha = np.exp( -0.5*((f*Tp - 1.0)/Sigma)**2 )

    return  0.5/np.pi *C* 0.3125*Hs*Hs*fpOvrf4/f *np.exp( -1.25*fpOvrf4 )* Gamma**Alpha
    
    
def printMat(mat):
    '''Print a matrix'''
    for i in range(mat.shape[0]):
        print( "\t".join(["{:+8.3e}"]*mat.shape[1]).format( *mat[i,:] ))
        
def printVec(vec):
    '''Print a vector'''
    print( "\t".join(["{:+8.3e}"]*len(vec)).format( *vec ))
    



class Model():


    def __init__(self, design, BEM=None, nTurbines=1, w=[], depth=300):
        '''
        Empty frequency domain model initialization function
        
        design : dict
            Dictionary of all the design info from turbine to platform to moorings
        nTurbines 
            could in future be used to set up any number of identical turbines
        '''
        
        self.fowtList = []
        self.coords = []
        
        self.nDOF = 0  # number of DOFs in system
        
        
    # ----- process turbine information -----------------------------------------
    # No processing actually needed yet - we pass the dictionary directly to RAFT.
    
    
    # ----- process platform information ----------------------------------------
    # No processing actually needed yet - we pass the dictionary directly to RAFT.
    
        
        # ----- process mooring information ----------------------------------------------
          
        self.ms = mp.System()
        
        self.ms.parseYAML(design['mooring'])
          
          
            
        self.depth = depth
        
        # If you're modeling OC3 spar, for example, import the manual yaw stiffness needed by the bridle config
        if 'yaw stiffness' in design['turbine']:
            self.yawstiff = design['turbine']['yaw stiffness']
        else:
            self.yawstiff = 0
        
        # analysis frequency array
        if len(w)==0:
            w = np.arange(.05, 3, 0.05)  # angular frequencies tp analyze (rad/s)
        
        self.w = np.array(w)
        self.nw = len(w)  # number of frequencies
        
        self.k = np.zeros(self.nw)  # wave number
        for i in range(self.nw):
            self.k[i] = waveNumber(self.w[i], self.depth)
        
        # set up the FOWT here  <<< only set for 1 FOWT for now <<<
        self.fowtList.append(FOWT(design, w=self.w, mpb=self.ms.BodyList[0], depth=depth))
        self.coords.append([0.0,0.0])
        self.nDOF += 6
        
        self.ms.BodyList[0].type = -1  # need to make sure it's set to a coupled type
        
        self.ms.initialize()  # reinitialize the mooring system to ensure all things are tallied properly etc.
        
        
        
    def addFOWT(self, fowt, xy0=[0,0]):
        '''adds an already set up FOWT to the frequency domain model solver.'''
    
        self.fowtList.append(fowt)
        self.coords.append(xy0)
        self.nDOF += 6
        
        # would potentially need to add a mooring system body for it too <<<
        
    
    def setEnv(self, Hs=8, Tp=12, V=10, beta=0, Fthrust=0):
    
        self.env = Env()
        self.env.Hs   = Hs   
        self.env.Tp   = Tp   
        self.env.V    = V    
        self.env.beta = beta
        self.Fthrust = Fthrust
    
        for fowt in self.fowtList:
            fowt.setEnv(Hs=Hs, Tp=Tp, V=V, beta=beta, Fthrust=Fthrust)
    
    
    def calcSystemProps(self):
        '''This gets the various static/constant calculations of each FOWT done.'''
        
        for fowt in self.fowtList:
            fowt.calcBEM()
            fowt.calcStatics()
            fowt.calcHydroConstants()
            #fowt.calcDynamicConstants()
        
        ## First get mooring system characteristics about undisplaced platform position (useful for baseline and verification)
        self.C_moor0 = self.ms.getCoupledStiffness(lines_only=True)                             # this method accounts for eqiuilibrium of free objects in the system
        self.F_moor0 = self.ms.getForces(DOFtype="coupled", lines_only=True)
        
    
    def calcMooringAndOffsets(self):
        '''Calculates mean offsets and linearized mooring properties for the current load case.
        setEnv and calcSystemProps must be called first.
        '''


        # Now find static equilibrium offsets of platform and get mooring properties about that point
        # (This assumes some loads have been applied)
        #self.ms.display=2
        
        self.ms.solveEquilibrium3(DOFtype="both", rmsTol=1.0E-5)     # get the system to its equilibrium
        print("Equilibrium'3' platform positions/rotations:")
        printVec(self.ms.BodyList[0].r6)
        
        self.ms.solveEquilibrium(DOFtype="both")
        print("Equilibrium platform positions/rotations:")        

        r6eq = self.ms.BodyList[0].r6
        printVec(r6eq)
        
        self.ms.plot()
                
        print("Surge: {:.2f}".format(r6eq[0]))
        print("Pitch: {:.2f}".format(r6eq[4]*180/np.pi))
        
        C_moor = self.ms.getCoupledStiffness(lines_only=True)
        F_moor = self.ms.getForces(DOFtype="coupled", lines_only=True)    # get net forces and moments from mooring lines on Body

        # manually add yaw spring stiffness as compensation until bridle (crow foot) configuration is added
        C_moor[5,5] += self.yawstiff
        
        self.C_moor = C_moor
        self.F_moor = F_moor
        
        
    
    def solveEigen(self):
        '''finds natural frequencies of system'''
        
    
        # total system coefficient arrays
        M_tot = np.zeros([self.nDOF,self.nDOF])       # total mass and added mass matrix [kg, kg-m, kg-m^2]
        C_tot = np.zeros([self.nDOF,self.nDOF])       # total stiffness matrix [N/m, N, N-m]
        
    
        # add in mooring stiffness from MoorPy system
        C_tot = np.array(self.C_moor0)
               
        # ::: a loop could be added here for an array :::
        fowt = self.fowtList[0]
        
        # range of DOFs for the current turbine
        i1 = 0
        i2 = 6            
        
        # add fowt's terms to system matrices (BEM arrays are not yet included here)        
        M_tot[i1:i2] += fowt.M_struc + fowt.A_hydro_morison   # mass
        C_tot[i1:i2] += fowt.C_struc + fowt.C_hydro           # stiffness
        
        # calculate natural frequencies (using eigen analysis to get proper values for pitch and roll - otherwise would need to base about CG if using diagonal entries only)
        eigenvals, eigenvectors = np.linalg.eig(np.matmul(np.linalg.inv(M_tot), C_tot))   # <<< need to sort this out so it gives desired modes, some are currently a bit messy
        
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
        
        print("natural frequencies from eigen values")
        printVec(fns)
        print("mode shapes from eigen values")
        printMat(modes)
        

        # alternative attempt to calculate natural frequencies based on diagonal entries (and taking pitch and roll about CG)
        if C_tot[0,0] == 0.0:
            zMoorx = 0.0
        else:
            zMoorx = C_tot[0,4]/C_tot[0,0]  # effective z elevation of mooring system reaction forces in x and y directions
        
        if C_tot[1,1] == 0.0:
            zMoory = 0.0
        else:
            zMoory = C_tot[1,3]/C_tot[1,1]
        
        zCG  = fowt.rCG_TOT[2]                    # center of mass in z
        zCMx = M_tot[0,4]/M_tot[0,0]              # effective z elevation of center of mass and added mass in x and y directions
        zCMy = M_tot[1,3]/M_tot[1,1]

        print("natural frequencies with added mass")
        fn = np.zeros(6)
        fn[0] = np.sqrt( C_tot[0,0] / M_tot[0,0] )/ 2.0/np.pi
        fn[1] = np.sqrt( C_tot[1,1] / M_tot[1,1] )/ 2.0/np.pi
        fn[2] = np.sqrt( C_tot[2,2] / M_tot[2,2] )/ 2.0/np.pi
        fn[5] = np.sqrt( C_tot[5,5] / M_tot[5,5] )/ 2.0/np.pi
        fn[3] = np.sqrt( (C_tot[3,3] + C_tot[1,1]*((zCMy-zMoory)**2 - zMoory**2) ) / (M_tot[3,3] - M_tot[1,1]*zCMy**2 ))/ 2.0/np.pi     # this contains adjustments to reflect rotation about the CG rather than PRP
        fn[4] = np.sqrt( (C_tot[4,4] + C_tot[0,0]*((zCMx-zMoorx)**2 - zMoorx**2) ) / (M_tot[4,4] - M_tot[0,0]*zCMx**2 ))/ 2.0/np.pi     # this contains adjustments to reflect rotation about the CG rather than PRP
        # note that the above lines use off-diagonal term rather than parallel axis theorem since rotation will not be exactly at CG due to effect of added mass
        printVec(fn)
                
    
    def solveStatics(self):
        '''Possibly a method to solve for the mean operating point (in conjunctoin with calcMooringAndOffsets)...'''
    
        # ::: a loop could be added here for an array :::
        fowt = self.fowtList[0]
        
        # range of DOFs for the current turbine
        i1 = 0
        i2 = 6  
        
        #C_tot0 = self.C_struc + self.C_hydro + C_moor0   # total system stiffness matrix about undisplaced position
        #W_tot0 = self.W_struc + self.W_hydro + W_moor0   # system mean forces and moments at undisplaced position

    
    
    def solveDynamics(self, nIter=15, tol=0.01):
        '''After all constant parts have been computed, call this to iterate through remaining terms 
        until convergence on dynamic response. Note that steady/mean quantities are excluded here.
        
        nIter = 2  # maximum number of iterations to allow
        '''


        # total system complex response amplitudes (this gets updated each iteration)
        XiLast = np.zeros([self.nDOF,self.nw], dtype=complex) + 0.1    # displacement and rotation complex amplitudes [m, rad]

        fig, ax = plt.subplots(3,1,sharex=True)                    #
        c = np.arange(nIter)                                       #
        c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))            # set up colormap to use to plot successive iteration restuls
        
        # range of DOFs for the current turbine
        i1 = 0
        i2 = 6        
        
        # sum up all linear (non-varying) matrices up front
        M_lin =               fowt.M_struc + fowt.A_BEM + fowt.A_hydro_morison # mass
        B_lin =               fowt.B_struc + fowt.B_BEM                        # damping (structural and linearized morison)
        C_lin = self.C_moor + fowt.C_struc              + fowt.C_hydro         # stiffness
        F_lin =                              fowt.F_BEM + fowt.F_hydro_iner    # excitation force complex amplitudes
        

        # start fixed point iteration loop for dynamics   <<< would a secant method solve be possible/better? <<<
        for iiter in range(nIter):

            # ::: re-zero some things that will be added to :::
            
            # total system coefficient arrays
            M_tot = np.zeros([self.nDOF,self.nDOF,self.nw])       # total mass and added mass matrix [kg, kg-m, kg-m^2]
            B_tot = np.zeros([self.nDOF,self.nDOF,self.nw])       # total damping matrix [N-s/m, N-s, N-s-m]
            C_tot = np.zeros([self.nDOF,self.nDOF,self.nw])       # total stiffness matrix [N/m, N, N-m]
            F_tot = np.zeros([self.nDOF,self.nw], dtype=complex)  # total excitation force/moment complex amplitudes vector [N, N-m]

            Z  = np.zeros([self.nDOF,self.nDOF,self.nw], dtype=complex)  # total system impedance matrix
            
            
            # ::: a loop could be added here for an array :::
            fowt = self.fowtList[0]
            
            # range of DOFs for the current turbine
            i1 = 0
            i2 = 6            
            
            # get linearized terms for the current turbine given latest amplitudes
            B_linearized, F_linearized = fowt.calcLinearizedTerms(XiLast)
            
            # calculate the response based on the latest linearized terms            
            Xi = np.zeros([self.nDOF,self.nw], dtype=complex)     # displacement and rotation complex amplitudes [m, rad]
                
            # add fowt's terms to system matrices (BEM arrays are not yet included here)            
            M_tot[i1:,:,:] = M_lin 
            B_tot[i1:,:,:] = B_lin + B_linearized 
            C_tot[i1:,:,:] = C_lin 
            F_tot[i1:  ,:] = F_lin + F_linearized
                
                
            for ii in range(self.nw):
                # form impedance matrix
                Z[:,:,ii] = -self.w[ii]**2 * M_tot[:,:,ii] + 1j*self.w[ii]*B_tot[:,:,ii] + C_tot[:,:,ii]
                
                # solve response (complex amplitude)
                Xi[:,ii] = np.matmul(np.linalg.inv(Z[:,:,ii]),  F_tot[:,ii] )
            
            
            # plots of surge response at each iteration for observing convergence
            ax[0].plot(self.w, np.abs(Xi[0,:]) , color=c[iiter], label=f"iteration {iiter}")
            ax[1].plot(self.w, np.real(Xi[0,:]), color=c[iiter], label=f"iteration {iiter}")
            ax[2].plot(self.w, np.imag(Xi[0,:]), color=c[iiter], label=f"iteration {iiter}")
            
            # check for convergence
            tolCheck = np.abs(Xi - XiLast) / ((np.abs(Xi)+tol))
            if (tolCheck < tol).all():
                print(f" Iteration {iiter}, converged, with largest tolCheck of {np.max(tolCheck)} < {tol}")
                break
            else:
                XiLast = 0.2*XiLast + 0.8*Xi    # use a mix of the old and new response amplitudes to use for the next iteration 
                                                # (uses hard-coded successive under relaxation for now)
                print(f" Iteration {iiter}, still going since largest tolCheck is {np.max(tolCheck)} >= {tol}")
            
            if iiter == nIter-1:
                print("WARNING - solveDynamics iteration did not converge to the tolerance.")
            
        # labels for convergence plots
        ax[1].legend()
        ax[0].set_ylabel("response magnitude")
        ax[1].set_ylabel("response, real")
        ax[2].set_ylabel("response, imag")
        ax[2].set_xlabel("frequency (rad/s)")
        fig.suptitle("Response convergence")
        
        
        # ------------------------------ preliminary plotting of response ---------------------------------

        fig, ax = plt.subplots(3,1, sharex=True)

        fowt = self.fowtList[0]
        
        ax[0].plot(self.w, np.abs(Xi[0,:])          , 'b', label="surge")
        ax[0].plot(self.w, np.abs(Xi[1,:])          , 'g', label="sway")
        ax[0].plot(self.w, np.abs(Xi[2,:])          , 'r', label="heave")
        ax[1].plot(self.w, np.abs(Xi[3,:])*180/np.pi, 'b', label="roll")
        ax[1].plot(self.w, np.abs(Xi[4,:])*180/np.pi, 'g', label="pitch")
        ax[1].plot(self.w, np.abs(Xi[5,:])*180/np.pi, 'r', label="yaw")
        ax[2].plot(self.w, fowt.zeta,                 'k', label="wave amplitude (m)")

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

        #ax[0].set_ylim([0, 1e6])
        #ax[1].set_ylim([0, 1e9])

        ax[0].set_ylabel("response magnitude (m)")
        ax[1].set_ylabel("response magnitude (deg)")
        ax[2].set_ylabel("wave amplitude (m)")
        ax[2].set_xlabel("frequency (rad/s)")


        return Xi  # currently returning the response rather than saving in the model object
        
        
        
    def calcOutputs(self, Xi):
        '''This is where various output quantities of interest are calculated based on the already-solved system response.'''
        
        
        
        
        '''
         # ---------- mooring line fairlead tension RAOs and constraint implementation ----------
        
         
         for il=1:Platf.Nlines
              
              #aNacRAO{imeto} = -(w').^2 .* (X{imeto}(:,1) + hNac*X{imeto}(:,5));      # Nacelle Accel RAO
                #aNac2(imeto) = sum( abs(aNacRAO{imeto}).^2.*S(:,imeto) ) *(w(2)-w(1));     # RMS Nacelle Accel

            TfairRAO{imeto}(il,:) = C_lf(il,:,imeto)*rao{imeto}(:,:)';  # get fairlead tension RAO for each line (multiply by dofs)
              #RMSTfair{imeto}(il) = sqrt( sum( (abs(TfairRAO{imeto}(il,:))).^2) / length(w) );
              #figure
            #plot(w,abs(TfairRAO{imeto}(il,:)))
              #d=TfairRAO{imeto}(il,:)
              RMSTfair{imeto}(il) = sqrt( sum( (abs(TfairRAO{imeto}(il,:)).^2).*S(:,imeto)') *(w(2)-w(1)) );
              #RMSTfair
              #sumpart = sum( (abs(TfairRAO{imeto}(il,:)).^2).*S(:,imeto)')
              #dw=(w(2)-w(1))
         end        
         
         [Tfair, il] = min( T_lf(:,imeto) );
         if Tfair - 3*RMSTfair{imeto}(il) < 0 && Xm < 1  # taut lines only
              disp([' REJECTING (mooring line goes slack)'])
              fitness = -1;
              return;  # constraint for slack line!!!
         end
         if grads
              disp(['mooring slackness: ' num2str(Tfair - 3*RMSTfair{imeto}(il))])
         end
         
         # ----------- dynamic pitch constraint ----------------------
         #disp('checking dynamic pitch');
         RMSpitch(imeto) = sqrt( sum( ((abs(rao{imeto}(:,5))).^2).*S(:,imeto) ) *(w(2)-w(1)) ); # fixed April 9th :(
         RMSpitchdeg = RMSpitch(imeto)*60/pi;
         if (Platf.spitch + RMSpitch(imeto))*180/pi > 10
              disp([' REJECTING (static + RMS dynamic pitch > 10)'])
              fitness = -1;
              return;  
         end    
         if grads
              disp(['dynamic pitch: ' num2str((Platf.spitch + RMSpitch(imeto))*180/pi)])
         end
         
         #figure(1)
         #plot(w,S(:,imeto))
         #hold on
         
         #figure()
         #plot(2*pi./w,abs(Xi{imeto}(:,5)))
         #ylabel('pitch response'); xlabel('T (s)')
         
         RMSsurge(imeto) = sqrt( sum( ((abs(rao{imeto}(:,1))).^2).*S(:,imeto) ) *(w(2)-w(1)) ); 
         RMSheave(imeto) = sqrt( sum( ((abs(rao{imeto}(:,3))).^2).*S(:,imeto) ) *(w(2)-w(1)) ); 
        ''' 
        
        
    
    def plot(self):
        '''plots the whole model, including FOWTs and mooring system...'''
        
        # for now, start the plot via the mooring system, since MoorPy doesn't yet know how to draw on other codes' plots
        fig, ax = self.ms.plot()
        #fig = plt.figure(figsize=(20/2.54,12/2.54))
        #ax = Axes3D(fig)

        # plot each FOWT
        for fowt in self.fowtList:
            fowt.plot(ax)
        




class FOWT():
    '''This class comprises the frequency domain model of a single floating wind turbine'''

    def __init__(self, design, w=[], mpb=None, depth=600):
        '''This initializes the FOWT object which contains everything for a single turbine's frequency-domain dynamics.
        The initializiation sets up the design description.
        
        Parameters
        ----------        
        design : dict
            Dictionary of the design...
        w
            Array of frequencies to be used in analysis
        '''


        # basic setup 
        self.nDOF = 6

        if len(w)==0:
            w = np.arange(.01, 3, 0.01)                              # angular frequencies tp analyze (rad/s)
        
        self.w = np.array(w)
        self.nw = len(w)                                             # number of frequencies
        self.k = np.zeros(self.nw)                                   # wave number
        
        self.depth = depth


        # member-based platform description 
        self.memberList = []                                         # list of member objects
        
        for mi in design['platform']['members']:
            self.memberList.append(Member(mi, self.nw))
            

        # mooring system connection 
        self.body = mpb                                              # reference to Body in mooring system corresponding to this turbine


        # turbine RNA description 
        self.mRNA    = design['turbine']['mRNA']
        self.IxRNA   = design['turbine']['IxRNA']
        self.IrRNA   = design['turbine']['IrRNA']
        self.xCG_RNA = design['turbine']['xCG_RNA']
        self.hHub    = design['turbine']['hHub']
        
       
        # initialize BEM arrays, whether or not a BEM sovler is used
        self.A_BEM = np.zeros([6,6,self.nw], dtype=float)                 # hydrodynamic added mass matrix [kg, kg-m, kg-m^2]
        self.B_BEM = np.zeros([6,6,self.nw], dtype=float)                 # wave radiation drag matrix [kg, kg-m, kg-m^2]
        self.F_BEM = np.zeros([6,  self.nw], dtype=complex)               # linaer wave excitation force/moment complex amplitudes vector [N, N-m]
        
        

    def setEnv(self, Hs=8, Tp=12, V=10, beta=0, Fthrust=0):
        '''For now, this is where the environmental conditions acting on the FOWT are set.'''

        # ------- Wind conditions
        #Fthrust = 800e3  # peak thrust force, [N]
        #Mthrust = self.hHub*Fthrust  # overturning moment from turbine thrust force [N-m]


        self.env = Env()
        self.env.Hs   = Hs   
        self.env.Tp   = Tp   
        self.env.V    = V    
        self.env.beta = beta 
        
        # make wave spectrum
        S = JONSWAP(self.w, Hs, Tp)
        
        for i in range(self.nw):
            self.k[i] = waveNumber(self.w[i], self.depth)
        
        # wave elevation amplitudes (these are easiest to use) - no need to be complex given frequency domain use
        self.zeta = np.sqrt(S)
            
        #Fthrust = 0
        #Fthrust = 800.0e3            # peak thrust force, [N]
        #Mthrust = self.hHub*Fthrust  # overturning moment from turbine thrust force [N-m]
        
        # add thrust force and moment to mooring system body
        self.body.f6Ext = Fthrust*np.array([np.cos(beta),np.sin(beta),0,-self.hHub*np.sin(beta), self.hHub*np.cos(beta), 0])



    def calcStatics(self):
        '''Fills in the static quantities of the FOWT and its matrices. 
        Also adds some dynamic parameters that are constant, e.g. BEM coefficients and steady thrust loads.'''
        
        rho = self.env.rho
        g   = self.env.g

        # structure-related arrays
        self.M_struc = np.zeros([6,6])                # structure/static mass/inertia matrix [kg, kg-m, kg-m^2]
        self.B_struc = np.zeros([6,6])                # structure damping matrix [N-s/m, N-s, N-s-m] (may not be used)
        self.C_struc = np.zeros([6,6])                # structure effective stiffness matrix [N/m, N, N-m]
        self.W_struc = np.zeros([6])                  # static weight vector [N, N-m]

        # hydrostatic arrays
        self.C_hydro = np.zeros([6,6])                # hydrostatic stiffness matrix [N/m, N, N-m]
        self.W_hydro = np.zeros(6)                    # buoyancy force/moment vector [N, N-m]
        

        # --------------- add in linear hydrodynamic coefficients here if applicable --------------------
        #[as in load them] <<<<<<<<<<<<<<<<<<<<<

        # --------------- Get general geometry properties including hydrostatics ------------------------

        # initialize some variables for running totals
        VTOT = 0.                   # Total underwater volume of all members combined
        mTOT = 0.                   # Total mass of all members [kg]
        AWP_TOT = 0.                # Total waterplane area of all members [m^2]
        IWPx_TOT = 0                # Total waterplane moment of inertia of all members about x axis [m^4]  
        IWPy_TOT = 0                # Total waterplane moment of inertia of all members about y axis [m^4]  
        Sum_V_rCB = np.zeros(3)     # product of each member's buoyancy multiplied by center of buoyancy [m^4]
        Sum_AWP_rWP = np.zeros(2)   # product of each member's waterplane area multiplied by the area's center point [m^3]
        Sum_M_center = np.zeros(3)  # product of each member's mass multiplied by its center of mass [kg-m] (Only considers the shell mass right now)
        
        self.msubstruc = 0          # total mass of just the members that make up the substructure [kg]
        msubstruc_sum = 0           # product of each substructure member's mass and CG, to be used to find the total substructure CG [kg-m]
        self.mshell = 0             # total mass of the shells/steel of the members in the substructure [kg]
        mballast = []               # list to store the mass of the ballast in each of the substructure members [kg]
        pballast = []               # list to store the density of ballast in each of the substructure members [kg]
        
        I44list = []                # list to store the I44 MoI about the PRP of each substructure member
        I55list = []                # list to store the I55 MoI about the PRP of each substructure member
        I66list = []                # list to store the I66 MoI about the PRP of each substructure member
        masslist = []               # list to store the mass of each substructure member
        
        # loop through each member
        for mem in self.memberList:
            
            # calculate member's orientation information (needed for later steps)
            mem.calcOrientation() 
            
            # ---------------------- get member's mass and inertia properties ------------------------------
            mass, center, mshell, mfill, pfill = mem.getInertia() # calls the getInertia method to calcaulte values
            
            
            # Calculate the mass matrix of the FOWT about the PRP
            self.W_struc += translateForce3to6DOF( center, np.array([0,0, -g*mass]) )  # weight vector
            self.M_struc += mem.M_struc     # mass/inertia matrix about the PRP
            
            Sum_M_center += center*mass     # product sum of the mass and center of mass to find the total center of mass [kg-m]
            
            
            # Tower calculations
            if mem.type <= 1:   # <<<<<<<<<<<< maybe find a better way to do the if condition
                self.mtower = mass                  # mass of the tower [kg]
                self.rCG_tow = center               # center of mass of the tower from the PRP [m]
            # Substructure calculations
            if mem.type > 1:
                self.msubstruc += mass              # mass of the substructure
                msubstruc_sum += center*mass        # product sum of the substructure members and their centers of mass [kg-m]
                self.mshell += mshell               # mass of the substructure shell material [kg]
                mballast.extend(mfill)              # list of ballast masses in each substructure member (list of lists) [kg]
                pballast.extend(pfill)              # list of ballast densities in each substructure member (list of lists) [kg/m^3]
                # Store substructure moment of inertia terms
                I44list.append(mem.M_struc[3,3])
                I55list.append(mem.M_struc[4,4])
                I66list.append(mem.M_struc[5,5])
                masslist.append(mass)
                
                
            # -------------------- get each member's buoyancy/hydrostatic properties -----------------------
            
            Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = mem.getHydrostatics(self.env)  # call to Member method for hydrostatic calculations
            
            # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices <<<<< needs updating (already about PRP)
            self.W_hydro += Fvec # translateForce3to6DOF( mem.rA, np.array([0,0, Fz]) )  # weight vector
            self.C_hydro += Cmat # translateMatrix6to6DOF(mem.rA, Cmat)                       # hydrostatic stiffness matrix
            
            VTOT    += V_UW    # add to total underwater volume of all members combined
            AWP_TOT += AWP
            IWPx_TOT += IWP + AWP*yWP**2
            IWPy_TOT += IWP + AWP*xWP**2
            Sum_V_rCB   += r_CB*V_UW
            Sum_AWP_rWP += np.array([xWP, yWP])*AWP


        # ------------------------- include RNA properties -----------------------------

        # Here we could initialize first versions of the structure matrix components. 
        # These might be iterated on later to deal with mean- or amplitude-dependent terms.
        #self.M_struc += structural.M_lin(q0,      self.turbineParams)        # Linear Mass Matrix
        #self.B_struc += structural.C_lin(q0, qd0, self.turbineParams, u0)    # Linear Damping Matrix
        #self.C_struc += structural.K_lin(q0, qd0, self.turbineParams, u0)    # Linear Stifness Matrix
        #self.W_struc += structural.B_lin(q0, qd0, self.turbineParams, u0)    # Linear RHS
        
        # below are temporary placeholders
        # for now, turbine RNA is specified by some simple lumped properties
        Mmat = np.diag([self.mRNA, self.mRNA, self.mRNA, self.IxRNA, self.IrRNA, self.IrRNA])            # create mass/inertia matrix
        center = np.array([self.xCG_RNA, 0, self.hHub])                                 # RNA center of mass location

        # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices
        self.W_struc += translateForce3to6DOF( center, np.array([0,0, -g*self.mRNA]) )  # weight vector
        self.M_struc += translateMatrix6to6DOF(center, Mmat)                       # mass/inertia matrix
        Sum_M_center += center*self.mRNA

     
        # ----------- process inertia-related totals ----------------
        
        mTOT = self.M_struc[0,0]                        # total mass of all the members
        rCG_TOT = Sum_M_center/mTOT                     # total CG of all the members
        self.rCG_TOT = rCG_TOT
        
        self.rCG_sub = msubstruc_sum/self.msubstruc     # solve for just the substructure mass and CG
        
        self.I44 = 0        # moment of inertia in roll due to roll of the substructure about the substruc's CG [kg-m^2]
        self.I44B = 0       # moment of inertia in roll due to roll of the substructure about the PRP [kg-m^2]
        self.I55 = 0        # moment of inertia in pitch due to pitch of the substructure about the substruc's CG [kg-m^2]
        self.I55B = 0       # moment of inertia in pitch due to pitch of the substructure about the PRP [kg-m^2]
        self.I66 = 0        # moment of inertia in yaw due to yaw of the substructure about the substruc's centerline [kg-m^2]
        
        # Use the parallel axis theorem to move each substructure's MoI to the substructure's CG
        x = np.linalg.norm([self.rCG_sub[1],self.rCG_sub[2]])   # the normalized distance between the x and x' axes
        y = np.linalg.norm([self.rCG_sub[0],self.rCG_sub[2]])   # the normalized distance between the y and y' axes
        z = np.linalg.norm([self.rCG_sub[0],self.rCG_sub[1]])   # the normalized distance between the z and z' axes
        for i in range(len(I44list)):
            self.I44 += I44list[i] - masslist[i]*x**2
            self.I44B += I44list[i]
            self.I55 += I55list[i] - masslist[i]*y**2
            self.I55B += I55list[i]
            self.I66 += I66list[i] - masslist[i]*z**2
        
        # Solve for the total mass of each type of ballast in the substructure
        self.pb = []                                                # empty list to store the unique ballast densities
        for i in range(len(pballast)):
            if pballast[i] != 0:                                    # if the value in pballast is not zero
                if self.pb.count(pballast[i]) == 0:                 # and if that value is not already in pb
                    self.pb.append(pballast[i])                     # store that ballast density value
        
        self.mballast = np.zeros(len(self.pb))                      # make an empty mballast list with len=len(pb)
        for i in range(len(self.pb)):                               # for each ballast density
            for j in range(len(mballast)):                          # loop through each ballast mass
                if np.float(pballast[j]) == np.float(self.pb[i]):   # but only if the index of the ballast mass (density) matches the value of pb
                    self.mballast[i] += mballast[j]                 # add that ballast mass to the correct index of mballast

        
        
        # ----------- process key hydrostatic-related totals for use in static equilibrium solution ------------------
        
        self.V = VTOT                                   # save the total underwater volume
        rCB_TOT = Sum_V_rCB/VTOT       # location of center of buoyancy on platform
        self.rCB = rCB_TOT

        if VTOT==0: # if you're only working with members above the platform, like modeling the wind turbine
            zMeta = 0
        else:
            zMeta   = rCB_TOT[2] + IWPx_TOT/VTOT  # add center of buoyancy and BM=I/v to get z elevation of metecenter [m] (have to pick one direction for IWP)

        self.C_struc[3,3] = -mTOT*g*rCG_TOT[2]
        self.C_struc[4,4] = -mTOT*g*rCG_TOT[2]
        
        # add relevant properties to this turbine's MoorPy Body
        self.body.m = mTOT
        self.body.v = VTOT
        self.body.rCG = rCG_TOT
        self.body.AWP = AWP_TOT
        self.body.rM = np.array([0,0,zMeta])
        # is there any risk of additional moments due to offset CB since MoorPy assumes CB at ref point? <<<



    def calcBEM(self):
        '''This generates a mesh for the platform and runs a BEM analysis on it.
        The mesh is only for non-interesecting members flagged with potMod=1.'''
        
        rho = self.env.rho
        g   = self.env.g
        
        # desired panel size (longitudinal and azimuthal)
        dz = 2.5
        da = 2.0

        # vertices array that will contain all panel vertices for writing to mesh file
        vertices = np.zeros([0,3])

        # go through members to be modeled with BEM and calculated their meshes
        for mem in self.memberList:                 
        
            if mem.potMod==1:
                vertices_i = memberMesh(mem.stations, mem.d, mem.rA, mem.rB, dz_max=dz, da_max=da)
                
                vertices = np.vstack([vertices, vertices_i])             # append the member's vertices to the master list
            
         
        # generate a mesh file (current example in WAMIT .gdf format)
        npan = int(vertices.shape[0]/4)

        f = open("platform.gdf", "w")
        f.write('gdf mesh \n')
        f.write('1.0   9.8 \n')
        f.write('0, 0 \n')
        f.write(f'{npan}\n')

        for i in range(npan*4):
            f.write(f'{vertices[i,0]:>10.3f} {vertices[i,1]:>10.3f} {vertices[i,2]:>10.3f}\n')
        
        f.close()
        
        
        # >>> this is where a BEM solve could be executed <<<
        
        
        # the BEM coefficients would be contained in the following at frequencies self.w:
        #self.A_BEM
        #self.B_BEM
        #self.F_BEM
        
        
        
    def calcHydroConstants(self):
        '''This computes the linear strip-theory-hydrodynamics terms.'''
        
        rho = self.env.rho
        g   = self.env.g

        # --------------------- get constant hydrodynamic values along each member -----------------------------

        self.A_hydro_morison = np.zeros([6,6])                # hydrodynamic added mass matrix, from only Morison equation [kg, kg-m, kg-m^2]
        self.F_hydro_iner    = np.zeros([6,self.nw],dtype=complex) # inertia excitation force/moment complex amplitudes vector [N, N-m]


        # loop through each member
        for mem in self.memberList:
            
            # loop through each node of the member
            for il in range(mem.n):
            
                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:
                    
                    # get wave kinematics spectra given a certain wave spectrum and location
                    mem.u[il,:,:], mem.ud[il,:,:], mem.pDyn[il,:] = getWaveKin(self.zeta, self.w, self.k, self.depth, mem.r[il,:], self.nw)


                    # ----- compute side effects ---------------------------------------------------------
                    
                    dl = 0.5*mem.dl if (il==0 or il==mem.n) else 1.0*mem.dl              # set dl to half if at the member end (the end result is similar to trapezoid rule integration)
                    v_i = 0.25*np.pi*mem.d[il]**2*dl                                     # member volume assigned to this node
                    
                    # added mass
                    Amat = rho*v_i *( mem.Ca_q*mem.qMat + mem.Ca_p1*mem.p1Mat + mem.Ca_p2*mem.p2Mat )  # local added mass matrix
                    
                    self.A_hydro_morison += translateMatrix3to6DOF(mem.r[il,:], Amat)    # add to global added mass matrix for Morison members
                    
                    # inertial excitation - Froude-Krylov
                    Imat = rho*v_i *( (1.+mem.Ca_q)*mem.qMat + (1.+mem.Ca_p1)*mem.p1Mat + (1.+mem.Ca_p2)*mem.p2Mat ) # local inertial excitation matrix
                    
                    for i in range(self.nw):                                             # for each wave frequency...
                    
                        mem.F_exc_iner[il,:,i] = np.matmul(Imat, mem.ud[il,:,i])         # add to global excitation vector (frequency dependent)
                        
                        self.F_hydro_iner[:,i] += translateForce3to6DOF( mem.r[il,:], mem.F_exc_iner[il,:,i])  # add to global excitation vector (frequency dependent)
                        
                    
                    # ----- add end effects for added mass, and excitation including dynamic pressure ------
                    
                    if il==0 or il==mem.n-1:
                    
                        v_i = np.pi*mem.d[il]**3/6.0                                     # volume assigned to this end surface
                        a_i = 0.25*np.pi*mem.d[il]**2                                    # end area
                            
                        # added mass
                        Amat = rho*v_i * mem.Ca_End*mem.qMat                             # local added mass matrix
                        
                        self.A_hydro_morison += translateMatrix3to6DOF(mem.r[il,:],Amat) # add to global added mass matrix for Morison members
                        
                        # inertial excitation
                        Imat = rho*v_i * (1+mem.Ca_End)*mem.qMat                         # local inertial excitation matrix
                        
                        for i in range(self.nw):                                         # for each wave frequency...
                        
                            F_exc_iner_temp = np.matmul(Imat, mem.ud[il,:,i])            # local inertial excitation force complex amplitude in x,y,z
                                               
                            if il == 0:
                                F_exc_iner_temp += mem.pDyn[il,i]*rho*a_i *mem.q         # add dynamic pressure - positive with q if end A
                            else:
                                F_exc_iner_temp -= mem.pDyn[il,i]*rho*a_i *mem.q         # add dynamic pressure - negative with q if end B
                            
                            mem.F_exc_iner[il,:,i] += F_exc_iner_temp                    # add to stored member force vector
                            
                            self.F_hydro_iner[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_iner_temp) # add to global excitation vector (frequency dependent)
                        

    def calcLinearizedTerms(self, Xi):
        '''The FOWT's dynamics solve iteration method. This calculates the amplitude-dependent linearized coefficients.
        
        Xi : complex array
            system response (just for this FOWT) - displacement and rotation complex amplitudes [m, rad]

        '''
        
        rho = self.env.rho
        g   = self.env.g

        # The linearized coefficients to be calculated
        
        B_hydro_drag = np.zeros([6,6])             # hydrodynamic damping matrix (just linearized viscous drag for now) [N-s/m, N-s, N-s-m]

        F_hydro_drag = np.zeros([6,self.nw],dtype=complex) # excitation force/moment complex amplitudes vector [N, N-m]


        # loop through each member
        for mem in self.memberList:
            
            # loop through each node of the member
            for il in range(mem.n):
                
                # node displacement, velocity, and acceleration (each [3 x nw])
                drnode, vnode, anode = getVelocity(mem.r[il,:], Xi, self.w)      # get node complex velocity spectrum based on platform motion's and relative position from PRP
                
                
                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:
                
                    # ----- compute side effects ------------------------
                    
                    dl = 0.5*mem.dl if (il==0 or il==mem.n) else 1.0*mem.dl              # set dl to half if at the member end (the end result is similar to trapezoid rule integration)
                    a_i = mem.d[il] * dl                                                 # member side cross-sectional area assigned to this node
                    
                    # water relative velocity over node (complex amplitude spectrum)  [3 x nw]
                    vrel = mem.u[il,:] - vnode
                    
                    # break out velocity components in each direction relative to member orientation [nw]
                    vrel_q  = vrel*mem.q[ :,None]     # (the ,None is for broadcasting q across all frequencies in vrel)
                    vrel_p1 = vrel*mem.p1[:,None]
                    vrel_p2 = vrel*mem.p2[:,None]
                    
                    # get RMS of relative velocity component magnitudes (real-valued)
                    vRMS_q  = np.linalg.norm( np.abs(vrel_q ) )                          # equivalent to np.sqrt( np.sum( np.abs(vrel_q )**2) /nw)
                    vRMS_p1 = np.linalg.norm( np.abs(vrel_p1) )
                    vRMS_p2 = np.linalg.norm( np.abs(vrel_p2) )
                    
                    # linearized damping coefficients in each direction relative to member orientation [not explicitly frequency dependent...] (this goes into damping matrix)
                    Bprime_q  = np.sqrt(8/np.pi) * vRMS_q  * 0.5*rho * np.pi*a_i * mem.Cd_q 
                    Bprime_p1 = np.sqrt(8/np.pi) * vRMS_p1 * 0.5*rho *       a_i * mem.Cd_p1
                    Bprime_p2 = np.sqrt(8/np.pi) * vRMS_p2 * 0.5*rho *       a_i * mem.Cd_p2
                    
                    Bmat = Bprime_q*mem.qMat + Bprime_p1*mem.p1Mat + Bprime_p2*mem.p2Mat # damping matrix for the node based on linearized drag coefficients 
                    
                    B_hydro_drag += translateMatrix3to6DOF(mem.r[il,:], Bmat)            # add to global damping matrix for Morison members
                                   
                    for i in range(self.nw):

                        mem.F_exc_drag[il,:,i] = np.matmul(Bmat, mem.u[il,:,i])          # get local 3d drag excitation force complex amplitude for each frequency [3 x nw]
                        
                        F_hydro_drag[:,i] += translateForce3to6DOF( mem.r[il,:], mem.F_exc_drag[il,:,i])   # add to global excitation vector (frequency dependent)
                        
                    
                    # ----- add end effects for added mass, and excitation including dynamic pressure ------
                    
                    if il==0 or il==mem.n-1:
                    
                        a_i = 0.25*np.pi*mem.d[il]**2                                    # end area
                        Bprime_End = np.sqrt(8/np.pi)*vRMS_q*0.5*rho*np.pi*a_i*mem.Cd_End 
                    
                        Bmat = Bprime_End*mem.qMat                                       # 
                        
                        B_hydro_drag += translateMatrix3to6DOF(mem.r[il,:], Bmat)        # add to global damping matrix for Morison members
                        
                        for i in range(self.nw):                                         # for each wave frequency...
                        
                            F_exc_drag_temp = np.matmul(Bmat, mem.u[il,:,i])             # local drag excitation force complex amplitude in x,y,z
                            
                            mem.F_exc_drag[il,:,i] += F_exc_drag_temp                    # add to stored member force vector
                            
                            F_hydro_drag[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_drag_temp) # add to global excitation vector (frequency dependent)
                    

        # save the arrays internally in case there's ever a need for the FOWT to solve it's own latest dynamics
        self.B_hydro_drag = B_hydro_drag
        self.F_hydro_drag = F_hydro_drag

        # return the linearized coefficients
        return B_hydro_drag, F_hydro_drag


    def plot(self, ax):
        '''plots the FOWT...'''
        
        
        # loop through each member and plot it
        for mem in self.memberList:
        
            mem.calcOrientation()  # temporary
        
            mem.plot(ax)
            
        # in future should consider ability to animate mode shapes and also to animate response at each frequency 
        # including hydro excitation vectors stored in each member
            
#