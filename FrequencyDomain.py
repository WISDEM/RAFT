# 2020-05-03: This is a start at a frequency-domain floating support structure model for WEIS-WISDEM.
#             Based heavily on the GA-based optimization framework from Hall 2013

# 2020-05-23: Starting as a script that will evaluate a design based on properties/data provided in separate files.


import numpy as np
import matplotlib.pyplot as plt



## This class represents linear (for now cylinderical) components in the substructure. 
#  It is meant to correspond to Member objects in the WEIS substructure ontology, but containing only 
#  the properties relevant for the Level 1 frequency-domain model, as well as additional data strucutres
#  used during the model's operation.
class Member:

    def __init__(self, strin, nw):
        '''Initialize a Member. For now, this function accepts a space-delimited string with all member properties.
        
        
        '''
    
        # note: haven't decided yet how to lump masses and handle segments <<<<
    
        entries = strin.split()
        
        self.id = np.int(entries[0])
        self.dA  = np.int(entries[2])                      # diameter of (lower) node
        self.dB  = np.int(entries[2])
        self.rA = np.array(entries[3:6], dtype=np.double)  # x y z of lower node
        self.rB = np.array(entries[6:9], dtype=np.double)
        self.t  = 0.06           # shell thickness [m]
        
        rAB = self.rB-self.rA
        self.l = np.linalg.norm(rAB)  # member length
        
        self.q = rAB/self.l           # member axial unit vector
        self.p1 = np.zeros(3)              # member transverse unit vectors (to be filled in later)
        self.p2 = np.zeros(3)              # member transverse unit vectors
        
        
        self.Cd_q  = 0.1  # drag coefficients
        self.Cd_p1 = 1.0
        self.Cd_p2 = 1.0
        self.Ca_q  = 0.0  # added mass coefficients
        self.Ca_p1 = 1.0
        self.Ca_p2 = 1.0
        
        
        self.n = 5 # number of nodes per member
        self.dl = self.l/self.n   #lumped node lenght (I guess, for now) <<<
        
        self.w = 1    # mass per unit length (kg/m)
        
        self.r  = np.zeros([self.n,3])  # undisplaced node positions along member  [m]
        self.d  = np.zeros([self.n,3])  # local diameter along member [m]
        for i in range(self.n):
            self.r[i,:] = self.rA + (i/(self.n-1))*rAB                # spread evenly for now
            self.d[i]   = self.dA + (i/(self.n-1))*(self.dB-self.dA)  # spread evenly since uniform taper
        
        # complex frequency-dependent amplitudes of quantities at each node along member (to be filled in later)
        self.dr = np.zeros([self.n,3,nw], dtype=complex)  # displacement
        self.v  = np.zeros([self.n,3,nw], dtype=complex)  # velocity
        self.a  = np.zeros([self.n,3,nw], dtype=complex)  # acceleration
        self.u  = np.zeros([self.n,3,nw], dtype=complex)  # wave velocity
        self.ud = np.zeros([self.n,3,nw], dtype=complex)  # wave acceleration
        
        
    def getDirections(self):
        '''Returns the direction unit vector for the member based on its end positions'''
        
        q  =  self.q
        p1 = np.zeros(3) # <<<< ???
        p2 = np.zeros(3) # <<<< ???
        
        return q, p1, p2
    
    
    
    # >>>>>>>>>>>>> @shousner's method for mass/inertia to go here <<<<<<<<<<<<<<<<<<
    def getInertia(self):
        rho_steel = 7850 #[kg/m^3] Density of steel

        V_UW = (np.pi/4)*(1/3)*(self.dA**2+self.dB**2+self.dA*self.dB)*self.l		#[m^3] Total enclosed underwater volume of member

        v_steel = ((self.t/2)*(self.dA+self.dB)-self.t**2)*np.pi*self.l			#[m^3] Volume of steel of the member
        
        if self.dA >= self.dB:
            center_calc = ((self.dB/(4*self.dA))+(1/4))*self.l
            center = self.rA + (self.q*center_calc)
        else:
            center_calc = ((self.dA/(4*self.dB))+(1/4))*self.l
            center = self.rB - (self.q*center_calc)

        # Method done in HydroDyn paper from Matt
        # center_calc = (self.l/4)*((self.dA**2 + 2*self.dA*self.dB + 3*self.dB**2)/(4*(self.dA**2 + self.dA*self.dB + self.dB**2)))
        # center = self.rA + (self.q+center_calc)

        # Moment of Inertia

        ro = ((self.dA+self.dB)/2)/2
        ri = (((self.dA-(2*self.t))+(self.dB-(2*self.t)))/2)/2
        #I_rad = (1/12)*(rho_steel*v_steel)*(3*(ro**2 + ri**2) + 4*self.l**2) # About the end (node) of the member
        #I_ax = (1/2)*(rho_steel*v_steel)*(ro**2 + ri**2)
        I_rad = (1/12)*(rho_steel*self.l*np.pi)*(3*(ro**4 - ri**4) + 4*self.l**2*(ro**2 - ri**2))
        I_ax = (1/2)*(rho_steel*np.pi*self.l)*(ro**4 - ri**4)
        # (future work) Can expand these calcs for an actual frustum and not just a cylinder with the average of the top and bottom diameters
        # (future work) To find the total moment of inertia of all members for any possible future calcs, bring each member's MOI to the origin

        return V_UW, v_steel, center, I_rad, I_ax
        
    
    
    def getHydrostatics(self):
        '''Calculates member hydrostatic properties, namely buoyancy and stiffness matrix'''
        
            
        # partially submerged case
        if self.r[0,2]*self.r[-1,2] < 0:    # if member crosses water plane
                
        #   # eventually should interpolate dwp=np.interp(0, self.r[:,2], self.d)
        #   dwp = self.d
        #   xwp = np.interp(0, self.r[:,2], self.r[:,0])
        #   ywp = np.interp(0, self.r[:,2], self.r[:,1])
            
            # ------------- get hydrostatic derivatives ---------------- 
            
            # angles
            beta = np.arctan(q[1],q[0])  # member incline heading from x axis
            phi  = np.arctan(np.sqrt(q[0]**2 + q[1]**2), q[2])  # member incline angle from vertical
            
            # precalculate trig functions
            cosPhi=np.cos(phi)
            sinPhi=np.sin(phi)
            tanPhi=np.tan(phi)
            cosBeta=np.cos(beta)
            sinBeta=np.sin(beta)
            tanBeta=sinBeta/cosBeta
            
            # derivatives from global to local 
            dPhi_dThx  = -sinBeta                     # \frac{d\phi}{d\theta_x} = \sin\beta
            dPhi_dThy  =  cosBeta
            dBeta_dThx = -cosBeta/tanBeta**2
            dBeta_dThy = -sinBeta/tanBeta**2
            
            # >>>>>>>>>>> warning, these are untapered-only so far >>>>>>>
            
            
            # buoyancy force and moment about end A
            Fz = -rho*g*pi*0.25*self.d**2*self.rA[2]/cosPhi
            M  = -rho*g*pi*( self.d*2/32*(2.0 + tanPhi**2) + 0.5*(self.rA[2]/cosPhi)**2)*sinPhi  # moment about axis of incline
            Mx = M*dPhi_dThx
            My = M*dPhi_dThy
            
            Fvec = np.zeros(6)                         # buoyancy force (and moment) vector (about PRP)
            Fvec[2] = Fz                               # vertical buoyancy force [N]
            Fvec[3] = Mx + Fz*self.r[1]                # moment about x axis [N-m]
            Fvec[4] = My - Fz*self.r[0]                # moment about y axis [N-m]
            
            # derivatives aligned with incline heading
            dFz_dz   = -rho*g*pi*0.25*self.d**2                  /cosPhi
            dFz_dPhi =  rho*g*pi*0.25*self.d**2*self.rA[2]*sinPhi/cosPhi**2
            dM_dz    =  1.0*dFz_dPhi
            dM_dPhi  = -rho*g*pi*0.25*self.d**2 * (self.d**2/32*(2*cosPhi + sinPhi**2/CosPhi + 2*sinPhi/cosPhi**2) + 0.5*self.rA[2]**2*(sinPhi**2+1)/cosPhi**3)
            
            # <<<<<<<<<<<< (end warning) <<<<<<<<<
            
            
            # derivatives in global coordinates about platform reference point
            #dFz_dz is already taken care of
            dFz_dThx = -dFz_dz*self.rA[1] + dFz_dPhi*dPhi_dThx
            dFz_dThy =  dFz_dz*self.rA[0] + dFz_dPhi*dPhi_dThy

            dMx_dz   = -dFz_dz*self.rA[1] + dM_dz*dPhi_dThy  #  = dFz_dThx
            dMy_dz   =  dFz_dz*self.rA[0] + dM_dz*dPhi_dThx  #  = dFz_dThy
            
            dMx_dThx = ( dFz_dz*self.rA[1] + dFz_dPhi*dPhi_dThx)*self.rA[1] + dM_dPhi*dPhi_dThx*dPhi_dThx   
            dMx_dThy = (-dFz_dz*self.rA[0] + dFz_dPhi*dPhi_dThy)*self.rA[1] + dM_dPhi*dPhi_dThx*dPhi_dThy  
            dMy_dThx =-( dFz_dz*self.rA[1] + dFz_dPhi*dPhi_dThy)*self.rA[0] + dM_dPhi*dPhi_dThy*dPhi_dThx  
            dMy_dThy =-(-dFz_dz*self.rA[0] + dFz_dPhi*dPhi_dThy)*self.rA[0] + dM_dPhi*dPhi_dThy*dPhi_dThy  
            
            # <<< the above contains some parallel axis stuff. I should remove it and use translateMatrix instead <<<<<

            # fill in stiffness matrix
            Cmat = np.zeros([6,6]) # hydrostatic stiffness matrix (about PRP)
            Cmat[2,2] = dFz_dz
            Cmat[2,3] = dFz_dThx
            Cmat[2,4] = dFz_dThy
            Cmat[3,2] = dMx_dz
            Cmat[4,2] = dMy_dz   # ignoring symmetries for now, as a way to check equations
            Cmat[3,3] = dMx_dThx
            Cmat[3,4] = dMx_dThy
            Cmat[4,3] = dMy_dThx
            Cmat[4,4] = dMy_dThy
            
            
        
        # fully submerged case <<<< not done yet <<<<
        else:
            Fvec = np.zeros(6)        # buoyancy force (and moment) vector (about end A)
            Cmat = np.zeros([6,6]) # hydrostatic stiffness matrix (about end A)
            
        return Fvec, Cmat



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
def getWaveKin(zeta0, w, k, h, r):

    # inputs: wave elevation fft, wave freqs, wave numbers, depth, point position

    beta = 0  # no wave heading for now

    zeta = np.zeros(nw , dtype=complex ) # local wave elevation
    u  = np.zeros([3,nw], dtype=complex) # local wave kinematics velocity
    ud = np.zeros([3,nw], dtype=complex) # local wave kinematics acceleration
    
    for i in range(nw):
    
        # .............................. wave elevation ...............................
        zeta[i] = zeta0[i]* np.exp( -1j*(k[i]*(np.cos(beta)*r[0] + np.sin(beta)*r[1])))  # shift each zetaC to account for location
        
        
        # ...................... wave velocities and accelerations ............................
        
        z = r[2]
        
        # Calculate SINH( k*( z + h ) )/SINH( k*h ) and COSH( k*( z + h ) )/SINH( k*h )
        # given the wave number, k, water depth, h, and elevation z, as inputs.
        if (    k[i]   == 0.0  ):                   # When .TRUE., the shallow water formulation is ill-conditioned; thus, the known value of unity is returned.
            SINHNumOvrSIHNDen = 1.0
            COSHNumOvrSIHNDen = 99999
        elif ( k[i]*h >  89.4 ):                # When .TRUE., the shallow water formulation will trigger a floating point overflow error; however, with h > 14.23*wavelength (since k = 2*Pi/wavelength) we can use the numerically-stable deep water formulation instead.
            SINHNumOvrSIHNDen = np.exp(  k[i]*z );
            COSHNumOvrSIHNDen = np.exp(  k[i]*z );
        else:                                    # 0 < k*h <= 89.4; use the shallow water formulation.
            SINHNumOvrSIHNDen = np.sinh( k[i]*( z + h ) )/np.sinh( k[i]*h );
            COSHNumOvrSIHNDen = np.cosh( k[i]*( z + h ) )/np.sinh( k[i]*h );
        
        # Fourier transform of wave velocities 
        u[0,i] =    w[i]* zeta[i]*COSHNumOvrSIHNDen *np.cos(beta) 
        u[1,i] =    w[i]* zeta[i]*COSHNumOvrSIHNDen *np.sin(beta)
        u[2,i] = 1j*w[i]* zeta[i]*SINHNumOvrSIHNDen

        # Fourier transform of wave accelerations                   
        ud[:,i] = 1j*w[i]*u[:,i]
        
    return u, ud


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

    # note that the J term and I terms are zero in this case because the input is just a mass matrix (assumed to be about CG)

    H = getH(r)     # "anti-symmetric tensor components" from Sadeghi and Incecik
    
    Mout = np.zeros([6,6]) #, dtype=complex)
    
    # mass matrix  [m'] = [m]
    Mout[:3,:3] = Min
        
    # product of inertia matrix  [J'] = [m][H] + [J]  
    Mout[3:,:3] = np.matmul(Min, H)
    Mout[:3,3:] = Mout[3:,:3].T
    
    # moment of inertia matrix  [I'] = [H][m][H]^T + [J]^T [H] + [H]^T [J] + [I]
    
    Mout[3:,3:] = np.matmul(np.matmul(H,Min), H.T) 
    
    return Mout


def translateMatrix6to6DOF(r, Min):

    H = getH(r)     # "anti-symmetric tensor components" from Sadeghi and Incecik
    
    Mout = np.zeros([6,6]) #, dtype=complex)
    
    # mass matrix  [m'] = [m]
    Mout[:3,:3] = Min[:3,:3]
        
    # product of inertia matrix  [J'] = [m][H] + [J]
    Mout[3:,:3] = np.matmul(Min[:3,:3], H) + Min[3:,:3]
    Mout[:3,3:] = Mout[3:,:3].T
    
    # moment of inertia matrix  [I'] = [H][m][H]^T + [J]^T [H] + [H]^T [J] + [I]    
    Mout[3:,3:] = np.matmul(np.matmul(H,Min[:3,:3]), H.T) + np.matmul(Min[:3,3:], H) + np.matmul(H.T, Min[3:,:3]) + Min[3:,3:]
    
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
    
    This formulate for the JONSWAP spectrum is taken from FAST v7 and based
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
    
	


# ------------------------------- basic setup -----------------------------------------

nDOF = 6

w = np.arange(.01, 2, 0.01)  # angular frequencies tp analyze (rad/s)
nw = len(w)  # number of frequencies

k= np.zeros(nw)  # wave number


# ----------------------- member-based platform description --------------------------

# (hard-coded for now - eventually these will be provided as inputs instead)

# list of member objects
memberList = []

#                    number  type  diameter  xa   ya    za   xb   yb   zb
memberList.append(Member("1     x      12     0    0   -20    0    0    0  ", nw))
memberList.append(Member("2     x      12    30    0   -20   30    0    0  ", nw))
memberList.append(Member("3     x      12     0   30   -20    0   30    0  ", nw))
memberList.append(Member("4     x      8      0    0   -20   30    0  -20  ", nw))
memberList.append(Member("5     x      8      0    0   -20    0   30  -20  ", nw))


# ---------------- (future work) import hydrodynamic coefficient files ----------------



# ---------------------------- environmental conditions -------------------------------

depth = 200
rho = 1025
g = 9.81

pi = np.pi

# environmental condition(s)
Hs = [3.4, 5   , 6 ];
Tp = [4.6, 6.05, 10];
windspeed = [8, 12, 18];

S   = np.zeros([len(Hs), nw])   # wave spectrum
S2  = np.zeros([len(Hs), nw])   # wave spectrum
zeta= np.zeros([len(Hs), nw])  # wave elevation

T = 2*np.pi/w # periods



for imeto in range(len(Hs)):       # loop through each environmental condition

    # make wave spectrum (could replace with function call)
    S[imeto,:]  = JONSWAP(w, Hs[imeto], Tp[imeto])
    
    S2[imeto,:] = 1/2/np.pi *5/16 * Hs[imeto]**2*T *(w*Tp[imeto]/2/np.pi)**(-5) *np.exp( -5/4* (w*Tp[imeto]/2/np.pi)**(-4) )

    # wave elevation amplitudes (these are easiest to use) - no need to be complex given frequency domain use
    zeta[imeto,:] = np.sqrt(S[imeto,:])
    

# get wave number
for i in range(nw):
    k[i] = waveNumber(w[i], depth)


plt.plot(w/np.pi/2,  S[2,:], "r")
plt.plot(w/np.pi/2, S2[2,:], "b")
plt.xlabel("Hz")


# ignoring multiple DLCs for now <<<<<
#for imeto = 1:length(windspeeds)
imeto = 0



# ---------------------- set up key arrays -----------------------------

# structure-related arrays
M_struc = np.zeros([6,6])                # structure/static mass/inertia matrix [kg, kg-m, kg-m^2]
B_struc = np.zeros([6,6])                # structure damping matrix [N-s/m, N-s, N-s-m] (may not be used)
C_struc = np.zeros([6,6])                # structure effective stiffness matrix [N/m, N, N-m]
W_struc = np.zeros([6])                  # static weight vector [N, N-m]

# hydrodynamics-related arrays
A_hydro = np.zeros([6,6,nw])             # hydrodynamic added mass matrix [kg, kg-m, kg-m^2]
B_hydro = np.zeros([6,6,nw])             # hydrodynamic damping matrix (just linearized viscous drag for now) [N-s/m, N-s, N-s-m]
C_hydro = np.zeros([6,6])                # hydrostatic stiffness matrix [N/m, N, N-m]
W_hydro = np.zeros(6)                    # buoyancy force/moment vector [N, N-m]
F_hydro = np.zeros([6,nw],dtype=complex) # excitation force/moment complex amplitudes vector [N, N-m]

# moorings-related arrays
A_moor = np.zeros([6,6,nw])              # mooring added mass matrix [kg, kg-m, kg-m^2] (may not be used)
B_moor = np.zeros([6,6,nw])              # mooring damping matrix [N-s/m, N-s, N-s-m] (may not be used)
C_moor = np.zeros([6,6])                 # mooring stiffness matrix (linearized about platform offset) [N/m, N, N-m]
W_moor = np.zeros(6)                     # mean net mooring force/moment vector [N, N-m]

# final coefficient arrays
M_tot = np.zeros([6,6,nw])               # total mass and added mass matrix [kg, kg-m, kg-m^2]
B_tot = np.zeros([6,6,nw])               # total damping matrix [N-s/m, N-s, N-s-m]
C_tot = np.zeros([6,6,nw])               # total stiffness matrix [N/m, N, N-m]
F_tot = np.zeros([6,nw], dtype=complex)  # total excitation force/moment complex amplitudes vector [N, N-m]

Z     = np.zeros([6,6,nw], dtype=complex)               # total system impedance matrix

# system response 
Xi = np.zeros([6,nw], dtype=complex)    # displacement and rotation complex amplitudes [m, rad]


# --------------- add in linear hydrodynamic coefficients here if applicable --------------------

# TODO <<<


# --------------- Get general geometry properties including hydrostatics ------------------------

# loop through each member
for mem in memberList:
    
    q, p1, p2 = mem.getDirections()                # get unit direction vectors
    
    
    # ---------------------- get member's mass and inertia properties ------------------------------
    
    # >>>> call to @shousner's Member method(s) <<<<
    
    # the method should compute the mass/inertia matrix, instead of the placeholders below
    # mass = mem.l*mem.w     # kg
    # iner = 0               # kg-m^2 
    V_UW, v_steel, center, I_rad, I_ax = mem.getInertia()
    rho_steel = 7850 #[kg/m^3]
    mass = v_steel*rho_steel #[kg]
    Mmat = np.diag([mass, mass, mass, I_rad, I_rad, I_ax])
    Mmat[0,4] = mass*center[2]
    Mmat[4,0] = mass*center[2]
    Mmat[1,3] = -mass*center[2]
    Mmat[3,1] = -mass*center[2]
    Mmat[0,5] = -mass*center[1]
    Mmat[5,0] = -mass*center[1]
    Mmat[2,3] = mass*center[1]
    Mmat[3,2] = mass*center[1]
    Mmat[1,5] = mass*center[0]
    Mmat[5,1] = mass*center[0]
    Mmat[4,2] = -mass*center[0]
    Mmat[2,4] = -mass*center[0]
    # The off diagonals of the moment of inertia section are assumed to be zero for right now, since we're initially assuming cylinders
    
    #>>>>>>>>>>>>>>>>>>>>>>>> double check this is the way to do this since "center" is the CoG about the global coordinates<<<<<<<<<<<<<<<<<<<<<<
        
    # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices
    W_struc += translateForce3to6DOF( mem.rA, np.array([0,0, -g*mass]) )  # weight vector
    M_struc += translateMatrix6to6DOF(mem.rA, Mmat)                       # mass/inertia matrix

    # @shousner, fill in _struc arrays above


    # -------------------- get each member's buoyancy/hydrostatic properties -----------------------
    
    Fvec, Cmat = mem.getHydrostatics()  # call to Member method for hydrostatic calculations
    
    # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices <<<<< needs updating (already about PRP)
    W_hydro += Fvec # translateForce3to6DOF( mem.rA, np.array([0,0, Fz]) )  # weight vector
    C_hydro += Cmat # translateMatrix6to6DOF(mem.rA, Cmat)                       # hydrostatic stiffness matrix

        
        

# ----------------------------- any solving for operating point goes here ----------------------

# solve for mean offsets (operating point)
Fthrust = 800e3       # N
Mthrust = 100*Fthrust  # N-m

# >>> W and K for struc and hydro will come into play here to find mean pitch angle <<<



# ------- quas-static mooring analysis ------------------
# Initial attempt by @shousner

from MoorPy import *
# >>>>>>>>>>>>>>>>>>>>> MoorPy was copied over into FD repo so it's not the most updated MoorPy (idk how else to import it) <<<<<<<<<<<<<<<<<<

# Create list variables of points and lines to describe the layout of the mooring system
# ANCHOR/BODY POINTS
PointList=[] # Makes a list variable to hold a list of all the points

# Before the depth was 40, radius was 30. Now, depth is 200, so radius is 150
anchorR = 150
angle1 = np.pi      # angle of mooring line wrt positive x positive y
angle2 = np.pi/3
angle3 = -np.pi/3

#                 number, type,      location,              external force
# =============================================================================
# PointList.append(Point(1, 1, np.array([anchorR*np.cos(angle1), anchorR*np.sin(angle1), -depth]), np.array([0, 0, 0],dtype=float)))
# PointList.append(Point(2, 1, np.array([anchorR*np.cos(angle2), anchorR*np.sin(angle2), -depth]), np.array([0, 0, 0],dtype=float)))
# PointList.append(Point(3, 1, np.array([anchorR*np.cos(angle3), anchorR*np.sin(angle3), -depth]), np.array([0, 0, 0],dtype=float)))
# =============================================================================
PointList.append(Point(1, 1, np.array([-30, 0, -40]), np.array([0, 0, 0],dtype=float)))
PointList.append(Point(2, 1, np.array([15, 25, -40]), np.array([0, 0, 0],dtype=float)))
PointList.append(Point(3, 1, np.array([15, -25, -40]), np.array([0, 0, 0],dtype=float)))
PointList.append(Point(4, 0, np.array([0, 0, 0]), np.array([0, 0, 50276.2],dtype=float)))
# Calls the Point class to create a Point object to append to the Point list with the given inputs

# MOORING LINES
LineTypes = {}  # create empty dictionary for line types
LineList=[] # Makes a list variable to hold a list of all the mooring lines
#                           LineType, diameter [m], mass density in air [kg/m], extensional stiffness EA [N]
LineTypes['main'] = LineType('main', 0.02, 40, 100e6) # Describes the parameters of the mooring lines used. 40 = mlin = used to calculate w = [N/m]
# Calls the LineType class to create a LineType object to put into the LineTypes dictionary

#  number, unstretched length, it's LineType, number of segments in the line
# =============================================================================
# LineList.append(Line(1, 100, LineTypes['main'], nSegs=6))
# LineList.append(Line(2, 100, LineTypes['main'], nSegs=6))
# LineList.append(Line(3, 100, LineTypes['main'], nSegs=6)) # Goes through the init function of the Line class to initialize more variables
# =============================================================================
LineList.append(Line(1, 50, LineTypes['main'], nSegs=6))
LineList.append(Line(2, 50, LineTypes['main'], nSegs=6))
LineList.append(Line(3, 50, LineTypes['main'], nSegs=6)) # Goes through the init function of the Line class to initialize more variables
# Calls the Line class to create a Line object to append to the Line list with the given inputs

# attach ends -- calls the addLine function of the Point class with inputs of (the line number that is attached to it, and which one is endB)
# addLine also prints out to which point that line was attached to
PointList[0].addLine(1, 0)
PointList[3].addLine(1, 1)

PointList[1].addLine(2, 0)
PointList[3].addLine(2, 1)

PointList[2].addLine(3, 0)
PointList[3].addLine(3, 1)


# save room for new Body class



# initialize mooring system with provided positions	
for Point in PointList:
	Point.setPosition(Point.r, LineList)
# For each point in the Point List, call the setPosition function in the Point class that takes the point's position and the list of Lines as input
# That function loops through each Line in the Line list to set the Line's end position in setEndPosition which takes the point's location, and endB as input
# If endB is 0, which it is for the bottom seafloor points, setEndPosition sets self.rA = r = position of the point.
# If endB is 1, which it is for the top floating point, setEndPosition sets self.rB = r = position of the point.
# So it essentially creates the rA or rB value of the Line as the point position

# Run the staticSolve function for each mooring line
for Line in LineList:
	Line.staticSolve(depth)
# solves for the forces in the x, y, and z directions for each end of the line A and B, fA and fB
    
# For each point, call the getForces function
for Point in PointList:
	f = Point.getForces(LineList)
	print(f)
# getForces creates a variable f to start off with the given external forces on the point. THIS IS WHERE YOU SHOULD ADD THE WEIGHT/OTHER FORCES!!!!!
# It then adds on to f, either fA or fB, depending on the endB variable of the point. Pretty nifty whoever wrote this
    
# draw initial mooring system <<<<<<< turn this into a single function at some point
fig = plt.figure(figsize=(20/2.54,12/2.54))
ax = Axes3D(fig)

for Line in LineList:
	Line.DrawLine(0, ax)
	
ax.set_title("Mooring system at initialization")





# MooringEq is an error function used in solving static equilibrium
def MooringEq(X):

	i = 0 # index used to split off input positions X for each free object	
	
	f = np.zeros(len(X)) # blank list to hold net forces on all free Points 
	
	# update position of free Points
	for Point in PointList:
		if Point.type==0:
			Point.setPosition(X[i:i+3], LineList)  # update position of free Point
            # move the position of the point to the input of point X aka move the free point to location X
            # this also adjusts the ends of the lines accordingly
			i += 3 #use this for when you have more than one free point
			
	# solve profile and forces of all lines
	for Line in LineList:
		Line.staticSolve(depth)
	
	# calculate net forces on free Points (combines the static solver with the external forces)
	for Point in PointList:
		if Point.type==0:
			i -= 3
			f[i:i+3] = Point.getForces(LineList) # get residual forces on free point
	
	#Fthrust = 800e3       # N
	#Mthrust = 100*Fthrust  # N-m
	#Thrust = np.array([Fthrust, 0, 0])#, 0, Mthrust, 0])
	#Fvec = np.zeros(3)
	#W_struc = np.array([0, 0, -1.73143989e7])#, -1.03990907e8, 1.03990907, 0])
	#F = f+Thrust+Fvec+W_struc
	return f
# This returns the forces in x, y, and z that act on the free point due to the mooring system

# Now, we want to find the location that the free point will want to go to so that the net forces = 0 -> use the fsolve function

# set up initial positions vector of free objects
X0 = []
for Point in PointList:
	if Point.type==0:
		X0 += [Point.r[0], Point.r[1], Point.r[2]]  # add free Point position to vector
	
X0 = np.array(X0)

X1 = fsolve(MooringEq, X0, epsfcn=0.2)


print(X1)  # output the final coordinates of the free Point objects in the mooring system

# This is different from when running MoorPy_Example even though I have all the same input variables. To-Do

# =============================================================================
# K = np.zeros([3,3])
# dx = 0.01
# for i in range(len(K)):
#     X2 = np.array([X1[0],X1[1],X1[2]])
#     X2[i] = X2[i]+dx
#     K[:,i] = (MooringEq(X2)-MooringEq(X1))/dx
# # LINEAR, SO ONLY FOR SMALL PERTURBATIONS
# =============================================================================







# ------------------------- get wave kinematics along each member ---------------------------------

# loop through each member
for mem in memberList:
    
    # loop through each node of the member
    for il in range(mem.n):
        
        # function to get wave kinematics spectra given a certain spectrum and location
        mem.u[il,:,:], mem.ud[il,:,:] = getWaveKin(zeta[imeto,:], w, k, depth, mem.r[il,:])    



# ------------------- solve for platform dynamics, iterating until convergence --------------------


nIter = 2  # maximum number of iterations to allow

# start fixed point iteration loop for dynamics
for iiter in range(nIter):

    # ------ calculate linearized coefficients within iteration ------- <<< some of this could likely only be done once...>>>


    # loop through each member
    for mem in memberList:
        
        q, p1, p2 = mem.getDirections()                # get unit direction vectors
        
        # loop through each node of the member
        for il in range(mem.n):
            
            # node displacement, velocity, and acceleration (each [3 x nw])
            drnode, vnode, anode = getVelocity(mem.r[il,:], Xi, w)      # get node complex velocity spectrum based on platform motion's and relative position from PRP
            
            # water relative velocity over node (complex amplitude spectrum)  [3 x nw]
            vrel = mem.u[il,:] - vnode
            
            # break out velocity components in each direction relative to member orientation [nw]
            vrel_q  = vrel* q[:,None]
            vrel_p1 = vrel*p1[:,None]
            vrel_p2 = vrel*p2[:,None]
            
            # get RMS of relative velocity component magnitudes (real-valued)
            vRMS_q  = np.linalg.norm( np.abs(vrel_q ) )  # equivalent to np.sqrt( np.sum( np.abs(vrel_q )**2) /nw)
            vRMS_p1 = np.linalg.norm( np.abs(vrel_p1) )
            vRMS_p2 = np.linalg.norm( np.abs(vrel_p2) )
            
            # linearized damping coefficients in each direction relative to member orientation [not explicitly frequency dependent...] (this goes into damping matrix)
            Bprime_q  = np.sqrt(8/np.pi) * vRMS_q  * 0.5*rho * np.pi*mem.d[il]*mem.dl * mem.Cd_q 
            Bprime_p1 = np.sqrt(8/np.pi) * vRMS_p1 * 0.5*rho * mem.d[il]*mem.dl * mem.Cd_p1
            Bprime_p2 = np.sqrt(8/np.pi) * vRMS_p2 * 0.5*rho * mem.d[il]*mem.dl * mem.Cd_p2
            
            # convert to global orientation
            Bmat = Bprime_q*VecVecTrans(q) + Bprime_p1*VecVecTrans(p1) + Bprime_p2*VecVecTrans(p2)
            
            # excitation force based on linearized damping coefficients [3 x nw]
            F_exc_drag = np.zeros([3, nw], dtype=complex)  # <<< should set elsewhere <<<
            for i in range(nw):
                F_exc_drag[:,i] = np.matmul(Bmat, mem.u[il,:,i])  # get local 3d drag excitation force complex amplitude for each frequency
            
            
            # added mass...
            Amat = rho*0.25*np.pi*mem.d[il]**2*mem.dl *( mem.Ca_q*VecVecTrans(q) + mem.Ca_p1*VecVecTrans(p1) + mem.Ca_p2*VecVecTrans(p2) )
            
            
            # inertial excitation...
            Imat = rho*0.25*np.pi*mem.d[il]**2*mem.dl * ( (1+mem.Ca_q)*VecVecTrans(q) + (1+mem.Ca_p1)*VecVecTrans(p1) + (1+mem.Ca_p2)*VecVecTrans(p2) )
            
            F_exc_inert = np.zeros([3, nw], dtype=complex)  # <<< should set elsewhere <<<
            for i in range(nw):
                F_exc_inert[:,i] = np.matmul(Imat, mem.ud[il,:,i])  # get local 3d inertia excitation force complex amplitude for each frequency
            
            
            for i in range(nw):
            
                # non-frequency-dependent matrices
                #translateMatrix3to6DOF(mem.r[il,:], )
                B_hydro[:,:,i] += translateMatrix3to6DOF(mem.r[il,:], Bmat)
                A_hydro[:,:,i] += translateMatrix3to6DOF(mem.r[il,:], Amat)
                
                # frequency-dependent excitation vector
                F_hydro[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_drag[:,i])
                F_hydro[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_inert[:,i])



    # ----------------------------- solve matrix equation of motion ------------------------------
    
    for ii in range(nw):          # loop through each frequency
        
        
        # sum contributions for each term        
        M_tot[:,:,ii] = M_struc + A_hydro[:,:,ii] + A_moor[:,:,ii]        # mass
        B_tot[:,:,ii] = B_struc + B_hydro[:,:,ii] + B_moor[:,:,ii]        # damping
        C_tot[:,:,ii] = C_struc + C_hydro + C_moor                        # stiffness
        F_tot[:,  ii] = F_hydro[:,ii]                                     # excitation force (complex amplitude)
        
        
        # form impedance matrix
        Z[:,:,ii] = -w[ii]**2 * M_tot[:,:,ii] + 1j*w[ii]*B_tot[:,:,ii] + C_tot[:,:,ii];
        
        # solve response (complex amplitude)
        Xi[:,ii] = np.matmul(np.linalg.inv(Z[:,:,ii]),  F_tot[:,ii] )
    
    
    '''

    #Xi{imeto} = rao{imeto}.*repmat(sqrt(S(:,imeto)),1,6); # complex response!
    
    #aNacRAO{imeto} = -(w').^2 .* (rao{imeto}(:,1) + hNac*rao{imeto}(:,5));      # Nacelle Accel RAO
    #aNac2(imeto) = sum( abs(aNacRAO{imeto}).^2.*S(:,imeto) ) *(w(2)-w(1));     # RMS Nacelle Accel

    
    # ----------------- convergence check --------------------
    conv = abs(aNac2(imeto)/aNac2last - 1);
    #disp(['at ' num2str(iiter) ' iterations - convergence is ' num2str(conv)])
    if conv < 0.0001

         
         break
    else
         aNac2last = aNac2(imeto);
    end
    
    '''

# ------------------------------ preliminary plotting of response ---------------------------------

fig, ax = plt.subplots(2,1, sharex=True)

ax[0].plot(w/2/np.pi, np.abs(Xi[0,:]), 'b', label="surge")
ax[0].plot(w/2/np.pi, np.abs(Xi[1,:]), 'g', label="sway")
ax[0].plot(w/2/np.pi, np.abs(Xi[2,:]), 'r', label="heave")
ax[1].plot(w/2/np.pi, np.abs(Xi[3,:]), 'b', label="roll")
ax[1].plot(w/2/np.pi, np.abs(Xi[4,:]), 'g', label="pitch")
ax[1].plot(w/2/np.pi, np.abs(Xi[5,:]), 'r', label="yaw")

ax[0].legend()
ax[1].legend()

ax[0].set_ylabel("response magnitude (m)")
ax[1].set_ylabel("response magnitude (rad)")
ax[1].set_xlabel("frequency (Hz)")

plt.show()

 
 # ---------- mooring line fairlead tension RAOs and constraint implementation ----------
'''
 
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
    