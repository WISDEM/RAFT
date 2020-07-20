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
        self.dA  = np.float(entries[2])                      # diameter of (lower) node
        self.dB  = np.float(entries[2])
        self.rA = np.array(entries[3:6], dtype=np.double)  # x y z of lower node
        self.rB = np.array(entries[6:9], dtype=np.double)
        self.t  = 0.06           # shell thickness [m]
        
        self.l_fill = 0                    # length of member (from end A to B) filled with ballast [m]
        self.rho_fill = 1025               # density of ballast in member [kg/m^3]
        
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
        self.d  = np.zeros( self.n   )  # local diameter along member [m]
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
    
    
    
    def getInertia(self):
        
        # Volume of steel based on the shell thickness [m^3]
        dAi = self.dA - 2*self.t
        dBi = self.dB - 2*self.t
        V_outer = (np.pi/4)*(1/3)*(self.dA**2+self.dB**2+self.dA*self.dB)*self.l
        V_inner = (np.pi/4)*(1/3)*(dAi**2+dBi**2+dAi*dBi)*self.l
        v_steel = V_outer - V_inner         #[m^3] Volume of steel of the member  ((self.t/2)*(self.dA+self.dB)-self.t**2)*np.pi*self.l
        
        # Ballast (future work - this doesn't account for any angle in the member. If the member is horizontal, the ballast doesn't follow gravity)
        dB_fill = (dBi-dAi)*(self.l_fill/self.l) + dAi       # interpolated diameter of member where the ballast is filled to
        v_fill = (np.pi/4)*(1/3)*(dAi**2+dB_fill**2+dAi*dB_fill)*self.l_fill    #[m^3]
        m_fill = self.rho_fill*v_fill                                           #[kg]
        
        # Center of mass
        hco = self.l*((self.dA**2 + 2*self.dA*self.dB + 3*self.dB**2)/(4*(self.dA**2 + self.dA*self.dB + self.dB**2)))  
        hci = self.l*((dAi**2 + 2*dAi*dBi + 3*dBi**2)/(4*(dAi**2 + dAi*dBi + dBi**2)))
        hc_shell = ((hco*V_outer)-(hci*V_inner))/(V_outer-V_inner)  # [m] CoG of member shell in relation to bottom node @ self.rA
        
        hc_fill = self.l_fill*((dAi**2 + 2*dAi*dB_fill + 3*dB_fill**2)/(4*(dAi**2 + dAi*dB_fill + dB_fill**2)))
        
        hc = ((hc_fill*self.rho_fill*v_fill)+(hc_shell*rho_steel*v_steel))/((self.rho_fill*v_fill)+(rho_steel*v_steel))
        
        center = self.rA + (self.q*hc)

        # Moment of Inertia (equations from HydroDyn paper)
        # Calc I@end for outer solid - Calc I@end for "inner" solid - I_outer-I_inner = I_shell @end - PA theorem to calc I@CoG
        r1 = self.dA/2
        r2 = self.dB/2
        m = (r2-r1)/self.l
        r1i = (self.dA/2)-self.t
        r2i = (self.dB/2)-self.t
        mi = (r2i-r1i)/self.l
        if m==0:
            Ir_end_outer = (1/12)*(rho_steel*self.l*np.pi*r1**2)*(3*r1**2 + 4*self.l**2) #[kg-m^2]    about end node
            Ir_end_inner = (1/12)*(rho_steel*self.l*np.pi*r1i**2)*(3*r1i**2 + 4*self.l**2) #[kg-m^2]  about end node
            Ir_end = Ir_end_outer - Ir_end_inner                     # I_outer - I_inner = I_shell -- about end node
            I_rad = Ir_end - (rho_steel*v_steel)*hc**2   # about CoG
            
            I_ax_outer = (1/2)*rho_steel*np.pi*self.l*(r1**4)
            I_ax_inner = (1/2)*rho_steel*np.pi*self.l*(r1i**4)
            I_ax = I_ax_outer - I_ax_inner
        else:
            Ir_tip_outer = abs((np.pi/20)*(rho_steel/m)*(1+(4/m**2))*(r2**5-r1**5))                                 # outer, about tip
            Ir_end_outer = abs(Ir_tip_outer - ((rho_steel/(3*m**2))*np.pi*(r2**3-r1**3)*((r1/m)+2*hc)*r1))          # outer, about node
            Ir_tip_inner = abs((np.pi/20)*(rho_steel/mi)*(1+(4/mi**2))*(r2i**5-r1i**5))                             # inner, about tip
            Ir_end_inner = abs(Ir_tip_inner - ((rho_steel/(3*mi**2))*np.pi*(r2i**3-r1i**3)*((r1i/mi)+2*hc)*r1i))    # inner, about node
            Ir_end = Ir_end_outer - Ir_end_inner                                                                    # shell, about node
            I_rad = Ir_end - (rho_steel*v_steel)*hc**2                                                              # shell, about CoG by PAT
            
            I_ax_outer = (rho_steel*np.pi/(10*m))*(r2**5-r1**5)
            I_ax_inner = (rho_steel*np.pi/(10*mi))*(r2i**5-r1i**5)
            I_ax = I_ax_outer - I_ax_inner

        return v_steel, center, I_rad, I_ax, m_fill
        
    
    
    def getHydrostatics(self):
        '''Calculates member hydrostatic properties, namely buoyancy and stiffness matrix'''
        
            
        # partially submerged case
        if self.r[0,2]*self.r[-1,2] <= 0:    # if member crosses (or touches) water plane
                
            # angles
            beta = np.arctan2(q[1],q[0])  # member incline heading from x axis
            phi  = np.arctan2(np.sqrt(q[0]**2 + q[1]**2), q[2])  # member incline angle from vertical
            
            # precalculate trig functions
            cosPhi=np.cos(phi)
            sinPhi=np.sin(phi)
            tanPhi=np.tan(phi)
            cosBeta=np.cos(beta)
            sinBeta=np.sin(beta)
            tanBeta=sinBeta/cosBeta
                
            # -------------------- buoyancy and waterplane area properties ------------------------
                
            dWP = np.interp(0, self.r[:,2], self.d)       # diameter of member where its axis crosses the waterplane 
            xWP = np.interp(0, self.r[:,2], self.r[:,0])  # x coordinate where member axis cross the waterplane [m]
            yWP = np.interp(0, self.r[:,2], self.r[:,1])  # y coordinate where member axis cross the waterplane [m]
            AWP = (np.pi/4)*dWP**2                        # waterplane area of member [m^2]
            IWP = (np.pi/64)*dWP**4                       # waterplane moment of inertia [m^4] approximates the waterplane area as the shape of a circle
            
            LWP = abs(self.r[0,2])/cosPhi                 # get length of member that is still underwater. Assumes self.r is about global coords -> z=0 @ SWL
            
            # Total enclosed underwater volume
            V_UW = (np.pi/4)*(1/3)*(self.dA**2+dWP**2+self.dA*dWP)*LWP       #[m^3] 
            
            L_center = TaperCV(0.5*self.dA, 0.5*dWP, LWP) # distance from end A to center of buoyancy of member [m]
            
            r_center = self.rA + self.q*L_center          # absolute coordinates of center of volume [m]
        
            
            
        
            # >>>> question: should this function be able to use displaced/rotated values? <<<<
            
            # ------------- get hydrostatic derivatives ---------------- 
            
            # derivatives from global to local 
            dPhi_dThx  = -sinBeta                     # \frac{d\phi}{d\theta_x} = \sin\beta
            dPhi_dThy  =  cosBeta
            #dBeta_dThx = -cosBeta/tanBeta**2
            #dBeta_dThy = -sinBeta/tanBeta**2
            
            # note: below calculations are based on untapered case, but 
            # temporarily approximated for taper by using dWP (diameter at water plane crossing) <<< this is rough
            
            # buoyancy force and moment about end A
            Fz = rho*g* V_UW
            M  = -rho*g*pi*( dWP**2/32*(2.0 + tanPhi**2) + 0.5*(self.rA[2]/cosPhi)**2)*sinPhi  # moment about axis of incline
            Mx = M*dPhi_dThx
            My = M*dPhi_dThy
            
            Fvec = np.zeros(6)                         # buoyancy force (and moment) vector (about PRP)
            Fvec[2] = Fz                               # vertical buoyancy force [N]
            Fvec[3] = Mx + Fz*self.rA[1]                # moment about x axis [N-m]
            Fvec[4] = My - Fz*self.rA[0]                # moment about y axis [N-m]
            
            # derivatives aligned with incline heading
            dFz_dz   = -rho*g*pi*0.25*dWP**2                  /cosPhi
            dFz_dPhi =  rho*g*pi*0.25*dWP**2*self.rA[2]*sinPhi/cosPhi**2
            dM_dz    =  1.0*dFz_dPhi
            dM_dPhi  = -rho*g*pi*0.25*dWP**2 * (dWP**2/32*(2*cosPhi + sinPhi**2/cosPhi + 2*sinPhi/cosPhi**2) + 0.5*self.rA[2]**2*(sinPhi**2+1)/cosPhi**3)
            
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
            
            
            
            # fill in stiffness matrix
            Cmat = np.zeros([6,6]) # hydrostatic stiffness matrix (about PRP)
            '''
            Cmat[2,2] = -dFz_dz
            Cmat[2,3] = -dFz_dThx
            Cmat[2,4] = -dFz_dThy
            Cmat[3,2] = -dMx_dz
            Cmat[4,2] = -dMy_dz   # ignoring symmetries for now, as a way to check equations
            Cmat[3,3] = -dMx_dThx
            Cmat[3,4] = -dMx_dThy
            Cmat[4,3] = -dMy_dThx
            Cmat[4,4] = -dMy_dThy
            '''
            # normal approach to hydrostatic stiffness, using this temporarily until above fancier approach is verified
            Iwp = np.pi*dWP**4/64 # [m^4] Moment of Inertia of the waterplane
            Iwp = np.pi*dWP**4/64 # [m^4] Moment of Inertia of the waterplane
            Cmat[2,2] = -dFz_dz
            Cmat[2,3] = rho*g*(     -AWP*yWP    )
            Cmat[2,4] = rho*g*(      AWP*xWP    )
            Cmat[3,2] = rho*g*(     -AWP*yWP    )
            Cmat[3,3] = rho*g*(IWP + AWP*yWP**2 )
            Cmat[3,4] = rho*g*(      AWP*xWP*yWP)
            Cmat[4,2] = rho*g*(      AWP*xWP    )
            Cmat[4,3] = rho*g*(      AWP*xWP*yWP)
            Cmat[4,4] = rho*g*(IWP + AWP*xWP**2 )
            
            
        
        # fully submerged case 
        else:
        
            AWP = 0
            IWP = 0
            xWP = 0
            yWP = 0
        
            V_UW  = TaperV( 0.5*self.dA, 0.5*self.dB, self.l)  # displaced volume of member [m^3]
            
            alpha = TaperCV(0.5*self.dA, 0.5*self.dB, 1.0)  # relative location of center of volume from end A (0) to B (1)
            
            r_center = self.rA*(1.0-alpha) + self.rB*alpha  # absolute coordinates of center of volume [m]
        
            # buoyancy force (and moment) vector
            Fvec = translateForce3to6DOF( r_center, np.array([0, 0, rho*g*V_UW]) ) 
  
            # hydrostatic stiffness matrix (about end A)
            Cmat = np.zeros([6,6])  
            Cmat[3,3] = rho*g*V_UW * r_center[2]
            Cmat[4,4] = rho*g*V_UW * r_center[2]
            
        return Fvec, Cmat, V_UW, r_center, AWP, IWP, xWP, yWP


def TaperV(R1, R2, H):
    '''returns the volume of a cylindrical section, possibly with taper'''
    
    if R1 == R2:             # if just a cylinder
        return np.pi*R1*R1*H
        #taperCV = H/2.0

    elif R1 == 0:             # seperate this case out because it gives a divide by zero in general formula
        return 1./3.*np.pi*R2*R2*H;                                            # cone volume
        #taperCV = 3./4.*H                                                     # from base
    
    else:
        coneH = H/(1.-R2/R1);                                                  # conical height
        coneV = 1./3.*np.pi*R1*R1*coneH;                                       # cone volume
        coneVtip = 1./3.*np.pi*R2*R2*(coneH-H);                                # height from end of taper to cone tip
        return coneV-coneVtip;                                                 # taper volume
        #taperCV = ( coneV*1./4.*coneH - coneVtip*(1./4.*(coneH-H) + H) )/ taperV # from base
    
    return taperV
    
    
def TaperCV(R1, R2, H):
    '''returns the height of the center of buoyancy from the lower node'''
    return H*(R1**2 + 2*R1*R2 + 3*R2**2)*0.25/(R1**2 + R1*R2 + R2**2)
    

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
    
    


# ------------------------------- basic setup -----------------------------------------

nDOF = 6

w = np.arange(.01, 2, 0.01)  # angular frequencies tp analyze (rad/s)
nw = len(w)  # number of frequencies

k= np.zeros(nw)  # wave number


# ----------------------- member-based platform description --------------------------

# (hard-coded for now - set to DeepCwind Semi geometry - eventually these will be provided as inputs instead)

# list of member objects
memberList = []

#                    number  type  diameter  xa        ya      za     xb        yb     zb
memberList.append(Member("1     x    12.0   14.43376  25.0   -14.0   14.43376  25.0   0.0  ", nw))
memberList.append(Member("2     x    12.0  -28.86751   0.0   -14.0  -28.86751   0.0   0.0  ", nw))
memberList.append(Member("3     x    12.0   14.43376 -25.0   -14.0   14.43376 -25.0   0.0  ", nw))
memberList.append(Member("4     x    24.0   14.43376  25.0   -20.0   14.43376  25.0  -14.0  ", nw))
memberList.append(Member("5     x    24.0  -28.86751   0.0   -20.0  -28.86751   0.0  -14.0  ", nw))
memberList.append(Member("6     x    24.0   14.43376 -25.0   -20.0   14.43376 -25.0  -14.0  ", nw))
memberList.append(Member("7     x     6.5    0.0       0.0   -20.0    0.0       0.0   0.0  ", nw))





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

'''
plt.plot(w/np.pi/2,  S[2,:], "r")
plt.plot(w/np.pi/2, S2[2,:], "b")
plt.xlabel("Hz")
'''

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

# initialize some variables for running totals
VTOT = 0.                   # Total underwater volume of all members combined
mTOT = 0.                   # Total mass of all members [kg]
AWP_TOT = 0.                # Total waterplane area of all members [m^2]
IWPx_TOT = 0                # Total waterplane moment of inertia of all members about x axis [m^4]  
IWPy_TOT = 0                # Total waterplane moment of inertia of all members about y axis [m^4]  
Sum_V_rCB = np.zeros(3)     # product of each member's buoyancy multiplied by center of buoyancy [m^4]
Sum_AWP_rWP = np.zeros(2)   # product of each member's waterplane area multiplied by the area's center point [m^3]
Sum_M_center = np.zeros(3)  # product of each member's mass multiplied by its center of mass [kg-m] (Only considers the shell mass right now)

# loop through each member
for mem in memberList:
    
    q, p1, p2 = mem.getDirections()                # get unit direction vectors
    
    
    # ---------------------- get member's mass and inertia properties ------------------------------
    rho_steel = 7850 #[kg/m^3]

    v_steel, center, I_rad, I_ax, m_fill = mem.getInertia() # calls the getInertia method to calcaulte values
    
    mass = v_steel*rho_steel + m_fill #[kg]
    Mmat = np.diag([mass, mass, mass, I_rad, I_rad, I_ax]) # MOI matrix = Mmat[3:,3:] is 0 on off diags bc symmetry in cylinders
    # @mhall: Mmat as written above is the mass and inertia matrix about the member CG...@shousner: you betcha
  
    # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices
    W_struc += translateForce3to6DOF( center, np.array([0,0, -g*mass]) )  # weight vector
    M_struc += translateMatrix6to6DOF(center, Mmat)                       # mass/inertia matrix
    # @mhall: Using the diagonal Mmat, and calling the above function with the "center" coordinate, will give the mass/inertia about the PRP!
    # @shousner: center is the position vector of the CG of the member, from the global coordinates aka PRP
    
    
    # -------------------- get each member's buoyancy/hydrostatic properties -----------------------
    
    Fvec, Cmat, V_UW, r_CB, AWP, IWP, xWP, yWP = mem.getHydrostatics()  # call to Member method for hydrostatic calculations
    
    # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices <<<<< needs updating (already about PRP)
    W_hydro += Fvec # translateForce3to6DOF( mem.rA, np.array([0,0, Fz]) )  # weight vector
    C_hydro += Cmat # translateMatrix6to6DOF(mem.rA, Cmat)                       # hydrostatic stiffness matrix
    VTOT    += V_UW    # add to total underwater volume of all members combined
    AWP_TOT += AWP
    IWPx_TOT += IWP + AWP*yWP**2
    IWPy_TOT += IWP + AWP*xWP**2
    Sum_V_rCB   += r_CB*V_UW
    Sum_AWP_rWP += np.array([xWP, yWP])*AWP
    Sum_M_center += center*mass

# ----------- process key hydrostatic-related totals for use in static equilibrium solution ------------------

mTOT = M_struc[0,0]
rCG_TOT = Sum_M_center/mTOT 

rCB_TOT = Sum_V_rCB/VTOT       # location of center of buoyancy on platform

zMeta   = rCB_TOT[2] + IWPx_TOT/VTOT  # add center of buoyancy and BM=I/v to get z elevation of metecenter [m] (have to pick one direction for IWP)
        
        

# ----------------------------- any solving for operating point goes here ----------------------

# solve for mean offsets (operating point)
Fthrust = 800e3       # N
Mthrust = 100*Fthrust  # N-m

# >>> W and K for struc and hydro will come into play here to find mean pitch angle <<<



# ---------------------------- quasi-static mooring analysis ---------------------------------------

#from MoorPy import *
import MoorPy as mp
from scipy.optimize import fsolve
# >>>>>>>>>>>>>>>>>>>>> MoorPy was copied over into FD repo so it's not the most updated MoorPy (idk how else to import it) <<<<<<<<<<<<<<<<<<

# This is initially hardcoded in but will eventually be able to read through an input file of mooring parameters
# Include weight, buoyancy, and thrust forces.

# ----------Initialization of the Mooring System Lines, Points, Bodies
BodyList=[] # Makes a list variable to hold a list of all the bodies in the system

# Create Body represent FOWT in MoorPy
#               number, type, xyz-roll-pitch-yaw position vector,   mass [kg], volume [m^3], center of gravity position vector, waterplane area, metacenter position vector
BodyList.append(mp.Body(1, 0, np.zeros(6), m=mTOT, v=VTOT, rCG=rCG_TOT, AWP=AWP_TOT, rM=np.array([0,0,zMeta])))

BodyList[0].f6Ext = np.array([Fthrust,0,0, 0,Mthrust,0])  # apply wind thrust force and moment on Body



anchorR = 150
angle = np.array([np.pi, np.pi/3, -np.pi/3]) # angle of mooring line wrt positive x positive y

PointList=[] # Makes a list variable to hold a list of all the attachment points of the mooring system
#                 number, type,      location,              external force
PointList.append(mp.Point(1, 1, np.array([anchorR*np.cos(angle[0]), anchorR*np.sin(angle[0]), -depth], dtype=float), np.zeros(3)))
PointList.append(mp.Point(2, 1, np.array([anchorR*np.cos(angle[1]), anchorR*np.sin(angle[1]), -depth], dtype=float), np.zeros(3)))
PointList.append(mp.Point(3, 1, np.array([anchorR*np.cos(angle[2]), anchorR*np.sin(angle[2]), -depth], dtype=float), np.zeros(3)))
# =============================================================================
# PointList.append(mp.Point(1, 1, np.array([-30, 0, -40], dtype=float), np.array([0, 0, 0], dtype=float)))
# PointList.append(mp.Point(2, 1, np.array([15, 25, -40], dtype=float), np.array([0, 0, 0], dtype=float)))
# PointList.append(mp.Point(3, 1, np.array([15, -25, -40], dtype=float), np.array([0, 0, 0], dtype=float)))
# =============================================================================
# Calls the Point class to create a Point object to append to the Point list with the given inputs

BodyID = 1

rRel = np.array([-2, 0, 0], dtype=float)
BodyList[BodyID-1].addPoint(4, rRel)
PointList.append(mp.Point(4, 1, rRel, np.array([0, 0, 0], dtype=float)))

rRel = np.array([1, 1, 0], dtype=float)
BodyList[BodyID-1].addPoint(5, rRel)
PointList.append(mp.Point(5, 1, rRel, np.array([0, 0, 0], dtype=float)))

rRel = np.array([1, -1, 0], dtype=float)
BodyList[BodyID-1].addPoint(6, rRel)
PointList.append(mp.Point(6, 1, rRel, np.array([0, 0, 0], dtype=float)))


LineTypes = {}  # create empty dictionary for line types
LineList=[] # Makes a list variable to hold a list of all the mooring lines
#                   LineType name, diameter [m], mass density in air [kg/m], extensional stiffness EA [N]
LineTypes['main'] = mp.LineType('main', 0.02, 40, 100e6) # Mooring Line characteristics. massden used to calculate w = [N/m]
# Calls the LineType class in MoorPy to create a LineType object to put into the LineTypes dictionary

LineLength = 250.

#  number, unstretched length, it's LineType, number of segments in the line
LineList.append(mp.Line(np.int(1), LineLength, LineTypes['main'], nSegs=np.int(6)))
LineList.append(mp.Line(np.int(2), LineLength, LineTypes['main'], nSegs=np.int(6)))
LineList.append(mp.Line(np.int(3), LineLength, LineTypes['main'], nSegs=np.int(6))) # Goes through the init function of the Line class to initialize more variables that show up in LineList
# Calls the Line class to create a Line object to append to the Line list with the given inputs


# attach ends of the lines to points 
# calls the addLine function in the Point class i.e. PointList[point ref].addLine(sets a LineID#, sets a variable endB (to be used later))
# addLine also prints out to which point that line was attached to
PointList[0].addLine(1, 0) # attached Line 1 (1) to Point 1 (0); 
PointList[3].addLine(1, 1) # attached Line 1 (1) to Point 4 (3)

PointList[1].addLine(2, 0) # attached Line 2 (2) to Point 2 (1)
PointList[4].addLine(2, 1) # attached Line 2 (2) to Point 5 (4)

PointList[2].addLine(3, 0) # attached Line 3 (3) to Point 3 (2)
PointList[5].addLine(3, 1) # attached Line 3 (3) to Point 6 (5)
# ------------------------

#depth = 40.

# ---------------------Initialization of mooring system with provided positions	in space
for Body in BodyList:
    Body.setPosition(Body.r6, PointList, LineList)
    
for Point in PointList:
	Point.setPosition(Point.r, LineList)

for Line in LineList:
	Line.staticSolve(depth)

for Point in PointList:
    f = Point.getForces(LineList)
    print(f)
# getForces creates a variable f to start off with the given external forces on the point. THIS IS WHERE YOU SHOULD ADD THE WEIGHT/OTHER FORCES!!!!!
# It then adds on to f, either fA or fB, depending on the endB variable of the point.
    
for Body in BodyList:
    f = Body.getForces(PointList, LineList)
# getForces (Body) sums all the forces and moments on the body. Starts off with any buoyancy force from itself, does an rCG X F of it's weight at the CoG, which can result in 6dof moments
# then calculates the forces on each of the attached points due to mooring, and adds that to the force vector
# ---------------------  



# -------------------MooringEq is an error function used to solve for the static equilibrium
def MooringEq(X):

    i = 0 # index used to split off input positions X for each free object  
    
    f = np.zeros(len(X)) # blank list to hold net forces on all free Points 
    
    # update position of free Bodies
    for Body in BodyList:
        if Body.type==0:
            Body.setPosition(X[i:i+6], PointList, LineList)  # update position of free Body
            i += 6
            
    # update position of free Points
    for Point in PointList:
        if Point.type==0:
            Point.setPosition(X[i:i+3], LineList)  # update position of free Point
            # set the position of the free point to location X, which also adjusts the ends of the lines accordingly
            i += 3 #use this for when you have more than one free point
            
    # solve profile and forces of all lines
    for Line in LineList:
        Line.staticSolve(depth)
    
    # calculate net forces on free Points (combines the static solver with the external forces)
    for Point in PointList:
        if Point.type==0:
            i -= 3
            f[i:i+3] = Point.getForces(LineList) # get residual forces on free point
            
    # calculate net forces on free Bodies
    for Body in BodyList:
        if Body.type==0:
            i -= 6
            f[i:i+6] = Body.getForces(PointList, LineList) # get residual forces on free point
    
    return f
# -------------------




# -----set up initial positions vector of free objects
X0 = []

for Body in BodyList:
    if Body.type==0:
        X0 += [Body.r6[0], Body.r6[1], Body.r6[2], Body.r6[3],Body.r6[4],Body.r6[5]]  # add free Body position and orientation to vector
        
for Point in PointList:
    if Point.type==0:
        X0 += [Point.r[0], Point.r[1], Point.r[2]]  # add free Point position to vector
    
X0 = np.array(X0)
# -----

X1 = fsolve(MooringEq, X0, epsfcn=0.2) # Iterates through and finds the X values where the return of MooringEq (f) becomes zero 
# epsfcnfloat, optional: A suitable step length for the forward-difference approximation of the Jacobian (for fprime=None). If epsfcn is less than the machine precision, it is assumed that the relative errors in the functions are of the order of the machine precision.

print(X1)  # output the final coordinates of the free Point objects in the mooring system
    

# ---------------Stiffness Matrix (This is a linear calculation, so it will only be accurate for small perturbations)  
K = np.zeros([6,6])
dx = 0.01
for i in range(len(K)):
    X2 = np.array([X1[0],X1[1],X1[2],X1[3],X1[4],X1[5]])
    X2[i] = X2[i]+dx
    K[:,i] = (MooringEq(X2)-MooringEq(X1))/dx
# the mooring stiffness force is the matrix K times the deltaX or the new r vector away from point X1; K@X2 is incorrect. K@(X2-X1) is correct.
# ----------------
   

MooringEq(X1)



# C_moor = Body.getStiffness(PointList, LineList, X1, depth)
# @shousner: I almost have it. I feel like I have the right idea, I'm just doing something wrong in staticSolve







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
    