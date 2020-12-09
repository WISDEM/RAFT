# 2020-05-03: This is a start at a frequency-domain floating support structure model for WEIS-WISDEM.
#             Based heavily on the GA-based optimization framework from Hall 2013

# 2020-05-23: Starting as a script that will evaluate a design based on properties/data provided in separate files.


import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../MoorPy')
import MoorPy as mp

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
        self.type = np.int(entries[1])
        self.dA  = np.float(entries[2])                      # diameter of (lower) node
        self.dB  = np.float(entries[3])
        self.rA = np.array(entries[4:7], dtype=np.double)  # x y z of lower node
        self.rB = np.array(entries[7:10], dtype=np.double)
        self.t  = np.float(entries[10])           # shell thickness [m]
        
        self.l_fill = np.float(entries[11])                  # length of member (from end A to B) filled with ballast [m]
        self.rho_fill = np.float(entries[12])                # density of ballast in member [kg/m^3]
        
        self.rho_steel = 8500 #[kg/m^3]    <<< shell density <<< needs to become a variable
        
        rAB = self.rB-self.rA
        self.l = np.linalg.norm(rAB)  # member length
                
        self.q = rAB/self.l           # member axial unit vector
        self.p1 = np.zeros(3)              # member transverse unit vectors (to be filled in later)
        self.p2 = np.zeros(3)              # member transverse unit vectors
        
        
        self.Cd_q  = 0.1  # drag coefficients
        self.Cd_p1 = 0.6
        self.Cd_p2 = 0.6
        self.Ca_End = 0.6
        self.Ca_q  = 0.0  # added mass coefficients
        self.Ca_p1 = 0.97
        self.Ca_p2 = 0.97
        self.Ca_End = 0.6
                     
        
        self.n = 10 # number of nodes per member
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
        self.pDyn=np.zeros([self.n,  nw], dtype=complex)  # dynamic pressure
        
        
    def getDirections(self):
        '''Returns the direction unit vector for the member based on its end positions'''
        
        q  =  self.q
        p1 = np.cross( np.array([0,1,0]), q) # transverse unit direction vector in X-Z plane
        p2 = np.cross( q, p1 )               # the other transverse unit vector
        
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
        
        hc = ((hc_fill*self.rho_fill*v_fill)+(hc_shell*self.rho_steel*v_steel))/((self.rho_fill*v_fill)+(self.rho_steel*v_steel))
        
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
            Ir_end_outer = (1/12)*(self.rho_steel*self.l*np.pi*r1**2)*(3*r1**2 + 4*self.l**2) #[kg-m^2]    about end node
            Ir_end_inner = (1/12)*(self.rho_steel*self.l*np.pi*r1i**2)*(3*r1i**2 + 4*self.l**2) #[kg-m^2]  about end node
            Ir_end_steel = Ir_end_outer - Ir_end_inner                     # I_outer - I_inner = I_shell -- about end node
            #Ir_end_steel = (1/12)*v_steel*self.rho_steel*(3*(r1**2 + r1i**2) + 4*self.l**2)
            
            Ir_end_fill = (1/12)*(self.rho_fill*self.l_fill*np.pi*r1i**2)*(2*r1i**2 + 4*self.l_fill**2) #[kg-m^2]  about end node
            
            Ir_end = Ir_end_steel + Ir_end_fill
            
            I_rad = Ir_end - ((self.rho_steel*v_steel)+m_fill)*hc**2
            
            #I_rad_steel = Ir_end - (self.rho_steel*v_steel)*hc**2   # about CoG
            
            #I_rad_fill = Ir_end_fill - m_fill*hc**2  # about CoG
            #I_rad = I_rad_steel + I_rad_fill   # sum of all masses about the CoG
            
            I_ax_outer = (1/2)*self.rho_steel*np.pi*self.l*(r1**4)
            I_ax_inner = (1/2)*self.rho_steel*np.pi*self.l*(r1i**4)
            I_ax_steel = I_ax_outer - I_ax_inner
            I_ax_fill = (1/2)*self.rho_fill*np.pi*self.l_fill*(r1i**4)
            I_ax = I_ax_steel + I_ax_fill
        else:
            Ir_tip_outer = abs((np.pi/20)*(self.rho_steel/m)*(1+(4/m**2))*(r2**5-r1**5))                                 # outer, about tip
            Ir_end_outer = abs(Ir_tip_outer - ((self.rho_steel/(3*m**2))*np.pi*(r2**3-r1**3)*((r1/m)+2*hc)*r1))          # outer, about node
            Ir_tip_inner = abs((np.pi/20)*(self.rho_steel/mi)*(1+(4/mi**2))*(r2i**5-r1i**5))                             # inner, about tip
            Ir_end_inner = abs(Ir_tip_inner - ((self.rho_steel/(3*mi**2))*np.pi*(r2i**3-r1i**3)*((r1i/mi)+2*hc)*r1i))    # inner, about node
            Ir_end = Ir_end_outer - Ir_end_inner                                                                    # shell, about node
            I_rad_steel = Ir_end - (self.rho_steel*v_steel)*hc**2                                                        # shell, about CoG by PAT
            
            I_ax_outer = (self.rho_steel*np.pi/(10*m))*(r2**5-r1**5)
            I_ax_inner = (self.rho_steel*np.pi/(10*mi))*(r2i**5-r1i**5)
            I_ax_steel = I_ax_outer - I_ax_inner
            
            if self.l_fill == 0:
                I_rad_fill = 0
                I_ax_fill = 0
            else:
                r2_fill = dB_fill/2
                mi_fill = (r2_fill-r1i)/self.l_fill 
                Ir_tip_fill = abs((np.pi/20)*(self.rho_steel/mi_fill)*(1+(4/mi_fill**2))*(r2_fill**5-r1i**5))
                Ir_end_fill = abs(Ir_tip_fill - ((self.rho_fill/(3*mi_fill**2))*np.pi*(r2_fill**3-r1i**3)*((r1i/mi_fill)+2*hc)*r1i))    # inner, about node
                I_rad_fill = Ir_end_fill - m_fill*hc**2   # about CoG

                I_ax_fill = (self.rho_fill*np.pi/(10*mi_fill))*(r2_fill**5-r1i**5)
            
            I_rad = I_rad_steel + I_rad_fill # about CoG
            I_ax = I_ax_steel + I_ax_fill 

        return v_steel, center, I_rad, I_ax, m_fill
        
    
    
    def getHydrostatics(self, env):
        '''Calculates member hydrostatic properties, namely buoyancy and stiffness matrix'''
        
        pi = np.pi
            
        # partially submerged case
        if self.r[0,2]*self.r[-1,2] <= 0:    # if member crosses (or touches) water plane
                
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
            Fz = env.rho*env.g* V_UW
            M  = -env.rho*env.g*pi*( dWP**2/32*(2.0 + tanPhi**2) + 0.5*(self.rA[2]/cosPhi)**2)*sinPhi  # moment about axis of incline
            Mx = M*dPhi_dThx
            My = M*dPhi_dThy
            
            Fvec = np.zeros(6)                         # buoyancy force (and moment) vector (about PRP)
            Fvec[2] = Fz                               # vertical buoyancy force [N]
            Fvec[3] = Mx + Fz*self.rA[1]                # moment about x axis [N-m]
            Fvec[4] = My - Fz*self.rA[0]                # moment about y axis [N-m]
            
            # derivatives aligned with incline heading
            dFz_dz   = -env.rho*env.g*pi*0.25*dWP**2                  /cosPhi
            dFz_dPhi =  env.rho*env.g*pi*0.25*dWP**2*self.rA[2]*sinPhi/cosPhi**2
            dM_dz    =  1.0*dFz_dPhi
            dM_dPhi  = -env.rho*env.g*pi*0.25*dWP**2 * (dWP**2/32*(2*cosPhi + sinPhi**2/cosPhi + 2*sinPhi/cosPhi**2) + 0.5*self.rA[2]**2*(sinPhi**2+1)/cosPhi**3)
            
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
            Cmat[2,3] = env.rho*env.g*(     -AWP*yWP    )
            Cmat[2,4] = env.rho*env.g*(      AWP*xWP    )
            Cmat[3,2] = env.rho*env.g*(     -AWP*yWP    )
            Cmat[3,3] = env.rho*env.g*(IWP + AWP*yWP**2 )
            Cmat[3,4] = env.rho*env.g*(      AWP*xWP*yWP)
            Cmat[4,2] = env.rho*env.g*(      AWP*xWP    )
            Cmat[4,3] = env.rho*env.g*(      AWP*xWP*yWP)
            Cmat[4,4] = env.rho*env.g*(IWP + AWP*xWP**2 )
            
            
        
        # fully submerged case 
        elif self.r[0,2] <= 0 and self.r[-1,2] <= 0:
        
            AWP = 0
            IWP = 0
            xWP = 0
            yWP = 0
        
            V_UW  = TaperV( 0.5*self.dA, 0.5*self.dB, self.l)  # displaced volume of member [m^3]
            
            alpha = TaperCV(0.5*self.dA, 0.5*self.dB, 1.0)  # relative location of center of volume from end A (0) to B (1)
            
            r_center = self.rA*(1.0-alpha) + self.rB*alpha  # absolute coordinates of center of volume [m]
        
            # buoyancy force (and moment) vector
            Fvec = translateForce3to6DOF( r_center, np.array([0, 0, env.rho*env.g*V_UW]) ) 
  
            # hydrostatic stiffness matrix (about end A)
            Cmat = np.zeros([6,6])  
            Cmat[3,3] = env.rho*env.g*V_UW * r_center[2]
            Cmat[4,4] = env.rho*env.g*V_UW * r_center[2]
            
        else: # if the members are fully above the surface
            
            AWP = 0
            IWP = 0
            xWP = 0
            yWP = 0
            V_UW = 0
            r_center = np.zeros(3)
            Fvec = np.zeros(6)
            Cmat = np.zeros([6,6])
            
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


    def __init__(self, memberList=[], nTurbines=1, ms=None, w=[], depth=300):
        '''
        Empty frequency domain model initialization function
        
        
        nTurbines 
            could in future be used to set up any number of identical turbines
        '''
        
        self.fowtList = []
        self.coords = []
        
        self.nDOF = 0  # number of DOFs in system
        
        # mooring system
        if ms==None:
            self.ms = mp.system  # if nothing passed, create a new blank MoorPy mooring system to be filled in later
        else:
            self.ms = ms
            
        self.depth = depth
            
        # analysis frequency array
        if len(w)==0:
            w = np.arange(.01, 3, 0.01)  # angular frequencies tp analyze (rad/s)
        
        self.w = np.array(w)
        self.nw = len(w)  # number of frequencies
        
        self.k = np.zeros(self.nw)  # wave number
        for i in range(self.nw):
            self.k[i] = waveNumber(self.w[i], self.depth)
        
        # if FOWT members were included, set up the FOWT here  <<< only set for 1 FOWT for now <<<
        if len(memberList)>0:
            self.fowtList.append(FOWT(memberList, w=self.w, mpb=ms.BodyList[0], depth=depth))
            self.coords.append([0.0,0.0])
            self.nDOF += 6
            
            ms.BodyList[0].type = -1  # need to make sure it's set to a coupled type
        
        ms.initialize()  # reinitialize the mooring system to ensure all things are tallied properly etc.
        
        
        
    def addFOWT(self, fowt, xy0=[0,0]):
        '''adds an already set up FOWT to the frequency domain model solver.'''
    
        self.fowtList.append(fowt)
        self.coords.append(xy0)
        self.nDOF += 6
        
        # would potentially need to add a mooring system body for it too <<<
        
    
    def setEnv(self, Hs=8, Tp=12, V=10, beta=0):
    
        self.env = Env()
        self.env.Hs   = Hs   
        self.env.Tp   = Tp   
        self.env.V    = V    
        self.env.beta = beta 
    
        for fowt in self.fowtList:
            fowt.setEnv(Hs=Hs, Tp=Tp, V=V, beta=beta)
    
    
    def calcSystemProps(self):
        '''This gets the various static/constant calculations of each FOWT done.'''
        
        for fowt in self.fowtList:
            fowt.calcStatics()
            fowt.calcHydroConstants()
            fowt.calcDynamicConstants()
    
    
    def calcMooringAndOffsets(self):
        '''Calculates mean offsets and linearized mooring properties for the current load case.
        setEnv and calcSystemProps must be called first.
        '''

        ## First get mooring system characteristics about undisplaced platform position (useful for baseline and verification)
        #C_moor = self.ms.getCoupledStiffness()                             # this method accounts for eqiuilibrium of free objects in the system
        #F_moor = self.ms.getForces(DOFtype="coupled", lines_only=True)


        # Now find static equilibrium offsets of platform and get mooring properties about that point
        # (This assumes some loads have been applied)
        self.ms.solveEquilibrium3(DOFtype="both")     # get the system to its equilibrium
        self.ms.plot()
        r6eq = self.ms.BodyList[0].r6
        print("Equilibirum platform positions/rotations:")
        printVec(r6eq)
        print("Surge: {:.2f}".format(r6eq[0]))
        print("Pitch: {:.2f}".format(r6eq[4]*180/np.pi))
        C_moor = self.ms.getCoupledStiffness()
        F_moor = self.ms.getForces(DOFtype="coupled", lines_only=True)    # get net forces and moments from mooring lines on Body

        # manually add yaw spring stiffness as compensation until bridle (crow foot) configuration is added
        C_moor[5,5] += 98340000.0
        
        self.C_moor = C_moor
        self.F_moor = F_moor
    
    
    
    
    def solveStatics(self):
        # <<<< what is this <<<<< ?
    
        # ::: a loop could be added here for an array :::
        fowt = self.fowtList[0]
        
        # range of DOFs for the current turbine
        i1 = 0
        i2 = 6  
        
        fowt.W_hydro + fowt.W_struc
    
    
    def solveDynamics(self):
        '''After all constant parts have been computed, call this to iterate through remaining terms 
        until convergence on dynamic response.'''


        # total system complex response amplitudes (this gets updated each iteration)
        Xi = np.zeros([self.nDOF,self.nw], dtype=complex) + 0.01    # displacement and rotation complex amplitudes [m, rad]

        nIter = 2  # maximum number of iterations to allow

        # start fixed point iteration loop for dynamics   <<< would a secant method solve be possible/better? <<<
        for iiter in range(nIter):

            # ::: re-zero some things that will be added to :::
            
            # total system coefficient arrays
            M_tot = np.zeros([self.nDOF,self.nDOF,self.nw])       # total mass and added mass matrix [kg, kg-m, kg-m^2]
            B_tot = np.zeros([self.nDOF,self.nDOF,self.nw])       # total damping matrix [N-s/m, N-s, N-s-m]
            C_tot = np.zeros([self.nDOF,self.nDOF,self.nw])       # total stiffness matrix [N/m, N, N-m]
            F_tot = np.zeros([self.nDOF,self.nw], dtype=complex)  # total excitation force/moment complex amplitudes vector [N, N-m]

            Z  = np.zeros([self.nDOF,self.nDOF,self.nw], dtype=complex)  # total system impedance matrix

            # add in mooring stiffness from MoorPy system
            for ii in range(self.nw):          # loop through each frequency
                C_tot[:,:,ii] = self.C_moor
            
            
            # ::: a loop could be added here for an array :::
            fowt = self.fowtList[0]
            
            # range of DOFs for the current turbine
            i1 = 0
            i2 = 6            
            
            # get linearized terms for the current turbine given latest amplitudes
            B_lin, F_lin = fowt.calcLinearizedTerms(Xi)
            
            # add fowt's terms to system matrices (BEM arrays are not yet included here)
            
            for ii in range(self.nw):          # loop through each frequency
                M_tot[:,:,ii] = M_tot[:,:,ii] + fowt.M_struc + fowt.A_hydro_morison   # mass
                B_tot[:,:,ii] = B_tot[:,:,ii] + fowt.B_struc + B_lin             # damping (structural and linearized morison)
                C_tot[:,:,ii] = C_tot[:,:,ii] + fowt.C_struc + fowt.C_hydro           # stiffness
                F_tot[:,  ii] = F_tot[:,  ii] + F_lin[:,ii] + fowt.F_hydro_iner[:,ii] # excitation force complex amplitudes
                
                # form impedance matrix
                Z[:,:,ii] = -self.w[ii]**2 * M_tot[:,:,ii] + 1j*self.w[ii]*B_tot[:,:,ii] + C_tot[:,:,ii]
                
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
        ax[2].set_xlabel("frequency (Hz)")

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
         
 
        




class FOWT():
    '''This class comprises the frequency domain model of a single floating wind turbine'''

    def __init__(self, memberStrings, w=[], mpb=None, depth=600):
        '''This initializes the FOWT object which contains everything for a single turbine's frequency-domain dynamics.
        The initializiation sets up the design description.
        
        Parameters
        ----------
        
        memberStrings
            List of strings describing each member
        w
            Array of frequencies to be used in analysis
        mpb
            Reference to the MoorPy Body that this FOWT will be attached to
        
        '''

        # store reference to the already-set-up MoorPy system (currently assuming the FOWT is represented by Body 1)
        #self.ms = mooringSystem

        # ------------------------------- basic setup -----------------------------------------

        self.nDOF = 6

        if len(w)==0:
            w = np.arange(.01, 3, 0.01)  # angular frequencies tp analyze (rad/s)
        
        self.w = np.array(w)
        self.nw = len(w)  # number of frequencies
        self.k = np.zeros(self.nw)  # wave number
        
        self.depth = depth


        # ----------------------- member-based platform description --------------------------

        # (hard-coded for now - set to OC3 Hywind Spar geometry - eventually these will be provided as inputs instead)

        # list of member objects
        self.memberList = []

        for memberString in memberStrings:
            self.memberList.append(Member(memberString, self.nw))



        # -------------------------- turbine RNA description (eventually these should be inputs) ------------------------
        mRotor = 227962 #[kg]
        mNacelle = 446036 #[kg]
        IxHub = 325671 #[kg-m^2]
        IzNacelle = 7326346 #[kg-m^2]
        IxBlades = 45671252 #[kg-m^2] MOI value from FAST file, don't know where MOI is about. Assuming about the hub
        xCG_Hub = -7.07 #[m] from yaw axis
        xCG_Nacelle = 2.687 #[m] from yaw axis


        self.mRNA = mRotor + mNacelle #[kg]
        self.IxRNA = IxBlades*(1 + 1 + 1) + IxHub # RNA moment of inertia about local x axis (assumed to be identical to rotor axis for now, as approx) [kg-m^2]
        self.IrRNA = IxBlades*(1 + .5 + .5) + IzNacelle # RNA moment of inertia about local y or z axes [kg-m^2]


        self.xCG_RNA = ((mRotor*xCG_Hub)+(mNacelle*xCG_Nacelle))/(mRotor+mNacelle)          # x location of RNA center of mass [m]

        #hHub    = 119.0                          # hub height above water line [m]
        self.hHub    = 118.0 

        # reference to Body in mooring system corresponding to this turbine
        self.body = mpb
            

        # ---------------------- set up key arrays (these are now created just in time)-----------------------------
        '''
        # structure-related arrays
        -self.M_struc = np.zeros([6,6])                # structure/static mass/inertia matrix [kg, kg-m, kg-m^2]
        -self.B_struc = np.zeros([6,6])                # structure damping matrix [N-s/m, N-s, N-s-m] (may not be used)
        -self.C_struc = np.zeros([6,6])                # structure effective stiffness matrix [N/m, N, N-m]
        -self.W_struc = np.zeros([6])                  # static weight vector [N, N-m]

        # hydrodynamics-related arrays
        self.A_hydro = np.zeros([6,6,nw])             # hydrodynamic added mass matrix [kg, kg-m, kg-m^2]
        self.B_hydro = np.zeros([6,6,nw])             # wave radiation drag matrix [kg, kg-m, kg-m^2]
        self.B_hydro_drag = np.zeros([6,6])           # linearized viscous drag matrix [kg, kg-m, kg-m^2]
        -self.C_hydro = np.zeros([6,6])                # hydrostatic stiffness matrix [N/m, N, N-m]
        -self.W_hydro = np.zeros(6)                    # buoyancy force/moment vector [N, N-m]
        self.F_hydro = np.zeros([6,nw],dtype=complex) # linaer wave excitation force/moment complex amplitudes vector [N, N-m]
        self.F_hydro_drag = np.zeros([6,nw],dtype=complex) # linearized drag wave excitation complex amplitudes vector [N, N-m]

        # moorings-related arrays
        self.A_moor = np.zeros([6,6,nw])              # mooring added mass matrix [kg, kg-m, kg-m^2] (may not be used)
        self.B_moor = np.zeros([6,6,nw])              # mooring damping matrix [N-s/m, N-s, N-s-m] (may not be used)
        self.C_moor = np.zeros([6,6])                 # mooring stiffness matrix (linearized about platform offset) [N/m, N, N-m]
        self.W_moor = np.zeros(6)                     # mean net mooring force/moment vector [N, N-m]
        '''


    def setEnv(self, Hs=8, Tp=12, V=10, beta=0):
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
        
        # wave elevation amplitudes (these are easiest to use) - no need to be complex given frequency domain use
        self.zeta = np.sqrt(S)
            
            
        Fthrust = 800.0e3            # peak thrust force, [N]
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

        # loop through each member
        for mem in self.memberList:
            
            q, p1, p2 = mem.getDirections()                # get unit direction vectors
            
            
            # ---------------------- get member's mass and inertia properties ------------------------------
            
            v_steel, center, I_rad, I_ax, m_fill = mem.getInertia() # calls the getInertia method to calcaulte values
            
            mass = v_steel*mem.rho_steel + m_fill #[kg]
            Mmat = np.diag([mass, mass, mass, I_rad, I_rad, I_ax]) # MOI matrix = Mmat[3:,3:] is 0 on off diags bc symmetry in cylinders
            # @mhall: Mmat as written above is the mass and inertia matrix about the member CG...@shousner: you betcha
          
            # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices
            self.W_struc += translateForce3to6DOF( center, np.array([0,0, -g*mass]) )  # weight vector
            self.M_struc += translateMatrix6to6DOF(center, Mmat)                       # mass/inertia matrix
            # @mhall: Using the diagonal Mmat, and calling the above function with the "center" coordinate, will give the mass/inertia about the PRP!
            # @shousner: center is the position vector of the CG of the member, from the global coordinates aka PRP
            Sum_M_center += center*mass
            
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

        # for now, turbine RNA is specified by some simple lumped properties
        Mmat = np.diag([self.mRNA, self.mRNA, self.mRNA, self.IxRNA, self.IrRNA, self.IrRNA])            # create mass/inertia matrix
        center = np.array([self.xCG_RNA, 0, self.hHub])                                 # RNA center of mass location

        # now convert everything to be about PRP (platform reference point) and add to global vectors/matrices
        self.W_struc += translateForce3to6DOF( center, np.array([0,0, -g*self.mRNA]) )  # weight vector
        self.M_struc += translateMatrix6to6DOF(center, Mmat)                       # mass/inertia matrix
        Sum_M_center += center*self.mRNA


        # ----------- process key hydrostatic-related totals for use in static equilibrium solution ------------------

        mTOT = self.M_struc[0,0]
        rCG_TOT = Sum_M_center/mTOT 
        self.rCG_TOT = rCG_TOT

        rCB_TOT = Sum_V_rCB/VTOT       # location of center of buoyancy on platform

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



# #print(stopt)
# # ------------------------------- sum all static matrices -----------------------------------------
# # this is to check totals from static calculations before hydrodynamic terms are added

# M_tot_stat = M_struc             
# C_tot_stat = C_struc + C_hydro + C_moor
# W_tot_stat = W_struc + W_hydro + W_moor




# print("hydrostatic stiffness matrix")
# printMat(C_hydro)    
    
# print("structural stiffness matrix")
# printMat(C_struc)
    
# print("mooring stiffness matrix")
# printMat(C_moor)
    

# print("total static mass matrix")
# printMat(M_tot_stat)
    
# print("total static stiffness matrix")
# printMat(C_tot_stat)
    
# print("total static forces and moments")
# printVec(W_tot_stat)


    def calcHydroConstants(self):

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
                
                    q, p1, p2 = mem.getDirections()                # get unit direction vectors
                    
                    
                    # set dl to half if at the member end (the end result is similar to trapezoid rule integration)
                    if il==0 or il==mem.n:
                        dl = 0.5*mem.dl
                    else:
                        dl = mem.dl
                    
                    
                    # get wave kinematics spectra given a certain wave spectrum and location
                    mem.u[il,:,:], mem.ud[il,:,:], mem.pDyn[il,:] = getWaveKin(self.zeta, self.w, self.k, self.depth, mem.r[il,:], self.nw)


                    # local added mass matrix
                    Amat = rho*0.25*np.pi*mem.d[il]**2*dl *( mem.Ca_q*VecVecTrans(q) + mem.Ca_p1*VecVecTrans(p1) + mem.Ca_p2*VecVecTrans(p2) )
                    
                    # add to global added mass matrix for Morison members
                    self.A_hydro_morison += translateMatrix3to6DOF(mem.r[il,:], Amat)
                    
                    
                    # local inertial excitation matrix
                    Imat = rho*0.25*np.pi*mem.d[il]**2*dl *( (1+mem.Ca_q)*VecVecTrans(q) + (1+mem.Ca_p1)*VecVecTrans(p1) + (1+mem.Ca_p2)*VecVecTrans(p2) )
                    
                    for i in range(self.nw):   # for each wave frequency...
                    
                        # local inertial excitation force complex amplitude in x,y,z
                        F_exc_inert = np.matmul(Imat, mem.ud[il,:,i])  
                    
                        # add to global excitation vector (frequency dependent)
                        self.F_hydro_iner[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_inert)


                    # add end effects for added mass, and excitation including dynamic pressure
                    if il==0:     # end A
                            
                        # local added mass matrix
                        Amat = rho*np.pi*mem.d[il]**3/6.0 *mem.Ca_End*VecVecTrans(q)                
                        
                        # add to global added mass matrix for Morison members
                        self.A_hydro_morison += translateMatrix3to6DOF(mem.r[il,:], Amat)
                        
                        
                        # local inertial excitation matrix
                        Imat = rho*np.pi*mem.d[il]**3/6.0 * (1+mem.Ca_End)*VecVecTrans(q)      
                        
                        for i in range(self.nw):   # for each wave frequency...
                        
                            # local inertial (plus dynamic pressure) excitation force complex amplitude in x,y,z
                            F_exc_inert = np.matmul(Imat, mem.ud[il,:,i]) + mem.pDyn[il,i]*rho*0.25*np.pi*mem.d[il]**2 *q  
                        
                            # add to global excitation vector (frequency dependent)
                            self.F_hydro_iner[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_inert)
                        
                        
                    elif il==mem.n-1:  # end B
                   
                        # local added mass matrix
                        Amat = rho*np.pi*mem.d[il]**3/6.0 *mem.Ca_End*VecVecTrans(q)                
                        
                        # add to global added mass matrix for Morison members
                        self.A_hydro_morison += translateMatrix3to6DOF(mem.r[il,:], Amat)
                        
                        
                        # local inertial excitation matrix
                        Imat = rho*np.pi*mem.d[il]**3/6.0 * (1+mem.Ca_End)*VecVecTrans(q)      
                        
                        for i in range(self.nw):   # for each wave frequency...
                        
                            # local inertial (plus dynamic pressure) excitation force complex amplitude in x,y,z
                            F_exc_inert = np.matmul(Imat, mem.ud[il,:,i]) - mem.pDyn[il,i]*rho*0.25*np.pi*mem.d[il]**2 *q  
                        
                            # add to global excitation vector (frequency dependent)
                            self.F_hydro_iner[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_inert)
                        

    def calcDynamicConstants(self):
        ''' get system properties in undisplaced position ----------------------------
         these are useful for verification, etc.
        
        Note: calcStatics and calcHydroConstants should be called before this method.
        '''

        # sum matrices to check totals from static calculations before hydrodynamic terms are added

        C_tot0 = self.C_struc + self.C_hydro # + C_moor0   # total system stiffness matrix about undisplaced position
        W_tot0 = self.W_struc + self.W_hydro #+ W_moor0   # system mean forces and moments at undisplaced position

        M = self.M_struc + self.A_hydro_morison          # total mass plus added mass matrix

        # do we want to relinearize structural properties about displaced position/orientation?  (Probably not)

        '''
        print("hydrostatic stiffness matrix")
        printMat(C_hydro)    
            
        print("structural stiffness matrix")
        printMat(C_struc)
            
        print("mooring stiffness matrix about undisplaced position")
        printMat(C_moor0)
            
        print("total static stiffness matrix about undisplaced position")
        printMat(C_tot0)
            
            

        print("total static mass matrix")
        printMat(M_struc)
            
        print("total added mass matrix")
        printMat(A_hydro_morison)

        print("total mass plus added mass matrix")
        printMat(M)
            
            
        print("total static forces and moments about undisplaced position")
        printVec(W_tot0)
        '''



        # calculate natural frequencies (using eigen analysis to get proper values for pitch and roll - otherwise would need to base about CG if using diagonal entries only)

        eigenvals, eigenvectors = np.linalg.eig(np.matmul(np.linalg.inv(M), C_tot0))   # <<< need to sort this out so it gives desired modes, some are currently a bit messy


        # alternative attempt to calculate natural frequencies based on diagonal entries (and taking pitch and roll about CG)

        zMoorx = C_tot0[0,4]/C_tot0[0,0]  # effective z elevation of mooring system reaction forces in x and y directions
        zMoory = C_tot0[1,3]/C_tot0[1,1]
        zCG  = self.rCG_TOT[2]                    # center of mass in z
        zCMx = M[0,4]/M[0,0]              # effective z elevation of center of mass and added mass in x and y directions
        zCMy = M[1,3]/M[1,1]

        print("natural frequencies without added mass")
        fn = np.zeros(6)
        fn[0] = np.sqrt( C_tot0[0,0] / self.M_struc[0,0] )/ 2.0/np.pi
        fn[1] = np.sqrt( C_tot0[1,1] / self.M_struc[1,1] )/ 2.0/np.pi
        fn[2] = np.sqrt( C_tot0[2,2] / self.M_struc[2,2] )/ 2.0/np.pi
        fn[5] = np.sqrt( C_tot0[5,5] / self.M_struc[5,5] )/ 2.0/np.pi
        zg = self.rCG_TOT[2]
        fn[3] = np.sqrt( (C_tot0[3,3] + C_tot0[1,1]*((zCG-zMoory)**2 - zMoory**2) ) / (self.M_struc[3,3] - self.M_struc[1,1]*zg**2 ))/ 2.0/np.pi     # this contains adjustments to reflect rotation about the CG rather than PRP
        fn[4] = np.sqrt( (C_tot0[4,4] + C_tot0[0,0]*((zCG-zMoorx)**2 - zMoorx**2) ) / (self.M_struc[4,4] - self.M_struc[0,0]*zg**2 ))/ 2.0/np.pi     # this contains adjustments to reflect rotation about the CG rather than PRP
        printVec(fn)


        print("natural frequencies with added mass")
        fn = np.zeros(6)
        fn[0] = np.sqrt( C_tot0[0,0] / M[0,0] )/ 2.0/np.pi
        fn[1] = np.sqrt( C_tot0[1,1] / M[1,1] )/ 2.0/np.pi
        fn[2] = np.sqrt( C_tot0[2,2] / M[2,2] )/ 2.0/np.pi
        fn[5] = np.sqrt( C_tot0[5,5] / M[5,5] )/ 2.0/np.pi
        fn[3] = np.sqrt( (C_tot0[3,3] + C_tot0[1,1]*((zCMy-zMoory)**2 - zMoory**2) ) / (M[3,3] - M[1,1]*zCMy**2 ))/ 2.0/np.pi     # this contains adjustments to reflect rotation about the CG rather than PRP
        fn[4] = np.sqrt( (C_tot0[4,4] + C_tot0[0,0]*((zCMx-zMoorx)**2 - zMoorx**2) ) / (M[4,4] - M[0,0]*zCMx**2 ))/ 2.0/np.pi     # this contains adjustments to reflect rotation about the CG rather than PRP
        # note that the above lines use off-diagonal term rather than parallel axis theorem since rotation will not be exactly at CG due to effect of added mass
        printVec(fn)



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
            
            q, p1, p2 = mem.getDirections()                # get unit direction vectors
            
            # loop through each node of the member
            for il in range(mem.n):
                
                # node displacement, velocity, and acceleration (each [3 x nw])
                drnode, vnode, anode = getVelocity(mem.r[il,:], Xi, self.w)      # get node complex velocity spectrum based on platform motion's and relative position from PRP
                
                
                # only process hydrodynamics if this node is submerged
                if mem.r[il,2] < 0:
                
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
                    
                    # add to global damping matrix for Morison members
                    Btemp = translateMatrix3to6DOF(mem.r[il,:], Bmat)
                    
                    #breakpoint()
                    
                    B_hydro_drag += Btemp
                    
                    
                    
                    # excitation force based on linearized damping coefficients [3 x nw]
                    F_exc_drag = np.zeros([3, self.nw], dtype=complex)  # <<< should set elsewhere <<<
                    for i in range(self.nw):

                        # get local 3d drag excitation force complex amplitude for each frequency
                        F_exc_drag[:,i] = np.matmul(Bmat, mem.u[il,:,i])  
                    
                        # add to global excitation vector (frequency dependent)
                        F_hydro_drag[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_drag[:,i])

        # save the arrays internally in case there's ever a need for the FOWT to solve it's own latest dynamics
        self.B_hydro_drag = B_hydro_drag
        self.F_hydro_drag = F_hydro_drag

        # return the linearized coefficients
        return B_hydro_drag, F_hydro_drag

