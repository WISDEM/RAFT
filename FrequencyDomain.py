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
		self.d  = np.int(entries[2])
		self.rA = np.array(entries[3:6], dtype=np.double)
		self.rB = np.array(entries[6:9], dtype=np.double)
		
		self.rAB = self.rB-self.rA
		self.l = np.linalg.norm(self.rAB)  # member length
		
		self.q = self.rAB/self.l           # member axial unit vector
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
		
		self.r  = np.zeros([self.n,3])  # undisplaced node positions along member 
		for i in range(self.n):
			self.r[i,:] = self.rA + (i/(self.n-1))*self.rAB  # spread evenly for now
		
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
	

## Get node complex velocity spectrum based on platform motion's and relative position from PRP		
def getVelocity(r, Xi, ws):
				
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
		if (    k[i]   == 0.0  ):  					# When .TRUE., the shallow water formulation is ill-conditioned; thus, the known value of unity is returned.
			SINHNumOvrSIHNDen = 1.0
			COSHNumOvrSIHNDen = 99999
		elif ( k[i]*h >  89.4 ):  				# When .TRUE., the shallow water formulation will trigger a floating point overflow error; however, with h > 14.23*wavelength (since k = 2*Pi/wavelength) we can use the numerically-stable deep water formulation instead.
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
	
	k = 0                          	#initialize  k outside of loop
									#error tolerance
	k1 = omega*omega/g                 	#deep water approx of wave number
	k2 = omega*omega/(np.tanh(k1*h)*g)    	#general formula for wave number
	while np.abs(k2 - k1)/k1 > e:          	#repeate until converges
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
	


# ------------------------------- basic setup -----------------------------------------


nDOF = 6


w = np.arange(.05, 5, 0.25)  # angular frequencies tp analyze (rad/s)
nw = len(w)  # number of frequencies

k= np.zeros(nw)  # wave number


# ----------------------- member-based platform description --------------------------

# list of member objects
memberList = []

#                    number  type  diameter  xa   ya    za   xb   yb   zb
memberList.append(Member("1     x      12     0    0   -20    0    0    0  ", nw))
memberList.append(Member("2     x      12    30    0   -20   30    0    0  ", nw))
memberList.append(Member("3     x      12     0   30   -20    0   30    0  ", nw))
memberList.append(Member("4     x      8      0    0   -20   30    0  -20  ", nw))
memberList.append(Member("5     x      8      0    0   -20    0   30  -20  ", nw))








# --------------------------- basic platform properties ------------------------------
'''
As  =  
CM  =  
Ixa = 
Iya = 
Iza = 
Awp = 
Iwp = 
V   =   
CB  =  
'''


# ---------------- create Morison-based platform coefficient matrices -----------------




# ---------------- (future work) import hydrodynamic coefficient files ----------------










# ---------------------------- environmental conditions -------------------------------

depth = 200
rho = 1025
g = 9.81

# environmental condition(s)
Hs = [2.4, 3.4, 5];
Tp = [4.1, 4.6, 6.05];
windspeed = [8, 12, 18];

S   = np.zeros([len(Hs), nw])   # wave spectrum
zeta= np.zeros([len(Hs), nw])  # wave elevation

T = 2*np.pi/w # periods



for imeto in range(len(Hs)):       # loop through each environmental condition

	# make wave spectrum (could replace with function call)
	S[imeto,:] = 1/2/np.pi *5/16 * Hs[imeto]**2*T *(w*Tp[imeto]/2/np.pi)**(-5) *np.exp( -5/4* (w*Tp[imeto]/2/np.pi)**(-4) )

	# wave elevation amplitudes (these are easiest to use) - no need to be complex given frequency domain use
	zeta[imeto,:] = np.sqrt(S[imeto,:])
	

# get wave number
for i in range(nw):
	k[i] = waveNumber(w[i], depth)



# ignoring multiple DLCs for now <<<<<
#for imeto = 1:length(windspeeds)
imeto = 0

# ------------------ get wave kinematics along each member ---------------------

# loop through each member
for mem in memberList:
	
	# loop through each node of the member
	for il in range(mem.n):
		
		# function to get wave kinematics spectra given a certain spectrum and location
		mem.u[il,:,:], mem.ud[il,:,:] = getWaveKin(zeta[imeto,:], w, k, depth, mem.r[il,:])    


# ---------------------- set up key arrays -----------------------------

# response 
Xi = np.zeros([6,nw], dtype=complex)

# constant coefficient arrays
W_tot = np.zeros([6])
M_ss  = np.zeros([6,6]) # structure/static

m = 0.0            # total platform mass
mr = np.zeros(3)   # product of mass by location
v = 0.0            # total platform displacement
vr = np.zeros(3)   # product of volume by location


# ---------- add in linear hydrodynamic coefficients here if applicable --------------------
# TODO


# --------------- Get general geometry properties including hydrostatics ------------------------

# TODO: find a smart way to get GM by linearizing hydrostatic response to pitch/roll! <<<

# loop through each member
for mem in memberList:
	
	q, p1, p2 = mem.getDirections()                # get unit direction vectors
	
	# loop through each node of the member
	for il in range(mem.n):
		# mass/inertia, weight, and hydrostatics
		mass = mem.dl*mem.w  # kg
		iner = 0               # kg-m^2 
		Mmat = np.diag([mass, mass, mass, iner, iner, iner])
		vol = mem.dl*np.pi/4*mem.d**2
		buoyancy = vol*rho*g - mass*g  # what about partially submerged segments? <<<<<<<<<<<<<<<<<
		
		
		# add to totals
		m += mass
		mr += mass*mem.r[il,:]
		v+= vol
		vr += vol*mem.r[il,:]
		
		# now convert everything to be about PRP and add to matrices
		W_tot += translateForce3to6DOF( mem.r[il,:], np.array([0,0, -buoyancy]) )  # weight/buoyancy
		M_ss = translateMatrix6to6DOF(mem.r[il,:], Mmat)  # mass/inertia


	# get each member's waterplane area, moment of inertia, and derivatives w.r.t heave,roll,pitch
	if mem.r[0,2]*mem.r[-1,2] < 0:    # if member crosses water plane
			
		# eventually should interpolate dwp=np.interp(0, mem.r[:,2], mem.d)
		dwp = mem.d
		xwp = np.interp(0, mem.r[:,2], mem.r[:,0])
		ywp = np.interp(0, mem.r[:,2], mem.r[:,1])
		
		# get derivatives too!!
		
		
		
		awp = np.pi/4*dwp**2 /np.cos(mem.phi)
		Iwpx= np.pi/4*dwp**4 * (np.cos(beta)/np.cos(mem.phi)**3 + np.sin(beta)/np.cos(mem.phi)) # check<<<<
		Iwpy= np.pi/4*dwp**4 * (np.sin(beta)/np.cos(mem.phi)**3 + np.cos(beta)/np.cos(mem.phi))
		
		
		# sum...
		
		
# calculate hydrostatic stiffness matrix from above
		...


# solve for mean offsets (operating point)
Fthrust = 800e3       # N
Mthust = 100*Fthrust  # N-m



nIter = 2  # maximum number of iterations to allow

# start fixed point iteration loop for dynamics
for iiter in range(nIter):

	# ------ calculate linearized coefficients within iteration ------- <<< some of this could likely only be done once...>>>

	# varying coefficient arrays
	M_tot = np.zeros([6,6,nw])
	B_tot = np.zeros([6,6,nw])
	C_tot = np.zeros([6,6,nw])
	F_tot = np.zeros([6,nw], dtype=complex)


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
			vrel_q  = vrel* q[:,None]   # <<<<<<<<<<<<<< what to do about complex values here????
			vrel_p1 = vrel*p1[:,None]
			vrel_p2 = vrel*p2[:,None]
			
			# RMS relative velocity components				
			vRMS_q  = np.sqrt( np.sum( vrel_q **2) /nw)
			vRMS_p1 = np.sqrt( np.sum( vrel_p1**2) /nw)
			vRMS_p2 = np.sqrt( np.sum( vrel_p2**2) /nw)
			
			# linearized damping coefficients in each direction relative to member orientation [not explicitly frequency dependent...] (this goes into damping matrix)
			Bprime_q  = np.sqrt(8/np.pi) * vRMS_q  * 0.5*rho * np.pi*mem.d*mem.dl * mem.Cd_q 
			Bprime_p1 = np.sqrt(8/np.pi) * vRMS_p1 * 0.5*rho * mem.d*mem.dl * mem.Cd_p1
			Bprime_p2 = np.sqrt(8/np.pi) * vRMS_p2 * 0.5*rho * mem.d*mem.dl * mem.Cd_p2
			
			# convert to global orientation
			Bmat = Bprime_q*VecVecTrans(q) + Bprime_p1*VecVecTrans(p1) + Bprime_p2*VecVecTrans(p2)
			
			# excitation force based on linearized damping coefficients [3 x nw]
			F_exc_drag = np.zeros([3, nw], dtype=complex)  # <<< should set elsewhere <<<
			for i in range(nw):
				F_exc_drag[:,i] = np.matmul(Bmat, mem.u[il,:,i])  # get local 3d drag excitation force complex amplitude for each frequency
			
			
			# added mass...
			Amat = rho*0.25*np.pi*mem.d**2*mem.dl *( mem.Ca_q*VecVecTrans(q) + mem.Ca_p1*VecVecTrans(p1) + mem.Ca_p2*VecVecTrans(p2) )
			
			
			# inertial excitation...
			Imat = rho*0.25*np.pi*mem.d**2*mem.dl * ( (1+mem.Ca_q)*VecVecTrans(q) + (1+mem.Ca_p1)*VecVecTrans(p1) + (1+mem.Ca_p2)*VecVecTrans(p2) )
			
			F_exc_inert = np.zeros([3, nw], dtype=complex)  # <<< should set elsewhere <<<
			for i in range(nw):
				F_exc_inert[:,i] = np.matmul(Imat, mem.ud[il,:,i])  # get local 3d inertia excitation force complex amplitude for each frequency
			
			
			for i in range(nw):
			
				# non-frequency-dependent matrices
				#translateMatrix3to6DOF(mem.r[il,:], )
				B_tot[:,:,i] = translateMatrix3to6DOF(mem.r[il,:], Bmat)
				M_tot[:,:,i] = translateMatrix3to6DOF(mem.r[il,:], Amat)
				
				# frequency-dependent excitation vector
				F_tot[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_drag[:,i])
				F_tot[:,i] += translateForce3to6DOF( mem.r[il,:], F_exc_inert[:,i])




	# ---------------- solve matrix equation of motion -----------------
	'''
	for ii in range(nw):
		
		
		# sum contributions for each term
		
		M_tot[:,:,ii] = M_wt{imeto} + M_p + permute(A_h(ii,:,:), [2 3 1]) + A_mor[:,:,ii]; # note reordering of hydro dimensions
		
		B_tot[:,:,ii] = B_wt{imeto} + permute(B_h(ii,:,:), [2 3 1]) + Bvisc[:,:,ii];
		
		C_tot[:,:,ii] = C_wt{imeto} + C_p + C_h + C_l(:,:,imeto);
			
		F_tot[:,ii] = F_h(ii,1,:)*sqrt(S(imeto,ii))
		
		# form impedance matrix
		
		Z[:,:,ii] = -w[ii]**2 * M_tot[:,:,ii] + 1i*w[ii]*B_tot[:,:,ii] + C_tot[:,:,ii];
		
		
		
		Xi[:,ii] = np.linalg.matmul(np.linalg.invert(Z[:,:,ii]),  F_tot[:,ii] )  # response # reordering dimensions back to (freq, dof)
	

	#Xi{imeto} = rao{imeto}.*repmat(sqrt(S(:,imeto)),1,6); # complex response!
	
	#aNacRAO{imeto} = -(w').^2 .* (rao{imeto}(:,1) + hNac*rao{imeto}(:,5));      # Nacelle Accel RAO
	#aNac2(imeto) = sum( abs(aNacRAO{imeto}).^2.*S(:,imeto) ) *(w(2)-w(1));   	# RMS Nacelle Accel

	
	# ----------------- convergence check --------------------
	conv = abs(aNac2(imeto)/aNac2last - 1);
	#disp(['at ' num2str(iiter) ' iterations - convergence is ' num2str(conv)])
	if conv < 0.0001

		 
		 break
	else
		 aNac2last = aNac2(imeto);
	end

			
			>>>> this part still old Matlab code >>>

			# damping in each DOF for each cylinder's member
			Bbvisc11 = sum( Bbprime(icyl,:) .* abs(cos((icyl-0.5)*2*pi/NF) ));
			Bbvisc22 = sum( Bbprime(icyl,:) .* abs(sin((icyl-0.5)*2*pi/NF) ));
			Bbvisc33 = sum( Bbprime(icyl,:)); 
			Bbvisc15 = sum( Bbprime(icyl,:) .* abs(cos((icyl-0.5)*2*pi/NF) )) * z1;
			Bbvisc24 = sum( Bbprime(icyl,:) .* abs(sin((icyl-0.5)*2*pi/NF) )) * z1;
			# neglecting B16 and B26 damping coupling due to axial symmetry
			Bbvisc44 = sum( Bbprime(icyl,:) .* abs(sin((icyl-0.5)*2*pi/NF)) .* (ybar.^2 + z1^2));
			Bbvisc55 = sum( Bbprime(icyl,:) .* abs(cos((icyl-0.5)*2*pi/NF)) .* (xbar.^2 + z1^2)); # note I'm not including moment of inertia, just x^2+y^2
			Bbvisc66 = sum( Bbprime(icyl,:) .* (xbar.^2 + ybar.^2) );  # this is wrong!!!

			# added mass in each DOF for each tendon
			C_a = 0.97; # added mass coefficient [from Jonkman OCC phase IV]
			A_mor11 = sum( pi/4*db^2*lb/ns .* abs(cos((icyl-1)*2*pi/NF) ))* C_a;
			A_mor22 = sum( pi/4*db^2*lb/ns .* abs(sin((icyl-1)*2*pi/NF)  ))* C_a;
			A_mor33 =      pi/4*db^2*lb * C_a;
			A_mor15 = sum( pi/4*db^2*lb/ns .* abs(cos((icyl-1)*2*pi/NF) )) * z1 * C_a;
			A_mor24 = sum( pi/4*db^2*lb/ns .* abs(sin((icyl-1)*2*pi/NF) )) * z1 * C_a;
			A_mor44 = sum( pi/4*db^2*lb/ns .* abs(sin((icyl-1)*2*pi/NF)) .* (ybar.^2 + z1^2)) * C_a;
			A_mor55 = sum( pi/4*db^2*lb/ns .* abs(cos((icyl-1)*2*pi/NF)) .* (xbar.^2 + z1^2)) * C_a;
			A_mor66 = sum( pi/4*db^2*lb/ns .* (xbar.^2 + ybar.^2) ) * C_a;  # this is wrong!!!
		
		
		
		# bottoms / heave plates	
		
		# A3(:,icyl) = Xi{imeto}(:,3) + yfloat*Xi{imeto}(:,4) - xfloat*Xi{imeto}(:,5); # heave motion
		# Arms(icyl) = sqrt( sum( (abs(A3(:,icyl))).^2 ) /length(w));
		
		# Bvisc33 = 2/3*rho*dc^3*w*(0.2 + 0.5*(2*pi*Arms(icyl)/dc)); #  Consider getting rid of w dependency with a simpler model? (also in B44, B55)
		
		
			
	'''

 
 # ---------- mooring line fairlead tension RAOs and constraint implementation ----------
'''
 
 for il=1:Platf.Nlines
	  
	  #aNacRAO{imeto} = -(w').^2 .* (X{imeto}(:,1) + hNac*X{imeto}(:,5));      # Nacelle Accel RAO
		#aNac2(imeto) = sum( abs(aNacRAO{imeto}).^2.*S(:,imeto) ) *(w(2)-w(1));   	# RMS Nacelle Accel

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
    