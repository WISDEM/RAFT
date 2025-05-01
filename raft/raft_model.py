# RAFT's main model class

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import yaml
import time 

try:
    import pickle5 as pickle
except:
    import pickle
    
import moorpy as mp
import raft.raft_fowt as fowt
from raft.helpers import *
from moorpy.helpers import dsolve2, set_axes_equal, dsolvePlot
import copy
#import F6T1RNA as structural    # import turbine structural model functions

raft_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TwoPi = 2.0*np.pi

class Model():


    def __init__(self, design, nTurbines=1):
        '''
        Empty frequency domain model initialization function

        design : dict
            Dictionary of all the design info from turbine to platform to moorings
        nTurbines
            could in future be used to set up any number of identical turbines
        '''

        self.fowtList = []      # list of FOWT objects
        self.coords = []        # list of FOWT reference coordinates in x and y (also stored inside each FOWT as x_ref, y_ref [m]

        self.nDOF = 0  # number of FOWT-level DOFs in the system - normally will be 6*len(fowtList)


        # parse settings
        if not 'settings' in design:    # if settings field not in input data
            design['settings'] = {}     # make an empty one to avoid errors
        
        min_freq     = getFromDict(design['settings'], 'min_freq', default=0.01, dtype=float)  # [Hz] lowest frequency to consider, also the frequency bin width 
        max_freq     = getFromDict(design['settings'], 'max_freq', default=1.00, dtype=float)  # [Hz] highest frequency to consider
        self.XiStart = getFromDict(design['settings'], 'XiStart' , default=0.1 , dtype=float)  # sets initial amplitude of each DOF for all frequencies
        self.nIter   = getFromDict(design['settings'], 'nIter'   , default=15  , dtype=int  )  # sets how many iterations to perform in Model.solveDynamics()
        
        self.w = np.arange(min_freq, max_freq+0.5*min_freq, min_freq) *2*np.pi  # angular frequencies to analyze (rad/s)
        self.nw = len(self.w)  # number of frequencies
        
        
        # water depth and wave number        
        self.depth = getFromDict(design['site'], 'water_depth', dtype=float)
        self.k = np.zeros(self.nw)  # wave number
        for i in range(self.nw):
            self.k[i] = waveNumber(self.w[i], self.depth)
        
        # ----- parse array section if it exists -----
        
        if 'array' in design:  # if array info is given, RAFT will run in array mode
        
            self.nFOWT = len(design['array']['data'])
            
            # some checks/updates of the dictionary for compatibility
            if 'turbine' in design and not 'turbines' in design:  # if a single turbine is listed, make it a list for more consistent parsing
                design['turbines'] = [design['turbine']]
            if 'platform' in design and not 'platforms' in design: 
                design['platforms'] = [design['platform']]
            if 'mooring' in design and not 'moorings' in design: 
                design['moorings'] = [design['mooring']]
            
            # form dictionary of fowt array data
            fowtInfo = [dict(zip( design['array']['keys'], row)) for row in design['array']['data']]
            
            # if array_mooring section exists, create an array-level MoorPy system
            if 'array_mooring' in design:
                
                # quick hackish inclusion of bathymetry (should improve with better methods in MoorPy)
                if 'bathymetry' in design['array_mooring']:
                    print('Including bathymetry in array-level MoorPy.')
                    bath_file = design['array_mooring']['bathymetry']
                    self.ms = mp.System(depth=self.depth, bathymetry=bath_file)
                else:
                    self.ms = mp.System(depth=self.depth)
            
                # set up a coupled MoorPy body for each FOWT
                for i in range(self.nFOWT):
                    self.ms.addBody(-1, [fowtInfo[i]['x_location'], fowtInfo[i]['y_location'], 0,0,0,0])
                # load the MD style input file (this is the only option supported right now)
                if 'file' in design['array_mooring']:
                    self.ms.load(design['array_mooring']['file'], clear=False)  # add the array level mooring system to the already created bodies
                else:
                    raise Exception("When using 'array_mooring', a MoorDyn-style input file must be provided as 'file'.")
            else:
                self.ms = None
                
            # go through each turbine in the list and set it up...
            for i in range(self.nFOWT):
            
                x_ref = fowtInfo[i]['x_location']
                y_ref = fowtInfo[i]['y_location']
                headj = fowtInfo[i]['heading_adjust']
            
                design_i = {}   # just make a temporary design dictionary for the FOWT (could make this a stored list of all)
                
                design_i['site'] = design['site']
                
                if fowtInfo[i]['turbineID'] == 0:
                    design_i.pop('turbine', None)  # if no turbine, make sure the entry isn't in the design dictionary
                else:
                    design_i['turbine'] = design['turbines'][fowtInfo[i]['turbineID']-1]
                
                if fowtInfo[i]['platformID'] == 0:
                    design_i['platform'] = None
                    print("Warning: platforms MUST be included for the time being.")
                else:
                    design_i['platform'] = design['platforms'][fowtInfo[i]['platformID']-1]
                    
                if fowtInfo[i]['mooringID'] == 0:  # no mooring on this FOWT (array-level moorings may be used instead)
                    design_i['mooring'] = None
                else:
                    design_i['mooring'] = design['moorings'][fowtInfo[i]['mooringID']-1]
                
                if self.ms:
                    mpb = self.ms.bodyList[i]  # reference to the FOWT's body in the array level MoorPy system
                else:
                    mpb = None
                
                
                self.fowtList.append(fowt.FOWT(design_i, self.w, mpb, depth=self.depth,
                                       x_ref=x_ref, y_ref=y_ref, heading_adjust=headj))
                                               
                self.coords.append([x_ref, y_ref])
                self.nDOF += 6
            
        
        
        else:  # normal single-FOWT mode
            
            # This is the original approach. It assumes a single turbine, platform, and mooring section are given.
            
            self.nFOWT = 1
            
            # Note: its mooring system will be put in the FOWT now rather than existing at the model/array level.
            # # process mooring information 
            self.ms = None

            # set up the FOWT here
            self.fowtList.append(fowt.FOWT(design, self.w, None, depth=self.depth))
            self.coords.append([0.0,0.0])
            self.nDOF += 6
        
        self.design = design # save design dictionary for possible later use/reference

        # Set mooring current modeling mode (0: no current; 1: uniform current included in MoorPy)
        if 'mooring' in design:
            self.mooring_currentMod = getFromDict(design['mooring'], 'currentMod', default=0, dtype=int)
        else:
            self.mooring_currentMod = 0

        # Initialize array-level mooring system if it exists
        if self.ms:
            self.ms.initialize()
        
        self.results = {}     # dictionary to hold all results from the model
        
        
        


    def addFOWT(self, fowt, xy0=[0,0]):
        '''(not used currently) Adds an already set up FOWT to the frequency domain model solver.'''

        self.fowtList.append(fowt)
        self.coords.append(xy0)
        self.nDOF += 6

        # would potentially need to add a mooring system body for it too <<<

    def analyzeUnloaded(self, ballast=0, heave_tol = 1):
        '''This calculates the system properties under undloaded coonditions: equilibrium positions, natural frequencies, etc.
        
        ballast: flag to ballast the FOWTs to achieve a certain heave offset'''
        
        # >>> this whole method needs to be updated or possibly removed <<<
        
        if len(self.fowtList) > 1:
            raise Exception('analyzeUnloaded is an old method that only works for a single FOWT.')
        
        # need to zero out external loads
        self.fowtList[0].setPosition(np.zeros(6))
        self.fowtList[0].D_hydr0 = np.zeros(6)
        self.fowtList[0].f_aero0 = np.zeros([6,self.fowtList[0].nrotors])
        
        
        # get mooring system characteristics about undisplaced platform position (useful for baseline and verification)
        self.C_moor0 = np.zeros([6,6])
        self.F_moor0 = np.zeros(6)
        
        if self.ms:
            try: 
                self.C_moor0 += self.ms.getCoupledStiffness(lines_only=True)        
                self.F_moor0 += self.ms.getForces(DOFtype="coupled", lines_only=True)
            except Exception as e:
                raise RuntimeError('An error occured when getting linearized mooring properties in undisplaced state: '+e.message)
        
        if self.fowtList[0].ms:
            try: 
                self.C_moor0 += self.fowtList[0].ms.getCoupledStiffness(lines_only=True)        
                self.F_moor0 += self.fowtList[0].ms.getForces(DOFtype="coupled", lines_only=True)
            except Exception as e:
                raise RuntimeError('An error occured when getting linearized mooring properties in undisplaced state: '+e.message)
        
        
        # calculate the system's constant properties
        for fowt in self.fowtList:
        
            # apply any ballast adjustment if requested
            if ballast == 1:
                print('adjusting ballast fill levels')
                self.adjustBallast(fowt, heave_tol=heave_tol)  
            elif ballast == 2:
                print('adjusting ballast densities')
                self.adjustBallastDensity(fowt)        
            
            # compute FOWT static and constant hydrodynamic properties
            fowt.calcStatics()
            fowt.calcHydroConstants()  # includes rotor when underwater
        
        
        self.results['properties'] = {}   # signal this data is available by adding a section to the results dictionary
            
        # calculate platform offsets and mooring system equilibrium state
        self.solveStatics(None)  # passing none should imply no load case (no WWC)
        self.results['properties']['offset_unloaded'] = self.fowtList[0].Xi0
        
        # TODO: add printing of summary info here - mass, stiffnesses, etc

    
    def analyzeCases(self, display=0, meshDir=os.path.join(os.getcwd(),'BEM'), RAO_plot=False):
        '''This runs through all the specified load cases, building a dictionary of results.'''
        
        nCases = len(self.design['cases']['data'])
        
        self.results['properties'] = {}  # signal that the properties calcs will be done
        
        # set up output arrays for load cases >>> put these into an initialization function <<<
        
        self.results['case_metrics'] = {}
        self.results['mean_offsets'] = []

        
        # calculate the system's constant properties
        for fowt in self.fowtList:
            fowt.setPosition([fowt.x_ref, fowt.y_ref,0,0,0,0])
            fowt.calcStatics()

        for i, fowt in enumerate(self.fowtList):
            fowt.calcBEM(meshDir=meshDir)
        
            
        # loop through each case
        for iCase in range(nCases):
        
            if display > 0:
                print(f"\n--------------------- Running Case {iCase+1} ----------------------")
                print(self.design['cases']['data'][iCase])
        
            # form dictionary of case parameters
            case = dict(zip( self.design['cases']['keys'], self.design['cases']['data'][iCase]))            
            case['iCase'] = iCase # We use iCase to name the output files
            
            if np.isscalar(case['wave_heading']):  # deal with the typical case of just one set of waves specified
                nWaves = 1
            else:
                nWaves = len(case['wave_heading'])
            
            # initialize dictionary of case results
            self.results['case_metrics'][iCase] = {}
            
            # solve system operating point / mean offsets for this load case
            self.solveStatics(case, display=display)
            
            # >>> add a flag that stores what case has had solveStatics to ensure consistency <<<
          
            # solve system dynamics            
            self.solveDynamics(case, RAO_plot=RAO_plot, display=display)

            # Solve system operating point / mean offsets again, but now including mean wave forces.
            # We actually wouldn't need to do that if the QTFs are computed externally, but all the wave information 
            # is currently computed only when solveDynamics is called. Should work on that.
            if any(fowt.potSecOrder > 0 for fowt in self.fowtList):
                if display > 1:
                    print('Recomputing equilibrium position, now with wave mean drift')
                self.solveStatics(case)

                # zero out the mean wave forces to avoid using old values in the next case
                for i, fowt in enumerate(self.fowtList): 
                    fowt.Fhydro_2nd_mean *= 0

            
            # >>> need to decide if I want to store Xi0 and Xi in the FOWTs or work with them directly here. <<<
            
            # process outputs that are specific to the floating unit (initialize dictionary for case and turb index)
            for i, fowt in enumerate(self.fowtList):
                self.results['case_metrics'][iCase][i] = {}
                fowt.saveTurbineOutputs(self.results['case_metrics'][iCase][i],case)            
                nTowers = fowt.ntowers
                nRotors = fowt.nrotors
                
                if display > 0:
        
                    metrics = self.results['case_metrics'][iCase][i]
                
                    # print statistics table
                    print(f"-------------------- FOWT {i+1} Case {iCase+1} Statistics --------------------")
                    print("Response channel     Average     RMS         Maximum     Minimum")
                    print(f"surge (m)          {metrics['surge_avg'] :10.2e}  {metrics['surge_std'] :10.2e}  {metrics['surge_max'] :10.2e}  {metrics['surge_min'] :10.2e}")
                    print(f"sway (m)           {metrics['sway_avg' ] :10.2e}  {metrics['sway_std' ] :10.2e}  {metrics['sway_max' ] :10.2e}  {metrics['sway_min'] :10.2e}")
                    print(f"heave (m)          {metrics['heave_avg'] :10.2e}  {metrics['heave_std'] :10.2e}  {metrics['heave_max'] :10.2e}  {metrics['heave_min'] :10.2e}")
                    print(f"roll (deg)         {metrics['roll_avg' ] :10.2e}  {metrics['roll_std' ] :10.2e}  {metrics['roll_max' ] :10.2e}  {metrics['roll_min'] :10.2e}")
                    print(f"pitch (deg)        {metrics['pitch_avg'] :10.2e}  {metrics['pitch_std'] :10.2e}  {metrics['pitch_max'] :10.2e}  {metrics['pitch_min'] :10.2e}")
                    print(f"yaw (deg)          {metrics[  'yaw_avg'] :10.2e}  {metrics[  'yaw_std'] :10.2e}  {metrics['yaw_max'  ] :10.2e}  {metrics['yaw_min'] :10.2e}")
                    for i in range(nTowers):
                        print(f"nacelle acc. (m/s) {metrics['AxRNA_avg'][i] :10.2e}  {metrics['AxRNA_std'][i] :10.2e}  {metrics['AxRNA_max'][i] :10.2e}  {metrics['AxRNA_min'][i] :10.2e}")
                    for i in range(nTowers):
                        print(f"tower bending (Nm) {metrics['Mbase_avg'][i] :10.2e}  {metrics['Mbase_std'][i] :10.2e}  {metrics['Mbase_max'][i] :10.2e}  {metrics['Mbase_min'][i] :10.2e}")
                    for i in range(nRotors):
                        if fowt.rotorList[i].Zhub < 0:
                            speed = getFromDict(case, 'current_speed', shape=0, default=1.0)
                        else:
                            speed = getFromDict(case, 'wind_speed', shape=0, default=10.0)
                        if fowt.rotorList[i].aeroServoMod > 1 and speed > 0.0:
                            print(f"rotor speed (RPM)  {metrics['omega_avg'][i] :10.2e}  {metrics['omega_std'][i] :10.2e}  {metrics['omega_max'][i] :10.2e}  {metrics['omega_min'][i] :10.2e}")
                            print(f"blade pitch (deg)  {metrics['bPitch_avg'][i] :10.2e}  {metrics['bPitch_std'][i] :10.2e} ")
                            print(f"rotor power        {metrics['power_avg'][i] :10.2e} ")
                    print(f"-----------------------------------------------------------")

               
 
            # process array-level mooring tension outputs
            if self.ms:
                
                self.results['case_metrics'][iCase]['array_mooring'] = {}
                
                nLines = len(self.ms.lineList) 
                T_moor_amps = np.zeros([nWaves+1, 2*nLines, self.nw], dtype=complex)  # mooring tension amplitudes for each excitation source and line end
                
                C_moor, J_moor = self.ms.getCoupledStiffness(lines_only=True, tensions=True) # get stiffness matrix and tension jacobian matrix
                T_moor = self.ms.getTensions()  # get line end mean tensions
                
            
                for ih in range(nWaves+1):
                    for iw in range(self.nw):
                        T_moor_amps[ih,:,iw] = np.matmul(J_moor, self.Xi[ih,:,iw])   # FFT of mooring tensions
            
                self.results['case_metrics'][iCase]['array_mooring']['Tmoor_avg'] = T_moor
                self.results['case_metrics'][iCase]['array_mooring']['Tmoor_std'] = np.zeros(2*nLines)
                self.results['case_metrics'][iCase]['array_mooring']['Tmoor_max'] = np.zeros(2*nLines)
                self.results['case_metrics'][iCase]['array_mooring']['Tmoor_min'] = np.zeros(2*nLines)
                self.results['case_metrics'][iCase]['array_mooring']['Tmoor_PSD'] = np.zeros([ 2*nLines, self.nw ])
                
                
                for iT in range(2*nLines):
                    TRMS = getRMS(T_moor_amps[:,iT,:]) # estimated mooring line RMS tension [N]
                    self.results['case_metrics'][iCase]['array_mooring']['Tmoor_std'][iT] = TRMS
                    self.results['case_metrics'][iCase]['array_mooring']['Tmoor_max'][iT] = T_moor[iT] + 3*TRMS
                    self.results['case_metrics'][iCase]['array_mooring']['Tmoor_min'][iT] = T_moor[iT] - 3*TRMS
                    self.results['case_metrics'][iCase]['array_mooring']['Tmoor_PSD'][iT,:] = getPSD(T_moor_amps[:,iT,:], self.w[0]) # PSD in N^2/(rad/s)
                    #self.results['case_metrics']['array_mooring']['Tmoor_DEL'][iCase,iT] = 
                
                if display > 0:
            
                    metrics = self.results['case_metrics'][iCase]['array_mooring']
                
                    # print statistics table
                    print(f"-------------------- Mooring Case {iCase+1} Statistics --------------------")
                    print("Response channel     Average     RMS         Maximum     Minimum")
                    for i in range(nLines):
                        j = i+nLines
                        print(f"line {i} tension (N) {metrics['Tmoor_avg'][j]:10.2e}  {metrics['Tmoor_std'][j]:10.2e}  {metrics['Tmoor_max'][j]:10.2e}  {metrics['Tmoor_min'][j]:10.2e}")
                    print(f"-----------------------------------------------------------")
                
                self.T_moor_amps = T_moor_amps  # save for future processing!
    

    def solveEigen(self, display=0):
        '''Compute the natural frequencies and mode shapes of the floating 
        system. When there is a single FOWT, this should give the same result
        as FOWT.solveEigen.
        
        Returns
        -------
        fns : array
            List of natural frequencies [Hz].
        modes : 2D array
            List of mode shapes (eigenvectors) corresponding to the natural 
            frequencies.
        '''

        # total system coefficient arrays
        M_tot = np.zeros([self.nDOF,self.nDOF])       # total mass and added mass matrix [kg, kg-m, kg-m^2]
        C_tot = np.zeros([self.nDOF,self.nDOF])       # total stiffness matrix [N/m, N, N-m]
        
        # include each FOWT's individual mass and stiffness
        for i, fowt in enumerate(self.fowtList):
            i1 = i*6                                              # range of DOFs for the current turbine
            i2 = i*6+6
            
            M_tot[i1:i2, i1:i2] += fowt.M_struc + fowt.A_hydro_morison + fowt.A_BEM[:,:,0] # Mass. Using added mass at w=0 because it is closer to the expected natural frequencies than w=inf
            C_tot[i1:i2, i1:i2] += fowt.C_struc + fowt.C_hydro + fowt.C_moor
            
            # add any additional yaw stiffness that isn't included in the MoorPy model (e.g. if a bridle isn't modeled)
            C_tot[i1+5, i1+5] += fowt.yawstiff
            
        # include array-level mooring stiffness
        if self.ms:
            C_tot += self.ms.getCoupledStiffnessA(lines_only=True)
        
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
        for i in range(self.nDOF-1,-1, -1):
            vec = np.abs(eigenvectors[i,:])  # look at each row (DOF) at a time (use reverse order to pick out rotational DOFs first)

            for j in range(self.nDOF):       # now do another loop in case the index was claimed previously

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
            print("Mode   "+"".join([f"{i+10:3d}"  for i in range(self.nDOF)]))
            print("Fn (Hz)"+"".join([f"{fn:10.4f}" for fn in fns]))
            print("")
            for i in range(self.nDOF):
                print(f"DOF {i+1}  "+"".join([f"{modes[i,j]:10.4f}" for j in range(self.nDOF)]))
            print("-----------------------------------------------------------")

        # store results
        self.results['eigen'] = {}   # signal this data is available by adding a section to the results dictionary
        self.results['eigen']['frequencies'] = fns
        self.results['eigen']['modes'      ] = modes
  
        return fns, modes
    
    
    def solveStatics(self, case, display=0):
        '''
        
        Old notes: To support nonlinear hydrostatics and multiple moorpy instances, this needs to
        become its own solve equilibrium process

        hopefully can just use dsolve2 and its default step func rather than something special
        
        the eval_func will involve:
        - mooring eq (array level and each turbine if applicable)
        - hydrostatics update (should roll,pitch,heave be solved separate from surge sway yaw?)
        - one of the prior two steps should also give device orientation and heave
        - get loads from wind (eventually floris), wave drift, and current (affected by submergence)
        - return total loads
        
        
        statics_mod - 0: linearized hydrostatics; 1: hydrostatics are updated each iteration based on new poses
        forcing_mod - 0: don't update environmental loads; 1: loads are updated each iteration based on new poses
        
        New change: supports either a single wind speed or a list (where there is one wind speed per turbine)
        '''
        
        statics_mod = 0
        forcing_mod = 0
        
        if statics_mod == 0:  # if using linearized hydrostatics approach, get the matrices
            K_hydrostatic = [] #np.zeros([self.nDOF, self.nDOF])   # this will be the constant hydrostatic stiffness matrix--buoyancy and weight terms
            F_undisplaced = np.zeros(self.nDOF)  # force and moment vector before any displacements
        if forcing_mod == 0:  # if using constant environmental mean forcing
            F_env_constant = np.zeros(self.nDOF)  # constant environmental force and moment vector
        
        
        X_initial = np.zeros(self.nDOF)  # position vector of all FOWTs
        
        if case:
            caseorig = copy.deepcopy(case) # save original case data in new dict
            if type(case['wind_speed']) == list :
                if display > 1:  print('List of wind speeds found!')
                
                if len(case['wind_speed']) != len(self.fowtList):
                    raise IndexError("List of wind speeds must be the same length as the list of wind turbines")
            
        # set initial values before solving        
        for i, fowt in enumerate(self.fowtList):
            
            if display > 1:  print(f"FOWT {i+1:}")
        
            X_initial[6*i:6*i+6] = np.array([fowt.x_ref, fowt.y_ref,0,0,0,0])
            fowt.setPosition(X_initial[6*i:6*i+6])      # zero platform offsets
            if case:
                fowt.calcTurbineConstants(case, ptfm_pitch=0)  # for turbine forces >>> still need to update to use current fowt pose <<<
            fowt.calcStatics() # Recompute statics because turbine heading may have changed due to yaw control
            
            if statics_mod == 0:
                K_hydrostatic.append(fowt.C_struc + fowt.C_hydro)
                F_undisplaced[6*i:6*i+6           ] += fowt.W_struc + fowt.W_hydro
                
                if display > 1:  print(" F_undisplaced "+"  ".join(["{:+8.2e}"]*6).format(*F_undisplaced[6*i:6*i+6]))

            if forcing_mod == 0 and case:
                
                # If list of wind speeds, set each turbine case with corresponding wind speed
                if type(caseorig['wind_speed']) == list :
                    case['wind_speed'] = caseorig['wind_speed'][i]
                    if display > 1: 
                        print('Fowt ' + str(i))
                        print(case)

                fowt.calcHydroConstants()
                F_env_constant[6*i:6*i+6] = np.sum(fowt.f_aero0, axis=1) + fowt.calcCurrentLoads(case)

                # Add mean drift if it was already computed.
                # For multiple waves in a given case, it is simply the sum of the mean drifts for each wave.
                # This is not strictly correct, as we would need to compute the QTFs for the combinations between wave headings, but this is a starting point
                if hasattr(fowt, 'Fhydro_2nd_mean'):
                    F_meandrift = np.sum(fowt.Fhydro_2nd_mean, axis=0)
                    F_env_constant[6*i:6*i+6] += F_meandrift
                
                if display > 1:  print(" F_env_constant"+"  ".join(["{:+8.2e}"]*6).format(*F_env_constant[6*i:6*i+6]))
        
        
        # ----- Pass case water current information to MoorPy -----
        
        currentMod = 0  # current modeling mode for MoorPy
        currentU = np.zeros(3)  # uniform current velocity for MoorPy [m/s]
        if case and self.mooring_currentMod > 0:
            cur_speed = getFromDict(case, 'current_speed', shape=0, default=0.0)
            cur_heading = getFromDict(case, 'current_heading', shape=0, default=0)
            if cur_speed > 0:
                currentMod = 1
                currentU = np.array([cur_speed*np.cos(np.radians(cur_heading)),
                                     cur_speed*np.sin(np.radians(cur_heading)), 0])
        
        # Apply current to MoorPy
        if self.ms:
            self.ms.currentMod = currentMod
            self.ms.current = np.array(currentU)
        for fowt in self.fowtList:
            if fowt.ms:
                fowt.ms.currentMod = currentMod
                fowt.ms.current = np.array(currentU)
        
        
        # ----- calculate platform offsets and mooring system equilibrium state -----
        
        # figure out some settings to the equilibrium solve
        db = np.array([30, 30, 5, 0.1, 0.1, 0.1]*len(self.fowtList))  # array for max step size (used manually in step func)
        tols = np.array([0.05,0.05,0.05, 0.005,0.005,0.005]*len(self.fowtList)) # create vector of tolerances - tol = 0.05  rtol = tol/10
        
        
        '''Calculates mean offsets and linearized mooring properties for the current load case.
        setEnv and calcSystemProps must be called first.  This will ultimately become a method for solving mean operating point.
        Mean offsets are saved in the FOWT object.
        '''        
        
        def eval_func_equil(X, args):

            display = args['display']
            
            # set latest positions of each FOWT
            for i, fowt in enumerate(self.fowtList):
                r6 = X[6*i:6*i+6]
                fowt.setPosition(r6)                  # this updates the fowt's position and its own MoorPy system's state (including new F and K)
                if self.ms:
                    self.ms.bodyList[i].setPosition(r6)   # FOWT body in array level MoorPy system
            
            # update array-level mooring system's internal equilibrium (free DOFs only)
            if self.ms:
                self.ms.solveEquilibrium()


            # get updated forces on each FOWT and sum them up
            Fnet = np.zeros(self.nDOF)  # net forces and moments on each DOF across all platforms [N,N,N,Nm,Nm,Nm,N...]
            
            for i, fowt in enumerate(self.fowtList):
                
                Xi0 = X[6*i:6*i+6] - np.array([fowt.x_ref, fowt.y_ref,0,0,0,0])  # fowt mean offset from its reference position

                # update FOWT hydrostatic loads
                if statics_mod == 0 :  # constant linear hydrostatics option
                    Fnet[6*i:6*i+6] += F_undisplaced[6*i:6*i+6]  # add original hydrostatics forces
                    Fnet[6*i:6*i+6] += -np.matmul(K_hydrostatic[i], Xi0) # use stiffness matrix to add hydrostatic reaction forces based on offsets
                elif statics_mod == 1: # switch for whether to recompute hydrostatics
                    fowt.calcStatics()
                    Fnet[6*i:6*i+6] += fowt.W_struc  # weight
                    Fnet[6*i:6*i+6] += fowt.W_hydro  # buoyancy
                    #breakpoint()
                else: 
                    raise Exception('Invalid statics_mod value')
                
                
                # if it's a loaded case, include mean environmental loads
                if case:    # <<<<<<
                    
                    if forcing_mod == 0:  # constant loads approach
                        Fnet[6*i:6*i+6] += F_env_constant[6*i:6*i+6]
                    
                    elif forcing_mod == 1:  # updated loads approach
                    
                        # If list of wind speeds, set each turbine case with corresponding wind speed
                        if type(caseorig['wind_speed']) == list :
                            case['wind_speed'] = caseorig['wind_speed'][i]
                        
                        fowt.calcTurbineConstants(case, ptfm_pitch=r6[4])  # for turbine forces >>> still need to update to use current fowt pose <<<
                        fowt.calcStatics() # Recompute statics because turbine heading may have changed due to yaw control
                        fowt.calcHydroConstants()  # prep for drag force and mean drift

                        Fnet[6*i:6*i+6] += np.sum(fowt.f_aero0, axis=1)  # sum mean turbine force across turbines                        
                        Fnet[6*i:6*i+6] += fowt.calcCurrentLoads(case)  # current drag force  i.e. fowt.D_hydro

                        # mean drift force
                        if hasattr(fowt, 'Fhydro_2nd_mean'):
                            F_meandrift = np.sum(fowt.Fhydro_2nd_mean, axis=0) 
                            Fnet[6*i:6*i+6] += F_meandrift 

                        
                    # This could eventually include FLORIS. If it's slow, FLORIS could be updated only every 5 or 10 iterations...
                
                # mooring forces (includes if currents were updated above)
                Fnet[6*i:6*i+6] += fowt.F_moor0 # fowt.ms.bodyList[0].getForces(lines_only=True)  # individual mooring forces
                if self.ms:
                    Fnet[6*i:6*i+6] += self.ms.bodyList[i].getForces(lines_only=True)     # array-level mooring forces
                
            
            # note that the above also calculates many stiffnes terms that are used in step_func_equil
            
            if display > 1:
                print("Net forces")
                printVec(Fnet)
                
                RMSeForce  = np.linalg.norm([Fnet[6*i  :6*i+3] for i in range(self.nFOWT)])
                RMSeMoment = np.linalg.norm([Fnet[6*i+3:6*i+6] for i in range(self.nFOWT)])
                print(f"Iteration RMS force and moment errors: {RMSeForce:8.2e} {RMSeMoment:8.2e}")
            
            Y = Fnet
            oths = dict(status=1)                # other outputs - returned as dict for easy use
           
            return Y, oths, False
        
        
        def step_func_equil(X, args, Y, oths, Ytarget, err, tol_, iter, maxIter):
            '''This function will get the stiffness of the array, ideally analytically.
            Most stiffness terms should have already been calculated during RAFT functions
            called by eval_func_equil for the current position iteration.
            '''
            
            K = np.zeros ([self.nDOF,self.nDOF])    # total stiffness matrix to be filled in
            
            # add array mooring system stiffness (if applicable)
            if self.ms:
                Kmoor = self.ms.getCoupledStiffnessA(lines_only=True)
                K += Kmoor
            
            # get stiffness of each fowt (hydrostatics, individual mooring, etc.)
            for i, fowt in enumerate(self.fowtList):
                K6 = np.zeros([6,6])

                if statics_mod == 0:
                    K6 += K_hydrostatic[i]
                else:
                    K6 += fowt.C_struc + fowt.C_hydro
                
                if fowt.ms:
                    K6 += fowt.ms.getCoupledStiffnessA(lines_only=True)

                K[6*i:6*i+6, 6*i:6*i+6] += K6
            
            # could get any stiffness effects from wakes or currents, though probably negligible
            
            # TODO: if there isn't any array-level stiffness coupling, could simply solve each fowt individually <<<

            
            # --- adjust positions according to stiffness matrix to move toward net zero forces ---
            
            kmean = np.mean(K.diagonal()) # mean value of diagonal stiffness entries
            
            for i in range(self.nDOF):   # go through DOFs and adjust any zero stiffness diagonals
                if K[i,i] == 0:
                    K[i,i] = kmean   # apply some stifness just to keep things working...                    
                elif K[i,i] < 0:
                    pass #breakpoint() <<<
            
            try:
                if self.nDOF > 36: # if huge, count on the system being sparse and use a sparse solver
                    # import relevant packages
                    import warnings
                    from scipy.sparse import csr_matrix
                    from scipy.sparse.linalg import spsolve, MatrixRankWarning

                    with warnings.catch_warnings():
                        warnings.simplefilter("error", category=MatrixRankWarning)
                        Kcsr = csr_matrix(K)
                        dX = spsolve(Kcsr, Y)
                
                else:  # normal approach
                    dX = np.linalg.solve(K, Y)   # calculate position adjustment according to Newton's method

                    if np.linalg.det(K) < 0:
                        print(f" XXXX Determinant is {np.linalg.det(K)} while sum of dx*y is {sum(dX*Y)}")
                   
                    # check sign for backward result (potentially a result of bad numerics?) and strengthen diagonals if so to straighten it out
                    for iTry in range(10):
                        if sum(dX*Y) < 0:
                            print(" XXXX sum(dX*Y) is negative so enlarging the diagonals")
                            for i in range(self.nDOF):
                                K[i,i] += 0.1*abs(K[i,i]) # increase the diagonal entries as a hack
                        
                            dX = np.linalg.solve(K, Y)  
                            
                        else:  # (this is when things are good)
                            #print(f" UPDATEdet is {np.linalg.det(K)} while sum of dx*y is {sum(dX*Y)}  after {iTry} adjustments")
                            break
              
            except Exception as ex:
                print(f"EXCEPTION  "+str(ex))
                
                print("trying to enlarge the diagonals")
                for i in range(self.nDOF):
                    K[i,i] += K[i,i] # double the diagonal entries as a hack
                    
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", category=MatrixRankWarning)
                    Kcsr = csr_matrix(K)
                    dX = spsolve(Kcsr, Y)
                    print('worked')    
                except Exception as e2:
                    dX = Y/np.diag(K)
                    print('failed'+str(e2)+" after "+str(ex))
            
            return dX
        
        
        # Now find static equilibrium offsets 
        X, Y, info = dsolve2(eval_func_equil, X_initial, step_func=step_func_equil, 
                             tol=tols, a_max=1.6, maxIter=20, display=0, args={'display': display} ) #, dodamping=True)

        if display > 1:
            RMSeForce  = np.linalg.norm([Y[6*i  :6*i+3] for i in range(self.nFOWT)])
            RMSeMoment = np.linalg.norm([Y[6*i+3:6*i+6] for i in range(self.nFOWT)])
            if RMSeForce > 1000 or RMSeMoment > 1000:
                print('Warning: RMS error of equilibrium forces or moments exceeds 1000.')
        
        if display > 0:
            print('New Equilibrium Position', X)
            print('Remaining Forces on the Model (N)', Y)
        
        self.Xs2 = info['Xs']    # List of positions as it finds equilibrium for every iteration
        self.Es2 = info['Es']    # List of errors that the forces are away from 0, which in this case, is the same as the forces
        
        if case and 'iCase' in case:
            self.results['mean_offsets'].append(self.Xs2[-1])  # save the final equilibrium position for this case
        
        for i, fowt in enumerate(self.fowtList):
            print(f"Found mean offets of FOWT {i+1} with surge = {fowt.Xi0[0]: .2f} m,  sway  = {fowt.Xi0[1]: .2f},  and heave = {fowt.Xi0[2]: .2f} m")
            print(f"                                 roll  = {fowt.Xi0[3]*180/np.pi: .2f} deg, pitch = {fowt.Xi0[4]*180/np.pi: .2f} deg, and yaw   = {fowt.Xi0[5]*180/np.pi: .2f} deg")

        
        #dsolvePlot(info) # plot solver convergence trajectories
        
        ''' TODO: following sections should be checked and streamlined >>>

        
        try:
            C_moor, J_moor = self.ms.getCoupledStiffness(lines_only=True, tensions=True) # get stiffness matrix and tension jacobian matrix
            F_moor = self.ms.getForces(DOFtype="coupled", lines_only=True)    # get net forces and moments from mooring lines on Body
            T_moor = self.ms.getTensions()
        except Exception as e:
            raise RuntimeError('An error occured when getting linearized mooring properties in offset state: '+e.message)
            
        # add any additional yaw stiffness that isn't included in the MoorPy model (e.g. if a bridle isn't modeled)
        C_moor[5,5] += fowt.yawstiff

        self.C_moor = C_moor
        self.J_moor = J_moor        # jacobian of mooring line tensions w.r.t. coupled DOFs
        self.F_moor = F_moor
        self.T_moor = T_moor
        
        # store results
        self.results['means'] = []   # signal this data is available by adding a section to the results dictionary
        for i, fowt in enumerate(self.fowtList):
            self.results['means'].append({})
            self.results['means'][i]['aero force'  ] = fowt.f_aero0
            self.results['means'][i]['platform offset'  ] = fowt.r6
            self.results['means'][i]['mooring force'    ] = F_moor
            self.results['means'][i]['fairlead tensions'] = np.array([np.linalg.norm(self.ms.pointList[id-1].getForces()) for id in self.ms.bodyList[0].attachedP])
        
        
        # mean tower base bending moment
        m_turbine = np.zeros([len(self.fowtList), max([len(self.fowtList[j].mtower) for j in range(len(self.fowtList))])])
        zCG_turbine = np.zeros_like(m_turbine)
        zBase = np.zeros_like(m_turbine)
        hArm = np.zeros_like(m_turbine)
        self.results['means']['Mbase'] = np.zeros_like(m_turbine)
        for j in range(len(self.fowtList)):
            for i in range(len(self.fowtList[j].mtower)):
                m_turbine[j,i] = self.fowtList[j].mtower[i] + self.fowtList[j].mRNA[i]          # total masses of each turbine
                zCG_turbine[j,i] = (self.fowtList[j].rCG_tow[i][2]*self.fowtList[j].mtower[i]  # CoG of each turbine
                                    + self.fowtList[j].hHub[i]*self.fowtList[j].mRNA[i])/m_turbine[j,i]
                zBase[j,i] = self.fowtList[j].memberList[self.fowtList[j].nplatmems + i].rA[2]  # tower base elevation [m]
                hArm[j,i] = zCG_turbine[j,i] - zBase[j,i]                                                  # vertical distance from tower base to turbine CG [m]
                self.results['means']['Mbase'][j,i] = m_turbine[j,i]*self.fowtList[j].g * hArm[j,i]*np.sin(r6eq[4]) + transformForce(self.fowtList[j].f_aero0[:,i], offset=[0,0,-hArm[j,i]])[4] # mean moment from weight and thrust
        
                
        # update values based on offsets if applicable
        for fowt in self.fowtList:
            fowt.calcTurbineConstants(case, ptfm_pitch=fowt.Xi0[4])
            # fowt.calcHydroConstants(case)  (hydrodynamics don't account for offset, so far) 
            # <<<<< can change the above once we support nonlinear hydrostatics
        
        # (could solve mooring and offsets a second time, but likely overkill)
        # self.calcMooringAndOffsets()
        '''
        

    def solveDynamics(self, case, tol=0.01, conv_plot=0, RAO_plot=0, display=0):
        '''After all constant parts have been computed, call this to iterate through remaining terms
        until convergence on dynamic response. Note that steady/mean quantities are excluded here.
        '''
        
        iCase = None
        if 'iCase' in case:
            iCase = case['iCase']

        nIter = int(self.nIter) + 1         # maybe think of a better name for the first nIter
        XiStart = self.XiStart

        # fowt matrices
        M_lin = []
        B_lin = []
        C_lin = []
        F_lin = []
        
        if conv_plot:
            fig, ax = plt.subplots(3,1,sharex=True)
            c = np.arange(nIter+1)      # adding 1 again here so that there are no RuntimeErrors
            c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))      # set up colormap to use to plot successive iteration results

        # Loop through each fowt to calculate its independent response to wave excitation.
        # This is the iterative linearization stage to get individual impedance matrices.
        for i, fowt in enumerate(self.fowtList):
            i1 = i*6                                              # range of DOFs for the current turbine
            i2 = i*6+6
            
            # total FOWT complex response amplitudes (this gets updated each iteration)
            XiLast = np.zeros([fowt.nDOF,self.nw], dtype=complex) + XiStart    # displacement and rotation complex amplitudes [m, rad]
            
            # calculate linear wave excitation forces for this case  (THIS IS A NEW WAY OF DOING THIS)<<<
            fowt.calcHydroExcitation(case, memberList=fowt.memberList)
            
            # add up coefficients for any number of turbines
            if fowt.nrotors> 0:
                M_turb = np.sum(fowt.A_aero, axis=3)
                B_turb = np.sum(fowt.B_aero, axis=3)
            else:
                M_turb = np.zeros([6,6,self.nw])
                B_turb = np.zeros([6,6,self.nw])
                
            # >>>> NOTE: Turbulent wind excitation is currently disabled pending formulation checks/fixes <<<<
            if display > 0:
                print('Solving for system response to wave excitation in primary wave direction')

            # We can compute second-order hydrodynamic forces here if they are calculated using external QTF file.
            # In some cases, they may be very relevant to the motion RMS values, so should be included in the drag linearization process.          
            fowt.Fhydro_2nd = np.zeros([fowt.nWaves, fowt.nDOF, fowt.nw], dtype=complex) 
            fowt.Fhydro_2nd_mean = np.zeros([fowt.nWaves, fowt.nDOF])
            if fowt.potSecOrder==2:
                fowt.Fhydro_2nd_mean[0, :], fowt.Fhydro_2nd[0, :, :] = fowt.calcHydroForce_2ndOrd(fowt.beta[0], fowt.S[0,:], iCase=iCase, iWT=i)

            # We use this flag to know if we have computed the QTFs already. It's used when fowt.potSecOrder==1, 
            # so that we compute the QTFs including first-order motions only.
            flagComputedQTF = False

            # sum up all linear (non-varying) matrices up front, including potential summation across multiple rotors
            M_lin.append( M_turb + fowt.M_struc[:,:,None] + fowt.A_BEM + fowt.A_hydro_morison[:,:,None]        ) # mass
            B_lin.append( B_turb + fowt.B_struc[:,:,None] + fowt.B_BEM + np.sum(fowt.B_gyro, axis=2)[:,:,None] ) # damping
            C_lin.append(          fowt.C_struc   + fowt.C_moor        + fowt.C_hydro                          ) # stiffness
            F_lin.append( fowt.F_BEM[0,:,:] + fowt.F_hydro_iner[0,:,:] + fowt.Fhydro_2nd[0, :, :]) # consider only excitation from the primary sea state in the load case for now

            # start fixed point iteration loop for dynamics of the individual FOWT
            iiter = 0
            while iiter < nIter:
                
                # initialize/zero total system coefficient arrays
                M_tot = np.zeros([fowt.nDOF,fowt.nDOF,self.nw])       # total mass and added mass matrix [kg, kg-m, kg-m^2]
                B_tot = np.zeros([fowt.nDOF,fowt.nDOF,self.nw])       # total damping matrix [N-s/m, N-s, N-s-m]
                C_tot = np.zeros([fowt.nDOF,fowt.nDOF,self.nw])       # total stiffness matrix [N/m, N, N-m]
                F_tot = np.zeros([fowt.nDOF,self.nw], dtype=complex)  # total excitation force/moment complex amplitudes vector [N, N-m]

                Z  = np.zeros([fowt.nDOF,fowt.nDOF,self.nw], dtype=complex)  # total  fowt impedance matrix
                
                # get linearized terms for the current turbine given latest amplitudes
                B_linearized = fowt.calcHydroLinearization(XiLast)
                F_linearized = fowt.calcDragExcitation(0)  # just looking at first sea state (wave heading) for the sake of linearization
                
                # calculate the response based on the latest linearized terms
                Xi = np.zeros([fowt.nDOF,self.nw], dtype=complex)     # displacement and rotation complex amplitudes [m, rad]

                # add fowt's terms to system matrices (BEM arrays are not yet included here)
                M_tot[:,:,:] = M_lin[i]
                B_tot[:,:,:] = B_lin[i]           + B_linearized[:,:,None]
                C_tot[:,:,:] = C_lin[i][:,:,None]
                F_tot[:  ,:] = F_lin[i]           + F_linearized


                for ii in range(self.nw):
                    # form impedance matrix
                    Z[:,:,ii] = -self.w[ii]**2 * M_tot[:,:,ii] + 1j*self.w[ii]*B_tot[:,:,ii] + C_tot[:,:,ii]
                    
                    # solve response (complex amplitude)
                    Xi[:,ii] = np.linalg.solve(Z[:,:,ii], F_tot[:,ii])

                if conv_plot:
                    # Convergence Plotting
                    # plots of surge response at each iteration for observing convergence
                    ax[0].plot(self.w, np.abs(Xi[0,:]) , color=c[iiter], label=f"iteration {iiter}")
                    ax[1].plot(self.w, np.real(Xi[0,:]), color=c[iiter], label=f"iteration {iiter}")
                    ax[2].plot(self.w, np.imag(Xi[0,:]), color=c[iiter], label=f"iteration {iiter}")
                
                if any(np.isnan(Xi).ravel()):
                    raise Exception("Nan detected in response vector Xi.")
                
                
                # check for convergence
                tolCheck = np.abs(Xi - XiLast) / ((np.abs(Xi)+tol))
                if (tolCheck < tol).all():
                    if display > 1:
                        print(f" Iteration {iiter}, converged (largest change is {np.max(tolCheck):.5f} < {tol})")

                    if fowt.potSecOrder != 1 or flagComputedQTF:
                        break
                    else:
                        # If we are computing the QTFs internally, we need to consider first-order body motions forces to compute the QTFs.
                        # So, we use the motions obtained after the linearized drag loop above and then we loop again to include the 
                        # second-order motions in the drag linearization procedure. 
                        # This is important for cases where the second-order motions are large compared to the first-order motions.
                        iiter = 0
                        if display > 1:
                            print(f"Resolving for system response in primary wave direction, now with second-order wave loads.")

                        # Get the response amplitude operators (RAOs, i.e. motions for unit wave amplitude)
                        Xi0 = getRAO(Xi[i1:i2, :], fowt.zeta[0,:])
                                                
                        tic = time.perf_counter() # Time the QTF calculation
                        fowt.calcQTF_slenderBody(waveHeadInd=0, Xi0=Xi0, verbose=True, iCase=iCase, iWT=i)
                        toc = time.perf_counter()
                        if display > 1:
                            print(f"\n Time to compute QTFs for fowt {i}: {toc - tic:0.4f} seconds")

                        # After computing the QTFs internally, we can now compute the second-order hydrodynamic forces
                        fowt.Fhydro_2nd_mean[0, :], fowt.Fhydro_2nd[0, :, :] = fowt.calcHydroForce_2ndOrd(fowt.beta[0], fowt.S[0,:], iCase=iCase, iWT=i)
                        F_lin[i1:i2] += fowt.Fhydro_2nd[0, :, :]
                        flagComputedQTF = True # Flag that we have computed the QTFs already and we don't need to do it again.
                else:
                    XiLast = 0.2*XiLast + 0.8*Xi    # use a mix of the old and new response amplitudes to use for the next iteration
                                                    # (uses hard-coded successive under relaxation for now)
                    if display > 2:
                        print(f" Iteration {iiter}, unconverged (largest change is {np.max(tolCheck):.5f} >= {tol})")
        
                if iiter == nIter-1:
                    if display > 0:
                        print("WARNING - solveDynamics iteration did not converge to the tolerance.")

                iiter += 1
            
            if conv_plot:
                # labels for convergence plots
                ax[1].legend()
                ax[0].set_ylabel("response magnitude")
                ax[1].set_ylabel("response, real")
                ax[2].set_ylabel("response, imag")
                ax[2].set_xlabel("frequency (rad/s)")
                fig.suptitle("Response convergence")
        
        
            # Save the FOWT's impedance matrix
            fowt.Z = Z
        
        # Now that invididual FOWT impedences matrices have been found, construct the 
        # system-level matrices (in case of couplings) and compute the total response
        # including multiple excitation sources.
        
        # 0. Construct full system matrices
        
        Z_sys = np.zeros([self.nDOF,self.nDOF,self.nw], dtype=complex)   # total system impedance matrix
        
        # include each FOWT's individual impedance matrix
        for i, fowt in enumerate(self.fowtList):
            i1 = i*6                                              # range of DOFs for the current turbine
            i2 = i*6+6
            Z_sys[i1:i2, i1:i2] += fowt.Z
        
        # include array-level mooring stiffness
        if self.ms:
            Z_sys += self.ms.getCoupledStiffnessA(lines_only=True)[:,:,None]
        
        
        # >>> For arrays, we would want a sparse solver for Zinv. <<<
        
        # 1. get latest impedence matrix and invert it
        #Z = Z (already done)
        Zinv  = np.zeros([self.nDOF,self.nDOF,self.nw], dtype=complex)  # total system impedance matrix
        for iw in range(self.nw):
            Zinv[:,:,iw] = np.linalg.inv(Z_sys[:,:,iw])
        
        # 2. calculate response for each source of excitation
        # This is the sytem response tensor, including for each excitation type.
        self.Xi = np.zeros([self.fowtList[0].nWaves+1,self.nDOF,self.nw], dtype=complex)  
        
        # >>> TODO: need to make a system-level wave description, and nWaves value <<<
        
        # wave excitation
        for ih in range(fowt.nWaves):
            
            F_wave = np.zeros([self.nDOF, self.nw], dtype=complex)  # system wave excitation vector for this wave
        
            for i, fowt in enumerate(self.fowtList):
                i1, i2 = i*6, i*6+6
                
                # calculate linear and nonlinear wave excitation for this FOWT and case (consider phasing due to position in array)
                fowt.calcHydroExcitation(case, memberList=fowt.memberList)
                F_linearized = fowt.calcDragExcitation(ih)
                if fowt.potSecOrder==2 and ih > 0:
                    fowt.Fhydro_2nd_mean[ih, :], fowt.Fhydro_2nd[ih, :, :] = fowt.calcHydroForce_2ndOrd(fowt.beta[ih], fowt.S[ih,:])
                F_wave[i1:i2] = fowt.F_BEM[ih,:,:] + fowt.F_hydro_iner[ih,:,:] + F_linearized + fowt.Fhydro_2nd[ih,:,:]
                
            # compute system response
            for iw in range(self.nw):
                self.Xi[ih,:,iw] = np.matmul(Zinv[:,:,iw], F_wave[:,iw])
        

            # If we are computing the QTFs internally, we need to consider the motions induced by first-order hydrodynamic forces, which were computed above
            # TODO: Not very nice to keep the same code twice. Maybe we can move it to a function?            
            for i, fowt in enumerate(self.fowtList):
                i1, i2 = i*6, i*6+6
                if fowt.potSecOrder == 1:
                    # Don't recompute the QTFs for the first wave because it was already done above.
                    # Also, we would end up including second-order motions if we computed it again.
                    if ih > 0: 
                        Xi0 = getRAO(self.Xi[ih,i1:i2, :], fowt.zeta[ih,:])                        
                        fowt.calcQTF_slenderBody(waveHeadInd=ih, Xi0=Xi0, verbose=True, iCase=iCase, iWT=i)                        
                        fowt.Fhydro_2nd_mean[ih, :], fowt.Fhydro_2nd[ih, :, :] = fowt.calcHydroForce_2ndOrd(fowt.beta[ih], fowt.S[ih,:])
                
                    # Recompute the wave excitation forces and consequent motions to include second-order hydrodynamic forces
                    F_wave[i1:i2] = fowt.F_BEM[ih,:,:] + fowt.F_hydro_iner[ih,:,:] + F_linearized + fowt.Fhydro_2nd[ih, :, :]
                    for iw in range(self.nw):
                        self.Xi[ih,:,iw] = np.matmul(Zinv[:,:,iw], F_wave[:,iw])
        
        # rotor excitation
        '''
        F_rotor = np.zeros([self.nDOF, self.nw], dtype=complex)
        
        for i, fowt in enumerate(self.fowtList):
            F_rotor[i*6:i*6+6] = np.sum(fowt.F_aero, axis=2)
            
        for iw in range(self.nw):
            self.Xi[-1,:,iw] = np.matmul(Zinv[:,:,iw], F_rotor[:,iw])
        '''
        
        # store all the results in the FOWT object 
        for i, fowt in enumerate(self.fowtList):
            fowt.Xi = self.Xi[:, i*6:i*6+6, :]  # this overwrites the response in the FOWT with what's been calculated
        

        # ------------------------------ preliminary plotting of response ---------------------------------
        
        if RAO_plot:
            # response amplitude plotting (for first wave heading)
            
            for i, fowt in enumerate(self.fowtList):
            
                fig, ax = plt.subplots(7,1, sharex=True)
        
                ax[0].plot(self.w, np.abs(fowt.Xi[0,0,:])          , 'k' , label="magnitude")
                ax[1].plot(self.w, np.abs(fowt.Xi[0,1,:])          , 'k' )
                ax[2].plot(self.w, np.abs(fowt.Xi[0,2,:])          , 'k' )
                ax[3].plot(self.w, np.abs(fowt.Xi[0,3,:])*180/np.pi, 'k' )
                ax[4].plot(self.w, np.abs(fowt.Xi[0,4,:])*180/np.pi, 'k' )
                ax[5].plot(self.w, np.abs(fowt.Xi[0,5,:])*180/np.pi, 'k' )
                ax[6].plot(self.w, fowt.zeta[0,:]                  , 'k' )
        
                ax[0].plot(self.w, np.real(fowt.Xi[0,0,:])          , ':g', label='real')
                ax[1].plot(self.w, np.real(fowt.Xi[0,1,:])          , ':g')
                ax[2].plot(self.w, np.real(fowt.Xi[0,2,:])          , ':g')
                ax[3].plot(self.w, np.real(fowt.Xi[0,3,:])*180/np.pi, ':g')
                ax[4].plot(self.w, np.real(fowt.Xi[0,4,:])*180/np.pi, ':g')
                ax[5].plot(self.w, np.real(fowt.Xi[0,5,:])*180/np.pi, ':g')
                
                ax[0].plot(self.w, np.imag(fowt.Xi[0,0,:])          , ':r', label='imag')
                ax[1].plot(self.w, np.imag(fowt.Xi[0,1,:])          , ':r')
                ax[2].plot(self.w, np.imag(fowt.Xi[0,2,:])          , ':r')
                ax[3].plot(self.w, np.imag(fowt.Xi[0,3,:])*180/np.pi, ':r')
                ax[4].plot(self.w, np.imag(fowt.Xi[0,4,:])*180/np.pi, ':r')
                ax[5].plot(self.w, np.imag(fowt.Xi[0,5,:])*180/np.pi, ':r')
                
                ax[0].legend()
        
                ax[0].set_ylabel("Surge (m)")
                ax[1].set_ylabel("Sway (m)")
                ax[2].set_ylabel("Heave (m)")
                ax[3].set_ylabel("Roll (deg)")
                ax[4].set_ylabel("Pitch (deg)")
                ax[5].set_ylabel("Yaw (deg)")
                ax[6].set_ylabel("wave amplitude (m)")
                ax[6].set_xlabel("frequency (rad/s)")


        self.results['response'] = {}   # signal this data is available by adding a section to the results dictionary

        return self.Xi  # is it better to return the response or save it in the model object? Or in the FOWT objects? <<<



    def calcOutputs(self):
        '''This is where various output quantities of interest are calculated based on the already-solved system response.'''
        
        fowt = self.fowtList[0]   # just using a single turbine for now
        
        
        # ----- system properties outputs -----------------------------
        # all values about platform reference point (z=0) unless otherwise noted
        
        if 'properties' in self.results:
        
            self.results['properties']['tower mass'] = fowt.mtower
            self.results['properties']['tower CG'] = fowt.rCG_tow
            self.results['properties']['substructure mass'] = fowt.m_sub
            self.results['properties']['substructure CG'] = fowt.rCG_sub
            self.results['properties']['shell mass'] = fowt.m_shell
            self.results['properties']['ballast mass'] = fowt.m_ballast
            self.results['properties']['ballast densities'] = fowt.pb
            self.results['properties']['total mass'] = fowt.M_struc[0,0]
            self.results['properties']['total CG'] = fowt.rCG
            self.results['properties']['roll inertia at subCG']  = fowt.props['Ixx_sub']
            self.results['properties']['pitch inertia at subCG'] = fowt.props['Iyy_sub']
            self.results['properties']['yaw inertia at subCG']   = fowt.props['Izz_sub']
            
            self.results['properties']['buoyancy (pgV)'] = fowt.rho_water*fowt.g*fowt.V
            self.results['properties']['center of buoyancy'] = fowt.rCB
            self.results['properties']['C hydrostatic'] = fowt.C_hydro
            self.results['properties']['C system'] = fowt.C_struc + fowt.C_hydro + self.C_moor0
            
            # unloaded equilibrium <<< 
            
            self.results['properties']['F_lines0'] = self.F_moor0
            self.results['properties']['C_lines0'] = self.C_moor0
                    
            # 6DOF matrices for the support structure (everything but turbine) including mass, hydrostatics, and mooring reactions
            self.results['properties']['M support structure'] = fowt.M_struc_sub          # mass matrix (about PRP)
            self.results['properties']['A support structure'] = fowt.A_hydro_morison + fowt.A_BEM[:,:,-1]   # hydrodynamic added mass (currently using highest frequency of BEM added mass)
            self.results['properties']['C support structure'] = fowt.C_struc_sub + fowt.C_hydro + self.C_moor0  # stiffness

        return self.results
        
        
    

    def plotResponses(self):
        '''Plots the power spectral densities of the available response channels for each case.'''
        
        fig, ax = plt.subplots(6, 1, sharex=True, figsize=(6,6))
        
        # loop through each FOWT and plot its response (on the same figure for now)
        for i in range(self.nFOWT):
        
            
            nCases = len(self.results['case_metrics'])
            
            for iCase in range(nCases):
                metrics = self.results['case_metrics'][iCase][i]
                ax[0].plot(self.w/TwoPi, TwoPi*metrics['surge_PSD']    )  # surge
                ax[1].plot(self.w/TwoPi, TwoPi*metrics['heave_PSD']    )  # heave
                ax[2].plot(self.w/TwoPi, TwoPi*metrics['pitch_PSD']    )  # pitch [deg]
                ax[3].plot(self.w/TwoPi, TwoPi*metrics['AxRNA_PSD']    )  # nacelle acceleration
                ax[4].plot(self.w/TwoPi, TwoPi*metrics['Mbase_PSD']    )  # tower base bending moment (using FAST's kN-m)
                ax[5].plot(self.w/TwoPi, TwoPi*metrics['wave_PSD' ], label=f'FOWT {i+1}; Case {iCase+1}')  # wave spectrum

                # need a variable number of subplots for the mooring lines
                #ax2[3].plot(model.w/2/np.pi, TwoPi*metrics['Tmoor_PSD'][0,3,:]  )  # fairlead tension

        ax[0].set_ylabel('surge \n'+r'(m$^2$/Hz)')
        ax[1].set_ylabel('heave \n'+r'(m$^2$/Hz)')
        ax[2].set_ylabel('pitch \n'+r'(deg$^2$/Hz)')
        ax[3].set_ylabel('nac. acc. \n'+r'((m/s$^2$)$^2$/Hz)')
        ax[4].set_ylabel('twr. bend \n'+r'((Nm)$^2$/Hz)')
        ax[5].set_ylabel('wave elev.\n'+r'(m$^2$/Hz)')

        ax[-1].set_xlabel('frequency (Hz)')
        
        ax[-1].legend()
        fig.suptitle('RAFT power spectral densities')
        fig.tight_layout()


    def saveResponses(self, outPath):
        '''Save the power spectral densities of the available response channels for each case to an output file.'''
        
        chooseMetrics = ['wave_PSD', 'surge_PSD', 'heave_PSD', 'pitch_PSD', 'AxRNA_PSD', 'Mbase_PSD']
        metricUnit    = ['m^2/Hz', 'm^2/Hz', 'm^2/Hz', 'deg^2/Hz', '(m/s^2)^2/Hz', '(Nm)^2/Hz']
        
        for i in range(self.nFOWT):
        
            nCases = len(self.results['case_metrics'])
            
            for iCase in range(nCases):
                metrics = self.results['case_metrics'][iCase][i]
                with open(f'{outPath}_Case{iCase+1}_WT{i}.txt', 'w') as file:
                    # Write the header
                    file.write('Frequency [rad/s] \t')
                    for metric, unit in zip(chooseMetrics, metricUnit):
                        file.write(f'{metric} [{unit}] \t')
                    file.write('\n')

                    # Write the data
                    for iFreq in range(len(self.w)):
                        file.write(f'{self.w[iFreq]:.5f} \t')
                        for metric in chooseMetrics:
                            file.write(f'{np.squeeze(metrics[metric][iFreq]):.5f} \t')
                        file.write('\n')

                # if self.results['mean_offsets']:
                #     with open(f'{outPath}_Case{iCase+1}_WT{i}_meanOffsets.txt', 'w') as file:
                #         file.write('Surge [m] \t Sway [m] \t Heave [m] \t Pitch [deg] \t Roll [deg] \t Yaw [deg] \n')
                #         mean_offsets = self.results['mean_offsets'][iCase]
                #         file.write(f'{mean_offsets[0]:.5f} \t {mean_offsets[1]:.5f} \t {mean_offsets[2]:.5f} \t {mean_offsets[3]:.5f} \t {mean_offsets[4]:.5f} \t {mean_offsets[5]:.5f} \n')


    def plotResponses_extended(self):
        '''Plots more power spectral densities of the available response channels for each case.'''

        fig, ax = plt.subplots(9, 1, sharex=True)
        
        # loop through each FOWT and plot its response (on the same figure for now)
        for i in range(self.nFOWT):
        
            nCases = len(self.results['case_metrics'])
            
            for iCase in range(nCases):
                metrics = self.results['case_metrics'][iCase][i]
                ax[0].plot(self.w / TwoPi, TwoPi * metrics['surge_PSD'][:])  # surge
                ax[1].plot(self.w / TwoPi, TwoPi * metrics['sway_PSD'][:])  # surge
                ax[2].plot(self.w / TwoPi, TwoPi * metrics['heave_PSD'][:])  # heave
                ax[3].plot(self.w / TwoPi, TwoPi * metrics['pitch_PSD'][:])  # pitch [deg]
                ax[4].plot(self.w / TwoPi, TwoPi * metrics['roll_PSD'][:])  # pitch [deg]
                ax[5].plot(self.w / TwoPi, TwoPi * metrics['yaw_PSD'][:])  # pitch [deg]
                ax[6].plot(self.w / TwoPi, TwoPi * metrics['AxRNA_PSD'][:])  # nacelle acceleration
                ax[7].plot(self.w / TwoPi,
                           TwoPi * metrics['Mbase_PSD'][:])  # tower base bending moment (using FAST's kN-m)
                ax[8].plot(self.w / TwoPi, TwoPi * metrics['wave_PSD'][ :],
                           label=f'case {iCase + 1}')  # wave spectrum

                # need a variable number of subplots for the mooring lines
                # ax2[3].plot(model.w/2/np.pi, TwoPi*metrics['Tmoor_PSD'][0,3,:]  )  # fairlead tension

        ax[0].set_ylabel('surge \n' + r'(m$^2$/Hz)')
        ax[1].set_ylabel('sway \n' + r'(m$^2$/Hz)')
        ax[2].set_ylabel('heave \n' + r'(m$^2$/Hz)')
        ax[3].set_ylabel('pitch \n' + r'(deg$^2$/Hz)')
        ax[4].set_ylabel('roll \n' + r'(deg$^2$/Hz)')
        ax[5].set_ylabel('yaw \n' + r'(deg$^2$/Hz)')
        ax[6].set_ylabel('nac. acc. \n' + r'((m/s$^2$)$^2$/Hz)')
        ax[7].set_ylabel('twr. bend \n' + r'((Nm)$^2$/Hz)')
        ax[8].set_ylabel('wave elev.\n' + r'(m$^2$/Hz)')

        ax[-1].set_xlabel('frequency (Hz)')

        # if nCases > 1:
        ax[-1].legend()
        fig.suptitle('RAFT power spectral densities')

        
        

    def preprocess_HAMS(self, dw=0, wMax=0, dz=0, da=0):
        '''This generates a mesh for the platform, runs a BEM analysis on it
        using pyHAMS, and writes .1 and .3 output files for use with OpenFAST.
        The input parameters are useful for multifidelity applications where 
        different levels have different accuracy demands for the HAMS analysis.
        The mesh is only made for non-interesecting members flagged with potMod=1.
        
        PARAMETERS
        ----------
        dw : float
            Optional specification of custom frequency increment (rad/s).
        wMax : float
            Optional specification of maximum frequency for BEM analysis (rad/s). Will only be
            used if it is greater than the maximum frequency used in RAFT.
        dz : float
            desired longitudinal panel size for potential flow BEM analysis (m)
        da : float
            desired azimuthal panel size for potential flow BEM analysis (m)
        '''
        
        self.fowtList[0].calcBEM(dw=dw, wMax=wMax, dz=dz, da=da)


    def plot(self, ax=None, hideGrid=False, draw_body=True, color=None, nodes=0, 
             xbounds=None, ybounds=None, zbounds=None, plot_rotor=True, airfoils=False, 
             station_plot=[], zorder=2, figsize=(6,4), plot_fowt=True, plot_ms=True, 
             shadow=True, plot_water=False, plot_soil=False, mp_args={}):
        '''plots the whole model, including FOWTs and mooring system
        
        mp_args
            additional arguments passed to the array-level MoorPy plot method.
        '''

        # prepare arguments to MoorPy
        mp_args2 = dict(color=color, draw_body=draw_body, xbounds=xbounds, ybounds=ybounds, zbounds=zbounds)
        mp_args2.update(mp_args)
        
        
        # if axes not passed in, make a new figure
        if ax == None:    
            if self.ms:
                fig, ax = self.ms.plot(figsize=figsize, **mp_args2)
            else:   
                fig = plt.figure(figsize=figsize)
                ax = plt.axes(projection='3d')
                
                if xbounds != None:
                    ax.set_xlim(xbounds[0], xbounds[1])
                if ybounds != None:
                    ax.set_ylim(ybounds[0], ybounds[1])
                if zbounds != None:
                    ax.set_zlim(zbounds[0], zbounds[1])
        else:
            fig = ax.get_figure()
            if self.ms:
                self.ms.plot(ax=ax, **mp_args2)

        # plot each FOWT
        for fowt in self.fowtList:
            fowt.plot(ax, color=color, zorder=zorder, nodes=nodes, 
                    plot_rotor=plot_rotor, station_plot=station_plot, 
                    airfoils=airfoils, plot_ms=plot_ms, plot_fowt=plot_fowt, shadow=shadow, mp_args=mp_args)
        
        set_axes_equal(ax)
        
        if hideGrid:       
            ax.set_xticks([])    # Hide axes ticks
            ax.set_yticks([])
            ax.set_zticks([])     
            ax.grid(False)       # Hide grid lines
            ax.grid(b=None)
            ax.axis('off')
            ax.set_frame_on(False)

        r = 75 #0.1*np.maximum(vx.max(), vy.max())

        if plot_water:
            water_color = (0.122, 0.4667, 0.706)
            p_sea = Circle((0,0), r, color=water_color, alpha=0.3)
            ax.add_patch(p_sea)
            art3d.pathpatch_2d_to_3d(p_sea, z=0, zdir="z")
        if plot_soil:
            soil_color = (0.703125, 0.390625, 0.0)
            p_soil = Circle((0,0), r, color=soil_color, alpha=0.5)
            ax.add_patch(p_soil)
            art3d.pathpatch_2d_to_3d(p_soil, z=-self.depth, zdir="z")
            
        return fig, ax
    
    
    def plot2d(self, ax=None, hideGrid=False, draw_body=True, color=None, 
               station_plot=[], Xuvec=[1,0,0], Yuvec=[0,0,1], figsize=(6,4),
               plot_rotor=2):
        '''plots the whole model, including FOWTs and mooring system...'''

        # if axes not passed in, make a new figure
        if ax == None:    
            if self.ms:
                fig, ax = self.ms.plot2d(color=color, draw_body=draw_body, Xuvec=Xuvec, Yuvec=Yuvec, figsize=figsize)
            else:   
                fig, ax = plt.subplots(1,1, figsize=figsize)
            
        else:
            fig = ax.get_figure()
            if self.ms:
                self.ms.plot2d(ax=ax, color=color, draw_body=draw_body, Xuvec=Xuvec, Yuvec=Yuvec)

        # plot each FOWT
        for fowt in self.fowtList:
            fowt.plot2d(ax, color=color, plot_rotor=plot_rotor, Xuvec=Xuvec, Yuvec=Yuvec)
        
        ax.axis("equal")
        
        if hideGrid:       
            ax.set_xticks([])    # Hide axes ticks
            ax.set_yticks([])
            ax.grid(False)       # Hide grid lines
            ax.grid(b=None)
            ax.axis('off')
            ax.set_frame_on(False)
            
        return fig, ax
    
    
    def adjustBallast(self, fowt, heave_tol=1, l_fill_adj=1e-2, rtn=0, display=0):
        '''function to add or subtract the fill level of ballast in a member to get equilibrium heave close to 0
        fowt: the FOWT object that needs to be ballasted
        heave_tol: the tolerance acceptable for equilibrium heave [m]
        l_fill_adj: the amount you want the heave to change by in each iteration of the while loop [m]
        rtn: flag to output relevant data'''
        
        member_break_flag=False
        data = []
        
        # calculate the difference in theoretical mass and actual mass
        fowt.calcStatics()
        mass = (fowt.V*fowt.rho_water*fowt.g + self.F_moor0[2])/fowt.g
        dmass = mass - fowt.M_struc[0,0]
        sumFz = -fowt.M_struc[0,0]*fowt.g + fowt.V*fowt.rho_water*fowt.g + self.F_moor0[2]
        heave = sumFz/(fowt.rho_water*fowt.g*fowt.AWP)
        if display==1: print(mass, dmass, heave)
        
        # loop through each member and adjust the l_fill of each to match the volume needed to balance the mass
        for i,member in enumerate(fowt.memberList):
            if display==1: print('-------',i,member.rA)
            # organize the headings to work for this specific function
            if np.isscalar(member.headings):
                headings = [member.headings]
            else:
                headings = member.headings
            if display==1: print(headings)
            if member.heading != headings[0]:   # to ensure that only one member in a repeated member list is adjusted
                pass
            else:
                # organize the l_fill and rho_fill variables for this specific function
                if type(member.l_fill) is float:  # if there is only one section of ballast in the member, make it a list
                    l_fills = [member.l_fill]    
                    rho_fills = [member.rho_fill]
                else:
                    l_fills = member.l_fill
                    rho_fills = member.rho_fill
                if display==1: print(l_fills, rho_fills)
                
                # loop through each section of ballast in the member and adjust its l_fill to balance heave
                for j,ballast in enumerate(rho_fills):
                    if ballast > 0:                                         # only adjust the sections with existing ballast
                        if display==1: print(j, ballast)    
                        dvol = dmass/ballast                                # the volume required to balance heave
                        mdvol = dvol/len(headings)                          # the volume required per repeated member
                        err = 1e5                                           # initialize the error for the l_fill solver
                        l_fill = l_fills[j]                                 # set the current l_fill value
                        #l = member.stations[j+1]-member.stations[j]         # set the length of the submember with ballast
                        l = member.l        # assume that the sub-member fill level (specified in 'l_fills[j]') can reach the entire height of the member
                        if display==1: print(dvol, mdvol, l_fill, l)
                        if member.shape=='circular':
                            dAi = member.d[j] - 2*member.t[j]
                            dBi = member.d[j+1] - 2*member.t[j+1]
                            # calculate the initial volume in the current submember first
                            dBi_fill = (dBi-dAi)*(l_fill/l) + dAi           # interpolated diameter of frustum that ballast is filled to
                            V0 = FrustumVCV(dAi, dBi_fill, l_fill, rtn=1)
                            
                            # adjust the l_fill value of the submember by l_fill_adj until the new ballast volume settles on V0+mdvol
                            while abs(err) > 0.01*V0:
                                if l_fill >= l and mdvol < 0:       # if l_fill is more than the given submember length and volume needs to decrease
                                    l_fill += -l_fill_adj
                                elif l_fill >= l and mdvol > 0:     # if l_fill is more than the given submember length and volume needs to increase
                                    l_fill = l_fill
                                    break                           # end the while loop since this is the maximum l_fill can go for this submember
                                elif l_fill <= 0 and mdvol > 0:     # if l_fill is less than 0 and the volume needs to increase
                                    l_fill += l_fill_adj
                                elif l_fill <= 0 and mdvol < 0:     # if l_fill is less than 0 and the volume needs to decrease
                                    l_fill = l_fill
                                    break                           # end the whole loop since l_fill can't go below 0
                                else:
                                    l_fill += l_fill_adj*np.sign(err)   # otherwise, adjust by l_fill in the correct direction
                                
                                dBi_fill = (dBi-dAi)*(l_fill/l) + dAi
                                V = FrustumVCV(dAi, dBi_fill, l_fill, rtn=1)    # calculate the volume of the ballast with the new l_fill
                                err = V0+mdvol - V                              # ensure V0+mdvol = V to solve for the correct l_fill
                            l_fill = np.round(l_fill, 2)
                            
                        
                        elif member.shape=='rectangular':
                            slAi = member.sl[j] - 2*member.t[j]
                            slBi = member.sl[j+1] - 2*member.t[j+1]
                            # calculate the initial volume in the current submember first
                            slBi_fill = (slBi-slAi)*(l_fill/l) + slAi   # interpolated side lengths of frustum that ballast is filled to
                            V0 = FrustumVCV(slAi, slBi_fill, l_fill, rtn=1)
                            
                            # adjust the l_fill value of the submember by l_fill_adj until the new ballast volume settles on V0+mdvol                            
                            while abs(err)> 0.01*V0:
                                if l_fill >= l and mdvol < 0:       # if l_fill is more than the given submember length and volume needs to decrease
                                    l_fill += -l_fill_adj
                                elif l_fill >= l and mdvol > 0:     # if l_fill is more than the given submember length and volume needs to increase
                                    l_fill = l_fill
                                    break                           # end the while loop since this is the maximum l_fill can go for this submember
                                elif l_fill <= 0 and mdvol > 0:     # if l_fill is less than 0 and the volume needs to increase
                                    l_fill += l_fill_adj
                                elif l_fill <= 0 and mdvol < 0:     # if l_fill is less than 0 and the volume needs to decrease
                                    l_fill = l_fill
                                    break                           # end the whole loop since l_fill can't go below 0
                                else:
                                    l_fill += l_fill_adj*np.sign(err)   # otherwise, adjust by l_fill in the correct direction
                                
                                slBi_fill = (slBi-slAi)*(l_fill/l) + slAi
                                V = FrustumVCV(slAi, slBi_fill, l_fill, rtn=1)  # calculate the volume of the ballast with the new l_fill
                                err = V0+mdvol - V                              # ensure V0+mdvol = V to solve for the correct l_fill
                            l_fill = np.round(l_fill, 2)
                        
                        if display==1:  print('solved l_fill = ', l_fill)
                        # replace the solved for l_fill value in each repeated member
                        for k,heading in enumerate(headings):
                            if np.isscalar(fowt.memberList[i+k].l_fill):
                                fowt.memberList[i+k].l_fill = l_fill
                            else:
                                fowt.memberList[i+k].l_fill[j] = l_fill
                                
                        
                        # check if heave equilibrium was reached by only changing this ballast section of the member
                        fowt.calcStatics()
                        sumFz = -fowt.M_struc[0,0]*fowt.g + fowt.V*fowt.rho_water*fowt.g + self.F_moor0[2]
                        heave = sumFz/(fowt.rho_water*fowt.g*fowt.AWP)
                        if display==1: print('heave', heave, heave_tol)
                        if abs(heave) < heave_tol:  # congrats, you've ballasted to achieve the given heave tolerance
                            member_break_flag=True  # break out of the outer member for loop as well
                            data.append([member.rA, member.l_fill, member.rho_fill, heave])             # save data
                            break                   # break out of this inner for loop iterating between ballast sections in one member
                        else:                       # bummer, you have to keep ballasting in other members or sections of the same member
                            mass = (fowt.V*fowt.rho_water*fowt.g + self.F_moor0[2])/fowt.g
                            dmass = mass - fowt.M_struc[0,0]                                # newly evaluated change in mass you need
                            data.append([member.rA.tolist(), member.l_fill, member.rho_fill, heave])    # save data
                            
                if member_break_flag:
                    break
                        
        if rtn:
            return data


    def adjustBallastDensity(self, fowt):
        '''Adjusts ballast densities unifromly to trim FOWT in heave.
        fowt: the FOWT object that needs to be ballasted
        '''
        
        print("Adjusting ballast to trim heave.")
        
        # check for any instances of zero-density ballast and ensure the corresponding fill length is zero        
        for member in fowt.memberList:
            if type(member.l_fill) is float:  # if there is only one section of ballast in the member            
                if member.rho_fill == 0.0:
                    member.l_fill = 0.0
            else:
                for i in range(len(member.l_fill)):
                    if member.rho_fill[i] == 0.0:
                        member.l_fill[i] = 0.0
        
        # compute ballast and check initial offset
        fowt.calcStatics()
        sumFz = -fowt.M_struc[0,0]*fowt.g + fowt.V*fowt.rho_water*fowt.g + self.F_moor0[2]
        heave = sumFz/(fowt.rho_water*fowt.g*fowt.AWP)        
        print(f" Original sumFz is {sumFz/1000:.0f} kN and heave is ~{heave:.3f} m")
        
        # total up the ballast volume
        ballast_volume = 0.0        
        for member in fowt.memberList:
            ballast_volume += sum(member.vfill)
        
        # ensure there isn't zero ballast volume
        if ballast_volume <= 0:
            raise Exception("adjustBallastDenity can only be used for platforms that have some ballast volume.")
        
        # calculate required change in ballast densities to zero heave offset
        delta_rho_fill = sumFz/fowt.g/ballast_volume
        
        print(f" Adjusting fill density by {delta_rho_fill:.3f} kg/m over {ballast_volume:.3f} m3 of ballast")
        
        # apply the change to each ballasted (sub)member's fill densities
        for member in fowt.memberList:
            if type(member.l_fill) is float:  # if there is only one section of ballast in the member            
                if member.l_fill > 0.0:
                    member.rho_fill += delta_rho_fill
            else:
                for i in range(len(member.l_fill)):
                    if member.l_fill[i] > 0.0:
                        member.rho_fill[i] += delta_rho_fill
        
        # recompute ballast and check adjusted offset
        fowt.calcStatics()
        sumFz = -fowt.M_struc[0,0]*fowt.g + fowt.V*fowt.rho_water*fowt.g + self.F_moor0[2]
        heave = sumFz/(fowt.rho_water*fowt.g*fowt.AWP)
        
        print(f" New sumFz is {sumFz/1000:.0f} kN and heave is ~{heave:.3f} m")
        
        # return adjustment
        return delta_rho_fill
    
    
    def adjustWISDEM(self, old_wisdem_file, new_wisdem_file):
        '''
        This loads an existing WISDEM input file and adjusts the ballast in the members in WISDEM based on 
        a RAFT model that was created based on the original WISDEM model
        '''
        
        # read in the wisdem file that you want to adjust and save it as a wisdem_design dictionary
        import ruamel_yaml as ry
        reader = ry.YAML(typ="safe", pure=True)
        with open(old_wisdem_file, "r", encoding="utf-8") as f:
            wisdem_design = reader.load(f)
        
        fowt = self.fowtList[0]
        membersRAFT = fowt.memberList       # list of members in the RAFT model
        membersWISDEM = wisdem_design['components']['floating_platform']['members']     # list of members in the WISDEM model
        
        # Main adjuster section:
        # Loop through each member in WISDEM and make adjustments based on the data in RAFT
        # For right now, this only changes the ballast fill levels
        for wisdem_member in membersWISDEM:
            if 'ballasts' in wisdem_member['internal_structure'].keys():    # skip the wisdem member if there is no ballast section
                for raft_member in membersRAFT:
                    # determine if the raft_member is the same type as the current wisdem member (they don't have great identifiers to relate)
                    # first, find the bottom joint of the current wisdem member
                    for joint in wisdem_design['components']['floating_platform']['joints']:
                        if wisdem_member['joint1']==joint['name']:  # find the name of the bottom 
                            # if that joint's location is the same as the raft member's bottom node location AND both the wisdem member and the raft member have the same diameter
                            if str(joint['location'][2])[0:5] == str(raft_member.rA[2])[0:5] and wisdem_member['outer_shape']['outer_diameter']['values'][0]==raft_member.d[0]:
                                # adjust the volume of the wisdem member based on the similar raft member's fill level
                                
                                # assume the diameter is constant along the member's length in both WISDEM and RAFT
                                area = np.pi * ((raft_member.d[0]-2*raft_member.t[0])/2)**2
                                # update the volume of the wisdem member based on the l_fill value in RAFT
                                wisdem_member['internal_structure']['ballasts'][0]['volume'] = float(area*raft_member.l_fill[0])
                                
                                break   # stop looping through the rest of the joints
                    break   # stop looping through the rest of the raft members
        
        # save the adjusted wisdem design dictionary into a new wisdem yaml file
        yaml = ry.YAML()
        yaml.default_flow_style = None
        yaml.width = float("inf")
        yaml.indent(mapping=4, sequence=6, offset=3)
        yaml.allow_unicode = False
        with open(new_wisdem_file, "w", encoding="utf-8") as f:
            yaml.dump(wisdem_design, f)

    def powerThrustCurve(self,nfowt, nrotor, uhubs, heading, yaw, plot = False ):
        '''
        Calculates power thrust curve for input into FLORIS. NOTE the pitch angles assume that the wind is at a 0 deg heading
        
        Parameters
        ----------
        vhubs : array
            Array of wind speeds for power thrust curves.
        Returns
        -------
        None.

        '''
        #store case data then delete for now
        casedata = self.design['cases']['data']
        
        
        #store yaw mode then set to zero
        yaw = self.fowtList[nfowt].rotorList[nrotor].yaw
        yawmode = self.fowtList[nfowt].rotorList[nrotor].yaw_mode
        self.fowtList[nfowt].rotorList[nrotor].yaw_mode = 0
        
        
        self.design['cases']['data'] = []
        cp = []
        ct = []
        pitch = []
        
        power = []
        thrust = [] 
        for uhub in uhubs:
            if uhub >= 3 and uhub <=25: 
                self.design['cases']['data'] = [[uhub, heading, 0.1, 'operating', yaw, 'JONSWAP', 0,0,0]]
            else:
                self.design['cases']['data'] = [[uhub, heading, 0.1, 'parked', yaw, 'JONSWAP', 0,0,0]]
            case = dict(zip( self.design['cases']['keys'], self.design['cases']['data'][0]))   
            self.solveStatics(case=case)
            
            
            # Calculate platform pitch
            rot = self.fowtList[nfowt].rotorList[nrotor]
            turbine_tilt    = np.arctan2(rot.q[2], rot.q[0])  # [rad] front facing up is positive
            
            # Not sure how the pitch angle should be handled if wind comes at angle.....
            ### NOTE: this CCBlade print statement comes after the mean offset printouts
            loads, derivs = self.fowtList[nfowt].rotorList[nrotor].runCCBlade(uhub,  ptfm_pitch=turbine_tilt, )
            cp.append(loads["CP"][0])
            ct.append(loads["CT"][0])
            pitch.append(rad2deg(self.fowtList[nfowt].Xi0[4]))
            
            power.append(self.fowtList[nfowt].rotorList[nrotor].aero_power)
            thrust.append(self.fowtList[nfowt].rotorList[nrotor].aero_thrust)
        
        if plot:
            fig, ax = plt.subplots(3,1, sharex = True)
            plotvars = [cp, ct,pitch]
            ylabels = ["Power Coef", "Thrust Coef", "Ptfm Pitch (deg)"]
            for i in range(0, 3):
                ax[i].plot(uhubs, plotvars[i])
                ax[i].set_ylabel(ylabels[i])
            ax[2].set_xlabel('Wind Speed (m/s)')
            
            
            fig, ax = plt.subplots(2,1, sharex = True)
            plotvars = [power, thrust]
            ylabels = ["Power", "Thrust"]
            for i in range(0, 2):
                ax[i].plot(uhubs, plotvars[i])
                ax[i].set_ylabel(ylabels[i])
            ax[1].set_xlabel('Wind Speed (m/s)')
            
        self.design['cases']['data'] = casedata
        
        # return yaw to original
        self.fowtList[nfowt].rotorList[nrotor].yaw_mode = yawmode
        self.fowtList[nfowt].rotorList[nrotor].yaw = yaw 
        return cp, ct, pitch
    
    
    def florisCoupling(self, config, turbconfig, path ):
        '''
        Function to set up FLORIS interface, using parameters from the RAFT model. 
        Takes in base yaml files for floris configuration and turbine configuration and updates the
        turbine locations, shear, air density, rotor diameter, power-thrust curve. Iterates through each
        RAFT case and calculates turbine power matrix
        
        NOTE: this function writes a new turbine yaml file for each turbine. 

        Parameters
        ----------
        config : str
            Name of floris config yaml file for basis of floris interface
        turbconfig : List of str
            List of turbine yaml files that corresponds to RAFT turbine IDs 
        path : str
            Path to turbine library of yaml files

        '''
        from floris.tools import FlorisInterface
        
        # Setup FLORIS interface using base yaml file
        self.fi = FlorisInterface(config)
        
        # Update floris interface settings to match RAFT design
        self.fi.reinitialize(air_density = self.design["site"]["rho_air"], wind_shear = self.design["site"]["shearExp"])
        fowtInfo = [dict(zip( self.design['array']['keys'], row)) for row in self.design['array']['data']]
        self.fi.reinitialize(layout_x=[fowtInfo[j]["x_location"] for j in range(0,len(fowtInfo))], layout_y=[fowtInfo[j]["y_location"] for j in range(0,len(fowtInfo))])
        
        # create new turbine yaml file for each turbine with a unique turbine, platform, mooring, or heading adjustment
        # this is because these effect the pitch of the platform in the power-thrust curve
        # FLORIS reinitialize function calls on the turbine yaml files whenever reinitialize is called
      
        turblist = []
        uniqueLists =[]
        
        #iterate through lies of turbines
        for i in range(self.nFOWT):
            turbID = fowtInfo[i]['turbineID']
            
            #Check if turbine has unique platform, turbine, mooring, or rotation ... if so, calculate new power thrust curve
            IDList = [fowtInfo[i]['turbineID'], fowtInfo[i]['platformID'], fowtInfo[i]['mooringID'], fowtInfo[i]['heading_adjust']]
            if IDList in uniqueLists:
                for j, ulist in enumerate(uniqueLists):
                    if IDList == ulist:
                        ID = j
                print('Turbine ' +str(i+1) +' is not unqiue, using Turbine ' + str(ID)+" input")
            else:
                
                # If turbine is unique then create new ID/input file
                uniqueLists.append(IDList)
                ID = len(uniqueLists) - 1
                print('Turbine ' +str(i+1) +' is unqiue, creating Turbine ' + str(ID)+ " input")
                
                with open(turbconfig[turbID - 1]) as file:
                    turbData = yaml.safe_load(file)
                
                #Find turbine ID and use yaml file associated with that ID as basis
                turbData['hub_height'] = self.design["turbines"][turbID - 1]["hHub"]
                turbData['rotor_diameter'] = self.design["turbines"][turbID - 1]["blade"][0]["Rtip"]*2 # Check this (takes the rotor radius from first blade)
                turbData['ref_density_cp_ct'] = self.design["site"]["rho_air"]
                
                #RAFT may want to name the turbines
                turbData['turbine_type'] ='turb'+str(ID)+'_floating'
                
                #Cp and Ct curves already incorporate the floating tilt... so FLORIS can ignore
                turbData['floating_correct_cp_ct_for_tilt'] = False 
                
                #set up list of hub velocities to match floris 
                uhubs=list(np.arange(3,25,0.5))
                #uhubs.insert(0, 0)
                uhubs.append(25.02)
                uhubs.append(50)
                
                #Currently only setup to handle one rotor
                #FLORIS inputs Cp, Ct, Cq curves with a yaw misalignment of 0 and wind heading of 0
                #In reality, the mooring system stiffness would slightly change the Cp curve based on heading (because the pitch angle would change)
                power, thrust, pitch = self.powerThrustCurve(i, 0, uhubs, 0, yaw = 0, plot = False)
                turbData['power_thrust_table']['power'] = np.array(power).tolist()
                turbData['power_thrust_table']['thrust'] = np.array(thrust).tolist()
                turbData['power_thrust_table']['wind_speed'] = np.array(uhubs).tolist()
                
                print('len winds ', len(uhubs))
                print('len pitch ', len(pitch))
                
                
                #Set floating tilt table only for use in Empirical Gaussian wake model (wake deflection for pitch angle)
                turbData['floating_tilt_table']['wind_speeds'] = np.array(uhubs).tolist() # match roughly the wind speeds in example files
                turbData['floating_tilt_table']['tilt'] = np.array(pitch).tolist()
    
                with open(path+'\\'+'turb'+str(ID)+'.yaml', 'w') as file:
                    yaml.dump(turbData, file, sort_keys=False, default_flow_style=None)
            turblist.append('turb'+str(ID)+'.yaml')
            
        #reinitialize floris interface with updated turbine yaml files
        self.fi.reinitialize(turbine_type= turblist, turbine_library_path= path)
           
        self.turblist = turblist
    
    def florisFindEquilibrium(self, case, cutin, plotting = True, ax = None):
        
        if not hasattr(self, 'fi'):
            raise AttributeError("Need to initialize floris coupling first")
            
        fowtInfo = [dict(zip( self.design['array']['keys'], row)) for row in self.design['array']['data']]

        
        #FLORIS inputs the wind direction as direction wind is coming from (where the -X axis is 0)
        self.fi.reinitialize(wind_directions = [-case['wind_heading']+270], wind_speeds = [case['wind_speed']], turbulence_intensity= case['turbulence'])
        yaw_angles = np.ones([1,1,self.nFOWT]) 
        
        
        #calc yaw misalignment to input into FLORIS
        heading = case['wind_heading']
        
        #solve statics to find updated turbine positions
        self.solveStatics(case=case, display = 1)
    
        for nfowt in range(0, (self.nFOWT)):
            rot = self.fowtList[nfowt].rotorList[0]
            
            if rot.yaw_mode == 0:  # assume aligned, assumes platform yaw is accounted for by controller
                turbine_heading = np.radians(heading) # [rad]
                
            elif rot.yaw_mode == 1:  # use case info, assumes platform yaw is accounted for by contoller
                turbine_heading = np.radians(getFromDict(case, 'turbine_heading', shape=0, default=0.0))  # [deg]

            
            elif rot.yaw_mode == 2:  # use self.yaw value and add platform yaw (not sure about this option!)
                nac_yaw = rot.yaw
                turbine_heading = np.arctan2(rot.q[1], rot.q[0]) + nac_yaw  # [rad]
            
            elif rot.yaw_mode == 3: # use self.yaw value as total nacelle yaw, assumes platform yaw is accounted for by controller
                turbine_heading = rot.yaw  # [rad]
            else:
                raise Exception('Unsupported yaw_mode value. Must be 0, 1, or 2.')
            
            

            # inflow misalignment heading relative to turbine heading [deg]
            yaw_misalign = turbine_heading - np.radians(heading) 
            yaw_angles[0,0,nfowt] = np.degrees(yaw_misalign)
        
        print('Yaw misalignment angles: ', yaw_angles)
        
        
        winds = []
        xpositions = []
        ypositions = []
        powers = []
        
        #setting this at 100 iterations so that the floris-raft loop does not go on forever
        N = 100
        for n in range(0, N):
            
            #solve statics to find updated turbine positions
            self.solveStatics(case=case, display = 1)

            #update floris turbine positions
            if n > 0:
                xnew = [0.9*(self.fowtList[nfowt].Xi0[0] + fowtInfo[nfowt]["x_location"]) + 0.1*xpositions[-1][nfowt] for nfowt in range(len(self.fowtList))]
                ynew = [0.9*(self.fowtList[nfowt].Xi0[1]  + fowtInfo[nfowt]["y_location"]) + 0.1*ypositions[-1][nfowt] for nfowt in range(len(self.fowtList))]
                self.fi.reinitialize(layout_x=xnew, layout_y=ynew)
            else:
                xnew = [self.fowtList[nfowt].Xi0[0] + fowtInfo[nfowt]["x_location"]  for nfowt in range(len(self.fowtList))]
                ynew = [self.fowtList[nfowt].Xi0[1]  + fowtInfo[nfowt]["y_location"] for nfowt in range(len(self.fowtList))]
               
            self.fi.reinitialize(layout_x=xnew, layout_y=ynew)
            self.fi.calculate_wake(yaw_angles=yaw_angles)
            print('Turbine avg vels ', self.fi.turbine_average_velocities[0][0])
            print('Turbine eff vels ', self.fi.turbine_effective_velocities[0][0])
            
            #update wind speed list for RAFT
            case['wind_speed'] = list(self.fi.turbine_average_velocities[0][0])
            winds.append(self.fi.turbine_average_velocities[0][0])
            xpositions.append(xnew)
            ypositions.append(ynew)
            
            #return FLORIS turbine powers (in order of turbine list)
            if min(self.fi.turbine_effective_velocities[0][0] > cutin):
                powers.append(self.fi.get_turbine_powers()[0][0])
            else:
                powers.append([0])

            
            if n > 1:
                if min(self.fi.turbine_effective_velocities[0][0] > cutin):
                    
                    #check if turbine powers from recent iteration and previous iteration are within 10 W, and if so break
                    if max([np.abs(powers[-1][i] - powers[-2][i]) for i in range(0, len(powers[-1]))]) < 10:
                        if max([np.abs(xpositions[-1][i] - xpositions[-2][i]) for i in range(0, len(xpositions[-1]))]) < 0.01:
                            break
            
                # if below cut in wind speed, can't get powers from FLORIS. instead, check that xpositions converge
                else:
                    if max([np.abs(xpositions[-1][i] - xpositions[-2][i]) for i in range(0, len(xpositions[-1]))]) < 0.01 :
                            break
                    #
            #print warning if coupling does not converge in N iterations
            if n == N - 1:
                print('RAFT FLORIS coupling did not converge in '+str(N) +' iterations')
     

        if plotting:
            import floris.tools.visualization as wakeviz
            horizontal_plane = self.fi.calculate_horizontal_plane(
                x_resolution=200,
                y_resolution=100,
                height=90.0,
                yaw_angles=yaw_angles, 
            )
    
            y_plane = self.fi.calculate_y_plane(
                x_resolution=200,
                z_resolution=100,
                crossstream_dist=0.0,
                yaw_angles=yaw_angles,
            )
            cross_plane = self.fi.calculate_cross_plane(
                y_resolution=100,
                z_resolution=100,
                downstream_dist=630.0,
                yaw_angles=yaw_angles,
            )
    
            # Create the plots
            if ax == None:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            #ax_list = ax_list.flatten()
            wakeviz.visualize_cut_plane(horizontal_plane, ax=ax)
            self.plot2d(Yuvec = [0, 1, 0], ax = ax)
    
            cmap = plt.cm.get_cmap('viridis_r')
            
        
        #store wind, positions, power data as arrays (each row stores an iteration)
        winds = np.array(winds)
        xpositions = np.array(xpositions)
        ypositions = np.array(ypositions)
        powers = np.array(powers)
            
        return(winds,xpositions, ypositions, powers)         
    
    def florisCalcAEP(self,windrose, cutin, cutout, TI):
        import pandas as pd
        # Read the windrose information file and display
        df_wr = pd.read_csv(windrose)
        print("The wind rose dataframe looks as follows: \n\n {} \n".format(df_wr))
        
        windspeeds = list(df_wr['ws'])
        winddirs = list(df_wr['wd'])
        probabilities = list(df_wr['freq_val'])
        
        powers = []
        aeps = []
        
        for i in range(0, len(windspeeds)):
            
            #set up case with appropriate windspeed/wind dir/TI (???)
            if windspeeds[i] >= cutin and windspeeds[i] <= cutout:
                case = dict(zip( self.design['cases']['keys'], [windspeeds[i], winddirs[i], TI, 'operating', 0, 'JONSWAP', 0, 0, 0])) 
                winds,xpositions, ypositions, turbine_powers = self.florisFindEquilibrium(case = case, cutin = cutin, plotting = False)
                powers.append(turbine_powers[-1,:])
                aeps.append(turbine_powers[-1,:]*(probabilities[i]))
                
            # else:
            #     case = dict(zip( self.design['cases']['keys'], [windspeeds[i], winddirs[i], TI, 'parked', 0, 'JONSWAP', 0, 0, 0])) 

        
        return(powers, aeps)       

def runRAFT(input_file, turbine_file="", plot=0, ballast=False, station_plot=[]):
    '''
    This will set up and run RAFT based on a YAML input file.
    '''
    
    if input_file[-3:]=='pkl' or input_file[-6:]=='pickle':
        with open(input_file, 'rb') as pfile:
            design = pickle.load(pfile)
    elif not isinstance(input_file, dict):
        # open the design YAML file and parse it into a dictionary for passing to raft
        print("\n\nLoading RAFT input file: "+input_file)
        with open(input_file) as file:
            design = yaml.load(file, Loader=yaml.FullLoader)
    else:
        design = input_file
        print(f"'{design['name']}'")
    
    
       
    # Create and run the model
    print(" --- making model ---")
    model = Model(design)  
    print(" --- analyzing unloaded ---")
    model.analyzeUnloaded(ballast=ballast)
    print(" --- analyzing cases ---")
    model.analyzeCases(display=1)
    
    model.calcOutputs()
    
    if plot:
        #model.plot(station_plot=station_plot, zbounds=[-model.ms.depth, model.ms.depth + 2*model.ms.bodyList[0].r6[2]], hideGrid=True, draw_body=True)        
        model.plot(station_plot=station_plot)
        model.plotResponses()
    
    #model.preprocess_HAMS("testHAMSoutput", dw=0.1, wMax=10)
    
    return model



def runRAFTFarm(input_file, plot=0):
    '''
    This will set up and run RAFT "Farm" based on a YAML input file.
    '''
    
    if input_file[-3:]=='pkl' or input_file[-6:]=='pickle':
        with open(input_file, 'rb') as pfile:
            design = pickle.load(pfile)
    elif not isinstance(input_file, dict):
        # open the design YAML file and parse it into a dictionary for passing to raft
        print("\n\nLoading RAFTFarm input file: "+input_file)
        with open(input_file) as file:
            design = yaml.load(file, Loader=yaml.FullLoader)
    else:
        design = input_file
        print(f"'{design['name']}'")
    
    # Create and run the model
    print(" --- making model ---")
    model = Model(design)  
    print('**Note: RAFTFarm cannot run model.analyzeUnloaded()')
    print(" --- analyzing cases ---")
    model.analyzeCases(display=1)
    
    print('**Note: model.calcOutputs is not supported yet for multi-turbine Farm configurations')
    
    if plot: 
        model.plot()
        model.plotResponses()
    
    return model
    

    
    
if __name__ == "__main__":
    
    ### Run a Simple Model ###
    #model = runRAFT(os.path.join(raft_dir,'designs/Vertical_cylinder.yaml'), plot=1)

    ### Run a Reference FOWT Model ###
    #model = runRAFT(os.path.join(raft_dir,'designs/OC3spar.yaml'), plot=1)
    #model = runRAFT(os.path.join(raft_dir,'designs/OC4semi.yaml'), plot=1)
    model = runRAFT(os.path.join(raft_dir,'designs/VolturnUS-S.yaml'), plot=1)
    
    ### Run a MHK Model ###
    #model = runRAFT(os.path.join(raft_dir,'designs/FOCTT_example.yaml'), plot=1)
    #model = runRAFT(os.path.join(raft_dir,'designs/RM1_Floating.yaml'), plot=1)
    
    ### Run a RAFT Farm Model ###
    #model = runRAFTFarm(os.path.join(raft_dir,'designs/VolturnUS-S_farm.yaml'), plot=1)

    plt.show()
    
