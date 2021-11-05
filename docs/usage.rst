Usage and Workflow
==================

..
  customize code highlight color through "hll" span css

.. raw:: html

    <style> .highlight .hll {color:#000080; background-color: #eeeeff} </style>
    <style> .fast {color:#000080; background-color: #eeeeff} </style>
    <style> .stnd {color:#008000; background-color: #eeffee} </style>

.. role:: fast
.. role:: stnd



RAFT requires an input design YAML file to describe the design of a floating offshore wind turbine. RAFT currently does not do any 
design processes, but can be coupled with other programs like WISDEM and OpenFAST to optimize the design variables.
Using the input design YAML, RAFT will then compute the 6x6 matrices to fill out the frequency domain equations of motion.
The main outputs will be the response amplitude operators (RAOs) of the FOWT.


Input Design YAML
------------------

.. code-block:: python

    tower:
      dlsMax       :  5.0     # maximum node splitting section amount; can't be 0
    
      name      :  tower                     # [-]    an identifier (no longer has to be number)       
      type      :  1                         # [-]    
      rA        :  [ 0, 0,  15]              # [m]    end A coordinates
      rB        :  [ 0, 0, 144.582]          # [m]    and B coordinates
      shape     :  circ                      # [-]    circular or rectangular
      gamma     :  0.0                       # [deg]   twist angle about the member's z-axis
        
      # --- outer shell including hydro---
      stations  :  [ 15,  28,  28.001,  41,  41.001,  54,  54.001,  67,  67.001,  80,  80.001,  93,  93.001,  106,  106.001,  119,  119.001,  132,  132.001,  144.582 ]    # [-]    location of stations along axis. Will be normalized such that start value maps to rA and end value to rB
      d         :  [ 10,  9.964,  9.964,  9.967,  9.967,  9.927,  9.927,  9.528,  9.528,  9.149,  9.149,  8.945,  8.945,  8.735,  8.735,  8.405,  8.405,  7.321,  7.321,  6.5 ]    # [m]    diameters if circular or side lengths if rectangular (can be pairs)
      t         :  [ 0.082954,  0.082954,  0.083073,  0.083073,  0.082799,  0.082799,  0.0299,  0.0299,  0.027842,  0.027842,  0.025567,  0.025567,  0.022854,  0.022854,  0.02025,  0.02025,  0.018339,  0.018339,  0.021211,  0.021211 ]                     # [m]    wall thicknesses (scalar or list of same length as stations)
      Cd        :  0.0                       # [-]    transverse drag coefficient       (optional, scalar or list of same length as stations)
      Ca        :  0.0                       # [-]    transverse added mass coefficient (optional, scalar or list of same length as stations)
      # (neglecting axial coefficients for now)
      CdEnd     :  0.0                       # [-]    end axial drag coefficient        (optional, scalar or list of same length as stations)
      CaEnd     :  0.0                       # [-]    end axial added mass coefficient  (optional, scalar or list of same length as stations)
      rho_shell :  7850                      # [kg/m3]   material density

 

Running RAFT
------------

The function runRAFT.py creates an initial Model object based on the input design YAML. It then analyzes the system in an 
unloaded condition--without any load conditions. Then it analyzes the model in the number of DLCs specficied by the input YAML.
These two Model methods are the bulk of the workflow of RAFT.

.. code-block:: python

    def runRAFT(input_file, turbine_file=""):
      '''
      This will set up and run RAFT based on a YAML input file.
      '''
      
      # open the design YAML file and parse it into a dictionary for passing to raft
      print("Loading RAFT input file: "+input_file)
      
      with open(input_file) as file:
          design = yaml.load(file, Loader=yaml.FullLoader)
      
      print(f"'{design['name']}'")
      
      
      depth = float(design['mooring']['water_depth'])
      
      # for now, turn off potMod in the design dictionary to avoid BEM analysis
      #design['platform']['potModMaster'] = 1
      
      # read in turbine data and combine it in
      # if len(turbine_file) > 0:
      #   turbine = convertIEAturbineYAML2RAFT(turbine_file)
      #   design['turbine'].update(turbine)
      
      # Create and run the model
      print(" --- making model ---")
      model = raft.Model(design)  
      print(" --- analyizing unloaded ---")
      model.analyzeUnloaded()
      print(" --- analyzing cases ---")
      model.analyzeCases()
      
      model.plot()
      
      #model.preprocess_HAMS("testHAMSoutput", dw=0.1, wMax=10)
      
      plt.show()
    
    return model
    
  model = runRAFT(os.path.join(raft_dir,'designs/VolturnUS-S.yaml'))




analyzeCases
^^^^^^^^^^^^

.. code-block::

        # calculate the system's constant properties
        for fowt in self.fowtList:
            fowt.calcStatics()
            fowt.calcBEM()
            
        # loop through each case
        for iCase in range(nCases):
        
            print("  Running case")
            print(self.design['cases']['data'][iCase])
        
            # form dictionary of case parameters
            case = dict(zip( self.design['cases']['keys'], self.design['cases']['data'][iCase]))   

            # get initial FOWT values assuming no offset
            for fowt in self.fowtList:
                fowt.Xi0 = np.zeros(6)      # zero platform offsets
                fowt.calcTurbineConstants(case, ptfm_pitch=0.0)
                fowt.calcHydroConstants(case)
            
            # calculate platform offsets and mooring system equilibrium state
            self.calcMooringAndOffsets()
            
            # update values based on offsets if applicable
            for fowt in self.fowtList:
                fowt.calcTurbineConstants(case, ptfm_pitch=fowt.Xi0[4])
                # fowt.calcHydroConstants(case)  (hydrodynamics don't account for offset, so far)
            
            # (could solve mooring and offsets a second time, but likely overkill)
            
            # solve system dynamics
            self.solveDynamics(case)











Advice and Frequent Problems
----------------------------
   
   





