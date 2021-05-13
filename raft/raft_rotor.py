# RAFT's rotor class

import os
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator

import wisdem.inputs as sch
from wisdem.ccblade.ccblade import CCBlade, CCAirfoil

'''
try:
    import ccblade        as CCBlade, CCAirfoil  # for cloned ccblade
except:
    import wisdem.ccblade as CCblade, CCAirfoil  # for conda install wisdem
'''


# a class for the rotor structure, aerodynamics, and control in RAFT
class Rotor:

    def __init__(self, turbine):
        '''
        >>>> add mean offset parameters add move this to runCCBlade<<<<
        '''

        # (not worrying about rotor structure/mass yet, just aero)

        blade   = turbine['blade']
        airfoils= turbine['airfoils']
        env     = turbine['env']

        # these should account for mean offsets of platform (and possibly tower bending)
        tilt = 0 + turbine['shaft_tilt']
        yaw  = 0        

        # Set CCBlade flags
        tiploss = True # Tip loss model True/False
        hubloss = True # Hub loss model, True/False
        wakerotation = True # Wake rotation, True/False
        usecd = True # Use drag coefficient within BEMT, True/False

        # Set discretization parameters
        nSector = 4 # [-] - number of equally spaced azimuthal positions where CCBlade should be interrogated. The results are averaged across the n positions. 4 is a good first guess
        n_span = 30 # [-] - number of blade stations along span
        grid = np.linspace(0., 1., n_span) # equally spaced grid along blade span, root=0 tip=1
        n_aoa = 200 # [-] - number of angles of attack to discretize airfoil polars


        # Atmospheric boundary layer data
        # rho = wt_init['environment']["air_density"] # [kg/m3] - density of air
        # mu = wt_init['environment']["air_dyn_viscosity"] # [kg/(ms)] - dynamic viscosity of air
        # shearExp = wt_init['environment']["shear_exp"] # [-] - shear exponent
        
        
        af = []
        for i in range(airfoils['cl_interp'].shape[0]):
            af.append(CCAirfoil(np.rad2deg(airfoils['aoa']), airfoils['Re'], airfoils['cl_interp'][i, :, :, 0], 
                                                                             airfoils['cd_interp'][i, :, :, 0], 
                                                                             airfoils['cm_interp'][i, :, :, 0]))

        self.ccblade = CCBlade(
            blade['r'],                # (m) locations defining the blade along z-axis of blade coordinate system
            blade['chord'],            # (m) corresponding chord length at each section
            blade['theta'],            # (deg) corresponding :ref:`twist angle <blade_airfoil_coord>` at each section---positive twist decreases angle of attack.
            af,                        # CCAirfoil object
            turbine['Rhub'],           # (m) radius of hub
            blade['Rtip'],             # (m) radius of tip
            turbine['nBlades'],        # number of blades
            env['rho'],                # (kg/m^3) freestream fluid density
            env['mu'],                 # (kg/m/s) dynamic viscosity of fluid
            turbine['precone'],          # (deg) hub precone angle
            tilt,                      # (deg) hub tilt angle
            yaw,                       # (deg) nacelle yaw angle
            env['shearExp'],           # shear exponent for a power-law wind profile across hub
            turbine['Zhub'],           # (m) hub height used for power-law wind profile.  U = Uref*(z/hubHt)**shearExp
            nSector,                   # number of azimuthal sectors to descretize aerodynamic calculation.  automatically set to 1 if tilt, yaw, and shearExp are all 0.0.  Otherwise set to a minimum of 4.
            blade['precurve'],         # (m) location of blade pitch axis in x-direction of :ref:`blade coordinate system <azimuth_blade_coord>`
            blade['precurveTip'],      # (m) location of blade pitch axis in x-direction at the tip (analogous to Rtip)
            blade['presweep'],         # (m) location of blade pitch axis in y-direction of :ref:`blade coordinate system <azimuth_blade_coord>`
            blade['presweepTip'],      # (m) location of blade pitch axis in y-direction at the tip (analogous to Rtip)
            tiploss=tiploss,           # if True, include Prandtl tip loss model
            hubloss=hubloss,           # if True, include Prandtl hub loss model
            wakerotation=wakerotation, # if True, include effect of wake rotation (i.e., tangential induction factor is nonzero)
            usecd=usecd,               # If True, use drag coefficient in computing induction factors (always used in evaluating distributed loads from the induction factors).
            derivatives=True,          # if True, derivatives along with function values will be returned for the various methods
        )


    def runCCBlade(self):
        '''
        '''

        # Set environmental conditions, these must be arrays except for yaw
        # Uhub = np.array([3.        ,  4.22088938,  5.22742206,  6.0056444 ,  6.54476783,
        #         6.83731843,  6.87924056,  7.08852808,  7.54612388,  8.24568427,
        #         9.17751118, 10.32868661, 10.89987023, 13.22242806, 14.9248779 ,
        #        16.76700002, 18.72325693, 20.76652887, 22.86848978, 25. ]) # m/s
        # Omega_rpm = np.array([5.        , 5.        , 5.        , 5.        , 5.        ,
        #        5.        , 5.        , 5.03607599, 5.36117694, 5.8581827 ,
        #        6.52020323, 7.33806089, 7.49924093, 7.49924093, 7.49924093,
        #        7.49924093, 7.49924093, 7.49924093, 7.49924093, 7.49924093]) # rpm
        # pitch_deg = np.array([3.8770757 ,  3.58018171,  2.63824381,  1.62701287,  0.81082407,
        #         0.32645039,  0.25491167,  0.        ,  0.        ,  0.        ,
        #         0.        ,  0.        ,  0.        ,  8.14543778, 11.02202702,
        #        13.61534727, 16.04700926, 18.3599078 , 20.5677456 , 22.67114154]) # deg

        self.Uhub = np.array([9.])
        Omega_rpm = np.array([6.39408964])
        pitch_deg = np.array([0.])


        self.outputs = {}

        loads, derivs = self.ccblade.evaluate(self.Uhub, Omega_rpm, pitch_deg, coefficients=True)

        self.outputs["P"] = loads["P"]
        self.outputs["Mb"] = loads["Mb"]
        self.outputs["CP"] = loads["CP"]
        self.outputs["CMb"] = loads["CMb"]
        self.outputs["Fhub"] = np.array( [loads["T" ][0], loads["Y"  ][0], loads["Z"  ][0]])
        self.outputs["Mhub"] = np.array( [loads["Q" ][0], loads["My" ][0], loads["Mz" ][0]])
        self.outputs["CFhub"] = np.array([loads["CT"][0], loads["CY" ][0], loads["CZ" ][0]])
        self.outputs["CMhub"] = np.array([loads["CQ"][0], loads["CMy"][0], loads["CMz"][0]])


        print("Wind speed")
        print(self.Uhub)
        print("Aerodynamic power coefficient")
        print(self.outputs["CP"])

        self.J={}

        self.J["T", "Uhub"] = derivs['dT']['dUinf'][0]


        dP = derivs["dP"]
        self.J["P", "r"] = dP["dr"]
        self.J["P", "chord"] = dP["dchord"]
        self.J["P", "theta"] = dP["dtheta"]
        self.J["P", "Rhub"] = np.squeeze(dP["dRhub"])
        self.J["P", "Rtip"] = np.squeeze(dP["dRtip"])
        self.J["P", "hub_height"] = np.squeeze(dP["dhubHt"])
        self.J["P", "precone"] = np.squeeze(dP["dprecone"])
        self.J["P", "tilt"] = np.squeeze(dP["dtilt"])
        self.J["P", "yaw"] = np.squeeze(dP["dyaw"])
        self.J["P", "shearExp"] = np.squeeze(dP["dshear"])
        self.J["P", "V_load"] = np.squeeze(dP["dUinf"])
        self.J["P", "Omega_load"] = np.squeeze(dP["dOmega"])
        self.J["P", "pitch_load"] = np.squeeze(dP["dpitch"])
        self.J["P", "precurve"] = dP["dprecurve"]
        self.J["P", "precurveTip"] = dP["dprecurveTip"]
        self.J["P", "presweep"] = dP["dpresweep"]
        self.J["P", "presweepTip"] = dP["dpresweepTip"]

        # dT = derivs["dT"]
        # self.J["Fhub", "r"][0,:] = dT["dr"]     # 0 is for thrust force, 1 would be y, 2 z
        # self.J["Fhub", "chord"][0,:] = dT["dchord"]
        # self.J["Fhub", "theta"][0,:] = dT["dtheta"]
        # self.J["Fhub", "Rhub"][0,:] = np.squeeze(dT["dRhub"])
        # self.J["Fhub", "Rtip"][0,:] = np.squeeze(dT["dRtip"])
        # self.J["Fhub", "hub_height"][0,:] = np.squeeze(dT["dhubHt"])
        # self.J["Fhub", "precone"][0,:] = np.squeeze(dT["dprecone"])
        # self.J["Fhub", "tilt"][0,:] = np.squeeze(dT["dtilt"])
        # self.J["Fhub", "yaw"][0,:] = np.squeeze(dT["dyaw"])
        # self.J["Fhub", "shearExp"][0,:] = np.squeeze(dT["dshear"])
        # self.J["Fhub", "V_load"][0,:] = np.squeeze(dT["dUinf"])
        # self.J["Fhub", "Omega_load"][0,:] = np.squeeze(dT["dOmega"])
        # self.J["Fhub", "pitch_load"][0,:] = np.squeeze(dT["dpitch"])
        # self.J["Fhub", "precurve"][0,:] = dT["dprecurve"]
        # self.J["Fhub", "precurveTip"][0,:] = dT["dprecurveTip"]
        # self.J["Fhub", "presweep"][0,:] = dT["dpresweep"]
        # self.J["Fhub", "presweepTip"][0,:] = dT["dpresweepTip"]



    def calcAeroContributions(self, nw=0, U_amplitude=[]):
        '''Calculates stiffness, damping, added mass, and excitation coefficients
        from rotor aerodynamics. Results are w.r.t. nonrotating hub reference frame
        and assume constant rotor speed and no controls.
        '''

        Uinf = 10.  # inflow wind speed (m/s) <<< eventually should be consistent with rest of RAFT
        Hhub = 100.

        # extract derivatives of interest, interpolated for the current wind speed
        dT_dU  = np.interp(Uinf, self.Uhub, self.J["T", "Uhub"     ])
        #dT_dOm = np.interp(Uinf, self.Uhub, self.J["T", "Omega_rpm"])
        #dT_dPi = np.interp(Uinf, self.Uhub, self.J["T", "pitch_deg"])
        #dQ_dU  = np.interp(Uinf, self.Uhub, self.J["Q", "Uhub"     ])
        #dQ_dOm = np.interp(Uinf, self.Uhub, self.J["Q", "Omega_rpm"])
        #dQ_dPi = np.interp(Uinf, self.Uhub, self.J["Q", "pitch_deg"])
        # wish list
        # dMy_dU  = np.interp(Uinf, self.Uhub, self.J["My", "Uhub"     ])  # overturning moment about hub
        # dMy_dShearExp =
        # ...

        # coefficients to be filled in
        A_aero = np.zeros([6,6])                        # added mass
        B_aero = np.zeros([6,6])                        # damping
        C_aero = np.zeros([6,6])                        # stiffness
        F_aero0= np.zeros(6)                            # steady wind forces/moments
        F_aero = np.zeros([6,nw])                       # wind excitation spectra in each DOF

        # calculate hub aero coefficients (in nonrotating hub reference frame) - assuming rigid body and no control to start with
        B_aero[0,0] += dT_dU                            # surge damping
        #B_aero[0,4] += dT_dU*Hhub                       #
        #B_aero[4,0] += dT_dU*Hhub                       #
        #B_aero[4,4] += dT_dU*Hhub**2                    # pitch damping

        # calculate wind excitation force/moment spectra (in nonrotating hub reference frame)
        for i in range(nw):                             # loop through each frequency component
            F_aero[0,i] = U_amplitude[i]*dT_dU             # surge excitation
            #F_aero[4,i] = U_amplitude[i]*dT_dU*Hhub        # pitch excitation
            #F_aero[7,i] = U_amplitude*dQ_dU            # rotor torque excitation

        # calculate steady aero forces and moments
        F_aero0 = np.hstack((self.outputs["Fhub"],self.outputs["Mhub"]))

        return A_aero, B_aero, C_aero, F_aero0, F_aero



    def calcAeroServoContributions(self, nw=0, U_amplitude=[]):
        '''Calculates stiffness, damping, added mass, and excitation coefficients
        from rotor aerodynamics coupled with turbine controls.
        Results are w.r.t. nonrotating hub reference frame.
        '''

        Uinf = 10.  # inflow wind speed (m/s) <<< eventually should be consistent with rest of RAFT
        Hhub = 100.

        # extract derivatives of interest, interpolated for the current wind speed
        dT_dU  = np.interp(Uinf, self.Uhub, self.J["T", "Uhub"     ])
        dT_dOm = np.interp(Uinf, self.Uhub, self.J["T", "Omega_rpm"])
        dT_dPi = np.interp(Uinf, self.Uhub, self.J["T", "pitch_deg"])
        dQ_dU  = np.interp(Uinf, self.Uhub, self.J["Q", "Uhub"     ])
        dQ_dOm = np.interp(Uinf, self.Uhub, self.J["Q", "Omega_rpm"])
        dQ_dPi = np.interp(Uinf, self.Uhub, self.J["Q", "pitch_deg"])

        # coefficients to be filled in
        A_aero = np.zeros([6,6])                        # added mass
        B_aero = np.zeros([6,6])                        # damping
        C_aero = np.zeros([6,6])                        # stiffness
        F_aero0= np.zeros(6)                            # steady wind forces/moments
        F_aero = np.zeros([6,nw])                       # wind excitation spectra in each DOF

        # calculate nonzero matrix entries

        #...

        # calculate wind excitation force/moment spectra (will this change with control?)
        for i in range(nw):                             # loop through each frequency component
            F_aero[0,i] = U_amplitude[i]*dT_dU             # surge excitation

        return A_aero, B_aero, C_aero, F_aero0, F_aero