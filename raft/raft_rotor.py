# RAFT's rotor class

import os
import os.path as osp
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator

import wisdem.inputs as sch
from wisdem.ccblade.ccblade import CCBlade, CCAirfoil
from wisdem.commonse.utilities import arc_length

'''
try:
    import ccblade        as CCBlade, CCAirfoil  # for cloned ccblade
except:
    import wisdem.ccblade as CCblade, CCAirfoil  # for conda install wisdem
'''


# a class for the rotor structure, aerodynamics, and control in RAFT
class Rotor:

    def __init__(self):
        '''
        '''
        
        # (not worrying about rotor structure/mass yet, just aero)
               

        # Load wind turbine geometry yaml
        run_dir = os.path.dirname( os.path.realpath(__file__) ) + os.sep
        fname_input_wt = os.path.join(run_dir, "IEA-15-240-RWT.yaml")   # <<<<<<<<<< input
        wt_init = sch.load_geometry_yaml(fname_input_wt)

        # Set CCBlade flags
        tiploss = True # Tip loss model True/False
        hubloss = True # Hub loss model, True/False
        wakerotation = True # Wake rotation, True/False
        usecd = True # Use drag coefficient within BEMT, True/False

        yaw = 0.

        # Set discretization parameters
        nSector = 4 # [-] - number of equally spaced azimuthal positions where CCBlade should be interrogated. The results are averaged across the n positions. 4 is a good first guess
        n_span = 30 # [-] - number of blade stations along span
        grid = np.linspace(0., 1., n_span) # equally spaced grid along blade span, root=0 tip=1
        n_aoa = 200 # [-] - number of angles of attack to discretize airfoil polars



        ##########################################
        #  No need to change anything after this #
        ##########################################

        # Conversion of the yaml inputs into CCBlade inputs
        Rhub = 0.5 * wt_init["components"]["hub"]["diameter"] # [m] - hub radius
        precone = np.rad2deg(wt_init["components"]["hub"]["cone_angle"]) # [deg] - rotor precone angle
        tilt = np.rad2deg(wt_init["components"]["nacelle"]["drivetrain"]["uptilt"]) # [deg] -  nacelle uptilt angle
        B = wt_init["assembly"]["number_of_blades"] # [-] - number of blades
        blade = wt_init["components"]["blade"]["outer_shape_bem"]

        # Blade quantities
        rotor_diameter = wt_init["assembly"]["rotor_diameter"]
        blade_ref_axis = np.zeros((n_span, 3))
        blade_ref_axis[:, 0] = np.interp(grid, blade["reference_axis"]["x"]["grid"], blade["reference_axis"]["x"]["values"])
        blade_ref_axis[:, 1] = np.interp(grid, blade["reference_axis"]["y"]["grid"], blade["reference_axis"]["y"]["values"])
        blade_ref_axis[:, 2] = np.interp(grid, blade["reference_axis"]["z"]["grid"], blade["reference_axis"]["z"]["values"])
        if rotor_diameter != 0.0:
            blade_ref_axis[:, 2] = (blade_ref_axis[:, 2] * rotor_diameter / ((arc_length(blade_ref_axis)[-1] + Rhub) * 2.0))
        r = blade_ref_axis[1:-1, 2] + Rhub # [m] - radial position along straight blade pitch axis
        Rtip = blade_ref_axis[-1, 2] + Rhub
        chord = np.interp(grid[1:-1], blade["chord"]["grid"], blade["chord"]["values"]) # [m] - blade chord distributed along r
        theta = np.rad2deg(np.interp(grid[1:-1], blade["twist"]["grid"], blade["twist"]["values"])) # [deg] - blade twist distributed along r
        precurve = blade_ref_axis[1:-1, 0] # [m] - blade prebend distributed along r, usually negative for upwind rotors
        precurveTip = blade_ref_axis[-1, 0] # [m] - prebend at blade tip
        presweep = blade_ref_axis[1:-1, 1] # [m] - blade presweep distributed along r, usually positive
        presweepTip = blade_ref_axis[-1, 1] # [m] - presweep at blade tip

        # Hub height
        if wt_init["assembly"]["hub_height"] != 0.0:
            hub_height = wt_init["assembly"]["hub_height"]
        else:
            hub_height = wt_init["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["values"][-1] + wt_init["components"]["nacelle"]["drivetrain"]["distance_tt_hub"]

        # Atmospheric boundary layer data
        rho = wt_init['environment']["air_density"] # [kg/m3] - density of air
        mu = wt_init['environment']["air_dyn_viscosity"] # [kg/(ms)] - dynamic viscosity of air
        shearExp = wt_init['environment']["shear_exp"] # [-] - shear exponent

        # Airfoil data


        n_af = len(wt_init["airfoils"])
        af_used = blade["airfoil_position"]["labels"]
        af_position = blade["airfoil_position"]["grid"]
        n_af_span = len(af_used)
        if n_aoa / 4.0 == int(n_aoa / 4.0):
            # One fourth of the angles of attack from -pi to -pi/6, half between -pi/6 to pi/6, and one fourth from pi/6 to pi
            aoa = np.unique(np.hstack([np.linspace(-np.pi, -np.pi / 6.0, int(n_aoa / 4.0 + 1)),np.linspace(-np.pi / 6.0,np.pi / 6.0,int(n_aoa / 2.0),),np.linspace(np.pi / 6.0, np.pi, int(n_aoa / 4.0 + 1))]))
        else:
            aoa = np.linspace(-np.pi, np.pi, n_aoa)
            print(
                "WARNING: If you like a grid of angles of attack more refined between +- 30 deg, please choose a n_aoa in the analysis option input file that is a multiple of 4. The current value of "
                + str(n_aoa)
                + " is not a multiple of 4 and an equally spaced grid is adopted."
            )

        Re_all = []
        for i in range(n_af):
            for j in range(len(wt_init["airfoils"][i]["polars"])):
                Re_all.append(wt_init["airfoils"][i]["polars"][j]["re"])
        n_Re = len(np.unique(Re_all))

        n_tab = 1

        af_name = n_af * [""]
        r_thick = np.zeros(n_af)
        Re_all = []
        for i in range(n_af):
            af_name[i] = wt_init["airfoils"][i]["name"]
            r_thick[i] = wt_init["airfoils"][i]["relative_thickness"]
            for j in range(len(wt_init["airfoils"][i]["polars"])):
                Re_all.append(wt_init["airfoils"][i]["polars"][j]["re"])

        Re = np.array(sorted(np.unique(Re_all)))

        cl = np.zeros((n_af, n_aoa, n_Re, n_tab))
        cd = np.zeros((n_af, n_aoa, n_Re, n_tab))
        cm = np.zeros((n_af, n_aoa, n_Re, n_tab))

        # Interp cl-cd-cm along predefined grid of angle of attack
        for i in range(n_af):
            n_Re_i = len(wt_init["airfoils"][i]["polars"])
            Re_j = np.zeros(n_Re_i)
            j_Re = np.zeros(n_Re_i, dtype=int)
            for j in range(n_Re_i):
                Re_j[j] = wt_init["airfoils"][i]["polars"][j]["re"]
                j_Re[j] = np.argmin(abs(Re - Re_j[j]))
                cl[i, :, j_Re[j], 0] = np.interp(
                    aoa, wt_init["airfoils"][i]["polars"][j]["c_l"]["grid"], wt_init["airfoils"][i]["polars"][j]["c_l"]["values"]
                )
                cd[i, :, j_Re[j], 0] = np.interp(
                    aoa, wt_init["airfoils"][i]["polars"][j]["c_d"]["grid"], wt_init["airfoils"][i]["polars"][j]["c_d"]["values"]
                )
                cm[i, :, j_Re[j], 0] = np.interp(
                    aoa, wt_init["airfoils"][i]["polars"][j]["c_m"]["grid"], wt_init["airfoils"][i]["polars"][j]["c_m"]["values"]
                )

                if abs(cl[i, 0, j, 0] - cl[i, -1, j, 0]) > 1.0e-5:
                    cl[i, 0, j, 0] = cl[i, -1, j, 0]
                    print(
                        "WARNING: Airfoil "
                        + af_name[i]
                        + " has the lift coefficient at Re "
                        + str(Re_j[j])
                        + " different between + and - pi rad. This is fixed automatically, but please check the input data."
                    )
                if abs(cd[i, 0, j, 0] - cd[i, -1, j, 0]) > 1.0e-5:
                    cd[i, 0, j, 0] = cd[i, -1, j, 0]
                    print(
                        "WARNING: Airfoil "
                        + af_name[i]
                        + " has the drag coefficient at Re "
                        + str(Re_j[j])
                        + " different between + and - pi rad. This is fixed automatically, but please check the input data."
                    )
                if abs(cm[i, 0, j, 0] - cm[i, -1, j, 0]) > 1.0e-5:
                    cm[i, 0, j, 0] = cm[i, -1, j, 0]
                    print(
                        "WARNING: Airfoil "
                        + af_name[i]
                        + " has the moment coefficient at Re "
                        + str(Re_j[j])
                        + " different between + and - pi rad. This is fixed automatically, but please check the input data."
                    )

            # Re-interpolate cl-cd-cm along the Re dimension if less than n_Re were provided in the input yaml (common condition)
            for l in range(n_aoa):
                cl[i, l, :, 0] = np.interp(Re, Re_j, cl[i, l, j_Re, 0])
                cd[i, l, :, 0] = np.interp(Re, Re_j, cd[i, l, j_Re, 0])
                cm[i, l, :, 0] = np.interp(Re, Re_j, cm[i, l, j_Re, 0])

        # Interpolate along blade span using a pchip on relative thickness
        r_thick_used = np.zeros(n_af_span)
        cl_used = np.zeros((n_af_span, n_aoa, n_Re, n_tab))
        cl_interp = np.zeros((n_span, n_aoa, n_Re, n_tab))
        cd_used = np.zeros((n_af_span, n_aoa, n_Re, n_tab))
        cd_interp = np.zeros((n_span, n_aoa, n_Re, n_tab))
        cm_used = np.zeros((n_af_span, n_aoa, n_Re, n_tab))
        cm_interp = np.zeros((n_span, n_aoa, n_Re, n_tab))

        for i in range(n_af_span):
            for j in range(n_af):
                if af_used[i] == af_name[j]:
                    r_thick_used[i] = r_thick[j]
                    cl_used[i, :, :, :] = cl[j, :, :, :]
                    cd_used[i, :, :, :] = cd[j, :, :, :]
                    cm_used[i, :, :, :] = cm[j, :, :, :]
                    break

        # Pchip does have an associated derivative method built-in:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.derivative.html#scipy.interpolate.PchipInterpolator.derivative
        spline = PchipInterpolator
        rthick_spline = spline(af_position, r_thick_used)
        r_thick_interp = rthick_spline(grid[1:-1])


        # Spanwise interpolation of the airfoil polars with a pchip
        r_thick_unique, indices = np.unique(r_thick_used, return_index=True)
        cl_spline = spline(r_thick_unique, cl_used[indices, :, :, :])
        cl_interp = np.flip(cl_spline(np.flip(r_thick_interp)), axis=0)
        cd_spline = spline(r_thick_unique, cd_used[indices, :, :, :])
        cd_interp = np.flip(cd_spline(np.flip(r_thick_interp)), axis=0)
        cm_spline = spline(r_thick_unique, cm_used[indices, :, :, :])
        cm_interp = np.flip(cm_spline(np.flip(r_thick_interp)), axis=0)


        af = [None] * (n_span - 2)
        for i in range(n_span - 2):
            af[i] = CCAirfoil(np.rad2deg(aoa), Re, cl_interp[i, :, :, 0], cd_interp[i, :, :, 0], cm_interp[i, :, :, 0])

        self.ccblade = CCBlade(
            r,
            chord,
            theta,
            af,
            Rhub,
            Rtip,
            B,
            rho,
            mu,
            precone,
            tilt,
            yaw,
            shearExp,
            hub_height,
            nSector,
            precurve,
            precurveTip,
            presweep,
            presweepTip,
            tiploss=tiploss,
            hubloss=hubloss,
            wakerotation=wakerotation,
            usecd=usecd,
            derivatives=True,
        )

    
    def runCCblade(self):
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