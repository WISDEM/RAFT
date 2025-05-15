# tests RAFT FOWT functionality and results

import pytest
import numpy as np
from numpy.testing import assert_allclose
import yaml
import pickle
import raft
from raft.raft_rotor import Rotor
from raft.helpers import getFromDict
import os

'''
 Define files for testing
'''
# Name of the subfolder where the test data is located
test_dir = 'test_data'

# List of input file names to be tested
list_files = [
    'IEA15MW.yaml',
]

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# To avoid problems with different platforms, get the full path of the file
list_files = [os.path.join(current_dir, test_dir, file) for file in list_files]


'''
 Aux functions
'''
# Function used to create Rotor instance
# Not explicitly inside the fixture below so that we can also run this file as a script
# This code is a modified copy from parts of FOWT.__init__ and Model.__init__ 
# We should probably have a function for doing that to avoid code repetition.
def create_rotor(file):
    with open(file) as f:
        design = yaml.load(f, Loader=yaml.FullLoader)
        
    nrotors = getFromDict(design['turbine'], 'nrotors', dtype=int, shape=0, default=1)
    if nrotors==1: design['turbine']['nrotors'] = 1
    else: raise NotImplementedError('Multiple rotors not supported yet by the testing function.')
    
    if 'tower' in design['turbine']:
        if isinstance(design['turbine']['tower'], dict):                           # if a single tower is specified (rather than a list)
            design['turbine']['tower'] = [design['turbine']['tower']]*nrotors  # replicate the tower info for each rotor    

    # copy over site info into turbine dictionary
    design['turbine']['rho_air'       ] = getFromDict(design['site'], 'rho_air', shape=0, default=1.225)
    design['turbine']['mu_air'        ] = getFromDict(design['site'], 'mu_air', shape=0, default=1.81e-05)
    design['turbine']['shearExp_air'  ] = getFromDict(design['site'], 'shearExp_air', shape=0, default=0.12)
    design['turbine']['rho_water'     ] = getFromDict(design['site'], 'rho_water', shape=0, default=1025.0)
    design['turbine']['mu_water'      ] = getFromDict(design['site'], 'mu_water', shape=0, default=1.0e-03)
    design['turbine']['shearExp_water'] = getFromDict(design['site'], 'shearExp_water', shape=0, default=0.12)

    min_freq     = getFromDict(design['settings'], 'min_freq', default=0.01, dtype=float)  # [Hz] lowest frequency to consider, also the frequency bin width 
    max_freq     = getFromDict(design['settings'], 'max_freq', default=1.00, dtype=float)  # [Hz] highest frequency to consider        
    w = np.arange(min_freq, max_freq+0.5*min_freq, min_freq) *2*np.pi  # angular frequencies to analyze (rad/s)
    
    # Nacelle geometry (for hydro): if a single one (dictionary) is listed,
    # replicate it into a list for all rotors.
    if 'nacelle' in design['turbine']:
        if isinstance(design['turbine']['nacelle'], dict): 
            design['turbine']['nacelle'] = [design['turbine']['nacelle']]*nrotors

    return Rotor(design['turbine'], w, 0)


# Define a fixture to loop rotor instances with the index to loop the desired values as well
# Could also zip the lists with the desired values, but I think the approach below is simpler
@pytest.fixture(params=enumerate(list_files))
def index_and_rotor(request):
    index, file = request.param
    rotor = create_rotor(file)    
    return index, rotor


'''
 Test functions
'''
def test_calcAero(index_and_rotor, flagSaveValues=False):    
    '''
    Verify rotor.calcAero for many combinations of wind speed, heading, TI, and yaw mode.
    Some combinations seem to be outside the validity of CCBlade (e.g., yaw_mode=1, wind direction=90, and turbine_heading=0),
    but they are tested nonetheless.
    Set flagSaveValues to true to replace the true values file with the values calculated below
    '''
    index, rotor = index_and_rotor
    rotor.setPosition()

    if 'IEA15MW' in list_files[index]:
        U_rated = 10.59
    elif 'NREL5MW' in list_files[index]:
        U_rated = 11.4
    else:
        U_rated = 12
        print(f'Unknown turbine. Considering U_rated = {U_rated} m/s.')

    # We are going to loop through the following values to test a broad range of cases
    wind_speeds = np.sort(np.append([5, 15, 25], U_rated))
    wind_headings = [-45, 0, 45]
    TI            = [0, 0.5]
    yaw_modes     = [0, 1, 2, 3]
    values4control = [  # Values used for yaw misalignment or turbine heading depending on yaw_mode. One list per yaw_mode
                     [0],
                     [-15, 0, 15],
                     [-15, 0, 15],
                     [-15, 0, 15],
                     ]
    
    for idx_ym, ym in enumerate(yaw_modes):
        rotor.yaw_mode = ym
    
        true_values_file = list_files[index].replace('.yaml', f'_true_calcAero-yaw_mode{ym:d}.pkl')
        output_true_values = []
        idxTrueValues = 0

        for ws in wind_speeds:
            for wh in wind_headings:
                for ti in TI:
                    for v4c in values4control[idx_ym]:
                        if ym == 1: # This one needs turbine_heading
                            thisCase = {'wind_speed': ws, 'wind_heading': wh, 'turbulence': ti, 'turbine_status': 'operating', 'turbine_heading': v4c}
                        else:
                            thisCase = {'wind_speed': ws, 'wind_heading': wh, 'turbulence': ti, 'turbine_status': 'operating', 'yaw_misalign': v4c}
                        f_aero0, f_aero, a_aero, b_aero = rotor.calcAero(thisCase)

                        if flagSaveValues:
                            output_true_values.append({
                                'case': thisCase,
                                'f_aero0': f_aero0,
                                'f_aero': f_aero,
                                'a_aero': a_aero,
                                'b_aero': b_aero
                            })

                        else:
                            with open(true_values_file, 'rb') as f:
                                true_values = pickle.load(f)

                            assert_allclose(f_aero0, true_values[idxTrueValues]['f_aero0'], rtol=1e-5, atol=1e-5)
                            assert_allclose(f_aero, true_values[idxTrueValues]['f_aero'], rtol=1e-5, atol=1e-5)
                            assert_allclose(a_aero, true_values[idxTrueValues]['a_aero'], rtol=1e-5, atol=1e-5)
                            assert_allclose(b_aero, true_values[idxTrueValues]['b_aero'], rtol=1e-5, atol=1e-5)
                        idxTrueValues += 1

            if flagSaveValues:
                with open(true_values_file, 'wb') as f:
                    pickle.dump(output_true_values, f)
    # Do something similar for turbine class

'''
 To run as a script. Useful for debugging.
'''
if __name__ == "__main__":
    index = 0

    rotor = create_rotor(list_files[index])
    test_calcAero((index, rotor), flagSaveValues=False)

