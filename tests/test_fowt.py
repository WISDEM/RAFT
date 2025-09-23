# tests RAFT FOWT functionality and results

import pytest
import numpy as np
from numpy.testing import assert_allclose
import yaml
import pickle
import raft
import os

'''
 Define files for testing
'''
# Name of the subfolder where the test data is located
test_dir = 'test_data'

# List of input file names to be tested
list_files = [
    'OC3spar.yaml',
    'VolturnUS-S.yaml',
    'VolturnUS-S-pointInertia.yaml',
    'OC4semi-WAMIT_Coefs.yaml',
    'VolturnUS-S-flexible.yaml',
]


# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# To avoid problems with different platforms, get the full path of the file
list_files = [os.path.join(current_dir, test_dir, file) for file in list_files]

'''
 Aux functions
'''
# Function used to create FOWT instance
# Not explicitly inside the fixture below so that we can also run this file as a script
# 
def create_fowt(file):
    with open(file) as f:
        design = yaml.load(f, Loader=yaml.FullLoader)        

    if 'hydroPath' in design['platform']:
        design['platform']['hydroPath'] = os.path.join(current_dir, test_dir, design['platform']['hydroPath'])

    fowt = raft.Model(design).fowtList[0]
    fowt.setPosition(np.zeros(fowt.nDOF))
    fowt.calcStatics()
    return fowt

# Define a fixture to loop fowt instances with the index to loop the desired values as well
# Could also zip the lists with the desired values, but I think the approach below is simpler
@pytest.fixture(params=enumerate(list_files))
def index_and_fowt(request):
    index, file = request.param
    fowt = create_fowt(file)    
    return index, fowt


'''
 Test functions
'''
def test_statics(index_and_fowt, flagSaveValues=False):
    index, fowt = index_and_fowt
    true_values_file = list_files[index].replace('.yaml', '_true_statics.pkl')

    # Values that were computed by current RAFT
    computed_values = {
        'rCG': fowt.rCG,
        'rCG_sub': fowt.rCG_sub,
        'm_ballast': fowt.m_ballast,
        'M_struc': fowt.M_struc,
        'M_struc_sub': fowt.M_struc_sub,
        'C_struc': fowt.C_struc,
        'W_struc': fowt.W_struc,
        'rCB': fowt.rCB,
        'C_hydro': fowt.C_hydro,
        'W_hydro': fowt.W_hydro,
    }
    
    if flagSaveValues: # Save computed values as true values
        with open(true_values_file, 'wb') as f:
            pickle.dump(computed_values, f)
    else: # Load true values and compare        
        with open(true_values_file, 'rb') as f:
            true_values = pickle.load(f)

        for key in computed_values:
            assert_allclose(computed_values[key], true_values[key], rtol=1e-5, atol=1e-3)


def test_hydroConstants(index_and_fowt, flagSaveValues=False):
    index, fowt = index_and_fowt
    true_values_file = list_files[index].replace('.yaml', '_true_hydroConstants.pkl')

    fowt.calcHydroConstants()
    computed_values = {
        'A_hydro_morison': fowt.A_hydro_morison,
    }

    if flagSaveValues:
        with open(true_values_file, 'wb') as f:
            pickle.dump(computed_values, f)
    else:
        with open(true_values_file, 'rb') as f:
            true_values = pickle.load(f)
        for key in computed_values:
            assert_allclose(computed_values[key], true_values[key], rtol=1e-5, atol=1e-3)


def test_hydroExcitation(index_and_fowt, flagSaveValues=False):
    # Set flagSaveValues to true to replace the true values file with the values calculated below
    index, fowt = index_and_fowt        
    true_values_file = list_files[index].replace('.yaml', '_true_hydroExcitation.pkl')
    output_true_values = []
    
    list_wave_heading = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    list_wave_period  = [5, 10, 15, 20]
    list_wave_height  = [1, 2]
    
    idxTrueValues = 0
    for wave_heading in list_wave_heading:
        for wave_period in list_wave_period:
            for wave_height in list_wave_height:
                # Create case dictionary. The other necessary fields have default values within calcHydroExcitation.
                # Using the default values is useful to check if they were changed.
                testCase = {'wave_heading': wave_heading, 'wave_period': wave_period, 'wave_height': wave_height}

                fowt.calcHydroConstants()
                fowt.calcHydroExcitation(testCase, memberList=fowt.memberList)

                if flagSaveValues:
                    output_true_values.append({
                        'case': testCase,
                        'w': fowt.w,
                        'F_hydro_iner': fowt.F_hydro_iner,
                    })
                else:
                    with open(true_values_file, 'rb') as f:
                        true_values = pickle.load(f)

                    assert_allclose(fowt.F_hydro_iner, true_values[idxTrueValues]['F_hydro_iner'], rtol=1e-05, atol=1e-3)                    
                idxTrueValues += 1

    if flagSaveValues:
        with open(true_values_file, 'wb') as f:
            pickle.dump(output_true_values, f)


def test_hydroLinearization(index_and_fowt, flagSaveValues=False):
    # Set flagSaveValues to true to replace the true values file with the values calculated below
    index, fowt = index_and_fowt
    true_values_file = list_files[index].replace('.yaml', '_true_hydroLinearization.pkl')

    testCase = {'wave_spectrum': 'unit', 'wave_heading': 0, 'wave_period': 10, 'wave_height': 2} # Currently we need to specify wave period and height, even though they are not used for unit spectrum
    fowt.calcHydroExcitation(testCase, memberList=fowt.memberList) # Need wave kinematics

    phase_array = np.linspace(0, 2 * np.pi, fowt.nw * fowt.nDOF).reshape(fowt.nDOF, fowt.nw) # Needed an arbitrary motion amplitude. Assuming uniform amplitude with phases linearly spaced between 0 and 2pi. Times 6 for surge, sway, ..., yaw
    Xi = 0.1*np.exp(1j * phase_array)
    B_hydro_drag = fowt.calcHydroLinearization(Xi)   
    F_hydro_drag = fowt.calcDragExcitation(0)
    if flagSaveValues:
        with open(true_values_file, 'wb') as f:
            true_values = {'B_hydro_drag': B_hydro_drag, 'F_hydro_drag': F_hydro_drag}
            pickle.dump(true_values, f)
        return
    else:
        with open(true_values_file, 'rb') as f:
            true_values = pickle.load(f)

    # Check the linearized drag matrix
    assert_allclose(B_hydro_drag, true_values['B_hydro_drag'], rtol=1e-05, atol=1e-10)

    # Check the linearized drag excitation
    assert_allclose(F_hydro_drag, true_values['F_hydro_drag'], rtol=1e-05)


def test_calcCurrentLoads(index_and_fowt, flagSaveValues=False):
    index, fowt = index_and_fowt
    true_values_file = list_files[index].replace('.yaml', '_true_calcCurrentLoads.pkl')
    testCase = {'current_speed': 2.0, 'current_heading': 15}
    D = fowt.calcCurrentLoads(testCase)

    if flagSaveValues:
        with open(true_values_file, 'wb') as f:
            pickle.dump(D, f)
    else:
        with open(true_values_file, 'rb') as f:
            true_D = pickle.load(f)
        assert_allclose(D, true_D, rtol=1e-5, atol=1e-3)

def test_calcQTF_slenderBody(index_and_fowt, flagSaveValues=False):    
    # Set flagSaveValues to true to replace the true values file with the values calculated below    
    index, fowt = index_and_fowt      

    if fowt.potSecOrder == 1: # Check only cases that compute the QTFs
        true_values_file = list_files[index].replace('.yaml', '_true_calcQTF_slenderBody.pkl')
        
        testCase = {'wave_heading': 30, 'wave_period': 12, 'wave_height': 6} # Testing one case only
        fowt.calcHydroConstants() # Need to call this one before calcHydroExcitation
        fowt.calcHydroExcitation(testCase, memberList=fowt.memberList) # Need to call this one before calcQTF_slenderBody
        fowt.calcQTF_slenderBody(0) # Testing for the body considered to be fixed. test_model.py should take care of cases with motion

        if flagSaveValues:
            true_values={
                'case': testCase,
                'qtf': fowt.qtf,
            }
        else:
            with open(true_values_file, 'rb') as f:
                true_values = pickle.load(f)
            assert_allclose(fowt.qtf, true_values['qtf'], rtol=1e-05, atol=1e-3)

        if flagSaveValues:
            with open(true_values_file, 'wb') as f:
                pickle.dump(true_values, f)

def test_calcBEM(index_and_fowt, flagSaveValues=False):    
    # Set flagSaveValues to true to replace the true values file with the values calculated below    
    index, fowt = index_and_fowt      
    
    # Only test cases that include potential flow coefficients
    if fowt.potMod:
        true_values_file = list_files[index].replace('.yaml', '_true_BEM_forces.pkl')
        
        testCase = {'wave_heading': 30, 'wave_period': 12, 'wave_height': 6} # Testing one case only
        fowt.calcBEM() # Need to call this one before calcHydroExcitation

        if flagSaveValues:
            true_values={
                'case': testCase,
                'X_BEM': fowt.X_BEM,
                'A_BEM': fowt.A_BEM,
                'B_BEM': fowt.B_BEM,
            }
        else:
            with open(true_values_file, 'rb') as f:
                true_values = pickle.load(f)
            assert_allclose(fowt.X_BEM, true_values['X_BEM'], rtol=1e-05, atol=1e-3)
            assert_allclose(fowt.A_BEM, true_values['A_BEM'], rtol=1e-05, atol=1e-3)
            assert_allclose(fowt.B_BEM, true_values['B_BEM'], rtol=1e-05, atol=1e-3)

        if flagSaveValues:
            with open(true_values_file, 'wb') as f:
                pickle.dump(true_values, f)                


'''
 To run as a script. Useful for debugging and updating true values when needed.
'''
if __name__ == "__main__":
    flagSaveValues = False
    for index in range(len(list_files)):
        fowt = create_fowt(list_files[index])
        test_statics((index,fowt), flagSaveValues=flagSaveValues)
        
        fowt = create_fowt(list_files[index])
        test_hydroConstants((index,fowt), flagSaveValues=flagSaveValues)

        fowt = create_fowt(list_files[index])
        test_hydroExcitation((index,fowt), flagSaveValues=flagSaveValues)

        fowt = create_fowt(list_files[index])
        test_hydroLinearization((index,fowt), flagSaveValues=flagSaveValues)

        fowt = create_fowt(list_files[index])
        test_calcCurrentLoads((index,fowt), flagSaveValues=flagSaveValues)

        fowt = create_fowt(list_files[index])
        test_calcQTF_slenderBody((index,fowt), flagSaveValues=flagSaveValues)

        fowt = create_fowt(list_files[index])
        test_calcBEM((index,fowt), flagSaveValues=flagSaveValues)
