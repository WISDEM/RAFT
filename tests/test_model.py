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
    'VolturnUS-S.yaml',
    'OC3spar.yaml',
    'VolturnUS-S_farm.yaml'
]


# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# To avoid problems with different platforms, get the full path of the file
list_files = [os.path.join(current_dir, test_dir, file) for file in list_files]


# Single test case used in test_solveStatics and test_solveDynamics
# Multiple cases from the yaml file are tested in test_analyzeCases only

'''
 Aux functions
'''
# Function used to create FOWT instance
# Not explicitly inside the fixture below so that we can also run this file as a script
# 
def create_model(file):
    with open(file) as f:
        design = yaml.load(f, Loader=yaml.FullLoader)

    if 'array_mooring' in design: # Relative paths may be different in different platforms, so we make sure the path is correct
        if design['array_mooring']['file']:
            design['array_mooring']['file'] = os.path.join(current_dir, test_dir, design['array_mooring']['file'])

    model = raft.Model(design)
    return model

# Define a fixture to loop fowt instances with the index to loop the desired values as well
# Could also zip the lists with the desired values, but I think the approach below is simpler
@pytest.fixture(params=enumerate(list_files))
def index_and_model(request):
    index, file = request.param
    model = create_model(file)    
    return index, model

'''
 Test functions
'''
#===== model.solveStatics for different loading conditions
cases4solveStatics = {
    'wind':              {'wind_speed': 8, 'wind_heading': 30, 'turbulence': 0, 'turbine_status': 'operating', 'yaw_misalign': 0, 'wave_spectrum': 'JONSWAP', 'wave_period':  0, 'wave_height': 0, 'wave_heading':   0, 'current_speed': 0, 'current_heading':  0},
    'wave':              {'wind_speed': 0, 'wind_heading':  0, 'turbulence': 0, 'turbine_status': 'operating', 'yaw_misalign': 0, 'wave_spectrum': 'JONSWAP', 'wave_period': 10, 'wave_height': 4, 'wave_heading': -30, 'current_speed': 0, 'current_heading':  0},
    'current':           {'wind_speed': 0, 'wind_heading':  0, 'turbulence': 0, 'turbine_status': 'operating', 'yaw_misalign': 0, 'wave_spectrum': 'JONSWAP', 'wave_period':  0, 'wave_height': 0, 'wave_heading':   0, 'current_speed': 0.6, 'current_heading': 15},
    'wind_wave_current': {'wind_speed': 8, 'wind_heading': 30, 'turbulence': 0, 'turbine_status': 'operating', 'yaw_misalign': 0, 'wave_spectrum': 'JONSWAP', 'wave_period': 10, 'wave_height': 4, 'wave_heading': -30, 'current_speed': 0.6, 'current_heading': 15}
}

desired_X0 = {
    'wind': [
        np.array([ 1.27981309e+01,  1.05301196e+01, -5.00955468e-01, -3.08123709e-02,  5.84502589e-02, -7.01593570e-03]),
        np.array([ 1.09704957e+01,  5.30341749e+00, -8.08255233e-01, -2.20791574e-02,  4.00562243e-02, -1.88931043e-02]),        
        np.array([ 1.68407496e+01,  1.10109039e+01,  6.65635303e-01, -2.65746461e-02,  5.62504188e-02, -3.52280731e-02,  1.61816436e+03,  1.14496392e+01,  1.03847988e+00, -2.74957980e-02,  5.84431017e-02,  4.36756068e-02])
    ],
    'wave': [
        np.array([ 1.69712005e-02, -1.93781208e-17, -4.28261180e-01, -1.21300094e-18,  2.26746861e-05, -2.30847610e-23]),
        np.array([-1.64267049e-05, -2.83795893e-15, -6.65861624e-01,  3.88717546e-19, -5.94238978e-11, -4.02571352e-17]),
        np.array([-5.01177348e-01,  1.11798952e-15,  8.82461053e-01,  4.91932000e-17,  4.39038724e-04,  8.69456218e-19,  1.60050118e+03,  9.82053320e-16,  8.82460768e-01,  4.27743746e-17, -4.39066827e-04, -8.32305085e-19])
    ],
    'current': [
        np.array([ 3.07647856e+00,  8.09230061e-01, -4.29676672e-01,  6.33390732e-04, -2.49217661e-03,  3.80888009e-03]),
        np.array([ 3.86072176e+00,  9.22694246e-01, -6.74898762e-01, -2.64759824e-04,  9.82529767e-04, -1.03532699e-05]),
        np.array([ 3.24739802e+00,  1.08484956e+00,  8.42959914e-01,  7.16963134e-04, -1.22097638e-03, -5.87434156e-03,  1.60424961e+03,  1.10109258e+00,  9.21764906e-01,  7.58137041e-04, -2.11268701e-03,  6.56575162e-03])
    ],
    'wind_wave_current': [
        np.array([ 1.50128997e+01,  1.17794646e+01, -5.13873441e-01, -3.01301643e-02,  5.60700058e-02, -3.19071097e-03]),        
        np.array([ 1.51819382e+01,  5.67973367e+00, -8.56976995e-01, -2.23553755e-02,  4.09788544e-02, -2.60679623e-02]),
        np.array([ 2.05748182e+01,  1.20556081e+01,  6.26678004e-01, -2.58821266e-02,  5.43569306e-02, -3.84336757e-02,  1.62220604e+03,  1.29571972e+01,  1.07692445e+00, -2.66374984e-02,  5.70843751e-02,  5.55086139e-02])
    ]
}

def solveStatics(index_and_model, test_case_key):
    '''
    We test only the mean offsets and linearized mooring properties.
    '''        
    index, model = index_and_model
    testCase = cases4solveStatics[test_case_key]    
    model.solveStatics(testCase)
    for i, fowt in enumerate(model.fowtList):
        assert_allclose(fowt.r6, desired_X0[test_case_key][index][6*i:6*(i+1)], rtol=1e-05, atol=1e-10)

def test_solveStatics_Wind(index_and_model):
    solveStatics(index_and_model, 'wind')

def test_solveStatics_Wave(index_and_model):
    solveStatics(index_and_model, 'wave')

def test_solveStatics_Current(index_and_model):
    solveStatics(index_and_model, 'current')

def test_solveStatics_Wind_Wave_Current(index_and_model):
    solveStatics(index_and_model, 'wind_wave_current')



#===== model.solveEigen for different cases
cases4solveEigen = {
    'unloaded': {'wind_speed': 0, 'wind_heading': 0, 'turbulence': 0, 'turbine_status': 'idle', 'yaw_misalign': 0, 'wave_spectrum': 'JONSWAP', 'wave_period': 0, 'wave_height': 0, 'wave_heading': 0, 'current_speed': 0, 'current_heading': 0},
    'loaded':   {'wind_speed': 8, 'wind_heading': 30, 'turbulence': 0, 'turbine_status': 'operating', 'yaw_misalign': 0, 'wave_spectrum': 'JONSWAP', 'wave_period': 10, 'wave_height': 4, 'wave_heading': -30, 'current_speed': 0.6, 'current_heading': 15}
}

desired_fn = {
    'unloaded': [
        np.array([0.00780613, 0.00781769, 0.06073888, 0.03861193, 0.03862018, 0.01239692]),
        np.array([0.00796903, 0.00796903, 0.03245079, 0.03383781, 0.03384323, 0.15347415]),
        np.array([0.01074625, 0.00716318, 0.05084381, 0.03748606, 0.03783757, 0.01574022, 0.00756192, 0.00704588, 0.05086277, 0.03748700, 0.03779494, 0.01547133])
    ],
    'loaded': [
        np.array([0.00983467, 0.00711942, 0.06075478, 0.03837865, 0.03917143, 0.01326502]),
        np.array([0.00730352, 0.00939103, 0.03246224, 0.03384297, 0.03390545, 0.1555763 ]),
        np.array([0.01066207, 0.00721616, 0.05086057, 0.03789163, 0.03835356, 0.01771014, 0.00743523, 0.00644145, 0.05082023, 0.03678883, 0.03752444, 0.01331672])
    ],
}

desired_modes = {
    'unloaded': [
        np.array([[-9.99999802e-01, -3.58177566e-14, -5.47524722e-08, -4.39990138e-13, -9.90311437e-01, -2.35694740e-15],
                  [-2.09525757e-20,  9.99999800e-01, -1.41067090e-16,  9.90300308e-01, -1.02414825e-14, -1.13370282e-01],
                  [ 1.27034965e-06,  1.14269752e-18, -1.00000000e+00, -5.06576752e-16, -3.61024872e-05, -9.89217552e-20],
                  [-1.92236308e-20,  6.27336513e-04,  2.26769079e-18, -1.38943513e-01,  1.44181432e-15, -1.78996143e-04],
                  [ 6.30061473e-04,  2.26235065e-17,  1.61898091e-07, -6.16893988e-14, -1.38864162e-01, -2.13805104e-17],
                  [ 3.40207703e-20, -8.15805381e-05,  4.18327175e-18,  1.97802394e-05, -7.70377924e-18, -9.93552790e-01]]),
        np.array([[ 9.99999999e-01, -5.75186507e-10, -3.66356915e-10, -8.67852276e-14,  9.99898193e-01, -2.99686750e-14],
                  [ 3.92738961e-10, -9.99999999e-01, -3.31616203e-17,  9.99898194e-01,  3.23260804e-14,  2.06096172e-03],
                  [ 7.45752595e-11, -2.85321785e-18,  1.00000000e+00, -2.72406265e-18,  2.31375465e-10, -1.10751480e-16],
                  [ 2.12335674e-14, -5.40653002e-05,  4.24138273e-19, -1.42689340e-02, -4.31499026e-16, -2.94481141e-05],
                  [-5.40646648e-05,  3.10972493e-14, -4.68816970e-12, -1.23789014e-15,  1.42689404e-02, -1.94677827e-16],
                  [ 1.95726867e-18, -5.58669353e-09,  9.44626575e-17,  4.97628953e-06, -2.81458719e-18, -9.99997876e-01]]),
        np.array([[-7.07063968e-01,  1.71887064e-16, -1.30340226e-02, -2.82542470e-14,  6.94863002e-01,  3.85153580e-16,  7.07066034e-01, -6.53986496e-17,  1.25007531e-02,  2.80098589e-14,  6.92313443e-01,  5.76639921e-16],
                  [-7.25250255e-15,  7.07099398e-01,  1.75093624e-14, -6.97995773e-01, -3.82099078e-14,  6.98639459e-01,  7.80689701e-14, -7.07100871e-01, -1.97524808e-14,  6.97890097e-01, -2.17875817e-14,  6.96282021e-01],
                  [ 7.78086989e-03, -1.31547996e-19, -7.06986675e-01,  3.69809461e-15, -3.37914294e-02,  1.87852753e-17, -7.58851866e-03, -1.14893765e-18,  7.06996195e-01, -3.54499811e-15, -4.84772086e-02,  1.82168142e-17],
                  [-4.42008293e-18,  3.85533504e-04, -1.95389309e-16,  1.13310010e-01,  6.59117155e-15,  2.20799443e-03,  4.26954391e-17, -3.95913134e-04, -2.00243504e-16, -1.13574765e-01,  3.57431861e-15,  2.50916481e-03],
                  [ 4.66972697e-05, -6.69897247e-19, -2.08076011e-04, -5.37783516e-15,  1.26573873e-01, -7.03266775e-19, -1.90337265e-04,  7.23965014e-19,  1.52684219e-04,  5.23638666e-15,  1.35481349e-01, -4.06301593e-18],
                  [ 4.59941229e-17, -3.22196475e-03,  8.14219936e-18, -2.53785685e-03, -1.51955686e-16,  1.09079638e-01, -3.47492945e-16,  2.85012022e-03, -5.76771539e-18,  2.46481410e-03, -4.41443842e-17,  1.23225467e-01],
                  [ 7.07063969e-01,  1.23317808e-16,  1.30340204e-02,  5.92119539e-15,  6.94866599e-01,  5.63370023e-16,  7.07066037e-01, -2.39412617e-17,  1.25007650e-02,  7.52362237e-15, -6.92309081e-01,  1.94827127e-16],
                  [ 8.83463633e-17, -7.07099273e-01, -2.67391452e-17, -6.97934777e-01,  8.19753287e-15, -6.98638992e-01, -9.46449353e-15, -7.07100982e-01, -4.28411155e-18, -6.97951671e-01, -9.99882707e-15,  6.96282772e-01],
                  [ 7.78087658e-03,  3.67247159e-19, -7.06986551e-01,  1.17996039e-15,  3.37916947e-02,  1.50031146e-18,  7.58851869e-03, -5.80489109e-19, -7.06996319e-01,  6.76458394e-16, -4.84769765e-02,  8.65225494e-18],
                  [ 1.23387652e-19, -3.85533550e-04,  3.22484756e-18,  1.13300083e-01, -1.34298598e-15, -2.20798839e-03, -4.64803567e-18, -3.95913299e-04,  3.40496951e-18,  1.13584760e-01,  1.62276251e-15,  2.50916234e-03],
                  [-4.66966628e-05,  9.52249103e-19,  2.08075991e-04,  1.35742924e-15,  1.26574617e-01,  5.81538035e-18, -1.90337282e-04, -1.14034186e-18,  1.52684170e-04,  1.19298438e-15, -1.35480553e-01,  4.27926093e-19],
                  [-2.06368074e-18, -3.22196433e-03,  3.25247512e-19,  2.53764084e-03, -2.87786844e-17,  1.09079545e-01, -6.15608121e-17, -2.85012079e-03, -4.84193205e-19,  2.46503742e-03,  3.86906564e-17, -1.23225573e-01]])
    ],
    'loaded': [
        np.array([[ 9.81639518e-01, -1.90667569e-01,  5.58635270e-03, -2.16960992e-01, -9.69645100e-01,  2.37939853e-01],
                  [-1.90694149e-01, -9.81647754e-01,  2.99413815e-03, -9.65528488e-01,  2.05596001e-01, -6.23831444e-01],
                  [-4.32526222e-03,  3.62729497e-03,  9.99979905e-01,  1.91932750e-02,  2.27486960e-02,  3.72816030e-03],
                  [-1.85483819e-04, -5.04396308e-04, -5.71588748e-05,  1.39107957e-01, -2.72889617e-02,  1.42417390e-02],
                  [-9.21676930e-04,  9.45385064e-05,  1.18833713e-04, -3.10320447e-02, -1.27499112e-01,  8.02033132e-03],
                  [-8.42648011e-05, -3.80015575e-04, -3.09533732e-06, -5.23051246e-04,  6.70612966e-04,  7.44270048e-01]]),
        np.array([[-9.64378242e-01, -2.63590752e-01, -2.10946483e-02,  6.67618956e-01,  7.11326547e-01, -3.97038343e-03],
                  [ 2.64226439e-01, -9.64567735e-01, -1.08535955e-02, -7.44333038e-01,  7.02378251e-01,  2.46425477e-03],
                  [ 1.25912062e-02,  1.12874097e-02,  9.99718461e-01,  5.36744834e-03,  2.17700278e-02, -4.08761745e-04],
                  [ 1.12508015e-05, -8.58285900e-05,  2.09146973e-04,  1.06044546e-02, -1.00067204e-02, -1.76052849e-04],
                  [ 5.54046589e-05,  2.84236886e-05, -4.10342190e-04,  9.52540836e-03,  1.01447352e-02, -8.89069168e-05],
                  [ 6.73636151e-04, -1.25989426e-03,  3.44483770e-05,  4.61085742e-03, -1.53377328e-03,  9.99988979e-01]]),
        np.array([[ 7.70854059e-01, -1.07812429e-01, -1.77350920e-02,  4.61488770e-01, -7.63899755e-01, -6.95967137e-02,  5.96511338e-01, -2.06063675e-01,  4.83434640e-03,  9.11089053e-04, -1.64189060e-02, -1.91827040e-01],
                  [ 7.12894630e-04,  9.85495892e-01, -1.70569554e-04, -8.69004513e-01, -6.15773931e-01,  9.59986560e-01,  1.60092808e-01, -6.58931415e-02,  3.85255071e-05,  2.69400183e-03,  2.07853900e-02,  6.42269549e-02],
                  [-8.49004639e-03,  1.03127643e-03, -9.69149072e-01, -2.89598744e-02,  5.83786886e-02,  1.04835127e-02, -6.37368931e-03,  2.17759940e-03,  2.45771736e-01,  3.19500877e-03,  1.04316961e-02,  2.07860190e-03],
                  [ 1.10847995e-06,  5.18097972e-04,  2.50680055e-05,  1.42863424e-01,  9.82807749e-02,  8.34570035e-03,  8.61679818e-05, -4.44663162e-05, -5.10282407e-06, -3.29505827e-04, -3.52504651e-03, -1.24295242e-04],
                  [-3.54854708e-05, -4.13208082e-05, -2.95584349e-04,  8.70726154e-02, -1.44554441e-01,  2.81498585e-03, -2.01642662e-04,  8.14607760e-05,  1.16823232e-04, -1.13803058e-03, -6.86652703e-03, -1.01259382e-04],
                  [ 4.40951734e-05, -2.96826452e-03,  4.33851404e-05, -3.97247702e-03, -2.14221686e-03,  2.40287363e-01, -2.57087136e-04, -6.23931448e-05, -1.08806459e-05,  6.31744509e-05,  1.40409394e-04, -3.72663034e-03],
                  [-6.36164823e-01, -1.30976836e-01, -3.43526092e-03,  5.38615203e-02, -5.70951162e-02, -5.77084868e-02,  6.96414716e-01, -3.16333319e-01, -1.52114930e-02,  2.54386914e-01,  9.13458876e-01,  1.71545373e-01],
                  [ 3.10558308e-02,  2.74403116e-03,  6.16245587e-04, -8.32132056e-03,  2.13959116e-03, -1.10817902e-01, -3.65315185e-01, -9.23630791e-01,  2.41710349e-03,  9.52370375e-01, -3.60638566e-01, -9.57266208e-01],
                  [-6.19874722e-03, -1.23086225e-03,  2.45811398e-01, -2.91685570e-03,  5.73531313e-03, -6.61072333e-04,  7.09727546e-03, -1.70528912e-03,  9.69193270e-01, -5.15961019e-03,  4.38259206e-02,  5.00531709e-04],
                  [ 6.07068762e-05,  1.31255225e-05, -2.24894180e-05,  9.41804232e-04, -3.82564624e-04, -9.46639158e-06, -2.43595956e-04, -4.77939019e-04, -8.11900260e-05, -1.59702991e-01,  5.51650815e-02, -6.65205950e-04],
                  [ 6.86645321e-05,  2.24822361e-05, -5.72790751e-06,  5.78808453e-03, -3.79483287e-03,  3.58925623e-05, -1.97483474e-04, -7.66275028e-05, -1.82794774e-04,  5.21713847e-02,  1.72333332e-01,  1.18660611e-03],
                  [ 1.68761387e-03, -4.53419690e-04,  6.05608833e-06,  1.18365474e-04, -4.51452573e-05,  6.72993925e-03, -2.16024254e-03, -5.22996657e-03,  2.18283821e-05, -2.17776795e-03, -4.85272172e-05,  1.15197842e-01]])
    ],
}

def solveEigen(index_and_model, test_case_key):
    index, model = index_and_model
    testCase = cases4solveEigen[test_case_key]
    model.solveStatics(testCase)
    fns, modes = model.solveEigen()
    assert_allclose(fns, desired_fn[test_case_key][index], rtol=1e-05, atol=1e-5)
    # assert_allclose(modes, desired_modes[test_case_key][index], rtol=1e-05, atol=1e-5) # this one is too sensitive to machine precision because there are some very small values

def test_solveEigen_unloaded(index_and_model):
    solveEigen(index_and_model, 'unloaded')

def test_solveEigen_loaded(index_and_model):
    solveEigen(index_and_model, 'loaded')


#===== model.analyzeCases for multiple environmental conditions specified in the yaml file
def test_analyzeCases(index_and_model, plotPSDs=False, flagSaveValues=False):
    '''Solve cases listed in the yaml file
    Set flagSaveValues to true to replace the true values file with the values calculated below
    '''
    index, model = index_and_model    
    true_values_file = list_files[index].replace('.yaml', '_true_analyzeCases.pkl')
    metrics2check = ['wave_PSD', 'surge_PSD', 'sway_PSD', 'heave_PSD', 'roll_PSD', 'pitch_PSD', 'yaw_PSD', 'AxRNA_PSD', 'Mbase_PSD', 'Tmoor_PSD']
    
    model.analyzeCases()

    # Save or read the true values
    if flagSaveValues:
        with open(true_values_file, 'wb') as f:
            pickle.dump(model.results['case_metrics'], f)
        return # If saving, we don't need to check the results
    else:
        with open(true_values_file, 'rb') as f:
            true_values = pickle.load(f)

    # Check computed results against previously computed true values
    nCases = len(model.results['case_metrics'])
    for iCase in range(nCases):
        for ifowt in range(model.nFOWT):
            for imetric, metric in enumerate(metrics2check):
                if metric in model.results['case_metrics'][iCase][ifowt]:
                    assert_allclose(model.results['case_metrics'][iCase][ifowt][metric], true_values[iCase][ifowt][metric], rtol=1e-05, atol=1e-3)
                elif 'array_mooring' in model.results['case_metrics'][iCase] and metric in model.results['case_metrics'][iCase]['array_mooring']:
                    assert_allclose(model.results['case_metrics'][iCase]['array_mooring'][metric], true_values[iCase]['array_mooring'][metric], rtol=1e-05, atol=1e-3)
    
    if plotPSDs:
        import matplotlib.pyplot as plt
        for ifowt in range(model.nFOWT):
            fig, ax = plt.subplots(3, 3, figsize=(15, 10))
            for iCase in range(nCases):
                for imetric, metric in enumerate(metrics2check):
                    if metric in model.results['case_metrics'][iCase][ifowt]:
                        y = model.results['case_metrics'][iCase][ifowt][metric]
                    elif 'array_mooring' in model.results['case_metrics'][iCase] and metric in model.results['case_metrics'][iCase]['array_mooring']:
                        y = model.results['case_metrics'][iCase]['array_mooring'][metric]

                    if metric == 'Tmoor_PSD':
                        if iCase == 0:
                            fig2, ax2 = plt.subplots(y.shape[0], 1, figsize=(15, 10))
                        for i in range(y.shape[0]):
                            ax2[i].plot(model.w, y[i, :])
                            ax2[i].set_ylabel(f'Line channel {i+1}')
                            ax2[i].set_xlabel('Frequency (Hz)')
                        ax2[0].set_title(f'{metric}')
                    else:
                        # assert_allclose(model.results['case_metrics'][iCase][ifowt][metric], true_values[idxTrueValues][ifowt][metric], rtol=1e-05, atol=1e-5)
                        ax[imetric//3, imetric%3].plot(model.w, y, label=f'Case {iCase+1}')
                        ax[imetric//3, imetric%3].set_ylabel(metric)
                        ax[imetric//3, imetric%3].set_xlabel('Frequency (Hz)')
        plt.show()
            
'''
 To run as a script. Useful for debugging.
'''
if __name__ == "__main__":
    index = 0
    
    model = create_model(list_files[index])
    test_solveStatics_Wind((index,model))

    model = create_model(list_files[index])
    test_solveStatics_Wave((index,model))

    model = create_model(list_files[index])
    test_solveStatics_Current((index,model))

    model = create_model(list_files[index])
    test_solveStatics_Wind_Wave_Current((index,model))

    model = create_model(list_files[index])
    test_solveEigen_unloaded((index,model))

    model = create_model(list_files[index])
    test_solveEigen_loaded((index,model))

    model = create_model(list_files[index])
    test_analyzeCases((index,model), plotPSDs=True, flagSaveValues=False)