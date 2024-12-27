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
        np.array([ 1.27859724e+01,  1.04743741e+01, -5.01461649e-01, -3.15384028e-02,  5.81281076e-02, -3.43746581e-02]),        
        np.array([ 1.10592890e+01,  5.11496586e+00, -8.09395988e-01, -2.24265993e-02,  3.99006197e-02, -9.07379936e-02]),        
        np.array([ 1.67865792e+01,  1.13731771e+01,  6.65488155e-01, -2.70207231e-02,  5.60311224e-02, -5.26038064e-02,  1.61811231e+03,  1.07724225e+01,  1.03879124e+00, -2.80596479e-02,  5.80909846e-02, 1.40253302e-02])        
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
        np.array([ 1.49997209e+01,  1.17238481e+01, -5.14214304e-01, -3.09044778e-02,  5.57397680e-02, -2.97221514e-02]),        
        np.array([ 1.52163136e+01,  5.51267781e+00, -8.60650288e-01, -2.27020326e-02,  4.08245936e-02, -9.32725846e-02]),
        np.array([ 2.05122350e+01,  1.24080919e+01,  6.26685779e-01, -2.63488567e-02,  5.41410316e-02, -5.50626331e-02,  1.62214230e+03,  1.22408251e+01,  1.07732379e+00, -2.72746927e-02,  5.66805306e-02,  2.37652367e-02]),
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
        np.array([0.00983736, 0.00711657, 0.06075493, 0.03838041, 0.03917049, 0.01328116]),
        np.array([0.00730845, 0.00938709, 0.0324622 , 0.03384591, 0.03390244, 0.15556943]),
        np.array([0.01065848, 0.0072143 , 0.05086057, 0.03789044, 0.03835428, 0.01772179, 0.00740685, 0.00644188, 0.05081992, 0.03679251, 0.03751652, 0.01330533])
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
        np.array([[-9.81922529e-01, -1.88477064e-01,  5.59814332e-03,  2.63748246e-01, -9.73655930e-01, -3.87082707e-01],
                  [ 1.89231078e-01, -9.82070660e-01,  2.97505444e-03,  9.53805742e-01,  1.85565530e-01, -6.74212470e-01],
                  [ 4.31094006e-03,  3.59103444e-03,  9.99979896e-01, -1.98260433e-02,  2.35679747e-02,  2.85386382e-02],
                  [ 2.07667634e-04, -5.06600871e-04, -5.42200927e-05, -1.37288737e-01, -2.49650348e-02,  1.19947617e-02],
                  [ 9.07418350e-04,  1.10221067e-04,  1.20498177e-04,  3.80792883e-02, -1.27985492e-01,  7.47856564e-03],
                  [-5.43821519e-04, -6.70936303e-04, -2.69410018e-05,  4.09428687e-04,  7.58604920e-04,  6.28164204e-01]]),
        np.array([[-9.64511113e-01, -2.62972949e-01, -2.17836252e-02,  5.32494465e-01,  6.65652114e-01, -5.22431173e-03],
                  [ 2.63742574e-01, -9.64736977e-01, -9.50447095e-03, -8.46299535e-01,  7.45814008e-01,  1.88682345e-03],
                  [ 1.25631854e-02,  1.12200192e-02,  9.99717415e-01,  9.87477280e-04,  2.15061908e-02, -1.80917944e-03],
                  [ 1.44107118e-05, -8.44944185e-05,  1.89490724e-04,  1.20573641e-02, -1.06243716e-02, -1.76905542e-04],
                  [ 5.42775897e-05,  3.32352502e-05, -4.19851679e-04,  7.59840101e-03,  9.49345592e-03, -9.35690288e-05],
                  [ 5.74509767e-04, -1.37725479e-03,  1.26304547e-04,  4.75937331e-03, -1.79147442e-03,  9.99982916e-01]]),
        np.array([[-7.72780208e-01,  1.35468412e-01, -1.77495566e-02,  4.22842581e-01, -7.32826057e-01,  1.07921591e-01,  5.95578887e-01, -1.87722800e-01,  4.79726175e-03, -1.83129519e-03, -1.72205796e-02, -1.44056530e-01],
                  [-2.28519471e-03, -9.77326843e-01, -1.74843237e-04, -8.88837415e-01, -6.53268410e-01, -9.54807874e-01,  2.05369622e-01, -6.41688285e-02,  3.56670017e-05, -3.02582744e-03,  2.46120155e-02,  6.30228760e-02],
                  [ 8.50862335e-03, -1.27740285e-03, -9.69677742e-01, -2.66634251e-02,  5.60030809e-02, -1.43192612e-02, -6.36785160e-03,  1.98135470e-03,  2.43677208e-01, -3.31096809e-03,  1.06565881e-02,  1.48729311e-03],
                  [-2.14147548e-07, -5.13051234e-04,  2.28569369e-05,  1.46348523e-01,  1.03860372e-01, -8.38713717e-03,  1.11521615e-04, -4.39100546e-05, -3.88676030e-06,  3.89544548e-04, -4.17391596e-03, -1.20687744e-04],
                  [ 3.50229130e-05,  4.81508566e-05, -2.96471440e-04,  7.95770662e-02, -1.38834167e-01, -2.81132282e-03, -2.08014641e-04,  7.50030410e-05,  1.16277632e-04,  1.13920287e-03, -7.01170226e-03, -8.83950465e-05],
                  [-8.25205229e-05,  2.97095677e-03,  5.83651248e-05, -4.00418833e-03, -2.30212529e-03, -2.40166282e-01, -2.55349690e-04, -1.02689336e-04, -1.45133749e-05, -7.17160421e-05,  1.72592987e-04, -3.61621868e-03],
                  [ 6.33918522e-01,  1.62082701e-01, -3.39536374e-03,  5.09072525e-02, -5.43724190e-02,  8.05114239e-02,  7.01183592e-01, -2.94123355e-01, -1.51879157e-02, -2.87259623e-01,  9.22402696e-01,  2.36651590e-01],
                  [-2.89806557e-02, -1.40197163e-02,  6.27951672e-04, -7.98679610e-03,  2.09851086e-03,  1.10471065e-01, -3.33697652e-01, -9.34933041e-01,  2.50812940e-03, -9.42830766e-01, -3.36459026e-01, -9.52181619e-01],
                  [ 6.12262049e-03,  1.54700079e-03,  2.43717017e-01, -2.73706238e-03,  5.41124848e-03,  7.14441628e-04,  7.03656560e-03, -1.60801493e-03,  9.69722323e-01,  6.75971380e-03,  4.42384895e-02,  4.03303739e-03],
                  [-7.31124374e-05, -2.43839230e-05, -2.50078305e-05,  9.22017629e-04, -4.24008169e-04,  4.83394158e-06, -2.41322773e-04, -4.82461803e-04, -9.63686678e-05,  1.57899450e-01,  5.19950014e-02, -6.52497993e-04],
                  [-6.60885996e-05, -3.13030222e-05, -4.59073676e-06,  5.58053584e-03, -3.54713531e-03, -2.21919732e-05, -1.89441252e-04, -6.23442736e-05, -1.80039164e-04, -5.95345943e-02,  1.73855081e-01,  1.09011091e-03],
                  [-1.74789101e-03,  2.81715003e-04, -9.04124903e-07,  9.27870178e-05,  1.81765117e-06, -6.72796711e-03, -2.50526577e-03, -5.23512364e-03, -4.85249735e-06,  2.19802909e-03,  1.10782634e-05,  1.12199961e-01]]),
        np.array([[-7.72811610e-01,  1.41393935e-01, -1.77448545e-02,  4.16814597e-01, -7.31112620e-01,  1.16063188e-01,  5.94584233e-01, -1.86581467e-01,  4.79641782e-03, -1.97767472e-03, -1.66489013e-02, -1.36253924e-01],
                  [-2.87714383e-03, -9.75384474e-01, -1.47052786e-04, -8.91800458e-01, -6.55231561e-01, -9.53950275e-01,  2.14539220e-01, -6.35002554e-02,  2.83679231e-05, -3.03172142e-03,  2.43867571e-02,  6.29145314e-02],
                  [ 8.50714289e-03, -1.33371188e-03, -9.69681898e-01, -2.69481972e-02,  5.52001345e-02, -1.48513382e-02, -6.35890001e-03,  1.96906781e-03,  2.43661135e-01, -3.29086307e-03,  1.06839020e-02,  1.39118989e-03],
                  [ 2.76682885e-06, -5.11949394e-04,  1.81895770e-05,  1.46771949e-01,  1.04294678e-01, -8.38004516e-03,  1.13811511e-04, -4.27400619e-05, -2.68577189e-06,  3.87919652e-04, -4.13176783e-03, -1.20055852e-04],
                  [ 3.50209270e-05,  4.45102058e-05, -2.95736719e-04,  7.81880400e-02, -1.38696517e-01, -2.81435054e-03, -2.07566077e-04,  7.41403334e-05,  1.16088871e-04,  1.12666703e-03, -6.95586370e-03, -8.60577252e-05],
                  [-1.01071307e-04,  2.97278099e-03,  6.05949297e-05, -4.08318592e-03, -2.26650443e-03, -2.39856843e-01, -2.67800207e-04, -1.08377245e-04, -1.50836906e-05, -7.31632719e-05,  1.77062979e-04, -3.62290754e-03],
                  [ 6.33853776e-01,  1.68272957e-01, -3.39550283e-03,  4.96307547e-02, -5.41620294e-02,  8.02636391e-02,  7.00179405e-01, -2.93259206e-01, -1.51867240e-02, -2.89140044e-01,  9.24723567e-01,  2.37665124e-01],
                  [-2.95045824e-02, -1.74229529e-02,  6.22939884e-04, -7.63992421e-03,  2.05492440e-03,  1.10408825e-01, -3.31805602e-01, -9.35478473e-01,  2.49028810e-03, -9.42277858e-01, -3.29934057e-01, -9.53089376e-01],
                  [ 6.12040807e-03,  1.61001861e-03,  2.43700854e-01, -2.72246214e-03,  5.40410086e-03,  7.00015249e-04,  7.01985645e-03, -1.60878965e-03,  9.69726432e-01,  6.18386926e-03,  4.40569831e-02,  4.23944407e-03],
                  [-7.16317109e-05, -2.63248450e-05, -2.42200121e-05,  8.78714898e-04, -4.24314517e-04,  4.54694832e-06, -2.38273158e-04, -4.83749124e-04, -9.37329246e-05,  1.57710359e-01,  5.11775274e-02, -6.52243965e-04],
                  [-6.46851669e-05, -3.30541306e-05, -4.57568389e-06,  5.43236832e-03, -3.53689068e-03, -2.23866309e-05, -1.87599602e-04, -5.77467140e-05, -1.79935229e-04, -5.97447838e-02,  1.74408007e-01,  1.09485156e-03],
                  [-1.71938462e-03,  2.49209378e-04, -1.38408991e-06,  8.75424867e-05,  8.14297749e-06, -6.73389531e-03, -2.53410362e-03, -5.23613641e-03, -6.63627886e-06,  2.20871027e-03, -3.33064438e-05,  1.12151147e-01]])
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