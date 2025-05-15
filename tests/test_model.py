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
        np.array([ 1.09704957e+01,  5.30341749e+00, -8.08255233e-01, -2.20791574e-02,  4.00562243e-02, -1.88931043e-02]),
        np.array([ 1.31233298e+01,  1.07387644e+01, -5.24494379e-01, -1.81027406e-02,  3.71705447e-02, -6.61884531e-03]),
        np.array([ 1.31179344e+01,  1.07370555e+01, -5.38800860e-01, -1.83773802e-02,  3.77155419e-02, -6.62372479e-03]),
        np.array([ 1.69682164e+01,  1.13133692e+01,  6.43325363e-01, -1.49816934e-02,  3.67094643e-02, -3.46615026e-02,  1.61831746e+03,  1.16883178e+01,  1.01673549e+00, -1.50552064e-02,  3.79424789e-02,  4.38179867e-02])
    ],
    'wave': [
        np.array([-1.64267049e-05, -2.83795893e-15, -6.65861624e-01,  3.88717546e-19, -5.94238978e-11, -4.02571352e-17]),
        np.array([ 4.27925162e-01, -9.00035158e-17, -4.51814991e-01, -5.63389767e-18, -2.54250076e-02, -1.07219357e-22]),
        np.array([ 4.34028448e-01,  1.29311805e-15, -4.66112782e-01,  8.09445578e-17, -2.58031212e-02,  1.54046523e-21]),
        np.array([-3.28437405e-01,  1.37380291e-15,  8.59345726e-01,  6.09528763e-17, -2.31870486e-02,  9.89478513e-19,  1.60065726e+03,  9.12847486e-16,  8.59907935e-01,  3.91868383e-17, -2.40815624e-02, -8.63499424e-19])
    ],
    'current': [
        np.array([ 3.86072176e+00,  9.22694246e-01, -6.74898762e-01, -2.64759824e-04,  9.82529767e-04, -1.03532699e-05]),
        np.array([ 3.46491856e+00,  8.10382757e-01, -4.53718903e-01,  6.48535991e-04, -2.79078335e-02,  3.71621922e-03]),
        np.array([ 3.47177656e+00,  8.10749061e-01, -4.68029699e-01,  6.58432223e-04, -2.83226533e-02,  3.71570242e-03]),
        np.array([ 3.43402590e+00,  1.08780658e+00,  8.19500640e-01,  7.10970656e-04, -2.47671283e-02, -5.93700672e-03,  1.60441156e+03,  1.09887425e+00,  8.99148002e-01,  7.76337021e-04, -2.58189091e-02,  6.49661703e-03])
    ],
    'wind_wave_current': [
        np.array([ 1.51819382e+01,  5.67973367e+00, -8.56976995e-01, -2.23553755e-02,  4.09788544e-02, -2.60679623e-02]),
        np.array([ 1.53213197e+01,  1.19846300e+01, -5.37552077e-01, -1.74035124e-02,  3.48566650e-02, -2.72880384e-03]),
        np.array([ 1.53171271e+01,  1.19835302e+01, -5.51858993e-01, -1.76676200e-02,  3.53655443e-02, -2.73417235e-03]),
        np.array([ 2.07005182e+01,  1.23527653e+01,  6.04380340e-01, -1.43462329e-02,  3.48812794e-02, -3.78041365e-02,  1.62236585e+03,  1.32021152e+01,  1.05518596e+00, -1.41134650e-02,  3.65034177e-02,  5.57420290e-02])
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
        np.array([0.00796903, 0.00796903, 0.03245079, 0.03383781, 0.03384323, 0.15347415]),
        np.array([0.00782180, 0.00779927, 0.06073036, 0.03829455, 0.03823218, 0.01238992]),
        np.array([0.00782029, 0.00779718, 0.06072388, 0.03804270, 0.03797990, 0.01238741]),
        np.array([0.01074526, 0.00704213, 0.05083874, 0.03718830, 0.03746220, 0.01573330, 0.00756069, 0.00716294, 0.05085846, 0.03718910, 0.03751292, 0.01545850]),
    ],
    'loaded': [
        np.array([0.00730352, 0.00939103, 0.03246224, 0.03384297, 0.03390545, 0.1555763 ]),
        np.array([0.00987260, 0.00712184, 0.06074470, 0.03805314, 0.03879976, 0.01324559]),
        np.array([0.00986996, 0.00712007, 0.06073821, 0.03779950, 0.03855126, 0.01324305]),
        np.array([0.01065943, 0.00720493, 0.05085334, 0.03762357, 0.03800233, 0.01769749, 0.00744181, 0.00644391, 0.05081630, 0.03648770, 0.03722345, 0.01325040]),
    ],
}

desired_modes = {
    'unloaded': [
        np.array([[ 9.99999999e-01, -5.75186507e-10, -3.66356915e-10, -8.67852276e-14,  9.99898193e-01, -2.99686750e-14],
                  [ 3.92738961e-10, -9.99999999e-01, -3.31616203e-17,  9.99898194e-01,  3.23260804e-14,  2.06096172e-03],
                  [ 7.45752595e-11, -2.85321785e-18,  1.00000000e+00, -2.72406265e-18,  2.31375465e-10, -1.10751480e-16],
                  [ 2.12335674e-14, -5.40653002e-05,  4.24138273e-19, -1.42689340e-02, -4.31499026e-16, -2.94481141e-05],
                  [-5.40646648e-05,  3.10972493e-14, -4.68816970e-12, -1.23789014e-15,  1.42689404e-02, -1.94677827e-16],
                  [ 1.95726867e-18, -5.58669353e-09,  9.44626575e-17,  4.97628953e-06, -2.81458719e-18, -9.99997876e-01]]),
        np.array([[-9.99999807e-01,  4.86100318e-14, -1.63567304e-03, -5.45667034e-14, -9.89726132e-01,  2.04003988e-15],
                  [-3.37568579e-20, -9.99999728e-01, -6.29828842e-16, -9.90218873e-01,  1.25468775e-17,  5.86275118e-01],
                  [ 8.23145082e-05, -5.26031153e-18,  9.99998628e-01, -9.74821863e-16, -2.64733741e-02,  3.12143871e-18],
                  [ 6.39974802e-21, -6.48163950e-04,  1.35297744e-16,  1.39458388e-01, -2.50093227e-18,  1.89793501e-03],
                  [ 6.15814470e-04, -3.03812746e-17, -2.63383311e-04, -7.74519594e-15, -1.40503895e-01,  1.86798683e-17],
                  [ 3.49993342e-20,  3.51214780e-04, -2.70592359e-18, -4.23584082e-03, -7.96472948e-18,  8.10109797e-01]]),
        np.array([[-9.99999804e-01, -2.02386208e-14, -1.58908175e-03,  7.04718771e-14,  9.89362815e-01,  9.80392867e-17],
                  [ 8.77787527e-18,  9.99999723e-01, -1.18099471e-15,  9.89868473e-01,  2.01310527e-15,  5.88357005e-01],
                  [ 8.36235365e-05,  5.46622703e-18,  9.99998703e-01,  1.48235884e-15,  2.64929776e-02,  4.87039683e-18],
                  [-1.74576623e-19,  6.54453986e-04,  1.45778172e-16, -1.41921601e-01, -3.91525643e-17,  1.90878244e-03],
                  [ 6.21062713e-04,  1.08784745e-17, -2.61844545e-04,  1.01884350e-14,  1.43036159e-01, -2.88892380e-17],
                  [-1.01366640e-20, -3.53400694e-04, -4.85827394e-18,  4.32042891e-03, -2.36973537e-18,  8.08599030e-01]]),
        np.array([[ 7.07825719e-01, -3.15055001e-17, -1.32630905e-02, -2.19914192e-14, -8.64609540e-01, -4.24293074e-16,  7.06326558e-01, -7.31382701e-18,  9.61558420e-03,  1.32162934e-14, -4.93902244e-01, -4.29224066e-16],
                  [-4.89583244e-17,  7.85911667e-01,  4.52106432e-15, -8.56521395e-01,  5.28474752e-15, -7.93284272e-01,  3.76977173e-14, -6.14665922e-01, -5.22230082e-15,  5.25433719e-01,  1.64307174e-14, -6.54712295e-01],
                  [-7.76765072e-03,  5.73003997e-18, -7.96969430e-01,  2.78055517e-15,  1.88343250e-02, -1.51849968e-17, -7.58106566e-03, -4.04692741e-18,  6.03839362e-01, -1.60196034e-15,  5.13083496e-04, -1.30251428e-17],
                  [-3.34340760e-19,  4.27427506e-04, -3.85068942e-16,  1.40135770e-01, -7.45324253e-16, -2.90906806e-03,  2.04954676e-17, -3.25330035e-04, -5.30992268e-16, -8.63333133e-02, -2.79405527e-15, -2.68890829e-03],
                  [-5.37598733e-05, -1.30564706e-18,  2.92009853e-05, -4.24958991e-15, -1.69811876e-01,  1.64600958e-18, -1.75528113e-04,  9.68430673e-19, -6.94904252e-05,  2.46004337e-15, -8.77291683e-02,  2.51343153e-18],
                  [ 2.49224761e-19, -3.28472446e-03,  9.44014612e-18, -8.17009389e-03,  3.98091977e-17, -1.18508842e-01, -1.74458248e-16,  2.90949567e-03,  8.33507402e-18,  4.93133876e-03,  1.68009630e-16, -1.10277637e-01],
                  [-7.06301389e-01,  8.24649949e-17,  1.21359094e-02, -2.17933632e-16,  4.59748625e-01, -8.10522725e-17,  7.07804746e-01, -9.84864194e-17,  1.53870931e-02,  1.16752336e-14, -8.46545575e-01,  2.46067501e-16],
                  [-8.42294853e-18,  6.18325170e-01, -1.55808794e-17, -4.90097261e-01,  7.77558316e-15,  5.89473068e-01, -4.63212600e-15,  7.88774350e-01, -1.36301789e-17, -8.35220796e-01, -1.11866279e-14, -7.35135054e-01],
                  [-7.79477632e-03,  1.73669598e-18, -6.03751937e-01,  6.28609948e-16,  5.62953547e-02,  5.06431194e-18,  7.59532320e-03, -1.23691005e-19, -7.96899379e-01,  1.58470155e-15, -7.93424870e-02, -6.41876055e-18],
                  [-5.87906282e-20,  3.60238282e-04,  2.66938307e-18,  8.04358236e-02, -1.27892213e-15,  1.55576021e-03, -2.34040824e-18,  4.46937680e-04,  3.59783942e-18,  1.37275491e-01,  1.84260199e-15, -2.28372214e-03],
                  [ 3.48598674e-05,  1.02289764e-18,  3.66943933e-04,  1.51868457e-16,  9.33893614e-02, -5.08719895e-18, -2.05513389e-04, -6.32993020e-19,  4.20980484e-04,  2.10891028e-15, -1.59465553e-01, -3.61168296e-18],
                  [ 4.28338626e-19,  2.39454188e-03, -1.72295654e-19, -1.03376490e-03,  1.55565829e-17, -9.57409855e-02, -2.91825502e-17,  3.47247537e-03, -3.73375983e-19, -1.81895138e-03, -2.27130818e-17,  1.36947655e-01]]),
    ],
    'loaded': [
        np.array([[-9.64378242e-01, -2.63590752e-01, -2.10946483e-02,  6.67618956e-01,  7.11326547e-01, -3.97038343e-03],
                  [ 2.64226439e-01, -9.64567735e-01, -1.08535955e-02, -7.44333038e-01,  7.02378251e-01,  2.46425477e-03],
                  [ 1.25912062e-02,  1.12874097e-02,  9.99718461e-01,  5.36744834e-03,  2.17700278e-02, -4.08761745e-04],
                  [ 1.12508015e-05, -8.58285900e-05,  2.09146973e-04,  1.06044546e-02, -1.00067204e-02, -1.76052849e-04],
                  [ 5.54046589e-05,  2.84236886e-05, -4.10342190e-04,  9.52540836e-03,  1.01447352e-02, -8.89069168e-05],
                  [ 6.73636151e-04, -1.25989426e-03,  3.44483770e-05,  4.61085742e-03, -1.53377328e-03,  9.99988979e-01]]),
        np.array([[ 9.82150856e-01, -1.87921630e-01,  4.07842531e-03, -2.55747246e-01, -9.61277765e-01, -1.95088862e-01],
                  [-1.88040577e-01, -9.82177055e-01,  2.19364068e-03, -9.56013224e-01,  2.41727181e-01, -3.54696639e-01],
                  [-4.42335800e-03,  3.65499013e-03,  9.99989268e-01,  1.52938349e-03,  2.70364948e-03,  5.97092474e-03],
                  [-1.77150592e-04, -5.11302213e-04,  7.18018030e-05,  1.38821901e-01, -3.27231465e-02,  1.96677683e-02],
                  [-8.84556439e-04,  8.96831659e-05, -1.18293930e-04, -3.66668232e-02, -1.28150299e-01,  1.14171188e-02],
                  [ 2.38840973e-04, -2.53960948e-04, -3.07164190e-06, -3.70754867e-03,  3.52357057e-03,  9.14099451e-01]]),
        np.array([[ 9.82139356e-01, -1.87978504e-01,  4.09742411e-03, -2.55335162e-01, -9.61113534e-01, -1.86115374e-01],
                  [-1.88100651e-01, -9.82166170e-01,  2.20495571e-03, -9.55719194e-01,  2.41182127e-01, -3.60466674e-01],
                  [-4.42108632e-03,  3.65456219e-03,  9.99989165e-01,  1.69642242e-03,  2.90337493e-03,  5.93561528e-03],
                  [-1.78839975e-04, -5.15825845e-04,  7.15278350e-05,  1.41411099e-01, -3.32066770e-02,  2.00040562e-02],
                  [-8.92997676e-04,  9.06083742e-05, -1.17673782e-04, -3.72889317e-02, -1.30261073e-01,  1.15899907e-02],
                  [ 2.33228959e-04, -2.56262900e-04, -3.07295593e-06, -3.78629402e-03,  3.59110001e-03,  9.13704067e-01]]),
        np.array([[ 7.70914018e-01, -1.17395863e-01, -1.61715593e-02,  4.59701692e-01,  7.57267114e-01,  1.15686564e-01,  5.92241261e-01, -2.11326383e-01, -4.84261186e-03,  2.91306009e-03, -6.28586978e-03,  1.77870103e-01],
                  [ 2.76336921e-03,  9.83111079e-01,  7.71148014e-04, -8.70214552e-01,  6.25819046e-01, -9.61018495e-01,  1.74059753e-01, -6.40952347e-02,  2.21741739e-04,  3.07456670e-03,  1.53015857e-02, -6.52865349e-02],
                  [-8.46849828e-03,  1.12234659e-03, -9.62918857e-01, -2.91133365e-02, -1.60565961e-02, -1.07370021e-02, -6.33313466e-03,  2.23523718e-03, -2.69267884e-01,  3.07129058e-03,  9.13867744e-03, -1.89938445e-03],
                  [-6.92466645e-06,  5.06740412e-04, -1.38450410e-04,  1.43646953e-01, -1.01646527e-01, -8.92085192e-03,  8.34784720e-05, -3.97573892e-05, -4.00085053e-05, -2.94989477e-04, -2.66978184e-03,  1.33226866e-04],
                  [-3.80000792e-05, -3.18718207e-05, -1.63568996e-05,  8.70645571e-02,  1.46136988e-01, -3.32403955e-03, -1.88439589e-04,  7.75417900e-05, -4.50382342e-05, -1.13345175e-03, -6.23085504e-03,  9.89278222e-05],
                  [ 1.32346905e-04, -3.03127923e-03,  5.28930412e-05, -1.05060309e-02,  2.84137592e-03, -2.24436041e-01, -2.93715708e-04, -7.33395067e-05,  1.54107755e-05,  1.22940365e-04,  3.12930602e-04,  3.93381356e-03],
                  [-6.35823018e-01, -1.39839802e-01, -4.12875454e-03,  4.64975038e-02,  5.38456530e-02,  4.25073519e-02,  6.90307833e-01, -3.26425122e-01,  1.63428644e-02,  2.68287842e-01,  9.07672647e-01, -1.31558228e-01],
                  [ 3.60619974e-02,  1.19588699e-02,  4.65390067e-04, -6.12538918e-03, -2.13691125e-03,  1.03129205e-01, -3.77271639e-01, -9.19045505e-01, -1.64956437e-03,  9.47996995e-01, -3.70527401e-01,  9.65489660e-01],
                  [-6.21316965e-03, -1.32552092e-03,  2.69272852e-01, -2.30252304e-03, -5.03775015e-03,  5.34681309e-04,  7.05692246e-03, -1.78745500e-03, -9.62912928e-01,  2.06635847e-02,  6.83944327e-02,  1.94739912e-04],
                  [ 5.73877577e-05,  1.73869447e-05,  1.55155496e-05,  7.34971902e-04,  3.91788360e-04, -3.59118324e-05, -2.45527237e-04, -4.91453005e-04, -6.20538865e-05, -1.60713894e-01,  5.82005468e-02,  3.92650678e-04],
                  [ 5.35217661e-05,  2.69720933e-05, -7.92408612e-05,  4.79170446e-03,  3.69617884e-03, -5.29046880e-05, -2.03450513e-04, -6.51449327e-05,  4.25820931e-04,  5.51040523e-02,  1.74216186e-01, -1.49517626e-03],
                  [ 1.48946042e-03, -4.11837504e-04,  8.78835414e-06, -1.37529855e-05, -3.50704978e-05, -6.31055822e-03, -2.13519130e-03, -5.14879469e-03, -3.15665253e-05,  1.47542156e-03, -4.49606559e-03, -1.20832552e-01]])
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