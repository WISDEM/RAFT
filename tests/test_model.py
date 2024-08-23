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
        np.array([ 1.23308074e+01,  9.96299273e+00, -5.63051222e-01, -3.47319300e-02,  5.88193515e-02, -3.13139467e-02]),
        np.array([ 1.04801398e+01,  5.02962758e+00, -1.21386808e+00, -2.35757026e-02,  3.99624626e-02, -7.85577354e-02]),
        np.array([ 1.67861341e+01,  1.12637020e+01,  6.65451797e-01, -3.01629231e-02,  5.68383850e-02, -5.12690113e-02,  1.61811048e+03,  1.07595392e+01,  1.03868611e+00, -3.13622101e-02,  5.90181250e-02,  1.65113590e-02])
    ],
    'wave': [
        np.array([ 1.69339101e-02, -2.19691109e-17, -4.91406900e-01, -1.37478722e-18,  2.35053505e-05, -2.63285860e-23]),
        np.array([-1.63533660e-05,  1.89914515e-17, -1.07418524e+00,  4.44254453e-20, -6.51648961e-11,  5.07146892e-19]),
        np.array([-5.01177348e-01,  1.11798952e-15,  8.82461053e-01,  4.91932000e-17,  4.39038724e-04,  8.69456218e-19,  1.60050118e+03,  9.82053320e-16,  8.82460768e-01,  4.27743746e-17, -4.39066827e-04, -8.32305085e-19])
    ],
    'current': [
        np.array([ 2.96495111e+00,  7.77277632e-01, -4.92772350e-01,  6.30645806e-04, -2.48398289e-03,  3.66288997e-03]),
        np.array([ 3.59576771e+00,  8.67535608e-01, -1.08251964e+00, -2.62463715e-04,  9.74451604e-04, -8.31349467e-06]),
        np.array([ 3.24739802e+00,  1.08484956e+00,  8.42959914e-01,  7.16963134e-04, -1.22097638e-03, -5.87434156e-03,  1.60424961e+03,  1.10109258e+00,  9.21764906e-01,  7.58137041e-04, -2.11268701e-03,  6.56575162e-03])
    ],
    'wind_wave_current': [
        np.array([ 1.44986310e+01,  1.11595225e+01, -5.75437881e-01, -3.40997102e-02,  5.64241123e-02, -2.69780906e-02]),
        np.array([ 1.43823691e+01,  5.41621133e+00, -1.25942070e+00, -2.38570281e-02,  4.08817980e-02, -8.28288844e-02]),
        np.array([ 2.05127673e+01,  1.23010332e+01,  6.26628389e-01, -2.94743425e-02,  5.49413694e-02, -5.38145777e-02,  1.62214085e+03,  1.22293955e+01,  1.07721320e+00, -3.05889945e-02,  5.76177298e-02,  2.63915249e-02])
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
        np.array([0.00795671, 0.00796852, 0.06075768, 0.03869459, 0.03870316, 0.01264271]),
        np.array([0.00825187, 0.00825187, 0.03249699, 0.03396774, 0.03397318, 0.15411201]),
        np.array([0.01074625, 0.00716318, 0.05084381, 0.03748606, 0.03783757, 0.01574022, 0.00756192, 0.00704588, 0.05086277, 0.037487, 0.03779494, 0.01547133])
    ],
    'loaded': [
        np.array([0.00992087, 0.00725360, 0.06077308, 0.03845488, 0.03925070, 0.01347912]),
        np.array([0.00758328, 0.00954365, 0.03250385, 0.03397074, 0.03403013, 0.15619635]),
        np.array([0.01065828, 0.00721512, 0.05086059, 0.03788729, 0.03835768, 0.01772042, 0.00740785, 0.00644214, 0.05081994, 0.03679016, 0.03751815, 0.01330817])
    ],
}

desired_modes = {
    'unloaded': [
    np.array([[-9.99999786e-01, -3.69628255e-14, -5.92393033e-08,  3.69390050e-13, -9.90407230e-01,  2.37476448e-15],
              [-1.64342740e-20, -9.99999785e-01,  2.21748401e-15, -9.90395909e-01, -9.28989309e-15,  1.13111315e-01],
              [ 1.32725818e-06, -1.09367424e-18, -1.00000000e+00,  5.03338308e-16, -3.72703567e-05, -3.21985905e-20],
              [-2.01060505e-20, -6.50820983e-04,  7.97375920e-18,  1.38260415e-01,  1.30153801e-15,  1.86026525e-04],
              [ 6.53634932e-04,  2.32555603e-17,  1.67990911e-07,  5.15425870e-14, -1.38179295e-01, -1.93174076e-17],
              [ 3.52581337e-20,  8.14381329e-05, -3.09938113e-19, -2.04893482e-05, -7.64579139e-18,  9.93582304e-01]]),
    np.array([[-9.99999998e-01, -8.77317326e-10, -3.49092389e-10,  9.51748787e-14, -9.99898213e-01,  8.19041919e-14],
              [ 5.48405330e-10,  9.99999998e-01,  7.27159717e-16,  9.99898213e-01,  3.94504994e-14, -2.06090718e-03],
              [-4.39171514e-12, -2.82979776e-18,  1.00000000e+00, -2.33529070e-17, -2.44175484e-10, -1.24556570e-16],
              [ 3.17951848e-14,  5.79774952e-05, -9.45498954e-18, -1.42675408e-02, -5.28101734e-16,  2.94471678e-05],
              [ 5.79767388e-05,  5.08639911e-14, -4.94624527e-12,  1.35756650e-15, -1.42675477e-02,  7.68943465e-17],
              [ 2.98539512e-18,  5.86365307e-09, -1.19687007e-16,  4.97249022e-06, -6.01294694e-18,  9.99997876e-01]]),
    np.array([[-7.07063968e-01, -3.74685395e-16, -1.30340226e-02, -1.60671813e-14,  6.94863002e-01,  9.66992083e-16,  7.07066034e-01, -3.15852571e-16,  1.25007531e-02,  1.57579288e-14,  6.92313443e-01, -1.28257167e-15],
              [-7.38990947e-15, -7.07099398e-01,  1.35767112e-14, -6.97995773e-01, -1.35662629e-14,  6.98639459e-01,  7.67959167e-14, -7.07100871e-01, -2.37803834e-14,  6.97890097e-01, -3.89102557e-14, -6.96282021e-01],
              [ 7.78086989e-03,  5.89152321e-19, -7.06986675e-01,  2.99328737e-15, -3.37914294e-02,  6.39487187e-18, -7.58851866e-03, -1.30365421e-19,  7.06996195e-01, -2.83676434e-15, -4.84772086e-02, -4.64558666e-18],
              [-3.20363668e-18, -3.85533504e-04,  1.07975338e-16,  1.13310010e-01,  2.55366615e-15,  2.20799443e-03,  4.30294847e-17, -3.95913134e-04,  1.11899017e-16, -1.13574765e-01,  6.35008562e-15, -2.50916481e-03],
              [ 4.66972697e-05,  8.57519194e-19, -2.08076011e-04, -3.10988290e-15,  1.26573873e-01,  1.46043163e-17, -1.90337265e-04,  5.74331126e-19,  1.52684219e-04,  2.95172787e-15,  1.35481349e-01, -2.39632395e-17],
              [ 4.70920240e-17,  3.22196475e-03,  5.60384450e-18, -2.53785685e-03, -6.12778054e-17,  1.09079638e-01, -3.39630882e-16,  2.85012022e-03, -8.38258308e-18,  2.46481410e-03, -1.21728304e-16, -1.23225467e-01],
              [ 7.07063969e-01, -2.37331143e-16,  1.30340204e-02,  5.90571577e-15,  6.94866599e-01, -1.35828127e-16,  7.07066037e-01, -1.97101869e-16,  1.25007650e-02,  7.51688495e-15, -6.92309081e-01,  6.10240474e-16],
              [ 8.94135441e-17,  7.07099273e-01, -2.83912649e-17, -6.97934777e-01,  8.19419505e-15, -6.98638992e-01, -9.33378913e-15, -7.07100982e-01, -5.98404462e-18, -6.97951671e-01, -9.99244395e-15, -6.96282772e-01],
              [ 7.78087658e-03, -1.95962096e-18, -7.06986551e-01,  1.07133288e-15,  3.37916947e-02, -7.89237804e-18,  7.58851869e-03, -2.75750860e-18, -7.06996319e-01,  7.91490326e-16, -4.84769765e-02,  4.59637063e-18],
              [ 1.23930455e-19,  3.85533550e-04,  3.22021359e-18,  1.13300083e-01, -1.33744307e-15, -2.20798839e-03, -4.54682753e-18, -3.95913299e-04,  3.40008559e-18,  1.13584760e-01,  1.61838451e-15, -2.50916234e-03],
              [-4.66966628e-05, -6.02168729e-19,  2.08075991e-04,  1.24650597e-15,  1.26574617e-01, -2.68766314e-17, -1.90337282e-04, -5.78288382e-19,  1.52684170e-04,  1.29804171e-15, -1.35480553e-01,  2.42583609e-17],
              [-2.07169478e-18,  3.22196433e-03,  3.47370992e-19,  2.53764084e-03, -3.00544265e-17,  1.09079545e-01, -6.07293025e-17, -2.85012079e-03, -4.60846393e-19,  2.46503742e-03,  3.95189308e-17,  1.23225573e-01]])
    ],
    'loaded': [
        np.array([[-9.81585403e-01, -1.90444692e-01,  5.60636373e-03,  2.61109148e-01, -9.72951674e-01, -2.48954808e-01],
                  [ 1.90972601e-01, -9.81691027e-01,  2.98598602e-03,  9.54609237e-01,  1.89625248e-01, -7.04564808e-01],
                  [ 4.30115532e-03,  3.58660157e-03,  9.99979817e-01, -2.05888728e-02,  2.37610970e-02,  2.73834768e-02],
                  [ 2.14704370e-04, -5.26503533e-04, -5.76405278e-05, -1.36789720e-01, -2.53297824e-02,  1.27545275e-02],
                  [ 9.33009020e-04,  1.13103108e-04,  1.22902817e-04,  3.75196821e-02, -1.27281666e-01,  7.83833714e-03],
                  [-3.81412847e-04, -6.27356093e-04, -2.49187704e-05,  4.03025530e-04,  7.04147266e-04,  6.63804160e-01]]),
        np.array([[-9.66830820e-01, -2.54463348e-01, -2.18013501e-02,  5.54556509e-01,  6.82583894e-01, -5.49640427e-03],
                  [ 2.55113845e-01, -9.67025371e-01, -1.05669634e-02, -8.32009423e-01,  7.30340333e-01,  2.55330854e-03],
                  [ 1.24361406e-02,  1.04107480e-02,  9.99706364e-01,  1.49542751e-03,  2.18261023e-02, -1.57093904e-03],
                  [ 1.49554476e-05, -8.83772450e-05,  2.00173410e-04,  1.18522901e-02, -1.04032876e-02, -1.89858723e-04],
                  [ 5.81284799e-05,  3.18466527e-05, -4.17175071e-04,  7.91161777e-03,  9.73466040e-03, -1.06471968e-04],
                  [ 6.55998085e-04, -1.39394894e-03,  1.12727782e-04,  4.70012664e-03, -1.68969292e-03,  9.99980377e-01]]),
        np.array([[-7.72780208e-01,  1.35468412e-01, -1.77495566e-02,  4.22842581e-01, -7.32826057e-01,  1.07921591e-01,  5.95578887e-01, -1.87722800e-01,  4.79726175e-03, -1.83129519e-03, -1.72205796e-02, -1.44056530e-01],
                  [-2.28519471e-03, -9.77326843e-01, -1.74843237e-04, -8.88837415e-01, -6.53268410e-01, -9.54807874e-01,  2.05369622e-01, -6.41688285e-02,  3.56670017e-05, -3.02582744e-03,  2.46120155e-02,  6.30228760e-02],
                  [ 8.50862335e-03, -1.27740285e-03, -9.69677742e-01, -2.66634251e-02,  5.60030809e-02, -1.43192612e-02, -6.36785160e-03,  1.98135470e-03,  2.43677208e-01, -3.31096809e-03,  1.06565881e-02,  1.48729311e-03],
                  [-2.14147548e-07, -5.13051234e-04,  2.28569369e-05,  1.46348523e-01,  1.03860372e-01, -8.38713717e-03,  1.11521615e-04, -4.39100546e-05, -3.88676030e-06,  3.89544548e-04, -4.17391596e-03, -1.20687744e-04],
                  [ 3.50229130e-05,  4.81508566e-05, -2.96471440e-04,  7.95770662e-02, -1.38834167e-01, -2.81132282e-03, -2.08014641e-04,  7.50030410e-05,  1.16277632e-04,  1.13920287e-03, -7.01170226e-03, -8.83950465e-05],
                  [-8.25205229e-05,  2.97095677e-03,  5.83651248e-05, -4.00418833e-03, -2.30212529e-03, -2.40166282e-01, -2.55349690e-04, -1.02689336e-04, -1.45133749e-05, -7.17160421e-05,  1.72592987e-04, -3.61621868e-03],
                  [ 6.33918522e-01,  1.62082701e-01, -3.39536374e-03,  5.09072525e-02, -5.43724190e-02,  8.05114239e-02,  7.01183592e-01, -2.94123355e-01, -1.51879157e-02, -2.87259623e-01,  9.22402696e-01,  2.36651591e-01],
                  [-2.89806557e-02, -1.40197163e-02,  6.27951672e-04, -7.98679610e-03,  2.09851086e-03,  1.10471065e-01, -3.33697652e-01, -9.34933041e-01,  2.50812940e-03, -9.42830766e-01, -3.36459026e-01, -9.52181619e-01],
                  [ 6.12262049e-03,  1.54700079e-03,  2.43717017e-01, -2.73706238e-03,  5.41124848e-03,  7.14441628e-04,  7.03656560e-03, -1.60801493e-03,  9.69722323e-01,  6.75971380e-03,  4.42384895e-02,  4.03303739e-03],
                  [-7.31124374e-05, -2.43839230e-05, -2.50078305e-05,  9.22017629e-04, -4.24008169e-04,  4.83394158e-06, -2.41322773e-04, -4.82461803e-04, -9.63686678e-05,  1.57899450e-01,  5.19950014e-02, -6.52497993e-04],
                  [-6.60885996e-05, -3.13030222e-05, -4.59073675e-06,  5.58053583e-03, -3.54713531e-03, -2.21919732e-05, -1.89441252e-04, -6.23442736e-05, -1.80039164e-04, -5.95345943e-02,  1.73855081e-01,  1.09011091e-03],
                  [-1.74789101e-03,  2.81715003e-04, -9.04124904e-07,  9.27870178e-05,  1.81765117e-06, -6.72796711e-03, -2.50526577e-03, -5.23512364e-03, -4.85249735e-06,  2.19802909e-03,  1.10782634e-05,  1.12199961e-01]])
    ],
}

def solveEigen(index_and_model, test_case_key):
    index, model = index_and_model
    testCase = cases4solveEigen[test_case_key]
    model.solveStatics(testCase)
    fns, modes = model.solveEigen()
    assert_allclose(fns, desired_fn[test_case_key][index], rtol=1e-05, atol=1e-5)
    # assert_allclose(modes, desired_modes[test_case_key][index], rtol=1e-05, atol=1e-5) # Include this again after moorpy dev stuff is merged

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

    # Check compute results against previously computed true values
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
    index = 2
    
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