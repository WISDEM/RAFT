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
        np.array([ 1.28438307e+01,  1.05848402e+01, -5.01878050e-01, -3.45924657e-02,  6.07763610e-02, -7.33163218e-04]),
        np.array([ 1.10392627e+01,  5.47937781e+00, -8.10737114e-01, -2.35754162e-02,  4.08518032e-02, -4.38926341e-04]),
        np.array([ 1.69630173e+01,  1.09348921e+01,  6.63960710e-01, -3.01117314e-02,  5.83103656e-02, -3.14895787e-02,  1.61829100e+03,  1.16470215e+01,  1.03911368e+00, -3.12200491e-02,  6.07472513e-02,  5.11977260e-02])
    ],
    'wave': [
        np.array([ 1.69696027e-02, -1.94624266e-17, -4.28261180e-01, -1.21827819e-18,  2.27736256e-05, -2.31851928e-23]),
        np.array([-1.64267096e-05, -2.83798953e-15, -6.65861624e-01,  3.89165101e-19, -5.94923163e-11, -4.02571352e-17]),
        np.array([-5.01171107e-01,  1.12265322e-15,  8.82460799e-01,  4.93983911e-17,  4.40816824e-04,  8.73086054e-19,  1.60050117e+03,  9.86160743e-16,  8.82460515e-01,  4.29532988e-17, -4.40845041e-04, -8.35782892e-19])
    ],
    'current': [
        np.array([ 3.31606478e+00,  8.76649127e-01, -4.29804749e-01,  7.13975192e-04, -2.78201279e-03,  4.82726099e-03]),
        np.array([ 4.14990507e+00,  9.81787082e-01, -6.83135141e-01, -2.57954138e-04,  9.53354898e-04, -1.68733226e-03]),
        np.array([ 3.64309829e+00,  1.25671181e+00,  8.37893251e-01,  8.10978896e-04, -1.45559332e-03, -7.00683060e-03,  1.60467331e+03,  1.26718257e+00,  9.27223417e-01,  8.66672657e-04, -2.33801589e-03,  7.94952827e-03])
    ],
    'wind_wave_current': [
        np.array([ 1.52270685e+01,  1.19537187e+01, -5.16017809e-01, -3.38031018e-02,  5.81230157e-02,  4.05845525e-03]),
        np.array([ 1.55463140e+01,  5.84772219e+00, -8.72423735e-01, -2.38450761e-02,  4.17264117e-02, -1.12004164e-02]),
        np.array([ 2.11083924e+01,  1.21463305e+01,  6.19640875e-01, -2.93095661e-02,  5.61539294e-02, -3.56156840e-02,  1.62273747e+03,  1.32852890e+01,  1.08341396e+00, -3.02246269e-02,  5.92351986e-02,  6.61114028e-02])
    ]
}

def solveStatics(index_and_model, test_case_key):
    '''
    We test only the mean offsets and linearized mooring properties.
    '''        
    index, model = index_and_model
    testCase = cases4solveStatics[test_case_key]
    model.solveStatics(testCase)
    assert_allclose(model.Xs2[-1,:], desired_X0[test_case_key][index], rtol=1e-05, atol=1e-10)

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
        np.array([0.00780606, 0.00781762, 0.06073888, 0.03837504, 0.03838327, 0.01239692]),
        np.array([0.00796903, 0.00796903, 0.03245079, 0.03378968, 0.03379512, 0.15351997]),
        np.array([0.01074628, 0.00716311, 0.05084378, 0.03727538, 0.03762673, 0.01574016, 0.00756195, 0.0070458 , 0.05086275, 0.03727631, 0.03758448, 0.01547125])
    ],
    'loaded': [
        np.array([0.00994101, 0.00713163, 0.06075823, 0.03813716, 0.03895660, 0.01329661]),
        np.array([0.00732929, 0.00948936, 0.03246881, 0.03379983, 0.03386064, 0.1557293 ]),
        np.array([0.01066487, 0.00721761, 0.0508606 , 0.03768800, 0.03816006, 0.01776428, 0.0074401 , 0.00644314, 0.05082037, 0.03655718, 0.03730935, 0.01326208])
    ],
}

desired_modes = {
    'unloaded': [
    np.array([[-9.99999800e-01,  7.74110524e-14, -7.81639542e-08,  3.63727383e-13, -9.90233656e-01,  2.35052419e-15],
              [-1.50645716e-20, -9.99999799e-01,  1.82302412e-15, -9.90222240e-01, -8.98381843e-15,  1.13366579e-01],
              [ 1.27037619e-06, -1.20677319e-18, -1.00000000e+00,  5.00446939e-16, -3.59393585e-05,  2.56052002e-20],
              [-1.94980845e-20, -6.29269257e-04,  6.38309923e-18,  1.39498802e-01,  1.27031173e-15,  1.79686479e-04],
              [ 6.32009125e-04, -4.84880484e-17,  1.59265861e-07,  5.12124196e-14, -1.39417739e-01, -8.76141408e-18],
              [ 3.45213984e-20,  8.15760141e-05, -2.57113224e-19, -2.01095382e-05, -7.79767072e-18,  9.93553213e-01]]),
    np.array([[-9.99999999e-01, -1.83367548e-09, -3.78559472e-10,  1.22566326e-13,  9.99898166e-01, -8.43453780e-14],
              [-6.47577388e-10, -9.99999999e-01, -1.22936115e-17, -9.99898166e-01,  2.64544241e-14,  5.06332975e-08],
              [-7.45737942e-11, -2.94703394e-18,  1.00000000e+00,  2.69646127e-18,  2.39945338e-10, -5.78160252e-17],
              [-3.48544775e-14, -5.38228488e-05,  1.48032355e-19,  1.42708908e-02, -3.40316813e-16, -4.55225348e-11],
              [ 5.38222094e-05,  9.86921807e-14, -4.85241400e-12,  1.74855424e-15,  1.42708973e-02, -3.37313650e-18],
              [-4.20325865e-18, -6.58417575e-09,  9.92155815e-17, -1.34058884e-10, -3.17933873e-19, -1.00000000e+00]]),
    np.array([[-7.07063967e-01,  3.36773664e-16, -1.30001646e-02, -9.57063127e-15,  6.94718579e-01,  1.78564067e-15,  7.07066034e-01, -2.24375912e-16,  1.24745074e-02, -9.23450361e-15,  6.92109367e-01, -2.07406580e-15],
              [-1.11304152e-14,  7.07099398e-01,  2.14376628e-14, -6.97904696e-01, -3.53028979e-16,  6.98639163e-01,  5.46838362e-14, -7.07100871e-01, -1.94721767e-14, -6.97798073e-01, -3.40088238e-14, -6.96281283e-01],
              [ 7.78100718e-03, -8.07965718e-19, -7.06987300e-01,  2.71832750e-15, -3.34486928e-02, -8.69102607e-18, -7.58850931e-03, -5.30583706e-19,  7.06996659e-01,  2.55818379e-15, -4.80919318e-02,  1.01557456e-17],
              [-4.65488726e-18,  3.86666516e-04,  6.54118176e-16,  1.13867454e-01,  1.31382750e-16,  2.21898678e-03,  3.15353069e-17, -3.97109439e-04,  6.76947125e-16,  1.14138984e-01,  6.04604655e-15, -2.52207659e-03],
              [ 4.54831150e-05, -4.20741589e-19, -2.03196534e-04, -1.99223992e-15,  1.27454531e-01,  2.18652634e-17, -1.90510889e-04,  4.18917963e-19,  1.48947471e-04, -1.83918911e-15,  1.36656384e-01, -2.22330883e-18],
              [ 6.74183900e-17, -3.22182732e-03,  2.35067762e-18, -2.58280633e-03,  6.70901394e-20,  1.09081308e-01, -2.50798052e-16,  2.84996004e-03, -1.26938492e-17, -2.50831456e-03, -1.66882389e-16, -1.23229372e-01],
              [ 7.07063967e-01,  2.62050142e-16,  1.30001624e-02,  1.94896392e-14,  6.94722204e-01, -9.82869196e-16,  7.07066037e-01, -1.63611558e-16,  1.24745193e-02,  5.84901169e-15, -6.92104953e-01,  1.59785773e-15],
              [ 1.31283609e-16, -7.07099273e-01, -3.62034060e-17, -6.97844079e-01,  8.17129686e-15, -6.98638697e-01, -6.59054121e-15, -7.07100982e-01, -1.18181188e-17,  6.97859276e-01, -9.97024286e-15, -6.96282034e-01],
              [ 7.78101386e-03,  2.64425346e-18, -7.06987176e-01,  1.79098745e-15,  3.34489583e-02, -1.42973841e-17,  7.58850934e-03, -2.84297061e-18, -7.06996784e-01, -6.83149652e-17, -4.80916992e-02,  2.20026592e-17],
              [ 9.78996040e-20, -3.86666562e-04,  3.06041489e-18,  1.13857538e-01, -1.33748508e-15, -2.21898070e-03, -3.24077928e-18, -3.97109605e-04,  3.29523935e-18, -1.14148969e-01,  1.62362535e-15, -2.52207409e-03],
              [-4.54825051e-05,  5.23529562e-20,  2.03196515e-04,  3.71794937e-15,  1.27455288e-01, -1.14670243e-17, -1.90510906e-04, -5.60196060e-20,  1.48947422e-04,  1.10749430e-15, -1.36655573e-01,  2.02614619e-17],
              [-1.30493119e-18, -3.22182689e-03,  7.68059712e-19,  2.58258784e-03, -3.12521411e-17,  1.09081214e-01, -4.50732332e-17, -2.84996061e-03, -1.48271713e-19, -2.50854048e-03,  4.00564314e-17,  1.23229477e-01]])
    ],
    'loaded': [
        np.array([[ 9.82082825e-01, -1.88568163e-01,  5.75804353e-03, -2.04971892e-01, -9.68878689e-01,  4.69913115e-01],
                  [-1.88393927e-01, -9.82052880e-01,  3.07838536e-03, -9.68042265e-01,  2.09246251e-01, -5.77914127e-01],
                  [-4.47178356e-03,  3.71543352e-03,  9.99978675e-01,  2.00907393e-02,  2.33551781e-02, -4.15325431e-03],
                  [-1.85012759e-04, -5.08669713e-04, -6.07721867e-05,  1.40061843e-01, -2.76354216e-02,  1.30271145e-02],
                  [-9.52780836e-04,  8.79655814e-05,  1.21909420e-04, -2.93166139e-02, -1.27202706e-01,  7.10169184e-03],
                  [-3.38082205e-04, -3.43810558e-04,  3.26240522e-06, -5.25145341e-04,  6.19605643e-04,  6.67052874e-01]]),
        np.array([[-9.64244017e-01, -2.64015300e-01, -2.26040605e-02,  6.90552836e-01,  7.36708379e-01, -5.19594500e-03],
                  [ 2.64689202e-01, -9.64444405e-01, -1.28690858e-02, -7.23099834e-01,  6.75635202e-01,  4.28646700e-03],
                  [ 1.31359504e-02,  1.18866853e-02,  9.99661540e-01,  6.26502431e-03,  2.39352223e-02, -3.66794754e-05],
                  [ 1.00847496e-05, -8.87707232e-05,  2.40950087e-04,  1.03023595e-02, -9.62612171e-03, -2.04540655e-04],
                  [ 5.66697506e-05,  2.83592903e-05, -4.36857615e-04,  9.85341450e-03,  1.05073050e-02, -1.12597505e-04],
                  [ 7.38035982e-04, -1.26874312e-03,  1.38649983e-05,  4.57509376e-03, -1.37048817e-03,  9.99977286e-01]]),
        np.array([[-7.70618004e-01, -1.01396294e-01, -1.76863636e-02, -4.78738210e-01, -7.64094045e-01, -5.62589382e-02,  5.96078614e-01, -2.10129831e-01,  4.83221950e-03,  7.62496248e-04, -1.53809047e-02, -2.15382865e-01],
                  [ 3.31128187e-05,  9.87028046e-01, -2.02228062e-04,  8.59179001e-01, -6.15133982e-01,  9.60305617e-01,  1.50371938e-01, -6.62956810e-02,  4.77440099e-05,  2.60789257e-03,  1.93143198e-02,  6.54155619e-02],
                  [ 8.48806011e-03,  9.73348574e-04, -9.68890763e-01,  2.91972366e-02,  5.88924160e-02,  9.61220179e-03, -6.36462048e-03,  2.21942451e-03,  2.46792149e-01,  3.19419320e-03,  1.02002480e-02,  2.36375430e-03],
                  [-4.75499393e-06,  5.19673394e-04,  3.03644562e-05, -1.42117893e-01,  9.85892906e-02,  8.58685165e-03,  8.38172000e-05, -4.57480239e-05, -6.57860279e-06, -3.18635084e-04, -3.30585075e-03, -1.22928651e-04],
                  [ 3.33611957e-05, -4.42649988e-05, -2.90103623e-04, -9.13880509e-02, -1.45570722e-01,  2.91314922e-03, -2.02215530e-04,  8.38341391e-05,  1.14959762e-04, -1.14627610e-03, -6.71054896e-03, -1.07281564e-04],
                  [-1.80974645e-05, -2.93824356e-03,  3.95493912e-05,  3.91954134e-03, -2.23105870e-03,  2.43035930e-01, -2.57196426e-04, -5.11277810e-05, -9.93643467e-06,  6.14074709e-05,  1.28369186e-04, -3.59062926e-03],
                  [ 6.36530316e-01, -1.24430519e-01, -3.45004734e-03, -5.55750752e-02, -5.77546473e-02, -5.61695388e-02,  6.95948907e-01, -3.19397393e-01, -1.51967378e-02,  2.46698722e-01,  9.08537004e-01,  1.53456629e-01],
                  [-2.93813039e-02, -1.20277943e-04,  6.10320103e-04,  8.45026749e-03,  1.99711445e-03, -1.10489186e-01, -3.70996066e-01, -9.21628528e-01,  2.37643876e-03,  9.54230030e-01, -3.72663264e-01, -9.55399776e-01],
                  [ 6.22259232e-03, -1.16177874e-03,  2.46830872e-01,  3.05552343e-03,  5.73487381e-03, -7.10939318e-04,  7.12037744e-03, -1.70755412e-03,  9.68934281e-01, -5.36860588e-03,  4.32454516e-02, -6.93197660e-04],
                  [-5.90033931e-05,  1.07746184e-05, -2.24129016e-05, -9.34974577e-04, -3.47146118e-04, -7.24912356e-06, -2.46810078e-04, -4.80526434e-04, -7.89490650e-05, -1.61018609e-01,  5.68111742e-02, -6.29251996e-04],
                  [-7.15685807e-05,  2.01242662e-05, -5.56493998e-06, -5.81659624e-03, -3.72766594e-03,  3.72592624e-05, -2.02099751e-04, -8.94403667e-05, -1.78176912e-04,  5.10405601e-02,  1.72671036e-01,  1.21418495e-03],
                  [-1.73493268e-03, -5.06027292e-04,  8.59120303e-06, -1.37090081e-04, -6.62746708e-05,  6.60233927e-03, -2.04451365e-03, -5.32969144e-03,  3.12564525e-05, -2.19825898e-03, -3.13987387e-05,  1.13896884e-01]])
    ],
}

def solveEigen(index_and_model, test_case_key):
    index, model = index_and_model
    testCase = cases4solveEigen[test_case_key]
    model.solveStatics(testCase)
    fns, modes = model.solveEigen()
    assert_allclose(fns, desired_fn[test_case_key][index], rtol=1e-05, atol=1e-5) # Checking only frequencies for now
    assert_allclose(modes, desired_modes[test_case_key][index], rtol=1e-05, atol=1e-5) # Checking only first mode for now

def test_solveEigen_unloaded(index_and_model):
    solveEigen(index_and_model, 'unloaded')

def test_solveEigen_loaded(index_and_model):
    solveEigen(index_and_model, 'loaded')


#===== model.analyzeCases for multiple environmental conditions specified in the yaml file
def test_analyzeCases(index_and_model, plotPSDs=False):
    '''Solve cases listed in the yaml file'''
    index, model = index_and_model

    flagSaveValues = False # Set this flag to true to replace the true values file with the calculated values
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
    index = 1
    
    # model = create_model(list_files[index])
    # test_solveStatics_Wind((index,model))

    # model = create_model(list_files[index])
    # test_solveStatics_Wave((index,model))

    # model = create_model(list_files[index])
    # test_solveStatics_Current((index,model))

    # model = create_model(list_files[index])
    # test_solveStatics_Wind_Wave_Current((index,model))

    # model = create_model(list_files[index])
    # test_solveEigen_unloaded((index,model))

    # model = create_model(list_files[index])
    # test_solveEigen_loaded((index,model))

    model = create_model(list_files[index])
    test_analyzeCases((index,model), plotPSDs=True)