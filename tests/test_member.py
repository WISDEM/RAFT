# Test RAFT member functionality
# 
import pytest
import numpy as np
from numpy.testing import assert_allclose
import yaml
import raft
from raft.raft_member import Member
from raft.helpers import getFromDict
import os


'''
 Define files for testing
'''
# Name of the subfolder where the test data is located
test_dir = 'test_data'

# List of input file names to be tested
list_files = [
    'mem_srf_vert_circ_cyl.yaml',
    'mem_srf_vert_rect_cyl.yaml',
    'mem_srf_inc_rect_cyl.yaml',
]

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# To avoid problems with different platforms, get the full path of the file
list_files = [os.path.join(current_dir, test_dir, file) for file in list_files]


'''
 Desired values to compare with the results.
 Should be lists of the same length as list_files.
 List elements are indicated below.
'''

# Basic inertia properties
# Format is: (mshell, mfill, COGx, COGy, COGz)
desired_inertiaBasic = [
    (4.22944765e+05, 2.310786205e+06,  0       , 0       , -14.67635),
    (4.32731250e+05, 7.794637500e+05,  0       , 0       , -12.88153),
    (4.32731250e+05, 7.794637500e+05, -8.050958, 2.788911, -9.661150),
]




# 6x6 inertia matrix wrt to (0,0,0)
# Total value (sum of shell and ballast)
desired_inertiaMatrix = [
    np.array([[  2.73373097e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.01212027e+07,  0.00000000e+00],
              [  0.00000000e+00,  2.73373097e+06,  0.00000000e+00,  4.01212027e+07,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  2.73373097e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  4.01212027e+07,  0.00000000e+00,  7.32378508e+08,  0.00000000e+00,  0.00000000e+00],
              [ -4.01212027e+07,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  7.32378508e+08,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.76138073e+07]]),
    np.array([[  1.21219500e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.56149297e+07,  0.00000000e+00],
              [  0.00000000e+00,  1.21219500e+06,  0.00000000e+00,  1.56149297e+07,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  1.21219500e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  1.56149297e+07,  0.00000000e+00,  3.23452553e+08,  0.00000000e+00,  0.00000000e+00],
              [ -1.56149297e+07,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.30580956e+08,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.73816357e+07]]),
    np.array([[  1.21219483e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.17111957e+07, -3.38070373e+06],
              [  0.00000000e+00,  1.21219483e+06,  0.00000000e+00,  1.17111957e+07,  0.00000000e+00, -9.75932974e+06],
              [  0.00000000e+00,  0.00000000e+00,  1.21219483e+06,  3.38070373e+06,  9.75932974e+06,  0.00000000e+00],
              [  0.00000000e+00,  1.17111957e+07,  3.38070373e+06,  2.04657143e+08,  4.36209409e+07, -1.43470721e+08],
              [ -1.17111957e+07,  0.00000000e+00,  9.75932974e+06,  4.36209409e+07,  3.15470081e+08,  4.96994023e+07],
              [ -3.38070373e+06, -9.75932974e+06,  0.00000000e+00, -1.43470721e+08,  4.96994023e+07,  1.51287459e+08]])
]


# Some hydrostatic quantities
# Fvec[2], Fvec[3], Fvec[4], Cmat[2,2], Cmat[3,3], Cmat[4,4], r_center[0], r_center[1], r_center[2], xWP, yWP
# Other hydrostatic values, such as AWP and IWP, are directly included in the others
desired_hydrostatics = [
    (1.53244611e+07, 0.00000000e+00, 0.00000000e+00, 7.6622305e+05, -1.4859830614e+08, -1.4859830614e+08,  0.00000000, 0.00000000, -10.000000, 0, 0),
    (1.18853055e+07, 0.00000000e+00, 0.00000000e+00, 5.9426527e+05, -1.1707025918e+08, -1.1404829645e+08,  0.00000000, 0.00000000, -10.000000, 0, 0),
    [1.18853038e+07, 5.14645542e+07, 1.48566298e+08, 7.9235359e+05, -8.7729607928e+07, -8.4742368515e+07, -6.25000000, 2.16504285, -7.5000000, -1.776357e-15, -1.428571e-05]
]



# Hydrodynamic added mass matrix
# Because those are analytical values, the test cases need to have a very fine discretization (dlsMax) otherwise the moments won't match
desired_Ahydro = [
    np.array([[  1.32780754e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.32780754e+07,  0.00000000e+00],
              [  0.00000000e+00,  1.32780754e+06,  0.00000000e+00,  1.32780754e+07,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  1.79514370e+05,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  1.32780754e+07,  0.00000000e+00,  1.77041005e+08,  0.00000000e+00,  0.00000000e+00],
              [ -1.32780754e+07,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.77041005e+08,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]),
    np.array([[  1.81732500e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -1.81732500e+07,  0.00000000e+00],
              [  0.00000000e+00,  1.51443750e+06,  0.00000000e+00,  1.51443750e+07,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  9.34949911e+04,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  1.51443750e+07,  0.00000000e+00,  2.01925000e+08,  0.00000000e+00,  0.00000000e+00],
              [ -1.81732500e+07,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.42310000e+08,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]),
    np.array([[  1.11082965e+06,  1.39485495e+05, -8.07513003e+05, -7.02595984e+05, -1.33865034e+07, -3.27881716e+06],
              [  1.39485495e+05,  1.46517172e+06,  2.79729887e+05,  1.16016299e+07,  7.02607520e+05, -9.46520150e+06],
              [ -8.07513003e+05,  2.79729887e+05,  8.47173319e+05,  3.93458753e+06,  1.13582442e+07, -1.15359000e+01],
              [ -7.02595984e+05,  1.16016299e+07,  3.93458753e+06,  1.27374004e+08,  3.98141183e+07, -9.46517481e+07],
              [ -1.33865034e+07,  7.02607520e+05,  1.13582442e+07,  3.98141183e+07,  2.28516352e+08,  3.27880362e+07],
              [ -3.27881716e+06, -9.46520150e+06, -1.15359000e+01, -9.46517481e+07,  3.27880362e+07,  8.83414728e+07]])
]

# Hydrodynamic inertial excitation matrices
desired_Ihydro = [
    np.array([[  2.88993405e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.88993405e+07,  0.00000000e+00],
              [  0.00000000e+00,  2.88993405e+06,  0.00000000e+00,  2.88993405e+07,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  1.79514370e+05,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  2.88993405e+07,  0.00000000e+00,  3.85324540e+08,  0.00000000e+00,  0.00000000e+00],
              [ -2.88993405e+07,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.85324540e+08,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]),
    np.array([[  3.02887500e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -3.02887500e+07,  0.00000000e+00],
              [  0.00000000e+00,  2.72598750e+06,  0.00000000e+00,  2.72598750e+07,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  9.34949911e+04,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  2.72598750e+07,  0.00000000e+00,  3.63465000e+08,  0.00000000e+00,  0.00000000e+00],
              [ -3.02887500e+07,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.03850000e+08,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]),
    np.array([[  1.84865626e+06,  3.03325240e+05, -1.37507220e+06, -7.02587876e+05, -2.24730987e+07, -5.90187304e+06],
              [  3.03325240e+05,  2.61920875e+06,  4.76337581e+05,  2.06882224e+07,  7.02607520e+05, -1.70373620e+07],
              [ -1.37507220e+06,  4.76337581e+05,  1.37689490e+06,  6.55764639e+06,  1.89304070e+07, -1.96438885e+01],
              [ -7.02587876e+05,  2.06882224e+07,  6.55764639e+06,  2.25811692e+08,  6.16728285e+07, -1.70373138e+08],
              [ -2.24730987e+07,  7.02607520e+05,  1.89304070e+07,  6.16728285e+07,  3.82483179e+08,  5.90184883e+07],
              [ -5.90187304e+06, -1.70373620e+07, -1.96438885e+01, -1.70373138e+08,  5.90184883e+07,  1.59014651e+08]])
]


'''
 Aux functions
'''
# Function used to create member
# Not explicitly inside the fixture below so that we can also run this file as a script 
def create_member(file):
    with open(file) as f:
        design = yaml.load(f, Loader=yaml.FullLoader)

    if len(design['members']) != 1:
        raise ValueError(f'{file} should have only one member.')

    memData = design['members'][0]
    headings = getFromDict(memData, 'heading', shape=-1, default=0.)
    memData['headings'] = headings   # differentiating the list of headings/copies of a member from the individual member's heading
    
    # create member object
    if np.isscalar(headings):
        member = Member(memData, 0, heading=headings)
    else:
        raise ValueError(f'Cannot have more than one heading. Need a single member.')
    
    member.setPosition() # Set the initial position
    return member

# Define a fixture to loop member instances with the index to loop the desired values as well
# Could also zip the lists with the desired values, but I think the approach below is simpler
@pytest.fixture(params=enumerate(list_files))
def index_and_member(request):
    index, file = request.param
    member = create_member(file)
    return index, member


'''
 Test functions
'''
def test_inertia(index_and_member):
    index, member = index_and_member
    mass, cg, mshell, mfill, pfill = member.getInertia()
    assert_allclose([mshell, mfill[0], cg[0], cg[1], cg[2]], desired_inertiaBasic[index], rtol=1e-05, atol=0, verbose=True)
    assert_allclose(member.M_struc, desired_inertiaMatrix[index], rtol=1e-05, atol=0, verbose=True)

def test_hydrostatics(index_and_member):
    index, member = index_and_member
    Fvec, Cmat, _, r_center, _, _, xWP, yWP = member.getHydrostatics(rho=1025, g=9.81)
    assert_allclose([Fvec[2], Fvec[3], Fvec[4], Cmat[2,2], Cmat[3,3], Cmat[4,4], r_center[0], r_center[1], r_center[2], xWP, yWP], desired_hydrostatics[index], rtol=1e-05, atol=0, verbose=True)

def test_hydroConstants(index_and_member):
    index, member = index_and_member
    A_hydro, I_hydro = member.calcHydroConstants(sum_inertia=True, rho=1025, g=9.81)
    assert_allclose(A_hydro, desired_Ahydro[index], rtol=1e-05, atol=0, verbose=True)
    assert_allclose(I_hydro, desired_Ihydro[index], rtol=1e-05, atol=0, verbose=True)


'''
 To run as a script. Useful for debugging.
'''
if __name__ == "__main__":
    index = 2
    member = create_member(list_files[index])

    # member.setPosition(r6=[0,0,0, np.deg2rad(30), np.deg2rad(30), np.deg2rad(30)])
    # print(member.rA)
    # print(member.rB)


    test_inertia((index, member))
    test_hydrostatics((index, member))
    test_hydroConstants((index, member))




    