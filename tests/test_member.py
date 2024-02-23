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
# Running the same file twice just to check if the loop is working. Going to include other tests soon
list_files = [
    'mem_srf_vert_circ_cyl.yaml',
    'mem_srf_vert_rect_cyl.yaml',
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
    (422944.765, 2310786.205, 0, 0, -14.67635),
    (432731.250,  779463.750, 0, 0, -12.88153),
]

# 6x6 inertia matrix wrt to (0,0,0)
# Total value (sum of shell and ballast)
desired_inertiaMatrix = [
    np.array([[  2733730.97, 0, 0, 0, -40121202.70, 0.],
              [  0, 2733730.97, 0, 40121202.70, 0, 0.],
              [  0, 0, 2733730.97, 0, 0, 0.],
              [  0, 40121202.70, 0, 732378507.99, 0, 0.],
              [  -40121202.70, 0, 0, 0, 732378507.99, 0.],
              [  0, 0, 0, 0, 0, 37613807.34]]),
    np.array([[  1212195.0, 0, 0, 0, -15614929.74, 0.],
              [  0, 1212195.0, 0, 15614929.74, 0, 0.],
              [  0, 0, 1212195.0, 0, 0, 0.],
              [  0, 15614929.74, 0, 323452553.47, 0, 0.],
              [  -15614929.74, 0, 0, 0, 330580955.76, 0.],
              [  0, 0, 0, 0, 0, 17381635.74]])
]

# Some hydrostatic quantities
# Fvec[2], Fvec[3], Fvec[4], Cmat[2,2], Cmat[3,3], Cmat[4,4], r_center[0], r_center[1], r_center[2], xWP, yWP
# Other hydrostatic values, such as AWP and IWP, are directly included in the others
desired_hydrostatics = [
    (15324461.09, 0, 0, 766223.05, -148598306.14, -148598306.14, 0, 0, -10, 0, 0),
    (11885305.50, 0, 0, 594265.27, -117070259.18, -114048296.45, 0, 0, -10, 0, 0),
]

# Hydrodynamic added mass matrix
# Because those are analytical values, the test cases need to have a very fine discretization (dlsMax) otherwise the moments won't match
desired_Ahydro = [
    np.array([[  1327807.54, 0, 0, 0, -13278075.36, 0.],
              [  0, 1327807.54, 0, 13278075.36, 0, 0.],
              [  0, 0, 179514.37, 0, 0, 0.],
              [  0, 13278075.36, 0, 177041004.79, 0, 0.],
              [  -13278075.36, 0, 0, 0, 177041004.79, 0.],
              [  0, 0, 0, 0, 0, 0]]),
    np.array([[  1817325.00, 0, 0, 0, -18173250.00, 0.],
              [  0, 1514437.50, 0, 15144375.00, 0, 0.],
              [  0, 0, 93494.99112, 0, 0, 0.],
              [  0, 15144375.00, 0, 201925000.00, 0, 0.],
              [  -18173250.00, 0, 0, 0, 242310000.00, 0.],
              [  0, 0, 0, 0, 0, 0]]),
]


# Hydrodynamic inertial excitation matrices
desired_Ihydro = [
    np.array([[  2889934.05, 0, 0, 0, -28899340.49, 0.],
              [  0, 2889934.05, 0, 28899340.49, 0, 0.],
              [  0, 0, 179514.37, 0, 0, 0.],
              [  0, 28899340.49, 0, 385324539.83, 0, 0.],
              [  -28899340.49, 0, 0, 0, 385324539.83, 0.],
              [  0, 0, 0, 0, 0, 0]]),
    np.array([[  3028875.00, 0, 0, 0, -30288750.00, 0.],
              [  0, 2725987.50, 0, 27259875.00, 0, 0.],
              [  0, 0, 93494.99112, 0, 0, 0.],
              [  0, 27259875.00, 0, 363465000.00, 0, 0.],
              [  -30288750.00, 0, 0, 0, 403850000.00, 0.],
              [  0, 0, 0, 0, 0, 0]]),
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
    index = 1
    member = create_member(list_files[index])
    test_inertia((index, member))
    test_hydrostatics((index, member))
    test_hydroConstants((index, member))




    