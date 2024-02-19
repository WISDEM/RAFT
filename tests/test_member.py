# Test RAFT member functionality
# 

import pytest
import numpy as np
from numpy.testing import assert_allclose
import yaml
import raft
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Name of the subfolder where the test data is located
test_dir = 'test_data'

# List of input file names to be tested
# Running the same file twice just to check if the loop is working. Going to include other tests soon
list_files = ['float_vert_circ_cyl.yaml', 'float_vert_circ_cyl.yaml']

# To avoid problems with different platforms, get the full path of the file
list_files = [os.path.join(current_dir, test_dir, file) for file in list_files]


#=== Desired values

# Define the desired values for mshell and mfill
# Format is: (mshell, mfill, COGx, COGy, COGz)
desired_inertiaBasic = [
    (422944.765, 2310786.205, 0., 0., -14.67635),
    (422944.765, 2310786.205, 0., 0., -14.67635),
]

# 6x6 inertia matrix wrt to (0,0,0)
# Total value (sum of shell and ballast)
desired_inertiaMatrix = [
    np.array([[  2733730.96947283, 0., 0., 0., -40121202.69952862, 0.],
              [  0, 2733730.96947283, 0., 40121202.69952862, 0., 0.],
              [  0, 0., 2733730.96947283, 0., 0., 0.],
              [  0, 40121202.69952862, 0., 732378507.9902301, 0., 0.],
              [  -40121202.69952862, 0., 0., 0., 732378507.9902301, 0.],
              [  0, 0., 0., 0., 0., 37613807.3431156]]),
    np.array([[  2733730.96947283, 0., 0., 0., -40121202.69952862, 0.],
              [  0, 2733730.96947283, 0., 40121202.69952862, 0., 0.],
              [  0, 0., 2733730.96947283, 0., 0., 0.],
              [  0, 40121202.69952862, 0., 732378507.9902301, 0., 0.],
              [  -40121202.69952862, 0., 0., 0., 732378507.9902301, 0.],
              [  0, 0., 0., 0., 0., 37613807.3431156]])
]

# Desired values for hydrostatics
# Fvec[2], Cmat[2,2], Cmat[3,3], Cmat[4,4], r_center[0], r_center[1], r_center[2], xWP, yWP
# Other values, like AWP and IWP, are directly included in the others
desired_hydrostatics = [
    (15324461.090751378, 766223.0545375689, -148598306.13820934, -148598306.13820934, 0, 0, -10, 0, 0),
    (15324461.090751378, 766223.0545375689, -148598306.13820934, -148598306.13820934, 0, 0, -10, 0, 0),
]

# Function used to create member
# Not explicitly inside the fixture below so that it can be used when running this file as a script 
def create_member(file):
    with open(file) as f:
        design = yaml.load(f, Loader=yaml.FullLoader)
    model = raft.Model(design)
    member = model.fowtList[0].memberList[0]
    member.setPosition() # Set the initial position
    return member

# Define a fixture to loop member instances with the index to loop the desired values as well
# Could also zip the lists with the desired values, but I think the approach below is simpler
@pytest.fixture(params=enumerate(list_files))
def index_and_member(request):
    index, file = request.param
    member = create_member(file)
    return index, member

def test_inertia(index_and_member):
    index, member = index_and_member
    
    # Basic inertia properties
    mass, cg, mshell, mfill, pfill = member.getInertia()
    assert_allclose([mshell, mfill[0], cg[0], cg[1], cg[2]], desired_inertiaBasic[index], rtol=1e-05, atol=0, verbose=True)

    # Inertia matrix
    assert_allclose(member.M_struc, desired_inertiaMatrix[index], rtol=1e-05, atol=0, verbose=True)


def test_hydrostatics(index_and_member):
    index, member = index_and_member
    Fvec, Cmat, _, r_center, _, _, xWP, yWP = member.getHydrostatics()
    assert_allclose([Fvec[2], Cmat[2,2], Cmat[3,3], Cmat[4,4], r_center[0], r_center[1], r_center[2], xWP, yWP], desired_hydrostatics[index], rtol=1e-05, atol=0, verbose=True)


if __name__ == "__main__":    
    index = 0
    member = create_member(list_files[index])
    test_inertia((index, member))
    test_hydrostatics((index, member))




    