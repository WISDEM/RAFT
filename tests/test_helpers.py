# Test RAFT member functionality
# 
import pytest
import numpy as np
from numpy.testing import assert_allclose
from raft.helpers import *




'''
 Test functions
'''
def test_FrustumVCV():
    # Circular frustrum
    dA, dB, h = 2, 1, 2
    V, hc = FrustumVCV(dA, dB, h) 
    assert_allclose([V, hc], [3.665191429188092, 0.7857142857142856], rtol=1e-05, atol=0, verbose=True)

    # Rectangular frustrum
    dA, dB, h = [2, 1], [1, 0.5], 2
    V, hc = FrustumVCV(dA, dB, h)
    assert_allclose([V, hc], [2.3333333333333335, 0.7857142857142857], rtol=1e-05, atol=0, verbose=True)


def test_getKinematics():    
    # Arbitrary inputs
    r = [2, 2, 2] # Point of interest
    w  = np.array([0.5, 0.75]) # Frequencies
    Xi = np.array([[1, 2+1j], [0.1+0.2j, 0.3+0.4j], [0.5+0.6j, 0.7+0.8j], [0.9+1.0j, 1.1+1.2j], [1.3+1.4j, 1.5+1.6j], [1.7+1.8j, 1.9+2.0j]])   #Motion amplitudes. These high angles don't make physical sense, but ok for testing
    
    desired_values = np.array([
            [[ 0.2-8.00000000e-01j,  1.2+2.00000000e-01j] , [ 1.7+1.80000000e+00j,  1.9+2.00000000e+00j], [-0.3-2.00000000e-01j, -0.1-2.22044605e-16j]],
            [[ 4.00000000e-01+0.1j, -1.50000000e-01+0.9j] , [-9.00000000e-01+0.85j , -1.50000000e+00+1.425j], [ 1.00000000e-01-0.15j ,  1.66533454e-16-0.075j]],
            [[-0.05+2.0000000e-01j, -0.675-1.1250000e-01j], [-0.425  -4.5000000e-01j, -1.06875-1.1250000e+00j], [ 0.075  +5.0000000e-02j,  0.05625+1.2490009e-16j]]
        ])
    dr, v, a = getKinematics(r, Xi, w)
    assert_allclose([dr, v, a], desired_values, rtol=1e-05, atol=0, verbose=True)


def test_waveKin():
    # Arbitrary inputs
    w = np.array([0.1, 0.25, 0.5, 0.75])
    zeta0 = np.array([0.2, 0.2, 0.2, 0.2])
    beta, h = 30, 200
    r = [30, 45, -20]

    # Firt, check calculation of wave number
    k = np.zeros(w.shape)
    desired_k = np.array([0.00233623, 0.0071452, 0.02548611, 0.05733945])
    for i, iw in enumerate(w):
        k[i] = waveNumber(iw, h)
    assert_allclose(k, desired_k, rtol=1e-05, atol=0, verbose=True)

    # Now, check calculation of wave kinematics
    desired_u = np.array([[ 0.0069097100+0.0006448900j,  0.0073269700+0.0021436100j,  0.0048875900+0.0078728400j, -0.0048089800+0.0055581900j],
                          [-0.0442590100-0.0041307200j, -0.0469316700-0.0137305200j, -0.0313066500-0.0504281200j,  0.0308031300-0.0356020400j],
                          [-0.0016613100+0.0178002300j, -0.0119250300+0.0407604200j, -0.0510284000+0.0316793100j, -0.0360333000-0.0311762500j]])

    desired_ud = np.array([[-0.0000644885+0.0006909710j, -0.0005359019+0.0018317440j, -0.0039364177+0.0024438000j, -0.0041686415-0.0036067400j],
                           [ 0.0004130725-0.0044259010j,  0.0034326291-0.0117329200j,  0.0252140594-0.0156533200j,  0.0267015296+0.0231023400j],
                           [-0.0017800228-0.0001661310j, -0.0101901044-0.0029812600j, -0.0158396548-0.0255142000j,  0.0233821912-0.0270249700j]])

    desired_pDyn = np.array([1963.730340920+183.276331860j, 1703.156386190+498.282218140j, 637.171137130+1026.342526750j, -417.980049950+483.098446900j])

    u, ud, pDyn = getWaveKin(zeta0, beta, w, k, h, r, len(w))
    assert_allclose(u, desired_u, rtol=1e-05, atol=0, verbose=True)
    assert_allclose(ud, desired_ud, rtol=1e-05, atol=0, verbose=True)
    assert_allclose(pDyn, desired_pDyn, rtol=1e-05, atol=0, verbose=True)


def test_smallRotate():
    r = [1, 2, 3]
    th = deg2rad(np.array([5+3j, 3+5j, 4+3j]))
    rt = SmallRotate(r, th)
    desired_rt = np.array([ 0.01745329+0.15707963j, -0.19198622-0.10471976j, 0.12217305+0.01745329j])
    assert_allclose(rt, desired_rt, rtol=1e-05, atol=0, verbose=True)


def test_vecVecTrans():
    # Arbitrary vector
    v = np.array([0.7+1.2j, 1.5+0.4j, 3.0+2.3j])
    vvt = VecVecTrans(v)
    desired_vvt = np.array([[-0.95+1.68j, 0.57+2.08j, -0.66+5.21j], [0.57+2.08j, 2.09+1.2j, 3.58+4.65j], [-0.66+5.21j, 3.58+4.65j, 3.71+13.8j]])
    assert_allclose(vvt, desired_vvt, rtol=1e-05, atol=0, verbose=True)
    

def test_translateForce3to6DOF():
    # Arbitrary force and point
    Fin = np.array([0.5+3j, 2.0+1.5j, 3.0+0.7j])
    r   = np.array([1, 2, 3])
    Fout = translateForce3to6DOF(Fin, r)
    desired_Fout = np.array([0.5+3.0j, 2.0+1.5j, 3.0+0.7j,  0.0-3.1j, -1.5+8.3j, 1.0-4.5j])
    assert_allclose(Fout, desired_Fout, rtol=1e-05, atol=0, verbose=True)


def test_transformForce():
    offset = np.array([10, 20, 30])
    f_in = np.array([0.5+3j, 2.0+1.5j, 3.0+0.7j]) # For the tests with an arbitrary force vector    
    F_in = np.array([1.2+0.3j, 0.4+1.5j, 2.3+0.7j, 0.5+0.9j, 1.1+0.2j, 0.7+1.4j]) # For the tests with an arbitrary force and moment vector    
    orient_3 = np.array([0.1, 0.2, 0.3]) # For the tests with orientation given by 3 Euler angles    
    rotMat = rotationMatrix(*orient_3) # For the tests with orientation given by a rotation matrix

    # Tests with f_in
    # Both rotations should yield the same result
    desired = np.array([0.57300698+02.54908178j,   1.94679387+02.27765615j, 3.02186311+00.23337633j,   
                        2.03344603-63.66215798j, -13.02842176+74.13869023j, 8.00779917-28.20507416j])                        
    F_out = transformForce(f_in, offset=offset, orientation=orient_3)
    assert_allclose(F_out, desired, rtol=1e-05, atol=0, verbose=True)
    F_out = transformForce(f_in, offset=offset, orientation=rotMat)
    assert_allclose(F_out, desired, rtol=1e-05, atol=0, verbose=True)

    # Tests with F_in
    # Both rotations should yield the same result
    desired = np.array([  1.51572022+2.10897023e-02j,  0.64512428+1.49565656e+00j,   2.04362591+7.69783522e-01j,   
                         21.83717669-2.83806906e+01j, 26.20635997-6.66493243e+00j, -23.17224939+1.57407763e+01j])
    F_out = transformForce(F_in, offset=offset, orientation=orient_3)
    assert_allclose(F_out, desired, rtol=1e-05, atol=0, verbose=True)
    F_out = transformForce(F_in, offset=offset, orientation=rotMat)
    assert_allclose(F_out, desired, rtol=1e-05, atol=0, verbose=True)


def test_translateMatrix3to6DOF():
    Min = np.array([[0.73, 2.41, 3.88],
                    [1.25, 9.12, 5.79],
                    [5.37, 7.94, 8.63]])
    r   = np.array([10, 20, 30])
    Mout = translateMatrix3to6DOF(Min, r)

    desired_Mout = np.array([[ 7.300e-01,  2.410e+00,  3.880e+00,  5.300e+00, -1.690e+01,  9.500e+00],
                             [ 1.250e+00,  9.120e+00,  5.790e+00, -1.578e+02, -2.040e+01,  6.620e+01],
                             [ 5.370e+00,  7.940e+00,  8.630e+00, -6.560e+01,  7.480e+01, -2.800e+01],
                             [ 5.300e+00, -1.578e+02, -6.560e+01,  3.422e+03,  2.108e+03, -2.546e+03],
                             [-1.690e+01, -2.040e+01,  7.480e+01,  8.150e+02, -1.255e+03,  5.650e+02],
                             [ 9.500e+00,  6.620e+01, -2.800e+01, -1.684e+03,  1.340e+02,  4.720e+02]])
    assert_allclose(Mout, desired_Mout, rtol=1e-05, atol=0, verbose=True)


def test_translateMatrix6to6DOF():
    # Define Min as a 6x6 matrix
    Min = np.array([[ 0.57,   0.64,  0.88,  0.12,  0.34,  0.56],
                    [ 2.03, -13.02,  8.00,  0.78,  0.90,  0.12],
                    [ 1.11,  -0.15,  0.10,  0.34,  0.56,  0.78],
                    [ 0.12,   0.78,  0.34,  0.90,  0.12,  0.34],
                    [ 0.34,   0.90,  0.56,  0.12,  0.34,  0.56],
                    [ 0.56,   0.12,  0.78,  0.34,  0.56,  0.78]])
    r   = np.array([10, 20, 30])
    Mout = translateMatrix6to6DOF(Min, r)
    desired_Mout = np.array([[ 5.70000e-01,  6.40000e-01,  8.80000e-01, -1.48000e+00,  8.64000e+00, -4.44000e+00],
                             [ 2.03000e+00, -1.30200e+01,  8.00000e+00,  5.51380e+02, -1.82000e+01, -1.70680e+02],
                             [ 1.11000e+00, -1.50000e-01,  1.00000e-01,  6.84000e+00,  3.28600e+01, -2.29200e+01],
                             [-1.48000e+00,  5.51380e+02,  6.84000e+00, -1.64203e+04,  1.20352e+03,  4.66774e+03],
                             [ 8.64000e+00, -1.82000e+01,  3.28600e+01, -1.28480e+02, -6.44600e+01,  9.87600e+01],
                             [-4.44000e+00, -1.70680e+02, -2.29200e+01,  5.55574e+03, -3.45240e+02, -1.62722e+03]])
    assert_allclose(Mout, desired_Mout, rtol=1e-05, atol=0, verbose=True)


def test_rotateMatrix6():
    rotMat = rotationMatrix(0.1, 0.2, 0.3)

    # Test a 6x6 matrix
    Min = np.array([[ 0.57,   0.64,  0.88,  0.12,  0.34,  0.56],
                    [ 2.03, -13.02,  8.00,  0.78,  0.90,  0.12],
                    [ 1.11,  -0.15,  0.10,  0.34,  0.56,  0.78],
                    [ 0.12,   0.78,  0.34,  0.90,  0.12,  0.34],
                    [ 0.34,   0.90,  0.56,  0.12,  0.34,  0.56],
                    [ 0.56,   0.12,  0.78,  0.34,  0.56,  0.78]])
    Mout = rotateMatrix6(Min, rotMat)
    desired_Mout = np.array([[-1.23327412,   4.08056795, -0.95870608, 0.06516703, 0.15206293, 0.66964386],
                             [ 7.03270577, -11.42123791,  6.09625616, 0.51524892, 1.11098643, 0.18118973],
                             [ 1.67312218,  -1.16775529,  0.30451203, 0.34805446, 0.62871201, 0.62384654],
                             [ 0.06516703,   0.51524892,  0.34805446, 0.86182628, 0.37858592, 0.16449501],
                             [ 0.15206293,   1.11098643,  0.62871201, 0.37858592, 0.40719201, 0.55131878],
                             [ 0.66964386,   0.18118973,  0.62384654, 0.16449501, 0.55131878, 0.75098172]])
    assert_allclose(Mout, desired_Mout, rtol=1e-05, atol=0, verbose=True)

    # Test a 6x6x3 matrix
    Min = np.array([[[ 0.57,   0.64,  0.88], [ 0.12,  0.34,  0.56], [ 2.03, -13.02,  8.00], [ 0.78,  0.90,  0.12], [ 1.11,  -0.15,  0.10], [ 0.34,  0.56,  0.78]],
                    [[ 0.12,   0.78,  0.34], [ 0.90,  0.12,  0.34], [ 0.34,   0.90,  0.56], [ 0.12,  0.34,  0.56], [ 0.56,   0.12,  0.78], [ 0.34,  0.56,  0.78]],
                    [[ 0.57,   0.64,  0.88], [ 0.12,  0.34,  0.56], [ 2.03, -13.02,  8.00], [ 0.78,  0.90,  0.12], [ 1.11,  -0.15,  0.10], [ 0.34,  0.56,  0.78]],
                    [[ 0.12,   0.78,  0.34], [ 0.90,  0.12,  0.34], [ 0.34,   0.90,  0.56], [ 0.12,  0.34,  0.56], [ 0.56,   0.12,  0.78], [ 0.34,  0.56,  0.78]],
                    [[ 0.57,   0.64,  0.88], [ 0.12,  0.34,  0.56], [ 2.03, -13.02,  8.00], [ 0.78,  0.90,  0.12], [ 1.11,  -0.15,  0.10], [ 0.34,  0.56,  0.78]],
                    [[ 0.12,   0.78,  0.34], [ 0.90,  0.12,  0.34], [ 0.34,   0.90,  0.56], [ 0.12,  0.34,  0.56], [ 0.56,   0.12,  0.78], [ 0.34,  0.56,  0.78]]])
    Mout = rotateMatrix6(Min, rotMat)
    desired_Mout = np.array([[[ 1.10667892e+00, -2.94455342e+00,  2.69493519e+00], [-6.38822610e-03,  1.06052232e+00,  4.60482856e-01], [ 2.05965518e+00, -1.49706284e+01,  8.72839902e+00], [ 5.67436840e-01,  1.04967800e+00,  1.62554624e-01], [ 1.31869988e+00,  5.84615864e-02, -1.24634114e-01], [ 2.29581822e-01,  2.72270035e-01,  6.62394607e-01]],
                             [[ 1.80170383e-01,  2.64311316e-01,  9.42590356e-01], [ 8.96253958e-01,  5.44607145e-01,  5.10429910e-01], [ 8.53073720e-01, -2.52940264e+00,  2.43037183e+00], [ 1.57254780e-01,  6.44083517e-01,  5.23616022e-01], [ 8.55634323e-01,  2.08543415e-01,  8.66723141e-01], [ 4.18776248e-01,  5.58021190e-01,  8.82713161e-01]],
                             [[ 7.26993044e-01, -1.72740810e+00,  1.91052186e+00], [ 2.45449243e-01,  8.00156640e-01,  4.23652255e-01], [ 1.49706712e+00, -9.86005372e+00,  6.01463490e+00], [ 3.90806718e-01,  8.21200430e-01,  2.45102321e-01], [ 1.04459715e+00,  9.37773151e-02,  1.64911720e-01], [ 2.56928837e-01,  3.21778587e-01,  6.50722236e-01]],
                             [[ 5.67436840e-01,  1.04967800e+00,  1.62554624e-01], [ 1.57254780e-01,  6.44083517e-01,  5.23616022e-01], [ 3.90806718e-01,  8.21200430e-01,  2.45102321e-01], [-9.97525644e-02,  1.93835477e-01,  4.84108703e-01], [ 2.93305811e-01,  1.95777043e-01,  9.87426179e-01], [ 3.40111996e-01,  4.69105958e-01,  6.32540213e-01]],
                             [[ 1.31869988e+00,  5.84615864e-02, -1.24634114e-01], [ 8.55634323e-01,  2.08543415e-01,  8.66723141e-01], [ 1.04459715e+00,  9.37773151e-02,  1.64911720e-01], [ 4.85661578e-01,  1.06534995e+00,  3.65339310e-01], [ 1.36035814e+00,  1.40953419e-01,  3.19340109e-01], [ 3.64370617e-01,  4.61133778e-01,  8.97408217e-01]],
                             [[ 2.29581822e-01,  2.72270035e-01,  6.62394607e-01], [ 4.18776248e-01,  5.58021190e-01,  8.82713161e-01], [ 2.56928837e-01,  3.21778587e-01,  6.50722236e-01], [ 7.41105350e-02,  4.14955880e-01,  3.97735065e-01], [ 5.57871216e-01,  1.58957129e-01,  6.92775693e-01], [ 3.09394425e-01,  4.15211104e-01,  6.36551189e-01]]])
    assert_allclose(Mout, desired_Mout, rtol=1e-05, atol=0, verbose=True)

    
def test_RotFrm2Vect():
    rotMat = rotationMatrix(0.1, 0.2, 0.3)
    A = np.array([5, 0, 0])
    B = np.matmul(rotMat, A)
    R = RotFrm2Vect(A,B)
    
    assert_allclose(B, np.matmul(R, A), rtol=1e-05, atol=0, verbose=True)
    



'''
 To run as a script. Useful for debugging.
'''
if __name__ == "__main__":
    test_FrustumVCV()
    test_getKinematics()
    test_waveKin()
    test_smallRotate()
    test_vecVecTrans()
    test_translateForce3to6DOF()
    test_transformForce()
    test_translateMatrix3to6DOF()
    test_translateMatrix6to6DOF()
    test_rotateMatrix6()
    test_RotFrm2Vect()
    print("Running as a script")



    