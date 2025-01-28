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
    'VolturnUS-S-pointInertia.yaml'
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
# Structure related quantities
desired_rCG = [
    np.array([              0,               0, -78.03525272   ]),
    np.array([-3.78742736e-01,  7.80925410e-16, -1.91132267e+00]),
    np.array([-3.78616422e-01,  3.80441015e-19, -1.69266625e+00]),
]
desired_rCG_sub = [
    np.array([              0,               0, -89.91292526    ]),
    np.array([ 2.65203563e-15,  8.85162184e-16,  -1.51939447e+01]),
    np.array([ 8.62405052e-19,  4.31202526e-19,  -1.49410924e+01]),
]

desired_m_ballast = [
    np.array([6.5323524956e+06]),
    np.array([1.0569497625e+07, 2.42678207158787e+06]),    
    np.array([0.]),
]

desired_M_struc = [
    np.array([[ 8.08951257e+06,  0.00000000e+00, -3.63797881e-12,  0.00000000e+00, -6.31267158e+08,  0.00000000e+00],
              [ 0.00000000e+00,  8.08951257e+06,  0.00000000e+00,  6.31267158e+08,  0.00000000e+00,  0.00000000e+00],
              [ 3.63797881e-12,  0.00000000e+00,  8.08951257e+06,  0.00000000e+00,  3.25832739e-10,  0.00000000e+00],
              [ 0.00000000e+00,  6.31267158e+08,  0.00000000e+00,  6.77394404e+10,  0.00000000e+00, -8.06082047e+05],
              [-6.31267158e+08,  0.00000000e+00,  3.25832739e-10,  0.00000000e+00,  6.77302268e+10,  0.00000000e+00],
              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -8.06082047e+05,  0.00000000e+00,  1.18100818e+08]]),
    np.array([[ 1.91186826e+07,  0.00000000e+00, -7.72177272e-12,  0.00000000e+00, -3.65419714e+07, -1.49302650e-08],
              [ 0.00000000e+00,  1.91186826e+07,  0.00000000e+00,  3.65419714e+07,  0.00000000e+00, -7.24106215e+06],
              [ 1.30711139e-11,  0.00000000e+00,  1.91186826e+07,  1.49302650e-08,  7.24106215e+06,  0.00000000e+00],
              [ 0.00000000e+00,  3.65419714e+07,  1.49302650e-08,  4.26605281e+10,  9.54605639e-07,  1.08255551e+09],
              [-3.65419714e+07,  0.00000000e+00,  7.24106215e+06,  9.54605639e-07,  4.27128822e+10,  4.76371497e-07],
              [-1.49302650e-08, -7.24106215e+06,  0.00000000e+00,  1.08255551e+09,  4.76371497e-07,  2.06119358e+10]]),
    np.array([[ 1.91250610e+07,  0.00000000e+00, -7.72177272e-12,  0.00000000e+00, -3.23723453e+07, -7.27595761e-12],
              [ 0.00000000e+00,  1.91250610e+07,  0.00000000e+00,  3.23723453e+07,  0.00000000e+00, -7.24106215e+06],
              [ 1.30711139e-11,  0.00000000e+00,  1.91250610e+07,  7.27595761e-12,  7.24106215e+06,  0.00000000e+00],
              [ 0.00000000e+00,  3.23723453e+07,  7.27595761e-12,  4.25355736e+10,  2.32830644e-10,  1.08255551e+09],
              [-3.23723453e+07,  0.00000000e+00,  7.24106215e+06,  2.32830644e-10,  4.25879277e+10,  1.16415322e-10],
              [-7.27595761e-12, -7.24106215e+06,  0.00000000e+00,  1.08255551e+09,  1.16415322e-10,  2.06187038e+10]]),
]

desired_M_struc_sub = [
    np.array([[  7.48986700e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -6.73435852e+08,  0.00000000e+00],
              [  0.00000000e+00,  7.48986700e+06,  0.00000000e+00,  6.73435852e+08,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  7.48986700e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  6.73435852e+08,  0.00000000e+00,  6.43071810e+10,  0.00000000e+00,  0.00000000e+00],
              [ -6.73435852e+08,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  6.43071810e+10,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  9.00523426e+07]]),    
    np.array([[  1.68672649e+07,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.56280290e+08, -1.49302650e-08],
              [  0.00000000e+00,  1.68672649e+07,  0.00000000e+00,  2.56280290e+08,  0.00000000e+00,  4.47325874e-08],
              [  0.00000000e+00,  0.00000000e+00,  1.68672649e+07,  1.49302650e-08, -4.47325874e-08,  0.00000000e+00],
              [  0.00000000e+00,  2.56280290e+08,  1.49302650e-08,  1.49458996e+10,  9.54605639e-07,  4.76371497e-07],
              [ -2.56280290e+08,  0.00000000e+00, -4.47325874e-08,  9.54605639e-07,  1.49458996e+10,  4.76371497e-07],
              [ -1.49302650e-08,  4.47325874e-08,  0.00000000e+00,  5.96046448e-07,  4.76371497e-07,  2.05313182e+10]]),
    np.array([[ 1.68736433e+07,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.52110664e+08, -7.27595761e-12],
              [ 0.00000000e+00,  1.68736433e+07,  0.00000000e+00,  2.52110664e+08,  0.00000000e+00,  1.45519152e-11],
              [ 0.00000000e+00,  0.00000000e+00,  1.68736433e+07,  7.27595761e-12, -1.45519152e-11,  0.00000000e+00],
              [ 0.00000000e+00,  2.52110664e+08,  7.27595761e-12,  1.48209451e+10,  2.32830644e-10,  2.32830644e-10],
              [-2.52110664e+08,  0.00000000e+00, -1.45519152e-11,  2.32830644e-10,  1.48209451e+10,  1.16415322e-10],
              [-7.27595761e-12,  1.45519152e-11,  0.00000000e+00,  3.49245965e-10,  1.16415322e-10,  2.05380863e+10]]),
]

desired_C_struc = [
    np.array([[  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  6.19273082e+09,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  6.19273082e+09,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]),    
    np.array([[  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.58476739e+08,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.58476739e+08,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]),
    np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.17572707e+08, 0.00000000e+00, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.17572707e+08, 0.00000000e+00],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]),
]

desired_W_struc = [
    np.array([  0.00000000e+00,  0.00000000e+00, -7.93581183e+07,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]),
    np.array([  0.00000000e+00,  0.00000000e+00, -1.87554276e+08, -2.38651410e-07, -7.10348197e+07,  0.00000000e+00]),
    np.array([  0.00000000e+00,  0.00000000e+00, -1.87616848e+08, -5.82076609e-11, -7.10348197e+07,  0.00000000e+00]),
]

# Hydrostatic quantities
desired_rCB = [
    np.array([ 0.00000000e+00,  0.00000000e+00, -6.20656552e+01]),
    np.array([ 3.04454348e-15,  1.52227174e-15, -1.35855138e+01]),
    np.array([ 3.04454348e-15,  1.52227174e-15, -1.35855138e+01]),
]

desired_C_hydro = [
    np.array([[  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  3.33664089e+05,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -5.01003340e+09,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -5.01003340e+09,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]),
    np.array([[  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  4.30992285e+06, -7.45058060e-09,  2.23517418e-08,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00, -7.45058060e-09,  2.17117691e+09, -4.76837158e-07,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  2.23517418e-08, -4.76837158e-07,  2.17117691e+09,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]),
    np.array([[  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  4.30992285e+06, -7.45058060e-09,  2.23517418e-08,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00, -7.45058060e-09,  2.17117691e+09, -4.76837158e-07,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  2.23517418e-08, -4.76837158e-07,  2.17117691e+09,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]])
]

desired_W_hydro = [
    np.array([ 0.00000000e+00,  0.00000000e+00,  8.07357058e+07,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]),
    np.array([ 0.00000000e+00,  0.00000000e+00,  1.92243134e+08,  2.38418579e-07, -3.57627869e-07,  0.00000000e+00]),    
    np.array([ 0.00000000e+00,  0.00000000e+00,  1.92243134e+08,  2.38418579e-07, -3.57627869e-07,  0.00000000e+00]),
]


# Hydrodynamic quantities
desired_A_hydro_morison = [
    np.array([[  8.22881104e+06,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  -5.10692712e+08,  0.00000000e+00],
              [  0.00000000e+00,  8.22881104e+06,  0.00000000e+00, 5.10692712e+08,   0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  2.23242607e+05, 0.00000000e+00,   0.00000000e+00,  0.00000000e+00],
              [  0.00000000e+00,  5.10692712e+08,  0.00000000e+00, 4.09467123e+10,   0.00000000e+00,  0.00000000e+00],
              [ -5.10692712e+08,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,   4.09467123e+10,  0.00000000e+00],
              [  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,   0.00000000e+00,  0.00000000e+00]]),    
    np.array([[  1.23332103e+07,  4.65661287e-10,  0.00000000e+00,  7.45058060e-09, -1.54950929e+08, -2.98023224e-08],
              [  4.65661287e-10,  1.23332103e+07, -2.58493941e-26,  1.54950929e+08, -7.45058060e-09,  7.45058060e-08],
              [  0.00000000e+00, -2.58493941e-26,  1.09392236e+07,  1.49011612e-08, -1.49011612e-08, -9.18354962e-41],
              [  7.45058060e-09,  1.54950929e+08,  1.49011612e-08,  7.44302567e+09,  3.57627869e-07,  9.53674316e-07],
              [ -1.54950929e+08, -7.45058060e-09, -1.49011612e-08,  3.57627869e-07,  7.44302567e+09,  4.76837158e-07],
              [ -2.98023224e-08,  7.45058060e-08, -9.18354962e-41,  8.34465027e-07,  4.76837158e-07,  2.39620560e+10]]),
    np.array([[  1.23332103e+07,  4.65661287e-10,  0.00000000e+00,  7.45058060e-09, -1.54950929e+08, -2.98023224e-08],
              [  4.65661287e-10,  1.23332103e+07, -2.58493941e-26,  1.54950929e+08, -7.45058060e-09,  7.45058060e-08],
              [  0.00000000e+00, -2.58493941e-26,  1.09392236e+07,  1.49011612e-08, -1.49011612e-08, -9.18354962e-41],
              [  7.45058060e-09,  1.54950929e+08,  1.49011612e-08,  7.44302567e+09,  3.57627869e-07,  9.53674316e-07],
              [ -1.54950929e+08, -7.45058060e-09, -1.49011612e-08,  3.57627869e-07,  7.44302567e+09,  4.76837158e-07],
              [ -2.98023224e-08,  7.45058060e-08, -9.18354962e-41,  8.34465027e-07,  4.76837158e-07,  2.39620560e+10]]),              
]

desired_F_hydro_iner = [
    np.array([[[  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+1.29014243e-04j,  0.00000000e+00+2.18136450e+04j,  0.00000000e+00+3.20625334e+05j,  0.00000000e+00+4.47063970e+05j,  0.00000000e+00+3.72186312e+05j,  0.00000000e+00+2.73435771e+05j,  0.00000000e+00+1.96646986e+05j,  0.00000000e+00+1.42846361e+05j,  0.00000000e+00+1.05877244e+05j],
               [  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j],
               [  0.00000000e+00+0.00000000e+00j, -3.83018423e-06+0.00000000e+00j, -4.39079984e+03+0.00000000e+00j, -7.61182325e+04+0.00000000e+00j, -1.07109629e+05+0.00000000e+00j, -8.59303021e+04+0.00000000e+00j, -5.94885784e+04+0.00000000e+00j, -3.95278075e+04+0.00000000e+00j, -2.59659374e+04+0.00000000e+00j, -1.70032358e+04+0.00000000e+00j],
               [  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j],
               [  0.00000000e+00+0.00000000e+00j,  0.00000000e+00-7.11328048e-03j,  0.00000000e+00-1.01290133e+06j,  0.00000000e+00-1.16863323e+07j,  0.00000000e+00-1.21895057e+07j,  0.00000000e+00-7.55633166e+06j,  0.00000000e+00-4.23360829e+06j,  0.00000000e+00-2.38627057e+06j,  0.00000000e+00-1.38694212e+06j,  0.00000000e+00-8.34570909e+05j],
               [  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j]]]),    
    np.array([[[  0.00000000e+00+0.00000000e+00j,  5.27862388e-06+3.55202739e-04j,  2.70437607e+03+7.13012568e+04j,  1.39204693e+05+1.20341600e+06j,  6.19003712e+05+1.45322648e+06j,  1.26581615e+06+5.38094639e+05j,  1.54066254e+06-1.06700055e+05j,  9.73449433e+05-1.54735127e+05j, -7.25098731e+04-1.53049868e+05j, -5.50225753e+05-2.51854400e+05j],
               [  0.00000000e+00+0.00000000e+00j,  0.00000000e+00-6.77626358e-21j,  0.00000000e+00-9.09494702e-13j,  2.91038305e-11+0.00000000e+00j, -2.91038305e-11-1.45519152e-11j, -5.09317033e-11+2.91038305e-11j,  0.00000000e+00-2.18278728e-11j,  7.27595761e-12-1.81898940e-11j, -3.63797881e-12-9.09494702e-13j,  0.00000000e+00+1.36424205e-12j],
               [  0.00000000e+00+0.00000000e+00j,  4.36781698e-04-1.06335971e-06j,  1.12590567e+04-7.39559352e+02j, -3.67100235e+05-3.03329514e+04j, -9.33433420e+05-5.29491702e+04j, -7.56882469e+05+4.71609883e+04j, -3.08614019e+05+2.21762147e+05j, -3.10898556e+04+3.10700182e+05j,  4.30529645e+04+2.41235701e+05j,  4.57382992e+04+1.05934499e+05j],
               [  0.00000000e+00+0.00000000e+00j,  1.95156391e-18+4.87890978e-19j,  8.73114914e-11+4.36557457e-11j,  9.31322575e-10+9.31322575e-10j, -9.31322575e-10+1.86264515e-09j, -1.39698386e-09-9.31322575e-10j, -2.03726813e-10-8.14907253e-10j,  5.82076609e-10+0.00000000e+00j,  3.20142135e-10-1.09139364e-10j,  1.45519152e-11+5.09317033e-11j],
               [  0.00000000e+00+0.00000000e+00j, -5.25125306e-04-1.13258648e-04j, -1.88802798e+05-2.34251669e+05j, -5.03535064e+06-9.34753238e+06j, -9.66951530e+06-2.29999628e+07j, -9.07389415e+06-2.31167534e+07j, -7.24707025e+06-1.36646621e+07j, -4.98264131e+06-4.74474286e+06j, -1.64674075e+06-6.23579171e+04j,  4.05694750e+05+1.24236863e+06j],
               [  0.00000000e+00+0.00000000e+00j,  1.49077799e-19-2.71050543e-19j, -1.81898940e-11+7.27595761e-12j,  1.51339918e-09-6.98491931e-10j,  0.00000000e+00+1.16415322e-09j, -1.62981451e-09+1.97906047e-09j,  3.02679837e-09-1.48429535e-09j, -7.56699592e-10+5.23868948e-10j,  1.13504939e-09+9.24046617e-10j, -4.36557457e-11-9.31322575e-10j]]]),
    np.array([[[  0.00000000e+00+0.00000000e+00j,  5.27862388e-06+3.55202739e-04j,  2.70437607e+03+7.13012568e+04j,  1.39204693e+05+1.20341600e+06j,  6.19003712e+05+1.45322648e+06j,  1.26581615e+06+5.38094639e+05j,  1.54066254e+06-1.06700055e+05j,  9.73449433e+05-1.54735127e+05j, -7.25098731e+04-1.53049868e+05j, -5.50225753e+05-2.51854400e+05j],
               [  0.00000000e+00+0.00000000e+00j,  0.00000000e+00-6.77626358e-21j,  0.00000000e+00-9.09494702e-13j,  2.91038305e-11+0.00000000e+00j, -2.91038305e-11-1.45519152e-11j, -5.09317033e-11+2.91038305e-11j,  0.00000000e+00-2.18278728e-11j,  7.27595761e-12-1.81898940e-11j, -3.63797881e-12-9.09494702e-13j,  0.00000000e+00+1.36424205e-12j],
               [  0.00000000e+00+0.00000000e+00j,  4.36781698e-04-1.06335971e-06j,  1.12590567e+04-7.39559352e+02j, -3.67100235e+05-3.03329514e+04j, -9.33433420e+05-5.29491702e+04j, -7.56882469e+05+4.71609883e+04j, -3.08614019e+05+2.21762147e+05j, -3.10898556e+04+3.10700182e+05j,  4.30529645e+04+2.41235701e+05j,  4.57382992e+04+1.05934499e+05j],
               [  0.00000000e+00+0.00000000e+00j,  1.95156391e-18+4.87890978e-19j,  8.73114914e-11+4.36557457e-11j,  9.31322575e-10+9.31322575e-10j, -9.31322575e-10+1.86264515e-09j, -1.39698386e-09-9.31322575e-10j, -2.03726813e-10-8.14907253e-10j,  5.82076609e-10+0.00000000e+00j,  3.20142135e-10-1.09139364e-10j,  1.45519152e-11+5.09317033e-11j],
               [  0.00000000e+00+0.00000000e+00j, -5.25125306e-04-1.13258648e-04j, -1.88802798e+05-2.34251669e+05j, -5.03535064e+06-9.34753238e+06j, -9.66951530e+06-2.29999628e+07j, -9.07389415e+06-2.31167534e+07j, -7.24707025e+06-1.36646621e+07j, -4.98264131e+06-4.74474286e+06j, -1.64674075e+06-6.23579171e+04j,  4.05694750e+05+1.24236863e+06j],
               [  0.00000000e+00+0.00000000e+00j,  1.49077799e-19-2.71050543e-19j, -1.81898940e-11+7.27595761e-12j,  1.51339918e-09-6.98491931e-10j,  0.00000000e+00+1.16415322e-09j, -1.62981451e-09+1.97906047e-09j,  3.02679837e-09-1.48429535e-09j, -7.56699592e-10+5.23868948e-10j,  1.13504939e-09+9.24046617e-10j, -4.36557457e-11-9.31322575e-10j]]]),               
]

desired_current_drag = [
    np.array([1.66747692e+06, 4.46799093e+05,        0.0e+00, 2.67342887e+07, -9.97737237e+07,        0.0e+00]),
    np.array([2.64655964e+06, 6.47726496e+05, 7.60648090e-27, 8.77357984e+06, -3.65254345e+07, 1.15751779e+07]),    
    np.array([2.64655964e+06, 6.47726496e+05, 7.60648090e-27, 8.77357984e+06, -3.65254345e+07, 1.15751779e+07]),
]

'''
 Aux functions
'''
# Function used to create FOWT instance
# Not explicitly inside the fixture below so that we can also run this file as a script
# 
def create_fowt(file):
    with open(file) as f:
        design = yaml.load(f, Loader=yaml.FullLoader)        
    fowt = raft.Model(design).fowtList[0]
    fowt.setPosition(np.zeros(6))
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
def test_statics(index_and_fowt):
    index, fowt = index_and_fowt    

    # Structure related quantities
    assert_allclose(fowt.rCG, desired_rCG[index], rtol=1e-05, atol=1e-3)
    assert_allclose(fowt.rCG_sub, desired_rCG_sub[index], rtol=1e-05, atol=1e-3)
    assert_allclose(fowt.m_ballast, desired_m_ballast[index], rtol=1e-05, atol=1e-3)
    assert_allclose(fowt.M_struc, desired_M_struc[index], rtol=1e-05, atol=1e-3)
    assert_allclose(fowt.M_struc_sub, desired_M_struc_sub[index], rtol=1e-05, atol=1e-3)
    assert_allclose(fowt.C_struc, desired_C_struc[index], rtol=1e-05, atol=1e-3)
    assert_allclose(fowt.W_struc, desired_W_struc[index], rtol=1e-05, atol=1e-3)

    # Hydrostatic quantities
    assert_allclose(fowt.rCB, desired_rCB[index], rtol=1e-05, atol=1e-3)
    assert_allclose(fowt.C_hydro, desired_C_hydro[index], rtol=1e-05, atol=1e-3)
    assert_allclose(fowt.W_hydro, desired_W_hydro[index], rtol=1e-05, atol=1e-3)    


def test_hydroConstants(index_and_fowt):
    index, fowt = index_and_fowt
    fowt.calcHydroConstants() 
    assert_allclose(fowt.A_hydro_morison, desired_A_hydro_morison[index], rtol=1e-05, atol=1e-3)


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

    phase_array = np.linspace(0, 2 * np.pi, fowt.nw * 6).reshape(6, fowt.nw) # Needed an arbitrary motion amplitude. Assuming uniform amplitude with phases linearly spaced between 0 and 2pi. Times 6 for surge, sway, ..., yaw
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

def test_calcCurrentLoads(index_and_fowt):
    index, fowt = index_and_fowt    
    testCase = {'current_speed': 2.0, 'current_heading':15}
    D = fowt.calcCurrentLoads(testCase)

    assert_allclose(D, desired_current_drag[index], rtol=1e-05, atol=1e-3)

def test_calcQTF_slenderBody(index_and_fowt, flagSaveValues=False):    
    # Set flagSaveValues to true to replace the true values file with the values calculated below    
    index, fowt = index_and_fowt      

    if fowt.potSecOrder: # Check only cases that compute the QTFs
        true_values_file = list_files[index].replace('.yaml', '_true_calcQTF_slenderBody.pkl')
        
        testCase = {'wave_heading': 30, 'wave_period': 12, 'wave_height': 6} # Testing one case only
        fowt.calcHydroConstants() # Need to call this one before calcHydroExcitation
        fowt.calcHydroExcitation(testCase, memberList=fowt.memberList) # Need to call this one before calcQTF_slenderBody
        fowt.calcQTF_slenderBody(0) # Testing for the body considered to be fixed. Model class should take care of cases with motion

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


'''
 To run as a script. Useful for debugging.
'''
if __name__ == "__main__":
    index = 0

    fowt = create_fowt(list_files[index])
    test_statics((index,fowt))
    
    fowt = create_fowt(list_files[index])
    test_hydroConstants((index,fowt))

    fowt = create_fowt(list_files[index])
    test_hydroExcitation((index,fowt), flagSaveValues=False)

    fowt = create_fowt(list_files[index])
    test_hydroLinearization((index,fowt), flagSaveValues=False)

    fowt = create_fowt(list_files[index])
    test_calcCurrentLoads((index,fowt))

    fowt = create_fowt(list_files[index])
    test_calcQTF_slenderBody((index,fowt), flagSaveValues=False)

