# example script for running RAFT with second-order loads computed internally with the slender-body approximation based on Rainey's equation

import numpy as np
import matplotlib.pyplot as plt
import yaml
import raft
import os.path as path

# open the design YAML file and parse it into a dictionary for passing to raft
flNm = 'OC4semi-WAMIT_QTF'
with open(flNm + '.yaml') as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

# Create the RAFT model (will set up all model objects based on the design dict)
model = raft.Model(design)

# Evaluate the system properties and equilibrium position before loads are applied
model.analyzeUnloaded()

# Due to the linearization of the quadratic drag term in RAFT, the QTFs depend on the sea state specified in the input file.
# If more than one case is analyzed, the outputs are numbered sequentially.
# Two output files are generated:
# - The QTF, following WAMIT .12d file format. File name is qtf-slender_body-total_Head#p##_Case#_WT#.12d
# - The RAOs used to computed the QTFs, following WAMIT .4 file format. File name is qtf-slender_body-total_Head#p##_Case#_WT#.12d
# The Head#p## in the file name indicates the wave heading in degrees (p replaces the decimal point). 
# Case number starts at 1, but turbine at 0 in conformity with the rest of the code.
model.analyzeCases(display=1)

# 0.02
# 12.37