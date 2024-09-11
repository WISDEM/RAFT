# example script for running RAFT with second-order loads computed internally with the slender-body approximation based on Rainey's equation

import numpy as np
import matplotlib.pyplot as plt
import yaml
import raft
import os
import os.path as path

# open the design YAML file and parse it into a dictionary for passing to raft
flNm = 'OC4semi-WAMIT_Coefs'
current_dir = os.path.dirname(os.path.abspath(__file__))
flPath = path.join(current_dir, flNm + '.yaml')
with open(flPath) as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

# Create the RAFT model (will set up all model objects based on the design dict)
model = raft.Model(design)

# Evaluate the system properties and equilibrium position before loads are applied
model.analyzeUnloaded()

model.analyzeCases(display=1)

model.plotResponses()
plt.show()

# 0.02
# 12.37