# example script for running RAFT with second-order loads computed internally with the slender-body approximation based on Rainey's equation

import numpy as np
import matplotlib.pyplot as plt
import yaml
import raft
import os.path as path

# open the design YAML file and parse it into a dictionary for passing to raft
flNm = 'OC3spar-SlenderBodyQTF'
with open('./examples/' + flNm + '.yaml') as file:
    design = yaml.load(file, Loader=yaml.FullLoader)

# Create the RAFT model (will set up all model objects based on the design dict)
model = raft.Model(design)  

# Evaluate the system properties and equilibrium position before loads are applied
model.analyzeUnloaded()

# Compute natural frequencie
model.solveEigen()

# Simule the different load cases
model.analyzeCases(display=1)

# Plot the power spectral densities from the load cases
model.plotResponses()

# Visualize the system in its most recently evaluated mean offset position
model.plot(hideGrid=True)

# Save the response to a given output folder
outFolder = './examples/'
model.saveResponses(path.join(outFolder, flNm))

plt.show()

# 0.02
# 12.37