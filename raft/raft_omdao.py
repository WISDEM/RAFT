# Bringing in the OpenMDAO nomenclature
import openmdao.api as om

# Import the top level RAFT model, not sure exactly which class that is
from raft.raft_model import Model

import numpy as np

# Number of grid points per member
# This could also be specified as a class option like 'n_members', or specified some other way
npts = 10

class RAFT_OMDAO(om.ExplicitComponent):
    """RAFT OpenMDAO Wrapper API"""

    def initialize(self):
        self.options.declare("n_members")

    def setup(self):
        self.add_input('rotor_F', val=np.zeros(3), units='N', desc='Rotor shaft force vector')
        self.add_input('rotor_M', val=np.zeros(3), units='N', desc='Rotor shaft moment vector')
        self.add_input('Hsig_wave', val=0.0, units='m', desc='Significant wave height')

        n_members = self.options['n_members']
        for k in range(n_members):
            self.add_input(f'member{k}:s', val=np.zeros(npts), desc='Non-dimentional grid along member axis')
            self.add_input(f'member{k}:diameter', val=np.zeros(npts), units='m', desc='Member diameter at grid points')
            # and so on with other properties that RAFT needs

        self.add_output('nacelle_acceleration', val=0.0, units='m/s/s', desc='Maximum nacelle acceleration')

    def compute(self, inputs, outputs):
        n_members = self.options['n_members']
        design = {}
        for k in range(n_members):
            design[k] = inputs[f'member{k}:s']

        model = raft.Model(design, w=w, depth=depth)  # set up model

        model.setEnv(Hs=float(inputs['Hsig_wave']),
                     Tp=float(inputs['Tsig_wave']),
                     V=float(inputs['Uref']),
                     Fthrust=inputs['rotor_F'][0])  # set basic wave and wind info

        model.calcSystemProps()          # get all the setup calculations done within the model

        model.solveEigen()

        model.calcMooringAndOffsets()    # calculate the offsets for the given loading

        model.solveDynamics()            # put everything together and iteratively solve the dynamic response

        # Dump the outputs from RAFT into the OpenMDAO structure
        outputs['nacelle_acceleration'] = model.output['nacelle_acceleration']
        #...
        # or maybe something like
        for k in model.output:
            outputs[k] = model.output[k]
