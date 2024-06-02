import openmdao.api as om
import os
import yaml

from raft.omdao_raft import RAFT_Group

'''
This example runs RAFT given the exact inputs and options that WEIS provides RAFT in 15_RAFT_Studies
To generate the inputs and options, set the DEBUG_OMDAO flag to true in omdao_raft.py
To debug a specific instance of the opendmdao RAFT run, set the options and input files below using absolute paths
'''

weis_options_file = 'weis_options.yaml'
weis_inputs_file = 'weis_inputs.yaml'

# -----------------------------------
# OMDAO
# -----------------------------------

def test_omdao_raft():
    this_dir = os.path.dirname(__file__)

    # Load options directly generated in WEIS
    with open(os.path.join(this_dir,'test_data',weis_options_file)) as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)

    prob = om.Problem()
    prob.model = RAFT_Group(modeling_options=opt['modeling_options'],
                            analysis_options=opt['analysis_options'],
                            turbine_options=opt['turbine_options'],
                            mooring_options=opt['mooring_options'],
                            member_options=opt['member_options'])
    prob.setup()

    # -------------------------
    # inputs
    # -------------------------
    # Load options directly generated in WEIS
    with open(os.path.join(this_dir,'test_data',weis_inputs_file)) as file:
        inputs = yaml.load(file, Loader=yaml.FullLoader)

    for key, val in inputs.items():
        prob[key] = val


    prob.run_model()

if __name__=="__main__":
    test_omdao_raft()

