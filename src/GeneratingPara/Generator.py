import torch
import os
import time
from src.GeneratingPara.ReadMol import PreProcess

def Generate_file(mol, model, bool_OAT=False, write_xyz=True, waterbox=False, print_time=False):
    inputs = PreProcess(mol)
    start_time = time.time()
    if torch.cuda.is_available():
        for name in inputs.keys():
            inputs[name] = inputs[name].to('cuda')
    with torch.no_grad():
        parameters = model(inputs)

    parameters['vdw'][:,1] = (parameters['vdw'][:,1]/10).pow(2)
    parameters['bond'][:,0] *= 100
    parameters['angle'][:,0] *= 10
    parameters['angle'][:,1] *= 180/10

    if parameters['angle'].shape[1] == 4:
        parameters['angle'][:,2] /= 10
    parameters['charge'] /= 10

    for name in parameters:
        parameters[name] = parameters[name].cpu().tolist()
        
    # confId: the n-th conformation will be stored as .xyz file
    mol.write_key_file(parameters, mol.mol_name, bool_OAT, write_xyz, waterbox, confId=0)
    if print_time:
        print(mol.mol_name, time.time() - start_time, ' s')
    return None
