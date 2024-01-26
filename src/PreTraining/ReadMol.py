import os
import re
import numpy as np
import h5py
import itertools
import pandas as pd
import random
import copy
import torch

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') #hiding the warning messages

class Molecule():
    def __init__(self, mol_name, dataset, GAFF, bond_morse=False, bool_ub=False, max_len=100, max_ring=15, print_info=False):
        self.coord_inputs = None
        self.paras_GAFF = {}
        self.paras_index = None
        self.max_len = max_len # maximum number of atoms in a molecule
        self.max_ring = max_ring
        self.mol_name = mol_name
        self.bond_morse = bond_morse
        self.bool_ub = bool_ub
        assert dataset == 'ANI-1'
        self.dataset = dataset # name of dataset
        self.root = os.path.join('./data/', dataset)
        self.pdb_path = os.path.join(self.root,'pdb') 
        self.ac_path = os.path.join(self.root,'GAFF_ac2')
        self.frcmod_path = os.path.join(self.root,'frcmod')
        
        self.read_ac_file()
        if self.read_pdb_file(GAFF):
            self.existe = False
        else:
            self.existe = True
            if print_info:
                self.info()

    def info(self):
        print(("\n[===========] \x1b[91mMolecule Information\x1b[0m [===========]\n"))
        print('For molecule: ', self.mol_name, 'in', self.dataset)
        print('SMILES: ', self.smiles)
        print('Number of atoms:                  ', self.nb_atom)
        print('Number of bonds:                  ', len(self.bond))
        print('Number of angles:                 ', len(self.angle))
        print('Number of improper torsions:      ', len(self.imptors))
        print('Number of torsions:               ', len(self.torsion))
        print('Number of non bonded interaction: ', len(self.non_bonded))
        print('Number of all Parameters:         ', self.nb_para)
        print(("[====================] \x1b[91mEnd\x1b[0m [====================]\n"))
        
    def read_ac_file(self):
        filename = self.mol_name + '.ac'
        atom_type = []
        charge = []
        openfile = open(os.path.join(self.ac_path,filename), 'r')
        data = openfile.read().splitlines()
        for line in data:
            s = line.split()
            if len(s) > 7 and s[0] == 'ATOM':
                at = s[-1]
                atom_type.append(at)
                charge.append(float(s[-2]) * 10)
        openfile.close() 
        self.atom_type = atom_type
        self.paras_GAFF['charge'] = np.array(charge).reshape(-1,1)
    
    def read_pdb_file(self, GAFF):
        split_num = re.split(r'(_)',self.mol_name)
        s_num = split_num[0][-1]
        with h5py.File(self.root+'/ANI-1_release/ani_gdb_s0'+s_num+'.h5', 'r') as f:
            smiles = ''
            for j in f['gdb11_s0'+s_num]['gdb11_s0'+s_num+'-'+split_num[-1]]['smiles'][:]:
                smiles += j.decode()
            self.smiles = smiles
            energy = f['gdb11_s0'+s_num]['gdb11_s0'+s_num+'-'+split_num[-1]]['energies'][:]
        self.paras_GAFF['energy'] = (np.array(energy) - energy[0]) * 627.5094740931 # Energy unit: 1Hartree = 627.509... Kcal/mol
        self.nb_shot = len(self.paras_GAFF['energy'])
        
        filename = os.path.join(self.pdb_path,self.mol_name+'.pdb')
        TmpMol = AllChem.MolFromPDBFile(filename,removeHs=False)
        if TmpMol is None:
            print('Fail to read molecule by RDKIT!!!', self.mol_name)
            return 1
        template = AllChem.MolFromSmiles(self.smiles, sanitize=True)
        template = Chem.AddHs(template)
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(template, TmpMol)
        except:
            print(self.mol_name, 'wrong assign bond order')
            return 1

        if self.process_mol_with_RDKit(mol, template):
            print(self.mol_name, 'fail to process molecule')
            return 1
        
        missing_para = self.get_missing_para(GAFF)
        if missing_para is None:
            print(self.mol_name, 'fail to read frcmod file')
            return 1
        
        self.atom = self.get_atom_list(mol)
        self.bond, self.paras_GAFF['bond'] = self.get_bond_list(mol, GAFF, missing_para)
        self.angle, self.paras_GAFF['angle'] = self.get_angle_list(mol, GAFF, missing_para)
        self.imptors, self.paras_GAFF['imptors'] = self.get_imptors_list(mol, GAFF, missing_para)
        self.torsion, self.paras_GAFF['torsion'] = self.get_torsion_list(mol, GAFF, missing_para)
        self.non_bonded, self.vdw14, self.charge14, self.paras_GAFF['vdw']= self.get_non_bonded_list(mol, GAFF)
        self.nb_atom = len(self.atom)
        self.nb_para = 3 * len(self.atom)  + len(self.imptors) + 4*len(self.torsion)
        if self.bool_ub:
            self.nb_para += 4*len(self.angle) + 3*len(self.bond)
        else:
            self.nb_para += 2*len(self.angle) + 2*len(self.bond)
                
    def get_missing_para(self, GAFF):
        missing_para = {}
        filename = self.frcmod_path + '/' + self.mol_name
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'r') as file:
            current_section = ''
            for line in file:
                line = line.strip()
                if line.startswith('BOND') or line.startswith('ANGLE') or line.startswith('DIHE') or line.startswith('IMPROPER'):
                    current_section = line.lower()
                    if current_section not in missing_para:
                        missing_para[current_section] = {}
                elif line.startswith('NONBON'):
                    current_section = line.lower()
                elif line:
                    if current_section == 'bond':
                        values = line[7:].split()
                        name = line[:7].replace(' ','')
                        split_name = re.split(r'-', name)
                        new_name = split_name[1] + '-' + split_name[0]
                        
                        if self.bool_ub:
                            missing_para[current_section][name] = [float(values[0])/100, float(values[1]), 2]
                            missing_para[current_section][new_name] = [float(values[0])/100, float(values[1]), 2]
                        else:
                            missing_para[current_section][name] = [float(values[0])/100, float(values[1])]
                            missing_para[current_section][new_name] = [float(values[0])/100, float(values[1])]
                    elif current_section == 'angle':
                        values = line[10:].split()
                        name = line[:10].replace(' ','')
                        split_name = re.split(r'-', name)
                        new_name = split_name[2] + '-' + split_name[1] + '-' + split_name[0]
                        missing_para[current_section][name] = [float(values[0])/10, float(values[1])/180*10]
                        missing_para[current_section][new_name] = [float(values[0])/10, float(values[1])/180*10]
                    elif current_section == 'improper':
                        values = line[13:].split()
                        name = line[:13].replace(' ','')
                        missing_para[current_section][name] = [float(values[0])]
                    elif current_section == 'dihe':
                        values = line[13:].split()
                        name = line[:13].replace(' ','')
                        if name in missing_para[current_section]:
                            missing_para[current_section][name].append([float(values[0]), float(values[1]), float(values[2]), float(values[3])])
                        else:
                            missing_para[current_section][name] = [[float(values[0]), float(values[1]), float(values[2]), float(values[3])]]
                    
        all_torsion_name = []
        for name in missing_para['dihe']:
            torsion_para = [0., 0., 0., 0.]
            for all_para in missing_para['dihe'][name]:
                para2 = -1 if all_para[2] == 180. else 1
                para1 = all_para[1] / all_para[0] * para2
                PN = int(abs(all_para[3]))
                torsion_para[PN-1] = para1
            missing_para['dihe'][name] = torsion_para
            all_torsion_name.append(name)
            
        for name in all_torsion_name:
            split_name = re.split(r'-', name)
            new_name = split_name[3] + '-' + split_name[2] + '-' + split_name[1] + '-' + split_name[0]
            if new_name not in missing_para['dihe']:
                missing_para['dihe'][new_name] = missing_para['dihe'][name]
    
        return missing_para
    
    def get_paras_index(self):
        if self.paras_index is None:
            paras_index = {}
            paras_index['atom'] = np.arange(self.nb_atom).astype(int)
            paras_index['bond'] = np.array(self.bond).astype(int)
            paras_index['angle'] = np.array(self.angle).astype(int)
            paras_index['imptors'] = np.array(self.imptors).astype(int)
            paras_index['torsion'] = np.array(self.torsion).astype(int)
            paras_index['non-bonded'] = np.array(self.non_bonded).astype(int)
            return paras_index
        else:
            return self.paras_index
        
    def get_coord_inputs(self):
        split_num = re.split(r'(_)',self.mol_name)
        s_num = split_num[0][-1]
        with h5py.File(self.root+'/ANI-1_release/ani_gdb_s0'+s_num+'.h5', 'r') as f:
            coord = f['gdb11_s0'+s_num]['gdb11_s0'+s_num+'-'+split_num[-1]]['coordinates'][:]
        assert self.nb_shot == len(coord)
        coord_x = coord[:,:,0]
        coord_y = coord[:,:,1]
        coord_z = coord[:,:,2]

        coord_inputs = {'vdw14':self.vdw14, 'charge14':self.charge14}
        node_len = self.nodes_features.shape[1]
        edge_len = self.edges_features.shape[1]

        atom_index = np.arange(self.nb_atom).astype(int)
        bond_index = np.array(self.bond).T.astype(int)
        angle_index = np.array(self.angle).T.astype(int)
        imptors_index = np.array(self.imptors).T.astype(int)
        torsion_index = np.array(self.torsion).T.astype(int)  
        non_bonded_index = np.array(self.non_bonded).T.astype(int)
        coord_inputs['non-bonded'] = non_bonded_index

        # pre-compute d(E_bond)/d(L), d(E_agnle)/d(theta), ...
        # for bond
        x12 = np.take_along_axis(coord_x, bond_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_x, bond_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        y12 = np.take_along_axis(coord_y, bond_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_y, bond_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        z12 = np.take_along_axis(coord_z, bond_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_z, bond_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        length_bond = np.sqrt(np.power(x12,2) + np.power(y12,2) + np.power(z12,2)) # nb_shot, nb_bond
        coord_inputs['length_bond'] = length_bond

        # for angle 
        x12 = np.take_along_axis(coord_x, angle_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_x, angle_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        y12 = np.take_along_axis(coord_y, angle_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_y, angle_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        z12 = np.take_along_axis(coord_z, angle_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_z, angle_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        x32 = np.take_along_axis(coord_x, angle_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_x, angle_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        y32 = np.take_along_axis(coord_y, angle_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_y, angle_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        z32 = np.take_along_axis(coord_z, angle_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_z, angle_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        cos_angle1 = x12 * x32 + y12 * y32 + z12 * z32
        r12_2 = np.power(x12,2) + np.power(y12,2) + np.power(z12,2)
        r32_2 = np.power(x32,2) + np.power(y32,2) + np.power(z32,2)
        cos_theta_angle = cos_angle1 / np.sqrt(r12_2) / np.sqrt(r32_2)
        cos_theta_angle = np.clip(cos_theta_angle, -1, 1)
        theta_angle = np.arccos(cos_theta_angle)  # nb_shot, nb_angle
        coord_inputs['theta_angle'] = theta_angle
        if self.bool_ub:
            x13 = np.take_along_axis(coord_x, angle_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_x, angle_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
            y13 = np.take_along_axis(coord_y, angle_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_y, angle_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
            z13 = np.take_along_axis(coord_z, angle_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_z, angle_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
            length_angle = np.sqrt(np.power(x13,2) + np.power(y13,2) + np.power(z13,2))
            coord_inputs['length_angle'] = length_angle
        
        # for vdw
        x12 = np.take_along_axis(coord_x, non_bonded_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_x, non_bonded_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        y12 = np.take_along_axis(coord_y, non_bonded_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_y, non_bonded_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        z12 = np.take_along_axis(coord_z, non_bonded_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) - np.take_along_axis(coord_z, non_bonded_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        length_vdw = np.sqrt(np.power(x12,2) + np.power(y12,2) + np.power(z12,2)) # nb_shot, nb_vdw
        coord_inputs['length_vdw'] = length_vdw

        # for torsion  https://github.com/TinkerTools/tinker/blob/release/source/etors1.f
        x1 = np.take_along_axis(coord_x, torsion_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) # four points for torsion 
        y1 = np.take_along_axis(coord_y, torsion_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        z1 = np.take_along_axis(coord_z, torsion_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        x2 = np.take_along_axis(coord_x, torsion_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        y2 = np.take_along_axis(coord_y, torsion_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        z2 = np.take_along_axis(coord_z, torsion_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        x3 = np.take_along_axis(coord_x, torsion_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        y3 = np.take_along_axis(coord_y, torsion_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        z3 = np.take_along_axis(coord_z, torsion_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        x4 = np.take_along_axis(coord_x, torsion_index[3].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        y4 = np.take_along_axis(coord_y, torsion_index[3].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        z4 = np.take_along_axis(coord_z, torsion_index[3].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)

        x12 = x1 - x2 # -xba
        y12 = y1 - y2 # -yba
        z12 = z1 - z2 # -zba
        x13 = x1 - x3 # -xca
        y13 = y1 - y3 # -yca
        z13 = z1 - z3 # -zca
        x23 = x2 - x3 # -xcb
        y23 = y2 - y3 # -ycb
        z23 = z2 - z3 # -zcb
        x24 = x2 - x4 # -xdb
        y24 = y2 - y4 # -ydb
        z24 = z2 - z4 # -zdb
        x34 = x3 - x4 # -xdc
        y34 = y3 - y4 # -ydc
        z34 = z3 - z4 # -zdc
        x123 = y12 * z23 - z12 * y23 # xt
        y123 = z12 * x23 - x12 * z23 # yt
        z123 = x12 * y23 - y12 * x23 # zt
        x234 = y23 * z34 - z23 * y34 # xu
        y234 = z23 * x34 - x23 * z34 # yu
        z234 = x23 * y34 - y23 * x34 # zu
        x1234 = y234 * z123 - y123 * z234  # -xtu
        y1234 = z234 * x123 - z123 * x234  # -ytu
        z1234 = x234 * y123 - x123 * y234  # -ztu
        r23 = np.sqrt(np.power(x23,2) + np.power(y23,2) + np.power(z23,2))
        r12_2 = np.power(x12,2) + np.power(y12,2) + np.power(z12,2)
        r23_2 = np.power(x23,2) + np.power(y23,2) + np.power(z23,2)
        r34_2 = np.power(x34,2) + np.power(y34,2) + np.power(z34,2)
        r123_2 = np.power(x123,2) + np.power(y123,2) + np.power(z123,2)
        r234_2 = np.power(x234,2) + np.power(y234,2) + np.power(z234,2)

        sin_cos_n_theta = np.zeros((self.nb_shot, len(self.torsion), 8))
        sin_cos_n_theta[:,:,0] = (x23 * x1234 + y23 * y1234 + z23 * z1234) / np.sqrt(r123_2 * r234_2)/ r23 # sin(theta)
        sin_cos_n_theta[:,:,1] = (x123 * x234 + y123 * y234 + z123 * z234) / np.sqrt(r123_2 * r234_2) # cos(theta)
        sin_cos_n_theta[:,:,2] = 2. * sin_cos_n_theta[:,:,0] * sin_cos_n_theta[:,:,1] # sin(2*theta)
        sin_cos_n_theta[:,:,3] = 2. * sin_cos_n_theta[:,:,1] * sin_cos_n_theta[:,:,1] - 1. # cos(2*theta)
        sin_cos_n_theta[:,:,4] = sin_cos_n_theta[:,:,0] * sin_cos_n_theta[:,:,3] + sin_cos_n_theta[:,:,2] * sin_cos_n_theta[:,:,1] # sin(3*theta)
        sin_cos_n_theta[:,:,5] = sin_cos_n_theta[:,:,1] * sin_cos_n_theta[:,:,3] - sin_cos_n_theta[:,:,0] * sin_cos_n_theta[:,:,2] # cos(3*theta)
        sin_cos_n_theta[:,:,6] = 2. * sin_cos_n_theta[:,:,2] * sin_cos_n_theta[:,:,3] # sin(4*theta)
        sin_cos_n_theta[:,:,7] = 2. * sin_cos_n_theta[:,:,3] * sin_cos_n_theta[:,:,3] - 1. # cos(4*theta)
        coord_inputs['sin_cos_n_theta_torsion'] = sin_cos_n_theta     
        
        # for improper torsion https://github.com/TinkerTools/tinker/blob/release/source/eimptor1.f
        x1 = np.take_along_axis(coord_x, imptors_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) # four points for torsion 
        y1 = np.take_along_axis(coord_y, imptors_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        z1 = np.take_along_axis(coord_z, imptors_index[0].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        x2 = np.take_along_axis(coord_x, imptors_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        y2 = np.take_along_axis(coord_y, imptors_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        z2 = np.take_along_axis(coord_z, imptors_index[1].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        x3 = np.take_along_axis(coord_x, imptors_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        y3 = np.take_along_axis(coord_y, imptors_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        z3 = np.take_along_axis(coord_z, imptors_index[2].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)
        x4 = np.take_along_axis(coord_x, imptors_index[3].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        y4 = np.take_along_axis(coord_y, imptors_index[3].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1) 
        z4 = np.take_along_axis(coord_z, imptors_index[3].reshape(1,-1).repeat(self.nb_shot,axis=0), axis=1)

        x12 = x1 - x2 # -xba
        y12 = y1 - y2 # -yba
        z12 = z1 - z2 # -zba
        x13 = x1 - x3 # -xca
        y13 = y1 - y3 # -yca
        z13 = z1 - z3 # -zca
        x23 = x2 - x3 # -xcb
        y23 = y2 - y3 # -ycb
        z23 = z2 - z3 # -zcb
        x24 = x2 - x4 # -xdb
        y24 = y2 - y4 # -ydb
        z24 = z2 - z4 # -zdb
        x34 = x3 - x4 # -xdc
        y34 = y3 - y4 # -ydc
        z34 = z3 - z4 # -zdc
        x123 = y12 * z23 - z12 * y23 # xt
        y123 = z12 * x23 - x12 * z23 # yt
        z123 = x12 * y23 - y12 * x23 # zt
        x234 = y23 * z34 - z23 * y34 # xu
        y234 = z23 * x34 - x23 * z34 # yu
        z234 = x23 * y34 - y23 * x34 # zu
        x1234 = y234 * z123 - y123 * z234  # -xtu
        y1234 = z234 * x123 - z123 * x234  # -ytu
        z1234 = x234 * y123 - x123 * y234  # -ztu
        r23 = np.sqrt(np.power(x23,2) + np.power(y23,2) + np.power(z23,2))
        r123_2 = np.power(x123,2) + np.power(y123,2) + np.power(z123,2)
        r234_2 = np.power(x234,2) + np.power(y234,2) + np.power(z234,2)

        cos_imptors = (x123 * x234 + y123 * y234 + z123 * z234) / np.sqrt(r123_2 * r234_2)
        sin_imptors = (x23 * x1234 + y23 * y1234 + z23 * z1234) / np.sqrt(r123_2 * r234_2) / r23
        cos2_imptors = 2. * cos_imptors * cos_imptors - 1.
        sin2_imptors = 2. * sin_imptors * cos_imptors
        coord_inputs['cos2_imptors'] = cos2_imptors

        return coord_inputs
    
    def find_edge_index(self, i, j):
        for index in range(len(self.edges)):
            if i in self.edges[index] and j in self.edges[index]:
                if i == self.edges[index][0] and j == self.edges[index][1]:
                    return 2 * index, 2 * index + 1
                else:
                    return 2 * index + 1, 2 * index
        print(i,j,self.edges)
        raise RuntimeError('Fail to find correspopnding edge')
        
    def get_atom_list(self, mol=None):
        if mol is not None:
            atom_list = []
            for atom in mol.GetAtoms():
                atom_list.append([800+atom.GetIdx(), atom.GetSymbol(), '"'+atom.GetSymbol()+'"', atom.GetAtomicNum(), atom.GetMass(), atom.GetDegree()])
            atom_list = pd.DataFrame(atom_list, columns=['atom_index','atom_type','atom_index2','nb_proton','relative_mass','nb_linked_atom'])
            return atom_list
        else:
            return self.atom
            
    def get_bond_list(self, mol=None, GAFF=None, missing_para=None):
        if mol is not None:
            bond_list = []
            bond_para = []
            for bond in mol.GetBonds():
                idx1 = bond.GetBeginAtomIdx()
                idx2 = bond.GetEndAtomIdx()
                bond_type = self.atom_type[idx1]+'-'+self.atom_type[idx2]
                if bond_type in GAFF['bond']:
                    bond_list.append([idx1,idx2])
                    bond_para.append(GAFF['bond'][bond_type])
                elif bond_type in missing_para['bond']:
                    bond_list.append([idx1,idx2])
                    bond_para.append(missing_para['bond'][bond_type])
                else:
                    raise RuntimeError('Unknown bond type: ', self.mol_name, bond_type, self.smiles)
            bond_para = np.array(bond_para)
            bond_list = pd.DataFrame(bond_list, columns=['idx1', 'idx2'])
            return bond_list, bond_para
        else:
            return self.bond, self.paras_GAFF['bond']
        
    def get_angle_list(self, mol=None, GAFF=None, missing_para=None):
        if mol is not None:
            angle_list = []
            angle_para = []
            for atom in mol.GetAtoms():
                idx2 = atom.GetIdx()
                if atom.GetDegree() < 2:
                    continue
                neighbors = [i.GetIdx() for i in atom.GetNeighbors()]
                for i in range(len(neighbors)-1):
                    for j in range(i+1,len(neighbors)):
                        idx1 = neighbors[i]
                        idx3 = neighbors[j]
                        angle_type = self.atom_type[idx1]+'-'+self.atom_type[idx2]+'-'+self.atom_type[idx3]
                        if angle_type in GAFF['angle']:
                            angle_list.append([idx1, idx2, idx3])
                            angle_para.append(GAFF['angle'][angle_type])
                        elif angle_type in missing_para['angle']:
                            angle_list.append([idx1, idx2, idx3])
                            angle_para.append(missing_para['angle'][angle_type])
                        else:
                            raise RuntimeError('Unknown angle type: ', self.mol_name, angle_type, self.smiles)
                            
#                         compute the UB term
                        if self.bool_ub:
                            bond_type1 = self.atom_type[idx1]+'-'+self.atom_type[idx2]
                            bond_type2 = self.atom_type[idx2]+'-'+self.atom_type[idx3]
                            if bond_type1 in GAFF['bond']:
                                bond_length1 = GAFF['bond'][bond_type1][1]
                            elif bond_type1 in missing_para['bond']:
                                bond_length1 = missing_para['bond'][bond_type1][1]
                            else:
                                raise RuntimeError('Unknown bond type, fail to assign UB length')
                            if bond_type2 in GAFF['bond']:
                                bond_length2 = GAFF['bond'][bond_type2][1]
                            elif bond_type2 in missing_para['bond']:
                                bond_length2 = missing_para['bond'][bond_type2][1]
                            else:
                                raise RuntimeError('Unknown bond type, fail to assign UB length')
                            UB_length = np.sqrt(bond_length1**2+bond_length2**2-2*bond_length1*bond_length2*np.cos(angle_para[-1][1]/10*np.pi))
                            angle_para[-1] = angle_para[-1] + [1, UB_length]
                        
            angle_list = pd.DataFrame(angle_list, columns=['idx1', 'idx2', 'idx3'])
            angle_para = np.array(angle_para)
            return angle_list, angle_para
        else:
            return self.angle, self.paras_GAFF['angle']

    def get_imptors_list(self, mol=None, GAFF=None, missing_para=None):
        if mol is not None:
            imptors_list = []
            imptors_para = []
            imptors_dict = GAFF['imptors']
            for atom in mol.GetAtoms():
                idx3 = atom.GetIdx()
                if atom.GetDegree() != 3: # as the coefficients for atom.GetDegree()>4 are always 0, we just ignore those terms
                    continue

                num_H = 0
                H_idx = []
                other_idx = []
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1: # for H ydrogen atom
                        num_H += 1
                        H_idx.append(neighbor.GetIdx())
                    else:
                        other_idx.append(neighbor.GetIdx())
                if num_H == 1:
                    idx1,idx2,idx4 = [other_idx[0], other_idx[1], H_idx[0]]
                elif num_H == 2:
                    idx1,idx2,idx4 = [H_idx[0], H_idx[1], other_idx[0]]
                else:
                    idx_list = H_idx if H_idx else other_idx
                    idx1,idx2,idx4 = [idx_list[0], idx_list[1], idx_list[2]]
                
                final_composition = None
                final_type = None
                At1 = self.atom_type[idx1]
                At2 = self.atom_type[idx2]
                At3 = self.atom_type[idx3]
                At4 = self.atom_type[idx4]
                for name in imptors_dict:
                    at1, at2, at3, at4 = re.split(r'-', name)
                    if at3 != At3:
                        continue
                    if at4 == At1:
                        if (at1 == At2 or at1 == 'X') and (at2 == At4 or at2 =='X'):
                            final_composition = [idx2, idx4, idx3, idx1]
                            final_type = name
                        elif (at1 == At4 or at1 == 'X') and (at2 == At2 or at2 =='X'):
                            final_composition = [idx4, idx2, idx3, idx1]
                            final_type = name
                    elif at4 == At2:
                        if (at1 == At1 or at1 == 'X') and (at2 == At4 or at2 =='X'):
                            final_composition = [idx1, idx4, idx3, idx2]
                            final_type = name
                        elif (at1 == At4 or at1 == 'X') and (at2 == At1 or at2 =='X'):
                            final_composition = [idx4, idx1, idx3, idx2]
                            final_type = name
                    elif at4 == At4:
                        if (at1 == At1 or at1 == 'X') and (at2 == At2 or at2 =='X'):
                            final_composition = [idx1, idx2, idx3, idx4]
                            final_type = name
                        elif (at1 == At2 or at1 == 'X') and (at2 == At1 or at2 =='X'):
                            final_composition = [idx2, idx1, idx3, idx4]
                            final_type = name

                if final_composition is not None:
                    imptors_para.append(imptors_dict[final_type])
                    imptors_list.append(final_composition)
                else:
                    for new_idx1, new_idx2, new_idx4 in list(itertools.permutations([idx1,idx2,idx4],3)):
                        name = self.atom_type[new_idx1] + '-' + self.atom_type[new_idx2] + '-' + At3 + '-' + self.atom_type[new_idx4]
                        if name in missing_para['improper']:
                            final_composition = [new_idx1, new_idx2, idx3, new_idx4]
                            final_type = name
                            break
                    if final_composition is not None:
                        imptors_para.append(missing_para['improper'][final_type])
                        imptors_list.append(final_composition)
                    else:
                        pass
#                         raise RuntimeError('Unknown imptors type: ',self.mol_name, At1+'-'+At2+'-'+At3+'-'+At4,self.smiles)

            imptors_list = pd.DataFrame(imptors_list, columns=['idx1', 'idx2', 'idx3','idx4'])
            imptors_para = np.array(imptors_para)
            return imptors_list, imptors_para
        else:
            return self.imptors, self.paras_GAFF['imptors']
                        
    def get_torsion_list(self, mol=None, GAFF=None, missing_para=None):
        if mol is not None:
            torsion_list = []
            torsion_para = []
            torsion_dict = GAFF['torsion']
            for bond_center in mol.GetBonds():
                idx2 = bond_center.GetBeginAtomIdx()
                idx3 = bond_center.GetEndAtomIdx()
                Atom2 = mol.GetAtomWithIdx(idx2)
                Atom3 = mol.GetAtomWithIdx(idx3)

                if Atom2.GetDegree() < 2 or Atom3.GetDegree() < 2:
                    continue

                for b1 in Atom2.GetBonds():
                    if b1.GetIdx() == bond_center.GetIdx():
                        continue
                    idx1 = b1.GetOtherAtomIdx(idx2)
                    for b2 in Atom3.GetBonds():
                        if ((b2.GetIdx() == bond_center.GetIdx()) or (b2.GetIdx() == b1.GetIdx())):
                            continue
                        idx4 = b2.GetOtherAtomIdx(idx3)
                        # skip 3-membered rings
                        if idx4 == idx1:
                            continue
                        
                        At1 = self.atom_type[idx1]
                        At2 = self.atom_type[idx2]
                        At3 = self.atom_type[idx3]
                        At4 = self.atom_type[idx4]
                        final_type = None
                        for name in torsion_dict:
                            at1, at2, at3, at4 = re.split(r'-', name)
                            if (at1 == At1 or at1 == 'X') and at2 == At2 and at3 == At3 and (at4 == At4 or at4 =='X'):
                                final_type = name
                            elif (at1 == At4 or at1 == 'X') and at2 == At3 and at3 == At2 and (at4 == At1 or at4 =='X'):
                                final_type = name
                        if final_type is not None:
                            torsion_para.append(torsion_dict[final_type])
                            torsion_list.append([idx1, idx2, idx3, idx4])
                        else:
                            if At1 + '-' + At2 + '-' + At3 + '-' + At4 in missing_para['dihe']:
                                torsion_para.append(missing_para['dihe'][At1 + '-' + At2 + '-' + At3 + '-' + At4])
                                torsion_list.append([idx1, idx2, idx3, idx4])
                            else:
                                raise RuntimeError('Unknown torsion type: ',self.mol_name, At1+'-'+At2+'-'+At3+'-'+At4,self.smiles)
            torsion_list = pd.DataFrame(torsion_list, columns=['idx1', 'idx2', 'idx3', 'idx4'])
            torsion_para = np.array(torsion_para)
            
            return torsion_list, torsion_para
        else:
            return self.torsion, self.paras_GAFF['torsion']

    def get_non_bonded_list(self, mol=None, GAFF=None):
        if mol is not None:
            non_bonded_list = []
            bonded = []
            for i in range(len(self.bond)):
                tmp = [self.bond['idx1'][i], self.bond['idx2'][i]] if self.bond['idx1'][i] < self.bond['idx2'][i] else [self.bond['idx2'][i], self.bond['idx1'][i]]
                if tmp not in bonded:
                    bonded.append(tmp)
            for i in range(len(self.angle)):
                tmp = [self.angle['idx1'][i], self.angle['idx3'][i]] if self.angle['idx1'][i] < self.angle['idx3'][i] else [self.angle['idx3'][i], self.angle['idx1'][i]]
                if tmp not in bonded:
                    bonded.append(tmp)
            for i in range(len(self.atom)):
                for j in range(i+1,len(self.atom)):
                    tmp = [i,j]
                    if tmp not in bonded:
                        non_bonded_list.append(tmp)

            vdw_para = []
            for i in self.atom_type:
                vdw_para.append(GAFF['vdw'][i])
            vdw_para = np.array(vdw_para)
            
            vdw14 = []
            charge14 = []
            torsion14 = []
            const_charge14 = 1/1.2
            const_vdw14 = 1/2
            for i in range(len(self.torsion)):
                tmp = [self.torsion['idx1'][i], self.torsion['idx4'][i]] if self.torsion['idx1'][i] < self.torsion['idx4'][i] else [self.torsion['idx4'][i], self.torsion['idx1'][i]]
                if tmp not in torsion14:
                    torsion14.append(tmp)
            for i in non_bonded_list:
                if i in torsion14:
                    vdw14.append(const_vdw14)
                    charge14.append(const_charge14)
                else:
                    vdw14.append(1.)
                    charge14.append(1.)
            non_bonded_list = pd.DataFrame(non_bonded_list, columns=['idx1', 'idx2'])
            return non_bonded_list, vdw14, charge14, vdw_para
        else:
            return self.non_bonded, self.vdw14, self.charge14, self.paras_GAFF['vdw']
    
    def process_mol_with_RDKit(self, mol, template):
        template2mol = list(mol.GetSubstructMatch(template))
        mol2template = []
        for i in range(len(template2mol)):
            for j in range(len(template2mol)):
                if template2mol[j] == i:
                    mol2template.append(j)
                    break
        if template2mol == []:
            mol2template = list(template.GetSubstructMatch(mol))
            template2mol = []
            for i in range(len(mol2template)):
                for j in range(len(mol2template)):
                    if mol2template[j] == i:
                        template2mol.append(j)
                        break
                        
        if template2mol == []:
            print('Fail to match template and mol!!!', self.mol_name)
            return 1
        
        assert len(mol2template) == len(template2mol), self.mol_name + 'missing atoms'
        nodes_features = []
        edges = []
        edges_features = []

        node_len = self.max_len
        edge_len = self.max_len + self.max_ring
        
        all_ele = ['PAD', 'UNK', 'H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Li', 'Na', 'Mg', 'K', 'Ca']
        ele2index = {j : i for i,j in enumerate(all_ele)}
        num_ele = len(all_ele)
        
        mol_atoms = mol.GetAtoms()
        template_atoms = template.GetAtoms()
        template_bonds = template.GetBonds()
        if len(mol_atoms) <= node_len and len(template_bonds) > 0:  
            for i in range(len(mol_atoms)): 
                atom = template_atoms[mol2template[i]]
                node_features = [0] * 7
                node_features[0] = ele2index[atom.GetSymbol() if atom.GetSymbol() in all_ele else 'UNK'] # atomic numebr [all_ele]
                node_features[1] = atom.GetDegree() if atom.GetDegree()<=5 else 6              # degree of atom [0~5, 6=other]
                node_features[2] = int(atom.IsInRing())            # whether in ring
                node_features[3] = int(atom.GetHybridization())  # hybridization type [unspecified,s,sp,sp2,sp3,sp3d,sp3d2,other]
                node_features[4] = int(atom.GetIsAromatic())      # whether aromatic  
                node_features[5] = int(atom.GetChiralTag())     # chirality [CHI_UNSPECIFIED,CHI_TETRAHEDRAL_CW,CHI_TETRAHEDRAL_CCW,CHI_OTHER]
                node_features[6] = 3 + int(atom.GetFormalCharge())     # formal charge -3, -2, -1, 0 ,1, 2, 3
                nodes_features.append(node_features)
            node_features = [0] * 7
            node_features[0] = ele2index['PAD']
            node_features[6] = 3
            for _ in range(node_len - len(mol_atoms)):
                nodes_features.append(node_features)

            for bond in template_bonds:
                edge = [template2mol[bond.GetBeginAtomIdx()], template2mol[bond.GetEndAtomIdx()]]
                edges.append(edge)

                edge_features = [0] * 6                         # First two places are indices of connected nodes, third place is used as [MASKED EDGE]
                edge_features[0] = edge[0]
                edge_features[1] = edge[1]
                bond_type = int(bond.GetBondType()) if (int(bond.GetBondType())<=3 or int(bond.GetBondType())==12) else 0
                if bond_type == 12:
                    bond_type = 4
                edge_features[2] = bond_type + 1                # bond type [PAD,OTHERS,SINGLE,DOUBLE,TRIPLE,AROMATIC]
                edge_features[3] = int(bond.GetIsConjugated())  # whether conjugation
                edge_features[4] = int(bond.IsInRing())         # whether in the ring
                edge_features[5] = int(bond.GetStereo())        # stereo type [STEREONONE,STEREOANY,STEREOZ,STEREOE,STEREOCIS,STEREOTRANS]

                edges_features.append(edge_features)
            edge_features = [0] * 6
            edge_features[2] = 0
            assert edge_len >= len(template_bonds), 'Too Much Bonds!!!'
            for _ in range(edge_len - len(template_bonds)):
                edges_features.append(edge_features)
        else:
            raise RuntimeError('Bad molecule to generate features!!!')
        
        self.nodes_features = np.array(nodes_features)
        self.edges_features = np.array(edges_features)
        self.edges = edges
        
    def get_inputs_features(self):
        return self.nodes_features, self.edges_features
    
    def get_edges(self):
        return self.edges
    
    def get_GAFF_params(self):
        return self.paras_GAFF

def PreProcess(mol, bond_morse=False, bool_ub=False):
    targets = {'vdw':[], 'bond':[], 'angle':[], 'imptors':[], 'torsion':[], 'charge':[], 'energy':[]}
    inputs = {'nodes_features':[], 'edges_features':[], 'atom':[], 'bond':[], 'angle':[], 'torsion':[], 'imptors':[]}
    all_coord_inputs = []
    mol_params_index = {'atom':[0], 'bond':[0], 'angle':[0], 'imptors':[0], 'torsion':[0], 'energy':[0]}
    for i in range(len(mol)):
        
        # for targets
        paras = mol[i].get_GAFF_params()
        for name in targets:
            targets[name] += paras[name].tolist()
        mol_params_index['energy'].append(len(targets['energy']))
        
        # for inputs
        nodes_features, edges_features = mol[i].get_inputs_features()
        node_len = nodes_features.shape[0]
        edge_len = edges_features.shape[0]
        inputs['nodes_features'].append(nodes_features.tolist())
        inputs['edges_features'].append(edges_features.tolist())

        paras_index = copy.deepcopy(mol[i].get_paras_index())
        
        # atom/vdw/charge index
        inputs['atom'] += (paras_index['atom'] + i * node_len).tolist()
        mol_params_index['atom'].append(len(inputs['atom']))
        
        # bond index
        tmp = np.zeros(paras_index['bond'].shape)
        tmp[:,:2] = i * node_len + paras_index['bond'][:,:2]
        inputs['bond'] += tmp.tolist()
        mol_params_index['bond'].append(len(inputs['bond']))
        
        # angle index
        tmp = np.zeros(paras_index['angle'].shape)
        tmp[:,:3] = i * node_len + paras_index['angle'][:,:3]
        inputs['angle'] += tmp.tolist()
        mol_params_index['angle'].append(len(inputs['angle']))
        
        # imptorsindex
        tmp = np.zeros(paras_index['imptors'].shape)
        tmp[:,:4] = i * node_len + paras_index['imptors'][:,:4]
        inputs['imptors'] += tmp.tolist()
        mol_params_index['imptors'].append(len(inputs['imptors']))
                
        # torsion index
        tmp = np.zeros(paras_index['torsion'].shape)
        tmp[:,:4] = i * node_len + paras_index['torsion'][:,:4]
        inputs['torsion'] += tmp.tolist()
        mol_params_index['torsion'].append(len(inputs['torsion']))
        
        coord_inputs = mol[i].get_coord_inputs()
        for name in coord_inputs.keys():
            if name == 'non-bonded':
                coord_inputs[name] = torch.LongTensor(coord_inputs[name])
            else:
                coord_inputs[name] = torch.Tensor(coord_inputs[name])
        all_coord_inputs.append(coord_inputs)
        
    if len(targets['bond']) == 0:
        if bool_ub:
            targets['bond'] = np.zeros((0,3))
        else:
            targets['bond'] = np.zeros((0,2))
    if len(targets['angle']) == 0:
        if bool_ub:
            targets['angle'] = np.zeros((0,4))
        else:
            targets['angle'] = np.zeros((0,2))
    if len(targets['imptors']) == 0:
        targets['imptors'] = np.zeros((0,1))
    if len(targets['torsion']) == 0:
        targets['torsion'] = np.zeros((0,6))
    for name in targets:
        targets[name] = torch.Tensor(targets[name])
        
    if len(inputs['bond']) == 0:
        inputs['bond'] = np.zeros((0,2))
    if len(inputs['angle']) == 0:
        inputs['angle'] = np.zeros((0,3))
    if len(inputs['imptors']) == 0:
        inputs['imptors'] = np.zeros((0,4))
    if len(inputs['torsion']) == 0:
        inputs['torsion'] = np.zeros((0,4))
        
    for name in inputs:
        inputs[name] = torch.LongTensor(inputs[name])
        if 'atom' not in name and 'features' not in name:
            inputs[name] = inputs[name].transpose(0,1)
    
    return [inputs, targets, all_coord_inputs, mol_params_index]

def construct_dataloader(mol, bool_random=False):
    if bool_random:
        random.shuffle(mol)
    bsz = 256
    num_dataloader = int(np.ceil(len(mol)/bsz))

    dataloader = []
    for i in range(num_dataloader):
        dataloader.append(PreProcess(mol[i * bsz: (i+1) * bsz]))

    return dataloader

def para_process_mol(i, all_name, dataset, GAFF, bond_morse=False, bool_ub=False, max_len=100, nb_time_processing=10, print_info=False):
    mol = []
    for j in range(i*nb_time_processing, min((i+1)*nb_time_processing,len(all_name))):
        name = all_name[j]
        one_mol = Molecule(name, dataset, GAFF, bond_morse, bool_ub, max_len, print_info=print_info)
        if one_mol.existe:
            mol.append(one_mol)

    return i, mol
