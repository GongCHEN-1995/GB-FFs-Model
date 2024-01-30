import os
import numpy as np
import torch
import copy
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') #hiding the warning messages

class Molecule():
    def __init__(self, mol_name, bond_morse=False, bool_ub=False, max_len=100, max_ring=15, print_info=True):
        self.coord_inputs = None
        self.paras_index = None
        self.max_len = max_len # maximum number of atoms in a molecule
        self.max_ring = max_ring
        self.mol_name = mol_name
        self.bond_morse = bond_morse
        self.bool_ub = bool_ub
        self.root = './GeneratingPara/'
        self.pdb_path = os.path.join(self.root,'pdb')
        self.sdf_path = os.path.join(self.root,'sdf')
        self.smiles_path = os.path.join(self.root,'smiles')
        self.xyz_path = os.path.join(self.root,'xyz')
        
        self.read_smiles_file()
        self.read_sdf_pdb_file()
        if print_info:
            self.info()

    def info(self):
        print(("\n[===========] \x1b[91mMolecule Information\x1b[0m [===========]\n"))
        print('SMILES: ', self.smiles)
        print('Number of atoms:                  ', self.nb_atom)
        print('Number of bonds:                  ', len(self.bond))
        print('Number of angles:                 ', len(self.angle))
        print('Number of improper torsions:      ', len(self.imptors))
        print('Number of torsions:               ', len(self.torsion))
        print('Number of non bonded interaction: ', len(self.non_bonded))
        print('Number of all Parameters:         ', self.nb_para)
        print(("[====================] \x1b[91mEnd\x1b[0m [====================]\n"))
        
    def read_smiles_file(self):
        filename = self.mol_name + '.txt'
        openfile = open(os.path.join(self.smiles_path,filename), 'r')
        data = openfile.read().splitlines()
        split_data = data[0].split()
        if 'SMILES' in split_data[0]: 
            self.smiles = split_data[1]
        else:
            self.smiles = split_data[0]
        openfile.close() 
            
    def get_coordinate(self, mol):
        coord_x = np.zeros((self.nb_shot, self.nb_atom))
        coord_y = np.zeros((self.nb_shot, self.nb_atom))
        coord_z = np.zeros((self.nb_shot, self.nb_atom))
        positions = mol.GetConformers()
        for i in range(self.nb_shot):
            position = positions[i].GetPositions()
            coord_x[i,:] = position[:,0]
            coord_y[i,:] = position[:,1]
            coord_z[i,:] = position[:,2]
        return coord_x, coord_y, coord_z
    
    def read_sdf_pdb_file(self):
        filename = os.path.join(self.sdf_path,self.mol_name+'.sdf')
        if not os.path.exists(filename):
            filename = os.path.join(self.pdb_path,self.mol_name+'.pdb')
            TmpMol = AllChem.MolFromPDBFile(filename,removeHs=False)
        else:
            TmpMol = AllChem.SDMolSupplier(filename,removeHs=False)[0]
            
        if TmpMol is None:
            raise RuntimeError(self.mol_name + ' Fail to read molecule by RDKIT!!!')
        
        template = AllChem.MolFromSmiles(self.smiles, sanitize=True)
        if template is None:
            template = mol = TmpMol
        else:
            template = Chem.AddHs(template)
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(template, TmpMol)
            except:
                print(self.mol_name+': wrong assign bond order! Using the coordinate molecule now.')
                template = mol = TmpMol
        self.nb_shot = len(mol.GetConformers())
        self.process_mol_with_RDKit(mol, template)
        
        self.atom = self.get_atom_list(mol)
        self.bond = self.get_bond_list(mol)
        self.angle = self.get_angle_list(mol)
        self.imptors = self.get_imptors_list(mol)
        self.torsion = self.get_torsion_list(mol)
        self.non_bonded = self.get_non_bonded_list(mol)
        self.nb_atom = len(self.atom)
        self.nb_para = 3 * len(self.atom)  + len(self.imptors) + 4*len(self.torsion)

        if self.bool_ub:
            self.nb_para += 3*len(self.bond) + 4*len(self.angle) 
        else:
            self.nb_para += 2*len(self.bond) + 2*len(self.angle)
            
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
                atom_list.append([atom.GetIdx(), atom.GetSymbol(), '"'+atom.GetSymbol()+'"', atom.GetAtomicNum(), atom.GetMass(), atom.GetDegree()])
            atom_list = pd.DataFrame(atom_list, columns=['atom_index','atom_type','atom_index2','nb_proton','relative_mass','nb_linked_atom'])
            return atom_list
        else:
            return self.atom
            
    def get_bond_list(self, mol=None):
        if mol is not None:
            bond_list = []
            for bond in mol.GetBonds():
                idx1 = bond.GetBeginAtomIdx()
                idx2 = bond.GetEndAtomIdx()
                bond_list.append([idx1,idx2])
                
            bond_list = pd.DataFrame(bond_list, columns=['idx1', 'idx2'])
            return bond_list
        else:
            return self.bond
        
    def get_angle_list(self, mol=None):
        if mol is not None:
            angle_list = []
            for atom in mol.GetAtoms():
                idx2 = atom.GetIdx()
                if atom.GetDegree() < 2:
                    continue
                neighbors = [i.GetIdx() for i in atom.GetNeighbors()]
                for i in range(len(neighbors)-1):
                    for j in range(i+1,len(neighbors)):
                        idx1 = neighbors[i]
                        idx3 = neighbors[j]
                        angle_list.append([idx1, idx2, idx3])
            angle_list = pd.DataFrame(angle_list, columns=['idx1', 'idx2', 'idx3'])
            return angle_list
        else:
            return self.angle

    def get_imptors_list(self, mol=None):
        if mol is not None:
            imptors_list = []
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

                imptors_list.append([idx2, idx1, idx3, idx4])
                
            imptors_list = pd.DataFrame(imptors_list, columns=['idx1', 'idx2', 'idx3','idx4'])
            return imptors_list
        else:
            return self.imptors
                        
    def get_torsion_list(self, mol=None):
        if mol is not None:
            torsion_list = []
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
                        torsion_list.append([idx1, idx2, idx3, idx4])
            torsion_list = pd.DataFrame(torsion_list, columns=['idx1', 'idx2', 'idx3', 'idx4'])
            
            return torsion_list
        else:
            return self.torsion

    def get_non_bonded_list(self, mol=None):
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
            
            non_bonded_list = pd.DataFrame(non_bonded_list, columns=['idx1', 'idx2'])
            return non_bonded_list
        else:
            return self.non_bonded
  
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
        
        all_ele = ['PAD', 'UNK', 'H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Na', 'Mg', 'K', 'Ca']
        ele2index = {j : i for i,j in enumerate(all_ele)}
        num_ele = len(all_ele)
        
        mol_atoms = mol.GetAtoms()
        if self.max_len == 0:
            self.max_len = len(mol_atoms)
        template_atoms = template.GetAtoms()
        template_bonds = template.GetBonds()
        node_len = self.max_len
        edge_len = self.max_len + self.max_ring
        
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
    
    def write_key_file(self, parameters, filename, bool_OAT=False, write_xyz=True, waterbox=False, confId=0):            
        if waterbox:
            with open('./GeneratingPara/waterbox/waterbox.prm', 'r') as file:
                lines = file.readlines()
        if waterbox:
            nb_init = 401
        else:
            nb_init = 349

        if bool_OAT:
            OAT = {}
            atom_type = []
            for i in range(len(self.atom)):
                bool_new_atom_type = True
                for j in OAT:
                    if  abs(OAT[j][2] - parameters['vdw'][i][0]) < 1e-4 and abs(OAT[j][3] - parameters['vdw'][i][1]) < 1e-4 and abs(OAT[j][4] - parameters['charge'][i][0]) < 1e-4:
                        atom_type.append(j)
                        bool_new_atom_type = False
                        break
                if bool_new_atom_type:
                    atom_type.append('AT' + str(len(OAT)))
                    OAT['AT' + str(len(OAT))] = [len(OAT), i, parameters['vdw'][i][0], parameters['vdw'][i][1], parameters['charge'][i][0]]
                    
        self.nb_init = nb_init
        prm = open(os.path.join(self.root,'key',filename+'.prm'), 'w+')
#         prm.write('parameters /users/home/chengo/ext/GPU/params/amber99 \n')
# verbose
# integrator              verlet
# a-axis                  50
# archive
# ewald
# run-mode                legacy

        prm.write(\
'''      ##############################
      ##                          ##
      ##  Force Field Definition  ##
      ##                          ##
      ##############################
        \n''')
        if not self.bond_morse:
            prm.write('forcefield              GB-FFs GAFF\n')
        elif self.bool_ub:
            prm.write('forcefield              GB-FFs UB\n')
        else:
            prm.write('forcefield              GB-FFs MORSE\n')
            
        if self.bond_morse:
            prm.write('bondtype                MORSE')
        else:
            prm.write('bondtype                HARMONIC')
        prm.write('''
vdwindex                TYPE
vdwtype                 LENNARD-JONES
radiusrule              ARITHMETIC
radiustype              R-min
radiussize              RADIUS
epsilonrule             GEOMETRIC
torsionunit             1.0
imptorunit              1.0
vdw-14-scale            2.0
chg-14-scale            1.2
electric                332.06
dielectric              1.0


      #############################
      ##                         ##
      ##  Atom Type Definitions  ##
      ##                         ##
      #############################

        ''')
        prm.write('\n')
        if bool_OAT:
            for key, values in OAT.items():
                prm.write('atom %10d %4d %5s %8s %10d %10.3f %5d \n' %
                    (nb_init+values[0], nb_init+values[0], self.atom['atom_type'][values[1]], self.atom['atom_index2'][values[1]],
                        self.atom['nb_proton'][values[1]], self.atom['relative_mass'][values[1]],self.atom['nb_linked_atom'][values[1]]))                
        else:
            for i in range(len(self.atom)):
                prm.write('atom %10d %4d %5s %8s %10d %10.3f %5d \n' %
                    (nb_init+self.atom['atom_index'][i], nb_init+self.atom['atom_index'][i], self.atom['atom_type'][i], self.atom['atom_index2'][i],
                        self.atom['nb_proton'][i], self.atom['relative_mass'][i],self.atom['nb_linked_atom'][i]))
        if waterbox:
            for line in lines:
                tokens = line.strip().split()
                if line[:5] == 'atom ':
                    prm.write(line)
        prm.write(
        '''


      ################################
      ##                            ##
      ##  Van der Waals Parameters  ##
      ##                            ##
      ################################

        ''')
        prm.write('\n')
        if bool_OAT:
            for key, values in OAT.items():
                prm.write('vdw %11d %16.4f %8.4f \n' %
                        (nb_init+values[0], parameters['vdw'][values[1]][0], parameters['vdw'][values[1]][1]))
        else:
            for i in range(len(self.atom)):
                prm.write('vdw %11d %16.4f %8.4f \n' %
                        (nb_init+self.atom['atom_index'][i], parameters['vdw'][i][0], parameters['vdw'][i][1]))
        if waterbox:
            for line in lines:
                tokens = line.strip().split()
                if line[:4] == 'vdw ':
                    prm.write(line)
        prm.write(
        '''


      ##################################
      ##                              ##
      ##  Bond Stretching Parameters  ##
      ##                              ##
      ##################################

        '''
        )
        prm.write('\n')
        if bool_OAT:
            bond_used = {}
            for i in range(len(self.bond)):
                a, b = self.bond['idx1'][i], self.bond['idx2'][i]
                if atom_type[a] + '-' + atom_type[b] in bond_used:
                    for j in range(len(parameters['bond'][i])):
                        assert abs(bond_used[atom_type[a] + '-' + atom_type[b]][j] - parameters['bond'][i][j]) < 5e-3
                    continue
                if self.bond_morse:
                    if self.bool_ub:
                        prm.write('morse %10d %4d %16.2f %8.4f %8.4f \n' %
                                  (nb_init+OAT[atom_type[a]][0], nb_init+OAT[atom_type[b]][0], parameters['bond'][i][0]/4*parameters['bond'][i][2]*parameters['bond'][i][2], parameters['bond'][i][1], parameters['bond'][i][2]))
                    else:
                        prm.write('morse %10d %4d %16.2f %8.4f \n' %
                                  (nb_init+OAT[atom_type[a]][0], nb_init+OAT[atom_type[b]][0], parameters['bond'][i][0], parameters['bond'][i][1]))
                else:
                    prm.write('bond %10d %4d %16.2f %8.4f \n' %
                              (nb_init+OAT[atom_type[a]][0], nb_init+OAT[atom_type[b]][0], parameters['bond'][i][0], parameters['bond'][i][1]))
                bond_used[atom_type[a] + '-' + atom_type[b]] = bond_used[atom_type[b] + '-' + atom_type[a]] = parameters['bond'][i]
        else:
            for i in range(len(self.bond)):
                if self.bond_morse:
                    if self.bool_ub:
                        prm.write('morse %10d %4d %16.2f %8.4f %8.4f \n' %
                                  (nb_init+self.bond['idx1'][i], nb_init+self.bond['idx2'][i], parameters['bond'][i][0]/4*parameters['bond'][i][2]*parameters['bond'][i][2], parameters['bond'][i][1], parameters['bond'][i][2]))
                    else:
                        prm.write('morse %10d %4d %16.2f %8.4f \n' %
                                  (nb_init+self.bond['idx1'][i], nb_init+self.bond['idx2'][i], parameters['bond'][i][0], parameters['bond'][i][1]))
                else:
                    prm.write('bond %10d %4d %16.2f %8.4f \n' %
                              (nb_init+self.bond['idx1'][i], nb_init+self.bond['idx2'][i], parameters['bond'][i][0], parameters['bond'][i][1]))
        if waterbox:
            for line in lines:
                tokens = line.strip().split()
                if (not self.bond_morse and line[:5] == 'bond ') or (self.bond_morse and line[:6] == 'morse '):
                    prm.write(line)                    
        prm.write(
        '''


      ################################
      ##                            ##
      ##  Angle Bending Parameters  ##
      ##                            ##
      ################################

        ''')
        prm.write('\n')
        if bool_OAT:
            angle_used = {}
            for i in range(len(self.angle)):
                a, b, c = self.angle['idx1'][i], self.angle['idx2'][i], self.angle['idx3'][i]
                if atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c] in angle_used:
                    for j in range(len(parameters['angle'][i])):
                        assert abs(angle_used[atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c]][j] - parameters['angle'][i][j]) < 1e-3
                    continue
                prm.write('angle %9d %4d %4d %8.2f %8.2f \n' %
                            (nb_init+OAT[atom_type[a]][0], nb_init+OAT[atom_type[b]][0], nb_init+OAT[atom_type[c]][0], parameters['angle'][i][0], parameters['angle'][i][1]))
                angle_used[atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c]] = angle_used[atom_type[c] + '-' + atom_type[b] + '-' + atom_type[a]] = parameters['angle'][i]
        else:
            for i in range(len(self.angle)):
                prm.write('angle %9d %4d %4d %8.2f %8.2f \n' %
                            (nb_init+self.angle['idx1'][i], nb_init+self.angle['idx2'][i], nb_init+self.angle['idx3'][i], parameters['angle'][i][0], parameters['angle'][i][1]))
        if waterbox:
            for line in lines:
                tokens = line.strip().split()
                if line[:6] == 'angle ':
                    prm.write(line)
        prm.write(
        '''


      ################################
      ##                            ##
      ##   Urey-Bradley Parameters  ##
      ##                            ##
      ################################
      
        ''')
        if self.bool_ub:
            prm.write('\n')
            if bool_OAT:
                angle_used = {}
                for i in range(len(self.angle)):
                    a, b, c = self.angle['idx1'][i], self.angle['idx2'][i], self.angle['idx3'][i]
                    if atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c] in angle_used:
                        for j in range(len(parameters['angle'][i])):
                            assert abs(angle_used[atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c]][j] - parameters['angle'][i][j]) < 1e-3
                        continue
                    prm.write('ureyquartic %9d %4d %4d %10.2f %8.2f \n' %
                                (nb_init+OAT[atom_type[a]][0], nb_init+OAT[atom_type[b]][0], nb_init+OAT[atom_type[c]][0], parameters['angle'][i][2], parameters['angle'][i][3]))
                    angle_used[atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c]] = angle_used[atom_type[c] + '-' + atom_type[b] + '-' + atom_type[a]] = parameters['angle'][i]
            else:
                for i in range(len(self.angle)):
                    prm.write('ureyquartic %9d %4d %4d %10.2f %8.2f \n' %
                                (nb_init+self.angle['idx1'][i], nb_init+self.angle['idx2'][i], nb_init+self.angle['idx3'][i], parameters['angle'][i][2], parameters['angle'][i][3]))
            if waterbox:
                for line in lines:
                    tokens = line.strip().split()
                    if line[:12] == 'ureyquartic ':
                        prm.write(line)
        prm.write(
        '''


      #####################################
      ##                                 ##
      ##  Improper Torsional Parameters  ##
      ##                                 ##
      #####################################


        ''')
        ### Impropers ###
        prm.write('\n')
        if not self.imptors.empty:
            if bool_OAT:
                imptors_used = {}
                for i in range(len(self.imptors)):
                    a, b, c, d = self.imptors['idx1'][i], self.imptors['idx2'][i], self.imptors['idx3'][i], self.imptors['idx4'][i]
                    if atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c] + '-' + atom_type[d] in imptors_used:
                        for j in range(len(parameters['imptors'][i])):
                            assert abs(imptors_used[atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c] + '-' + atom_type[d]][j] - parameters['imptors'][i][j]) < 1e-3
                        continue
                    prm.write('imptors %7d %4d %4d %4d %12.3f %4.1f %2d \n' %
                            (nb_init+OAT[atom_type[a]][0], nb_init+OAT[atom_type[b]][0], nb_init+OAT[atom_type[c]][0], nb_init+OAT[atom_type[d]][0], parameters['imptors'][i][0], 180.0, 2))
                    imptors_used[atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c] + '-' + atom_type[d]] = imptors_used[atom_type[a] + atom_type[d] + '-' + atom_type[c] + '-' + atom_type[b]] = \
                    imptors_used[atom_type[b] + '-' + atom_type[a] + '-' + atom_type[c] + '-' + atom_type[c]] = imptors_used[atom_type[b] + atom_type[d] + '-' + atom_type[c] + '-' + atom_type[a]] = \
                    imptors_used[atom_type[d] + '-' + atom_type[a] + '-' + atom_type[c] + '-' + atom_type[b]] = imptors_used[atom_type[d] + atom_type[b] + '-' + atom_type[c] + '-' + atom_type[a]] = parameters['imptors'][i]
            else:
                for i in range(len(self.imptors)):
                    prm.write('imptors %7d %4d %4d %4d %12.3f %4.1f %2d \n' %
                            (nb_init+self.imptors['idx1'][i], nb_init+self.imptors['idx2'][i], nb_init+self.imptors['idx3'][i], nb_init+self.imptors['idx4'][i], parameters['imptors'][i][0], 180.0, 2))
            
        prm.write(
        '''


      ############################
      ##                        ##
      ##  Torsional Parameters  ##
      ##                        ##
      ############################

        ''')
        prm.write('\n')
        if not self.torsion.empty:
            if bool_OAT:
                torsion_used = {}                 
                for i in range(len(self.torsion)):
                    a, b, c, d = self.torsion['idx1'][i], self.torsion['idx2'][i], self.torsion['idx3'][i], self.torsion['idx4'][i]
                    if atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c] + '-' + atom_type[d] in torsion_used:
                        for j in range(len(parameters['torsion'][i])):
                            assert abs(torsion_used[atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c] + '-' + atom_type[d]][j] - parameters['torsion'][i][j]) < 1e-3
                        continue
                    gamma = np.zeros(4)
                    value = np.zeros(4)
                    for j in range(4):
                        if parameters['torsion'][i][j] < 0:
                            gamma[j] = 180.0
                        value[j] = abs(parameters['torsion'][i][j])
                    prm.write('torsion %7d %4d %4d %4d %12.3f %5.1f %2d %6.3f %5.1f %2d %6.3f %5.1f %2d %6.3f %5.1f %2d \n' %
                            (nb_init+OAT[atom_type[a]][0], nb_init+OAT[atom_type[b]][0], nb_init+OAT[atom_type[c]][0], nb_init+OAT[atom_type[d]][0], value[0], gamma[0], 1, value[1], gamma[1], 2, value[2], gamma[2], 3, value[3], gamma[3], 4))
                    torsion_used[atom_type[a] + '-' + atom_type[b] + '-' + atom_type[c] + '-' + atom_type[d]] = torsion_used[atom_type[d] + atom_type[c] + '-' + atom_type[b] + '-' + atom_type[a]] = parameters['torsion'][i]
            else:
                for i in range(len(self.torsion)):
                    gamma = np.zeros(4)
                    value = np.zeros(4)
                    for j in range(4):
                        if parameters['torsion'][i][j] < 0:
                            gamma[j] = 180.0
                        value[j] = abs(parameters['torsion'][i][j])
                    prm.write('torsion %7d %4d %4d %4d %12.3f %5.1f %2d %6.3f %5.1f %2d %6.3f %5.1f %2d %6.3f %5.1f %2d \n' %
                            (nb_init+self.torsion['idx1'][i], nb_init+self.torsion['idx2'][i], nb_init+self.torsion['idx3'][i], nb_init+self.torsion['idx4'][i], value[0], gamma[0], 1, value[1], gamma[1], 2, value[2], gamma[2], 3, value[3], gamma[3], 4))
        prm.write(
        '''
torsion       0    0    0    0        0.000   0.0  1  0.000 180.0  2  0.000   0.0  3  0.000   0.0  4

      ########################################
      ##                                    ##
      ##  Atomic Partial Charge Parameters  ##
      ##                                    ##
      ########################################

        ''')
        prm.write('\n')
        if bool_OAT:
            for key, values in OAT.items():
                prm.write('charge %11d %16.4f \n' %
                    (nb_init+values[0], parameters['charge'][values[1]][0]))
        else:       
            for i in range(len(self.atom)):
                prm.write('charge %11d %16.4f \n' %
                        (nb_init+self.atom['atom_index'][i], parameters['charge'][i][0]))
        if waterbox:
            for line in lines:
                tokens = line.strip().split()
                if line[:7] == 'charge ':
                    prm.write(line)
        prm.close()
        print('New Parameter File (' + self.mol_name + ') Written!')

        if write_xyz:
            filename = os.path.join(self.sdf_path,self.mol_name+'.sdf')
            if not os.path.exists(filename):
                filename = os.path.join(self.pdb_path,self.mol_name+'.pdb')
                TmpMol = AllChem.MolFromPDBFile(filename,removeHs=False)
            else:
                TmpMol = AllChem.SDMolSupplier(filename,removeHs=False)[0]
            template = AllChem.MolFromSmiles(self.smiles, sanitize=True)
            if template is None:
                template = TmpMol
                mol = TmpMol
            else:
                template = Chem.AddHs(template)
                try:
                    mol = AllChem.AssignBondOrdersFromTemplate(template, TmpMol)
                except:
                    template = mol = TmpMol
            coord_x, coord_y, coord_z = self.get_coordinate(mol)
            atoms = mol.GetAtoms()

            prm = open(os.path.join(self.xyz_path,self.mol_name+'.xyz'), 'w+')
            prm.write('    ' + str(self.nb_atom) + '  ' + self.mol_name + '\n')
            for j in range(self.nb_atom):
                if bool_OAT:
                    prm.write('%6d %2s %13.6f %11.6f %11.6f %5d' %
                        (j+1, self.atom['atom_type'][j], coord_x[confId,j], coord_y[confId,j], coord_z[confId,j], self.nb_init+OAT[atom_type[j]][0]))
                else:
                    prm.write('%6d %2s %13.6f %11.6f %11.6f %5d' %
                        (j+1, self.atom['atom_type'][j], coord_x[confId,j], coord_y[confId,j], coord_z[confId,j], self.nb_init+j))
                neighbors = [p.GetIdx()+1 for p in atoms[j].GetNeighbors()]
                neighbors.sort()
                for p in neighbors:
                     prm.write('%6d' % (p))
                prm.write('\n')
            prm.close()
            print('New XYZ File (' + self.mol_name + ') Written!')
        
def check_file(name):
    pdb_path = './GeneratingPara/pdb/'
    sdf_path = './GeneratingPara/sdf/'
    smiles_path = './GeneratingPara/smiles/'
    bool_pdb = False
    bool_sdf = False
    bool_smiles = False
    
    if os.path.exists(pdb_path + name + '.pdb'):
        bool_pdb = True
    if os.path.exists(sdf_path + name + '.sdf'):
        bool_sdf = True
    if os.path.exists(smiles_path + name + '.txt'):
        bool_smiles = True
    if not (bool_pdb or bool_sdf):
        if bool_smiles:
            os.system('obabel -ismi ' + smiles_path + name + '.txt -osdf -O ' + sdf_path + name + '.sdf --gen3d')
        else:
            raise RuntimeError(name + ': Must provide SMILES or SDF file or PDB file!')
        
    if not bool_smiles:
        if bool_sdf:
            os.system('obabel -isdf ' + sdf_path + name + '.sdf -osmi -O ' + smiles_path + name + '.txt')
        elif bool_pdb:
            os.system('obabel -ipdb ' + pdb_path + name + '.pdb -osmi -O ' + smiles_path + name + '.txt')
    return None

def PreProcess(mol):
    inputs = {'nodes_features':[], 'edges_features':[], 'atom':[], 'bond':[], 'angle':[], 'torsion':[], 'imptors':[]}

    nodes_features, edges_features = copy.deepcopy(mol.get_inputs_features())
    inputs['nodes_features'] = [nodes_features.tolist()]
    inputs['edges_features'] = [edges_features.tolist()]

    paras_index = copy.deepcopy(mol.get_paras_index())
    for name in inputs:
        if name in paras_index:
            inputs[name] = paras_index[name]
                                    
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
    
    return inputs
    