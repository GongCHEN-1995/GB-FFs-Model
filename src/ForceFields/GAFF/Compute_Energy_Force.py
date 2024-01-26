import torch
import torch.nn as nn
import numpy as np

# for GAFF
class ComputeEnergyForce(nn.Module):
    def __init__(self, bond_morse=False, bool_ub=False):
        super(ComputeEnergyForce, self).__init__()
        self.model_type = 'ComputeEnergyForce'
        self.charge = 18.222615
        self.set_bond_morse(bond_morse)
        self.set_bool_ub(bool_ub)
        
    def set_bond_morse(self, bond_morse=False):
        self.bond_morse = bond_morse
        
    def set_bool_ub(self, bool_ub=False):
        self.bool_ub = bool_ub

    def forward(self, inputs, paras, inputs_gradient=None):
        nb_shot = inputs['length_vdw'].size(0)
        if inputs_gradient:
            Force = torch.zeros((nb_shot, len(paras['vdw']), 3)).to(paras['vdw'].device) # nb_shot, nb_atom, 3
        else:
            Force = None
            
        Energy = {}
        # for bond 
        length_bond = inputs['length_bond']  # nb_shot, nb_bond
        K = paras['bond'][:,0].view(1,-1) * 100
        r0 = paras['bond'][:,1].view(1,-1)
        alpha = paras['bond'][:,2].reshape(1,-1) if paras['bond'].shape[1]==3 else 2*torch.ones(1, 1).to(paras['vdw'].device)
        if self.bond_morse:
            tmp = torch.exp(-alpha*(length_bond - r0))
            Energy['bond'] = 0.25 * K * (tmp - 1).pow(2)
        else:
            tmp = length_bond - r0
            Energy['bond'] = K * tmp.pow(2)

        if inputs_gradient:
            # Force_bond.size = (nb_shot, nb_bond, 6)
            tmp = tmp.view(nb_shot,-1,1)
            K = K.view(1,-1,1)
            if self.bond_morse:
                alpha = alpha.view(1,-1,1)
                Force_bond = inputs_gradient['dlength_bond'] * 0.5 * alpha * K * tmp * (1 - tmp)
            else:
                Force_bond = inputs_gradient['dlength_bond'] * 2 * K * tmp
            
            Force_bond = Force_bond.reshape(nb_shot, -1, 3)
            index_bond = inputs_gradient['bond_index'].reshape(-1)
            Force.index_add_(1, index_bond, Force_bond)

        # for angle 
        theta_angle = inputs['theta_angle'] # nb_shot, nb_angle
        K = paras['angle'][:,0].view(1,-1) * 10
        theta0 = paras['angle'][:,1].view(1,-1) * np.pi / 10
        Energy['angle'] = K * (theta_angle - theta0).pow(2)
        if inputs_gradient:
            # Force_angle.size = (nb_shot, nb_angle, 9)
            Force_angle = inputs_gradient['dtheta_angle'] * 2 * K.view(1,-1,1) * (theta_angle.view(nb_shot,-1,1) - theta0.view(1,-1,1))
            Force_angle = Force_angle.reshape(nb_shot, -1, 3)
            index_angle = inputs_gradient['angle_index'].reshape(-1)
            Force.index_add_(1, index_angle, Force_angle)
                
        # for urey-breadely term 
        if self.bool_ub:
            length_angle = inputs['length_angle']
            epsilon = paras['angle'][:,2].view(1,-1) / 10
            sigma = paras['angle'][:,3].view(1,-1)
            tmp = (sigma / length_angle).pow(2)
            Energy['UB'] = epsilon * (tmp - 1).pow(2)
            if inputs_gradient:
                tmp = tmp.view(nb_shot,-1,1)
                Force_angle_corr = inputs_gradient['dlength_angle'] * 4 * epsilon.view(1,-1,1) * (1 - tmp) * tmp / length_angle.view(nb_shot,-1,1)
                Force_angle_corr = Force_angle_corr.reshape(nb_shot, -1, 3)
                index_angle = inputs_gradient['angle_index'][:,[0,2]].reshape(-1)
                Force.index_add_(1, index_angle, Force_angle_corr)
        else:
            Energy['UB'] = torch.zeros(nb_shot, 1).to(paras['vdw'].device)
        # for vdw
        length_vdw = inputs['length_vdw']
        length_vdw6 = length_vdw.pow(6)
        sigma1 = torch.index_select(paras['vdw'][:,0], 0, inputs['non-bonded'][0])
        sigma2 = torch.index_select(paras['vdw'][:,0], 0, inputs['non-bonded'][1])
        epsilon1 = torch.index_select(paras['vdw'][:,1], 0, inputs['non-bonded'][0]) / 10 
        epsilon2 = torch.index_select(paras['vdw'][:,1], 0, inputs['non-bonded'][1]) / 10 
        sigma = sigma1 + sigma2
        tmp = sigma.pow(6) / length_vdw6
        epsilon = epsilon1 * epsilon2 * inputs['vdw14'].reshape(1,-1)
        Energy['vdw'] = epsilon * (tmp.pow(2) - 2 * tmp)
        if inputs_gradient:
            epsilon = epsilon.view(1,-1,1)
            tmp = tmp.view(nb_shot,-1,1)
            length_vdw = length_vdw.view(nb_shot,-1,1)
            # Force_vdw.size = (nb_shot, nb_non_bonded, 6)
            Force_vdw = inputs_gradient['dlength_vdw'] * 12. * epsilon * tmp * (1 - tmp) / length_vdw
            Force_vdw = Force_vdw.reshape(nb_shot, -1, 3)
            index_vdw = inputs_gradient['non-bonded_index'].reshape(-1)
            Force.index_add_(1, index_vdw, Force_vdw)

        # for charge
        length_vdw = length_vdw.view(nb_shot,-1)
        charge1 = self.charge / 10 * torch.index_select(paras['charge'], 0, inputs['non-bonded'][0]).view(1, -1)
        charge2 = self.charge / 10 * torch.index_select(paras['charge'], 0, inputs['non-bonded'][1]).view(1, -1)
        Energy['charge'] = charge1 * charge2 / length_vdw * inputs['charge14'].reshape(1,-1)
        if inputs_gradient:
            charge1 = charge1.view(1,-1,1)
            charge2 = charge2.view(1,-1,1)
            length_vdw = length_vdw.view(nb_shot,-1,1)
            # Force_vdw.size = (nb_shot, nb_non_bonded, 6)
            Force_charge = -inputs_gradient['dlength_vdw'] * charge1 * charge2 / length_vdw.pow(2) * inputs['charge14'].reshape(1,-1,1)
            Force_charge = Force_charge.reshape(nb_shot, -1, 3)
            index_charge = inputs_gradient['non-bonded_index'].reshape(-1)
            Force.index_add_(1, index_charge, Force_charge)

#       for torsion  #use sin(alpha) and sin(alpha) to replace cos(2alpha) and cos(3alpha)
        cos = inputs['sin_cos_n_theta_torsion'][:,:,1]
        cos2 = inputs['sin_cos_n_theta_torsion'][:,:,3]
        cos3 = inputs['sin_cos_n_theta_torsion'][:,:,5]
        cos4 = inputs['sin_cos_n_theta_torsion'][:,:,7]
        Energy['torsion'] = paras['torsion'][:,0].view(1,-1) * cos  + paras['torsion'][:,1].view(1,-1) * cos2 \
                          + paras['torsion'][:,2].view(1,-1) * cos3 + paras['torsion'][:,3].view(1,-1) * cos4
        if inputs_gradient:
            sin = inputs['sin_cos_n_theta_torsion'][:,:,0].view(nb_shot,-1,1)
            sin2 = inputs['sin_cos_n_theta_torsion'][:,:,2].view(nb_shot,-1,1)
            sin3 = inputs['sin_cos_n_theta_torsion'][:,:,4].view(nb_shot,-1,1)
            sin4 = inputs['sin_cos_n_theta_torsion'][:,:,6].view(nb_shot,-1,1)
            # Force_torsion.size = (nb_shot, nb_torsion, 12)
            Force_torsion = -inputs_gradient['dtheta_torsion'] * (paras['torsion'][:,0].view(1,-1,1) * sin  + \
                                                            2. *  paras['torsion'][:,1].view(1,-1,1) * sin2 + \
                                                            3. *  paras['torsion'][:,2].view(1,-1,1) * sin3 + \
                                                            4. *  paras['torsion'][:,3].view(1,-1,1) * sin4)
            Force_torsion = Force_torsion.reshape(nb_shot, -1, 3)
            index_torsion = inputs_gradient['torsion_index'].reshape(-1)
            Force.index_add_(1, index_torsion, Force_torsion)

        # for improper torsion
        cos2 = inputs['cos2_imptors']
        Energy['imptors'] = paras['imptors'][:,0].view(1,-1) * (1 - cos2)
        if inputs_gradient:
            # Force_imptors.size = (nb_shot, nb_imptors, 12)
            Force_imptors = -inputs_gradient['dcos2_imptors'] * paras['imptors'][:,0].view(1,-1,1)
            Force_imptors = Force_imptors.reshape(nb_shot, -1, 3)
            index_imptors = inputs_gradient['imptors_index'].reshape(-1)
            Force.index_add_(1, index_imptors, Force_imptors)

        return  Energy, Force
