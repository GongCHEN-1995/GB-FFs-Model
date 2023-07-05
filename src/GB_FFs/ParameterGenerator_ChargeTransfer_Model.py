import torch
from torch import Tensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

from src.GB_FFs.MolProcessingModel import GAT, SMU

class VDW(nn.Module):
    def __init__(self, d_model=128, activation='relu', leakyrelu=0.1):
        super(VDW, self).__init__()
        self.model_type = 'VDW'
        self.node = Linear(d_model,d_model)
        self.linear1 = Linear(d_model, d_model)
        self.linear2 = Linear(d_model, d_model)
        self.linear3 = Linear(d_model, 2)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'SMU':
            self.activation = SMU()
            
    def forward(self, x1:Tensor)->Tensor:
        out = self.activation(self.node(x1))
        out = self.linear3(self.activation(self.linear2(self.activation(self.linear1(out)))))
        return F.relu(out)

class BOND(nn.Module):
    def __init__(self, d_model=128, activation='relu', leakyrelu=0.1):
        super(BOND, self).__init__()
        self.model_type = 'BOND'
        self.d_model = d_model
        self.node = Linear(d_model, d_model)
        self.linear1 = Linear(d_model, d_model)
        self.linear2 = Linear(d_model, d_model)
        self.linear3 = Linear(d_model, 2)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'SMU':
            self.activation = SMU()
            
    def forward(self, x1:Tensor, x2:Tensor)->Tensor:
        out = self.activation(self.node(x1) + self.node(x2))
        out = self.linear3(self.activation(self.linear2(self.activation(self.linear1(out)))))
        return F.relu(out)

class BOND_MORSE(nn.Module):
    def __init__(self, d_model=128, activation='relu', leakyrelu=0.1):
        super(BOND_MORSE, self).__init__()
        self.model_type = 'BOND_MORSE'
        self.d_model = d_model
        self.node = Linear(d_model, d_model)
        self.linear1 = Linear(d_model, d_model)
        self.linear2 = Linear(d_model, d_model)
        self.linear3 = Linear(d_model, 3)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'SMU':
            self.activation = SMU()
            
    def forward(self, x1:Tensor, x2:Tensor)->Tensor:
        out = self.activation(self.node(x1) + self.node(x2))
        out = self.linear3(self.activation(self.linear2(self.activation(self.linear1(out)))))
        return F.relu(out)

class ANGLE(nn.Module):
    def __init__(self, d_model=128, activation='relu', leakyrelu=0.1):
        super(ANGLE, self).__init__()
        self.model_type = 'ANGLE'
        self.d_model = d_model
        self.node1 = Linear(d_model, d_model)
        self.node2 = Linear(d_model, d_model)
        self.linear1 = Linear(d_model, d_model)
        self.linear2 = Linear(d_model, d_model)
        self.linear3 = Linear(d_model, 2)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'SMU':
            self.activation = SMU()
            
    def forward(self, x1:Tensor, x2:Tensor, x3:Tensor)->Tensor:
        out = self.activation(self.node1(x1) + self.node2(x2) + self.node1(x3))
        out = self.linear3(self.activation(self.linear2(self.activation(self.linear1(out)))))
        out = torch.cat([out[:,:1],10*torch.sigmoid(out[:,1:])],dim=1)
        return F.relu(out)
    
class ANGLE_UB(nn.Module):
    def __init__(self, d_model=128, activation='relu', leakyrelu=0.1):
        super(ANGLE_UB, self).__init__()
        self.model_type = 'ANGLE_UB'
        self.d_model = d_model
        self.node1 = Linear(d_model, d_model)
        self.node2 = Linear(d_model, d_model)
        self.linear1 = Linear(d_model, d_model)
        self.linear2 = Linear(d_model, d_model)
        self.linear3 = Linear(d_model, 4)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'SMU':
            self.activation = SMU()
            
    def forward(self, x1:Tensor, x2:Tensor, x3:Tensor)->Tensor:
        out = self.activation(self.node1(x1) + self.node2(x2) + self.node1(x3))
        out = self.linear3(self.activation(self.linear2(self.activation(self.linear1(out)))))
        out = torch.cat([out[:,:1],10*torch.sigmoid(out[:,1:2]),out[:,2:]],dim=1)
        return F.relu(out)
    
class IMPTORS(nn.Module):
    def __init__(self, d_model=128, activation='relu', leakyrelu=0.1):
        super(IMPTORS, self).__init__()
        self.model_type = 'IMPTORS'
        self.d_model = d_model
        self.node1 = Linear(d_model, d_model)
        self.node2 = Linear(d_model, d_model)
        self.linear1 = Linear(d_model, d_model)
        self.linear2 = Linear(d_model, d_model)
        self.linear3 = Linear(d_model, 1)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'SMU':
            self.activation = SMU()
            
    def forward(self, x1:Tensor, x2:Tensor, x3:Tensor, x4:Tensor)->Tensor:
        out = self.activation(self.node1(x1)+self.node1(x2)+self.node2(x3)+self.node1(x4))      
        out = self.linear3(self.activation(self.linear2(self.activation(self.linear1(out)))))
        return out
    
class TORSION(nn.Module):
    def __init__(self, d_model=128,activation='relu', leakyrelu=0.1):
        super(TORSION, self).__init__()
        self.model_type = 'TORSION'
        self.d_model = d_model
        self.node = Linear(2*d_model, d_model)
        self.linear1 = Linear(d_model, d_model)
        self.linear2 = Linear(d_model, d_model)
        self.linear3 = Linear(d_model, 4)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'SMU':
            self.activation = SMU()
            
    def forward(self, x1:Tensor, x2:Tensor, x3:Tensor, x4:Tensor)->Tensor:
        z1 = torch.cat([x1,x2],dim=1)
        z2 = torch.cat([x4,x3],dim=1)
        out = self.activation(self.node(z1)+self.node(z2))      
        out = self.linear3(self.activation(self.linear2(self.activation(self.linear1(out)))))
        return out

# charge transfer model
class CHARGE(nn.Module):
    def __init__(self, d_model=128, activation='relu', leakyrelu=0.1):
        super(CHARGE, self).__init__()
        self.model_type = 'CHARGE'
        self.node = Linear(d_model,d_model)
        self.linear1 = Linear(d_model, d_model)
        self.linear2 = Linear(d_model, d_model)
        self.linear3 = Linear(d_model, 1)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'SMU':
            self.activation = SMU()
            
    def forward(self, x:Tensor)->Tensor:
        out = self.activation(self.node(x))
        out = self.linear3(self.activation(self.linear2(self.activation(self.linear1(out)))))
        return out
    
class ParaGenerator(nn.Module):
    def __init__(self, d_model=128, activation='relu', bond_morse=False, angle_ub=False, leakyrelu=0.1):
        super(ParaGenerator, self).__init__()
        self.model_type = 'ParaGenerator'
        self.d_model = d_model
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'SMU':
            self.activation = SMU()
            
        self.VDW = VDW(d_model=d_model, activation=activation, leakyrelu=leakyrelu)
        if bond_morse:
            self.BOND = BOND_MORSE(d_model=d_model, activation=activation, leakyrelu=leakyrelu)
        else:
            self.BOND = BOND(d_model=d_model, activation=activation, leakyrelu=leakyrelu)
        if angle_ub:
            self.ANGLE = ANGLE_UB(d_model=d_model, activation=activation, leakyrelu=leakyrelu)
        else:
            self.ANGLE = ANGLE(d_model=d_model, activation=activation, leakyrelu=leakyrelu)
        self.IMPTORS = IMPTORS(d_model=d_model, activation=activation, leakyrelu=leakyrelu)
        self.TORSION = TORSION(d_model=d_model, activation=activation, leakyrelu=leakyrelu)
        self.CHARGE = CHARGE(d_model=d_model, activation=activation, leakyrelu=leakyrelu)
        
    def forward(self, node_x:Tensor, edge_x:Tensor, inputs, charge_input_mask, charge_output_mask):
        outputs = {}
        node_x = node_x.reshape(-1, self.d_model)
        
        # VDW
        x1 = torch.index_select(node_x, 0, inputs['atom'])
        outputs['vdw'] = self.VDW(x1)
        
        # BOND
        x1 = torch.index_select(node_x, 0, inputs['bond'][0])
        x2 = torch.index_select(node_x, 0, inputs['bond'][1])
        outputs['bond'] = self.BOND(x1,x2)
        
        # ANGLE
        x1 = torch.index_select(node_x, 0, inputs['angle'][0])
        x2 = torch.index_select(node_x, 0, inputs['angle'][1])
        x3 = torch.index_select(node_x, 0, inputs['angle'][2])
        outputs['angle'] = self.ANGLE(x1,x2,x3)
        
        # IMPTORS
        x1 = torch.index_select(node_x, 0, inputs['imptors'][0])
        x2 = torch.index_select(node_x, 0, inputs['imptors'][1])
        x3 = torch.index_select(node_x, 0, inputs['imptors'][2])
        x4 = torch.index_select(node_x, 0, inputs['imptors'][3])
        outputs['imptors'] = self.IMPTORS(x1,x2,x3,x4)
        
        # TORSION
        x1 = torch.index_select(node_x, 0, inputs['torsion'][0])
        x2 = torch.index_select(node_x, 0, inputs['torsion'][1])
        x3 = torch.index_select(node_x, 0, inputs['torsion'][2])
        x4 = torch.index_select(node_x, 0, inputs['torsion'][3])

        outputs['torsion'] = self.TORSION(x1,x2,x3,x4)
        
        # CHARGE transfer
        transfer_charge = self.CHARGE(edge_x) #bsz, 2*edge_len, 1
        charge_output = torch.bmm(charge_output_mask, transfer_charge).squeeze() #bsz, node_len
        charge_input = torch.bmm(charge_input_mask, transfer_charge).squeeze() #bsz, node_len
        # for formal charge, 0:-3, 1:-2, 2:-1, 3:0, 4:1, 5:2, 6:3
        charge = charge_input - charge_output + (inputs['nodes_features'][:,:,6] - 3)
        charge = charge.reshape(-1)
        outputs['charge'] = torch.index_select(charge, 0, inputs['atom']).reshape(-1,1)
        outputs['transfer_charge'] = charge_input
        
        return outputs  
    
class OptimizedPara(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, num_heads=1, nb_layer=1, activation='relu', bond_morse=False, angle_ub=False, leakyrelu=0.1):
        super(OptimizedPara, self).__init__()
        self.model_type = 'OptimizedPara'
        self.GAT = GAT(d_model=d_model, dropout=dropout, num_heads=num_heads, nb_layer=nb_layer, activation=activation, leakyrelu=leakyrelu)
        self.ParaGenerator = ParaGenerator(d_model=d_model, activation=activation, bond_morse=bond_morse, angle_ub=angle_ub, leakyrelu=leakyrelu)
    def forward(self, inputs):
        node_x, edge_x, charge_input_mask, charge_output_mask = self.GAT(inputs['nodes_features'], inputs['edges_features'])
        outputs = self.ParaGenerator(node_x, edge_x, inputs, charge_input_mask, charge_output_mask)
        return outputs