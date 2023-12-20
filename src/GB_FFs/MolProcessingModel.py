import torch
from torch import Tensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.parameter import Parameter
from typing import Optional, Tuple, List
import numpy as np
import copy

class SMU(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
        self.mu = Parameter(torch.Tensor([0.5]))
    def forward(self, x):
        return x*(1+torch.erf(self.mu*x))/2
    
# module for Graph Attention Network
class EDGEAttention(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, num_heads=1, activation='relu', leakyrelu=0.1, bias=True):
        super(EDGEAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == self.d_model, "d_model must be divisible by num_heads"
        
        self.in_proj_weight1 = Parameter(torch.empty(d_model, 3 * d_model))

        if bias:
            self.in_proj_bias1 = Parameter(torch.empty(3 * d_model))
        else:
            self.register_parameter('in_proj_bias1', None)
            
        self._reset_parameters()
        
        self.dropout = nn.Dropout(dropout)
        self.linear1_edge = Linear(d_model, d_model)
        self.linear2_edge = Linear(d_model, d_model)
        self.norm1_edge = nn.modules.normalization.LayerNorm(d_model)
        self.norm2_edge = nn.modules.normalization.LayerNorm(d_model)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'SMU':
            self.activation = SMU()
            
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight1)
        if self.in_proj_bias1 is not None:
            nn.init.constant_(self.in_proj_bias1, 0.)
    
    def _MHA_linear(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        output = inputs.matmul(weight)
        if bias is not None:
            output += bias
        return output

    def forward(self, edge_x: Tensor, edge_mask: Tensor) -> Tensor:
        '''
        node_x.shape:      bsz,edge_len,d_model
        edge_mask.shape:   bsz,edge_len,edge_len
        
        It should be noted that here edge_len is doubled!!! 
        Because each edge will derive two directed edges!!!
        edge_x.shape[1] == 2*edge_features.shape[1]
        '''
        edge_x = edge_x.transpose(0,1) #edge_len, bsz, d_model
        bsz = edge_x.size(1)
        edge_len = edge_x.size(0)
        scaling = float(self.head_dim) ** -0.5

        # self-attention fot edge_x
        # query/key/value matrix for attention mechanism
        q_edge, k_edge, v_edge = self._MHA_linear(edge_x, self.in_proj_weight1, self.in_proj_bias1).chunk(3, dim=-1) 

        q_edge = q_edge.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,edge_len,head_dim
        k_edge = k_edge.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,edge_len,head_dim
        v_edge = v_edge.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,edge_len,head_dim
        
        # Q and K to compute the coefficients e
        attn_output_weights = torch.bmm(q_edge, k_edge.transpose(1, 2)) #bsz*num_heads, edge_len, edge_len
        attn_output_weights = attn_output_weights * scaling
        attn_output_weights = attn_output_weights.reshape(bsz, self.num_heads, edge_len, edge_len)
        attn_output_weights = attn_output_weights.masked_fill(edge_mask.unsqueeze(1),float("-inf"))
        attn_output_weights = attn_output_weights.view(bsz * self.num_heads, edge_len, edge_len) #bsz*num_heads, edge_len, edge_len
        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)  #bsz*num_heads, edge_len, edge_len
        attn_output_weights =  self.dropout(attn_output_weights)
        
        # Value and alpha to construct the new matrix
        attn_output = torch.bmm(attn_output_weights, v_edge)  # size = bsz*num_heads, edge_len, head_dim  
        edge_x2 = attn_output.transpose(0, 1).reshape(edge_len, bsz, self.d_model)
        
        # Followed by a multi-layer perception with skip connection
        edge_x = edge_x + self.dropout(edge_x2)
        edge_x = self.norm1_edge(edge_x)
        edge_x2 = self.linear2_edge(self.dropout(self.activation(self.linear1_edge(edge_x))))
        edge_x = edge_x + self.dropout(edge_x2)
        edge_x = self.norm2_edge(edge_x)
        
        return edge_x.transpose(0,1)
     
# module for Graph Attention Network
class NODEAttention(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, num_heads=1, activation='relu', leakyrelu=0.1, bias=True):
        super(NODEAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == self.d_model, "d_model must be divisible by num_heads"
        self.D = 128
        self.sqrt2_pi_inverse = 1 / np.sqrt(2 * np.pi)
        
        self.in_proj_weight1 = Parameter(torch.empty(d_model, 3 * d_model))
        self.in_proj_weight2 = Parameter(torch.empty(d_model, 2 * d_model))
        
        if bias:
            self.in_proj_bias1 = Parameter(torch.empty(3 * d_model))  
            self.in_proj_bias2 = Parameter(torch.empty(2 * d_model))
        else:
            self.register_parameter('in_proj_bias1', None)
            self.register_parameter('in_proj_bias2', None)
            
        self._reset_parameters()
        
        self.dropout = nn.Dropout(dropout)
        self.linear1_node = Linear(d_model, d_model)
        self.linear2_node = Linear(d_model, d_model)
        self.norm1_node = nn.modules.normalization.LayerNorm(d_model)
        self.norm2_node = nn.modules.normalization.LayerNorm(d_model)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'SMU':
            self.activation = SMU()
            
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight1)
        nn.init.xavier_uniform_(self.in_proj_weight2)
        if self.in_proj_bias1 is not None:
            nn.init.constant_(self.in_proj_bias1, 0.)
            nn.init.constant_(self.in_proj_bias2, 0.)
    
    def _MHA_linear(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        output = inputs.matmul(weight)
        if bias is not None:
            output += bias
        return output

    def forward(self, node_x: Tensor, edge_x: Tensor, node_mask: Tensor) -> Tensor:
        '''
        node_x.shape:      bsz,node_len,d_model
        edge_x.shape:      bsz,edge_len,d_mdoel
        node_mask.shape:   bsz,node_len,node_len+edge_len
        
        It should be noted that here edge_len is doubled!!! 
        Because each edge will derive two directed edges!!!
        edge_x.shape[1] == 2*edge_features.shape[1]
        '''
        node_x = node_x.transpose(0,1) #node_len, bsz, d_model
        edge_x = edge_x.transpose(0,1) #edge_len, bsz, d_model
        bsz = node_x.size(1)
        node_len = node_x.size(0)
        edge_len = edge_x.size(0)
        scaling = float(self.head_dim) ** -0.5
        
        # attention fot node_x
        # query/key/value matrix for attention mechanism
        q_node, k_node, v_node = self._MHA_linear(node_x, self.in_proj_weight1, self.in_proj_bias1).chunk(3, dim=-1) 
        k_edge, v_edge = self._MHA_linear(edge_x, self.in_proj_weight2, self.in_proj_bias2).chunk(2, dim=-1) 
                
        # Q remains the same
        q_node = q_node.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,node_len,head_dim
        
        # K and V have to consider the node, as well as the connected edges
        k_node = torch.cat([k_node,k_edge],dim=0)
        k_node = k_node.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,node_len+edge_len,head_dim
        v_node = torch.cat([v_node,v_edge],dim=0)
        v_node = v_node.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,node_len+edge_len,head_dim

        attn_output_weights = torch.bmm(q_node, k_node.transpose(1, 2)) #bsz*num_heads, node_len, node_len+edge_len
        attn_output_weights = attn_output_weights * scaling
                
        # Q and K to compute the coefficients e
        attn_output_weights = attn_output_weights.reshape(bsz, self.num_heads, node_len, -1)
        attn_output_weights = attn_output_weights.masked_fill(node_mask.unsqueeze(1),float("-inf"))
        attn_output_weights = attn_output_weights.view(bsz * self.num_heads, node_len, -1) #bsz*num_heads, node_len, node_len+edge_len
        
        
        # compute the attention weights alpha
        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)  #bsz*num_heads, node_len, node_len+edge_len
        attn_output_weights = self.dropout(attn_output_weights)
        
        # Value and alpha to construct the new matrix
        attn_output = torch.bmm(attn_output_weights, v_node)  # size = bsz*num_heads, node_len, head_dim  
        node_x2 = attn_output.transpose(0, 1).reshape(node_len, bsz, self.d_model)
        
        # Followed by a multi-layer perception with skip connection
        node_x = node_x + self.dropout(node_x2)
        node_x = self.norm1_node(node_x)
        node_x2 = self.linear2_node(self.dropout(self.activation(self.linear1_node(node_x))))
        node_x = node_x + self.dropout(node_x2)
        node_x = self.norm2_node(node_x)
        
        return node_x.transpose(0,1)

    # module for Graph Attention Network
class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, num_heads=1, activation='relu', leakyrelu=0.1):
        super(GraphAttentionLayer, self).__init__()
        
        self.node_attention = NODEAttention(d_model, dropout, num_heads, activation, leakyrelu)
        self.edge_attention = EDGEAttention(d_model, dropout, num_heads, activation, leakyrelu)

    def forward(self, node_x: Tensor, edge_x: Tensor, node_mask: Tensor, edge_mask: Tensor) -> Tensor:
        edge_x = self.edge_attention(edge_x, edge_mask)
        node_x = self.node_attention(node_x, edge_x, node_mask)
        
        return node_x, edge_x

class GAT(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, num_heads=1, nb_layer=1, activation='relu', leakyrelu=0.1):
        super(GAT, self).__init__()
        self.model_type = 'GAT'
        self.nb_layer = nb_layer
        self.d_layer1 = 128
        self.d_model = d_model
        self.node_attention1 = NODEAttention(d_model=self.d_layer1, dropout=dropout, num_heads=2, activation=activation, leakyrelu=leakyrelu)
        self.node_attention2 = NODEAttention(d_model=d_model, dropout=dropout, num_heads=num_heads, activation=activation, leakyrelu=leakyrelu)
        
#         transform features to vectors
        self.embedding1 = Feature_Embedding(d_model = self.d_layer1)
        self.embedding2 = Feature_Embedding(d_model = self.d_model)
        
        self.norm_node2 = nn.modules.normalization.LayerNorm(d_model)
        self.norm_edge1 = nn.modules.normalization.LayerNorm(self.d_layer1)
        self.norm_edge2 = nn.modules.normalization.LayerNorm(d_model)
        
        self.node_1_2 = Linear(self.d_layer1, d_model)
        self.edge_1_2 = Linear(self.d_layer1, d_model)
        self.layers1 = self._get_clones(GraphAttentionLayer(d_model=self.d_layer1, dropout=dropout, num_heads=4, activation=activation, leakyrelu=leakyrelu) , 4)
        self.layers2 = self._get_clones(GraphAttentionLayer(d_model=d_model, dropout=dropout, num_heads=num_heads, activation=activation, leakyrelu=leakyrelu) , nb_layer)
        
    def _get_clones(self, module, N):
        return nn.modules.container.ModuleList([copy.deepcopy(module) for i in range(N)])
        
    def forward(self, node_features:Tensor, edge_features:Tensor)->Tuple[Tensor,Tensor]:
        '''
        node_features.shape:      bsz,node_len,nb_atom_features
        edge_features.shape:      bsz,edge_len,nb_bond_features+2
        '''
        node_len = node_features.size(1)
        edge_len = edge_features.size(1)
        bsz = edge_features.size(0)

        # edge_features[:,:,:2] denotes the start node and end ndoe for corresponding edge
        # edge_features[:,:,:2] for padding edges are [0,0]
        edges_all = edge_features[:,:,:2].cpu().numpy().astype(int).tolist()
        for i in range(len(edges_all)):
            for j in range(len(edges_all[i])):
                if edges_all[i][j] == [0,0]:
                    del edges_all[i][j:]
                    break
        
        # Double the edge features to prepare for generating the directed edges
        edge_features = edge_features[:,:,2:].reshape(bsz*edge_len, -1).repeat(1,2).reshape(bsz, 2*edge_len, -1) # bsz,2*edge_len,num_edge_features
        
        
         # convert the features to embeddings
        node_x1, node_x1_start, node_x1_end, edge_x1 = self.embedding1(node_features, edge_features)
        node_x2, node_x2_start, node_x2_end, edge_x2 = self.embedding2(node_features, edge_features)
        
        '''
        Construct the masking matrices
        As the number of undirected bonds are different in each molecule, it has to be one by one.
        edge_mask_all is to update directed bond embeddings
        node_mask_all is to update atom embeddings
        charge_output_mask is to decide the charges flow in/out
        '''
        edge_mask_all = np.ones((bsz,2*edge_len,2*edge_len))
        node_mask_all = np.ones((bsz,node_len,node_len+2*edge_len))
        charge_output_mask = np.zeros((bsz,node_len,2*edge_len))
        for i in range(bsz):
            edges = edges_all[i]
                
            '''
            To store the start node and end node for directed edges
            
            Here is a simple example: n0-n1-n2-n3
            considering 4 connected nodes [n0,n1,n2,n3]
            Therefor there are three undirected edges [e0,e1,e2]  and e0=[n0,n1], e1=[n1,n2], e2=[n2,n3]
            e0-1: from n0 to n1                  e0-2: from n1 to n0
            e1-1: from n1 to n2                  e1-2: from n2 to n1
            e2-1: from n2 to n3                  e2-2: from n3 to n2
            dict_edge_node_start = {n0:[e0-1], n1:[e0-2,e1-1], n2[e1-2,e2-1], n3:[e2-2]}
            dict_edge_node_end   = {n0:[e0-2], n1:[e0-1,e1-2], n2[e1-1,e2-2], n3:[e2-1]}
            Therefore, if we consider n1, dict_edge_node_start[n1] = [e0-2,e1-1], 
            which means n1 is the start node of directed eges e0-2 and e1-1
            '''
            dict_edge_node_start = {i:[] for i in range(node_len)}
            dict_edge_node_end = {i:[] for i in range(node_len)}
            for j in range(len(edges)):
                dict_edge_node_end[edges[j][0]].append(2*j+1)
                dict_edge_node_end[edges[j][1]].append(2*j)
                dict_edge_node_start[edges[j][0]].append(2*j)
                dict_edge_node_start[edges[j][1]].append(2*j+1)
                
            '''
            edge2_index2 stores the start node of directed edges
            edge2_index3 stores the end node of directed edges
            edge_mask 0: not masking 1: masking  
            '''
            edge_mask = 1 - np.identity(2*edge_len)
            edge2_index2 = []
            edge2_index3 = []
            for j in range(len(edges)):
                mask_index = copy.deepcopy(dict_edge_node_end[edges[j][0]])
                mask_index.remove(2*j+1)
                edge_mask[2*j,mask_index] = 0
                mask_index = copy.deepcopy(dict_edge_node_end[edges[j][1]])
                mask_index.remove(2*j)
                edge_mask[2*j+1,mask_index] = 0
    
                edge2_index2 += [edges[j][0], edges[j][1]]
                edge2_index3 += [edges[j][1], edges[j][0]]

            '''
            node_mask_all:       0:not masking     1:masking
            charge_output_mask:  0:masking         1:not masking 
            '''
            for j in range(node_len):
                node_mask_all[i,j,[node_len+p for p in dict_edge_node_end[j]]] = 0
                node_mask_all[i,j,j] = 0
                charge_output_mask[i,j,[p for p in dict_edge_node_start[j]]] = 1
            # add the single masking matrix to batch size masking matrix
            edge_mask_all[i,:,:] = edge_mask
            edge2_index2 = torch.LongTensor(edge2_index2).to('cuda' if torch.cuda.is_available() else 'cpu')  
            edge2_index3 = torch.LongTensor(edge2_index3).to('cuda' if torch.cuda.is_available() else 'cpu')
            # construct the directed edges for Small layers: add start nodes
            edge_x1[i,:2*len(edges),:] += torch.index_select(node_x1_start[i,:,:], 0, edge2_index2)
            # construct the directed edges for Small layers: add end nodes
            edge_x1[i,:2*len(edges),:] += torch.index_select(node_x1_end[i,:,:], 0, edge2_index3)
            # construct the directed edges for Large layers: add start nodes
            edge_x2[i,:2*len(edges),:] += torch.index_select(node_x2_start[i,:,:], 0, edge2_index2)
            # construct the directed edges for Large layers: add end nodes
            edge_x2[i,:2*len(edges),:] += torch.index_select(node_x2_end[i,:,:], 0, edge2_index3)
            
        # put the masking matricies in GPU if necessary
        edge_mask_all = torch.LongTensor(edge_mask_all).to(torch.bool).to('cuda' if torch.cuda.is_available() else 'cpu')
        node_mask_all = torch.LongTensor(node_mask_all).to(torch.bool).to('cuda' if torch.cuda.is_available() else 'cpu')
        charge_output_mask = torch.Tensor(charge_output_mask).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Update in Small layers
        edge_x1 = self.norm_edge1(edge_x1)
        node_x1 = self.node_attention1(node_x1, edge_x1, node_mask_all)
        for mod in self.layers1:
            node_x1, edge_x1 = mod(node_x1, edge_x1, node_mask_all, edge_mask_all)
        
        # Update in Large Layers
        node_x2 = self.norm_node2(self.node_1_2(node_x1) + node_x2)
        edge_x2 = self.norm_edge2(self.edge_1_2(edge_x1) + edge_x2)
        node_x2 = self.node_attention2(node_x2, edge_x2, node_mask_all)
        for mod in self.layers2:
            node_x2, edge_x2 = mod(node_x2, edge_x2, node_mask_all, edge_mask_all)
        
        # charge_input_mask:   0:masking       1:not masking
        charge_input_mask = (~node_mask_all[:,:,node_len:]).to(torch.float32)
        return node_x2, edge_x2, charge_input_mask, charge_output_mask

class Feature_Embedding(nn.Module):
    def __init__(self, d_model=128):
        super(Feature_Embedding, self).__init__()
        self.model_type = 'Feature_Embedding'
        self.d_model = d_model
        self.node_element       = nn.Embedding(17, d_model)
        self.node_degree        = nn.Embedding(7, d_model)
        self.node_ring          = nn.Embedding(2, d_model)
        self.node_hybridization = nn.Embedding(8, d_model)
        self.node_aromatic      = nn.Embedding(2, d_model)
        self.node_chirality     = nn.Embedding(4, d_model)
        self.node_formalcharge  = nn.Embedding(7, d_model)
        self.node_start         = Linear(d_model, d_model)
        self.node_end           = Linear(d_model, d_model)
        
        self.edge_type        = nn.Embedding(6, d_model)
        self.edge_conjugation = nn.Embedding(2, d_model)
        self.edge_ring        = nn.Embedding(2, d_model)
        self.edge_stereotype  = nn.Embedding(6, d_model)
        self.norm_node = nn.modules.normalization.LayerNorm(d_model)
        self.norm_edge = nn.modules.normalization.LayerNorm(d_model)
        
    def forward(self, node_features:Tensor, edge_features:Tensor):
        # 0:PAD, 1:UNK, 2:H, 3:C, 4:N, 5:O, 6:S, 7:P, 8:F, 9:Cl, 10:Br, 11:I, 12:Li, 13:Na, 14:Mg, 15:K, 16:Ca
        # But the metal elements are not used!!!
        node_x_element       = self.node_element(node_features[:,:,0])
        # 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:others
        node_x_degreee       = self.node_degree(node_features[:,:,1])
        # 0:not in the ring, 1:in the ring
        node_x_ring          = self.node_ring(node_features[:,:,2])
        # 0:unspecified, 1:s, 2:sp, 3:sp2, 4:sp3, 5:sp3d, 6:sp3d2, 7:other
        node_x_hybridization = self.node_hybridization(node_features[:,:,3])
        # 0: not in aromatic environment, 1; in aromatic environment
        node_x_aromatic      = self.node_aromatic(node_features[:,:,4])
        # 0:CHI_UNSPECIFIED, 1:CHI_TETRAHEDRAL_CW, 2:CHI_TETRAHEDRAL_CCW, 3:CHI_OTHER
        node_x_chirality     = self.node_chirality(node_features[:,:,5])
        # 0:-3, 1:-2, 2:-1, 3:0, 4:1, 5:2, 6:3
        node_x_formalcharge  = self.node_formalcharge(node_features[:,:,6])
        
        node_x               = self.norm_node(node_x_element + node_x_degreee + node_x_ring + node_x_hybridization + node_x_aromatic + node_x_chirality + node_x_formalcharge)
        node_x_start         = self.node_start(node_x)
        node_x_end           = self.node_end(node_x)
        
        # 0:PAD, 1:OTHERS, 2:SINGLE, 3:DOUBLE, 4:TRIPLE, 5:AROMATIC
        edge_x_type        = self.edge_type(edge_features[:,:,0])
        # 0: no conjugation, 1:with conjugation
        edge_x_conjugation = self.edge_conjugation(edge_features[:,:,1])
        # 0:not in the ring, 1:in the ring
        edge_x_ring        = self.edge_ring(edge_features[:,:,2])
        # 0:STEREONONE, 1:STEREOANY, 2:STEREOZ, 3:STEREOE, 4:STEREOCIS, 5:STEREOTRANS
        edge_x_stereotype  = self.edge_stereotype(edge_features[:,:,3])
        edge_x = self.norm_edge(edge_x_type + edge_x_conjugation + edge_x_ring + edge_x_stereotype)
        
        return node_x, node_x_start, node_x_end, edge_x