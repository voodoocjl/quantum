import torch
from torch import nn
import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers


# """Multimodal fusion quantum circuit"""

n_qubits = 3
dev = qml.device('default.qubit', wires=n_qubits)
@qml.qnode(dev, interface="torch", diff_method="backprop")

def multimodal_fusion_vqc(inputs, weights, n_fusion_layers):
    def S(x):
        for m in range(3):
            qml.RX(x[m], wires=m)

    def WM(theta):
        StronglyEntanglingLayers(theta, wires=[0, 1, 2])

    for l in range(n_fusion_layers):
        S(inputs)
        WM(weights[l])
    
    return qml.state()
  

"""Hybrid quantum fusion network"""

class QFN(nn.Module):
    def __init__(self, input_dims, n_ansartz_layers, n_fusion_layers):
        super(QFN, self).__init__()
        
        self.input_dims = input_dims
        self.n_fusion_layers = n_fusion_layers
        self.q_params = nn.Parameter(torch.rand( n_fusion_layers, n_ansartz_layers, 3, 3))      

    def forward(self, x):
        outputs = []

        for reps_t in zip(*x): #(model_1, modal_2, modal_3), each is [33,2,[32,1]]
            multimodal_rep = [torch.stack(rep_field, dim = -2).squeeze(dim = -1) for rep_field in zip(*reps_t)] #[2,[32,3]]
            # output_rep = []
            # for _rep in multimodal_rep:
            #     output = [multimodal_fusion_vqc(__rep, self.q_params, self.n_fusion_layers) for __rep in _rep]
            #     output_rep.append(torch.stack(output))
            output_rep = torch.stack([multimodal_fusion_vqc(_rep, self.q_params, self.n_fusion_layers) for _rep in multimodal_rep[0]] ) #[32,8]
            
            outputs.append([output_rep.real, output_rep.imag])
        
        return outputs
