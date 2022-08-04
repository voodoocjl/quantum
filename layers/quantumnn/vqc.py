import torch
from torch import nn
import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers


# """Multimodal fusion quantum circuit"""

n_qubits = 3

dev = qml.device("lightning.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch", diff_method="adjoint")

def multimodal_fusion_vqc(inputs, weights, n_fusion_layers):
    dev_m = qml.device('default.qubit', wires=3)

    def S(x):
        for m in range(3):
            qml.RX(x[m], wires=m)

    def WM(theta):
        StronglyEntanglingLayers(theta, wires=[0, 1, 2])

    for l in range(n_fusion_layers):
        S(inputs)
        WM(weights[l])
    
    return qml.expval(qml.PauliZ(wires=0))

  

"""Hybrid quantum-classical fusion network"""
class QFN(nn.Module):
    def __init__(self, input_dims, n_fusion_layers):
        super(QFN, self).__init__()
        
        self.input_dims = input_dims
        self.n_fusion_layers = n_fusion_layers
        self.multimodal = multimodal_fusion_vqc(n_fusion_layers)

    def forward(self, x):
        expvals = torch.stack(x, dim=1)
        pred = multimodal_fusion_vqc(expvals, self.q_params, self.n_fusion_layers)

        return pred
