import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size = 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size = 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(inputs)
        x = self.bn2(x)
        x = self.relu(x)
        del inputs
        return x
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2,2))
        self.bn1 = nn.BatchNorm2d(out_c)
        
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn1(x)
        
        p = self.pool(x)
        del inputs
        return x,p
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size = 2, stride = 2, padding = 0)
        self.conv = conv_block(out_c+out_c, out_c)
        self.bn1 = nn.BatchNorm2d(out_c)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis = 1)
        x = self.conv(x)
        x = self.bn1(x)
        del inputs
        return x

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.templates.embeddings import AmplitudeEmbedding
from pennylane.operation import Tensor

from torch.nn.functional import normalize

num_qubits = 9
num_shots = 20000
dev = qml.device("default.qubit", wires=num_qubits, shots=num_shots)
@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weight):
    """
    The variational quantum circuit.
    """
    

    
    q_input_features = normalize(q_input_features, p=2.0, dim = 0)
    
    # Start from Amplitude Embadding to Embed features in the quantum node
    qml.AmplitudeEmbedding(features=q_input_features, wires=[0,1,2,3,5,6,7,8],pad_with=num_qubits-1, normalize=True)
    
    qml.RY(q_weight[0], wires=0)
    qml.RY(q_weight[1], wires=1)
    qml.RY(q_weight[2], wires=2)
    qml.RY(q_weight[3], wires=3)
    qml.RY(q_weight[4], wires=4)
    qml.RY(q_weight[5], wires=5)
    qml.RY(q_weight[6], wires=6)
    qml.RY(q_weight[7], wires=7)
    qml.RY(q_weight[8], wires=8)
    
    
    
    qml.RX(q_weight[9], wires=0)
    qml.RX(q_weight[10], wires=1)
    qml.RX(q_weight[11], wires=2)
    qml.RX(q_weight[12], wires=3)
    qml.RX(q_weight[13], wires=4)
    qml.RX(q_weight[14], wires=5)
    qml.RX(q_weight[15], wires=6)
    qml.RX(q_weight[16], wires=7)
    qml.RX(q_weight[17], wires=8)
    
    
    
    qml.IsingXX(q_weight[18], wires=[0,1])
    qml.IsingXX(q_weight[19], wires=[8,7])
    qml.IsingXX(q_weight[20], wires=[1,2])
    qml.IsingXX(q_weight[21], wires=[7,6])
    qml.IsingXX(q_weight[22], wires=[2,3])
    qml.IsingXX(q_weight[23], wires=[6,5])
    qml.IsingXX(q_weight[24], wires=[3,4])
    qml.IsingXX(q_weight[25], wires=[5,4])
    
    
    qml.CRY(q_weight[26], wires=[0,1])
    qml.CRY(q_weight[27], wires=[1,2])
    qml.CRY(q_weight[28], wires=[2,3])
    qml.CRY(q_weight[29], wires=[8,7])
    qml.CRY(q_weight[30], wires=[7,6])
    qml.CRY(q_weight[31], wires=[6,5])
    qml.CRY(q_weight[32], wires=[5,4])
    qml.CRY(q_weight[33], wires=[3,4])
    
    
    
    qml.RY(q_weight[34], wires=0)
    qml.RY(q_weight[35], wires=1)
    qml.RY(q_weight[36], wires=2)
    qml.RY(q_weight[37], wires=3)
    qml.RY(q_weight[38], wires=4)
    qml.RY(q_weight[39], wires=5)
    qml.RY(q_weight[40], wires=6)
    qml.RY(q_weight[41], wires=7)
    qml.RY(q_weight[42], wires=8)
    
    
    qml.IsingXY(q_weight[43], wires=[4,3])
    qml.IsingXY(q_weight[44], wires=[4,5])
    qml.IsingXY(q_weight[45], wires=[3,2])
    qml.IsingXY(q_weight[46], wires=[5,6])
    qml.IsingXY(q_weight[47], wires=[2,1])
    qml.IsingXY(q_weight[48], wires=[6,7])
    qml.IsingXY(q_weight[49], wires=[1,0])
    qml.IsingXY(q_weight[50], wires=[7,8])
    
    
    qml.CRX(q_weight[51], wires=[0,1])
    qml.CRX(q_weight[52], wires=[2,3])
    qml.CRX(q_weight[53], wires=[5,6])
    qml.CRX(q_weight[54], wires=[7,8])
    qml.CRX(q_weight[55], wires=[1,2])
    qml.CRX(q_weight[56], wires=[6,7])
    
    
    qml.RX(q_weight[57], wires=0)
    qml.RX(q_weight[58], wires=1)
    qml.RX(q_weight[59], wires=2)
    qml.RX(q_weight[60], wires=3)
    qml.RX(q_weight[61], wires=4)
    qml.RX(q_weight[62], wires=5)
    qml.RX(q_weight[63], wires=6)
    qml.RX(q_weight[64], wires=7)
    qml.RX(q_weight[65], wires=8)
    
    
    
    
    qml.IsingXX(q_weight[66], wires=[0,1])
    qml.IsingXX(q_weight[67], wires=[8,7])
    qml.IsingXX(q_weight[68], wires=[1,2])
    qml.IsingXX(q_weight[69], wires=[7,6])
    qml.IsingXX(q_weight[70], wires=[2,3])
    qml.IsingXX(q_weight[71], wires=[6,5])
    qml.IsingXX(q_weight[72], wires=[3,4])
    qml.IsingXX(q_weight[73], wires=[5,4])
    
    
    qml.CRY(q_weight[74], wires=[0,1])
    qml.CRY(q_weight[75], wires=[1,2])
    qml.CRY(q_weight[76], wires=[2,3])
    qml.CRY(q_weight[77], wires=[8,7])
    qml.CRY(q_weight[78], wires=[7,6])
    qml.CRY(q_weight[79], wires=[6,5])
    qml.CRY(q_weight[80], wires=[5,4])
    qml.CRY(q_weight[81], wires=[3,4])
    
    
    
    qml.RY(q_weight[82], wires=0)
    qml.RY(q_weight[83], wires=1)
    qml.RY(q_weight[84], wires=2)
    qml.RY(q_weight[85], wires=3)
    qml.RY(q_weight[86], wires=4)
    qml.RY(q_weight[87], wires=5)
    qml.RY(q_weight[88], wires=6)
    qml.RY(q_weight[89], wires=7)
    qml.RY(q_weight[90], wires=8)
    
    
    qml.IsingXY(q_weight[91], wires=[4,3])
    qml.IsingXY(q_weight[92], wires=[4,5])
    qml.IsingXY(q_weight[93], wires=[3,2])
    qml.IsingXY(q_weight[94], wires=[5,6])
    qml.IsingXY(q_weight[95], wires=[2,1])
    qml.IsingXY(q_weight[96], wires=[6,7])
    qml.IsingXY(q_weight[97], wires=[1,0])
    qml.IsingXY(q_weight[98], wires=[7,8])
    
    
    qml.CRX(q_weight[99], wires=[0,1])
    qml.CRX(q_weight[100], wires=[2,3])
    qml.CRX(q_weight[101], wires=[5,6])
    qml.CRX(q_weight[102], wires=[7,8])
    qml.CRX(q_weight[103], wires=[1,2])
    qml.CRX(q_weight[104], wires=[6,7])
    
    
    qml.RX(q_weight[105], wires=0)
    qml.RX(q_weight[106], wires=1)
    qml.RX(q_weight[107], wires=2)
    qml.RX(q_weight[108], wires=3)
    qml.RX(q_weight[109], wires=4)
    qml.RX(q_weight[110], wires=5)
    qml.RX(q_weight[111], wires=6)
    qml.RX(q_weight[112], wires=7)
    qml.RX(q_weight[113], wires=8)
    
    # Prob of state
    
    
    
    # Expectation values in the Z basis
    #exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]

    return qml.probs(wires=[0,1,2,3,4,5,6,7,8])
q_delta = 1
n_cir = 8
n_pram = 114
n_qubits = 9

class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        #self.pre_net = nn.Linear(512, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(n_pram))
        #self.post_net = nn.Linear(n_qubits, 2)

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        #pre_out = self.pre_net(input_features)
        q_in = torch.tanh(input_features) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        #out = torch.Tensor(0, (256,8,8))
        #q_out = q_out.to(device)
        ep = 0
        for elem in q_in:
            elemt = torch.flatten(elem)
            
            elemt = torch.split(elemt, 2**(n_qubits-1))
            
            for i in range(n_cir):
                if i ==0:
                    
                    q_out_elem = quantum_net(elemt[i], self.q_params).float().unsqueeze(0)[0]
                    
                    q_out = q_out_elem
                    
                    
                else:
                    q_out_elem = quantum_net(elemt[i], self.q_params).float().unsqueeze(0)[0]
                    
                    q_out = torch.cat((q_out, q_out_elem))
            #q_out = q_out.reshape(1,256,8,8)
            if (ep == 0):
                out = q_out
            else:
                out = torch.cat((out, q_out))
            ep+=1
            #print(q_out.shape)
        out = out.reshape(ep, 1024, 2, 2)
        
        # return the two-dimensional prediction from the postprocessing layer
        del q_out, q_out_elem, elemt, q_in
        return out
    
class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.e1 = encoder_block(1,4)
        self.e2 = encoder_block(4,8)
        self.e3 = encoder_block(8,16)
        self.e4 = encoder_block(16,32)
        self.e5 = encoder_block(32,64)
        self.e6 = encoder_block(64,128)
        self.e7 = encoder_block(128,256)
        self.e8 = encoder_block(256,512)
        self.e9 = encoder_block(512,1024)
        
        
        
        
        self.b = DressedQuantumNet()
        self.da = decoder_block(2048, 1024)
        self.db = decoder_block(1024, 512)
        self.dc = decoder_block(512, 256)
        self.dd = decoder_block(256, 128)
        self.d0 = decoder_block(128, 64)
        self.d1 = decoder_block(64, 32)
        self.d2 = decoder_block(32, 16)
        self.d3 = decoder_block(16, 8)
        self.d4 = decoder_block(8, 4)
        
        self.outputs = nn.Conv2d(4, 1, kernel_size=1,padding=0)
    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)        
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        s6, p6 = self.e6(p5)        
        s7, p7 = self.e7(p6)
        s8, p8 = self.e8(p7)
        
        
       
        
        b1 = self.b(p8)

        
        db = self.db(b1, s8)
        dc = self.dc(db, s7)
        dd = self.dd(dc, s6)
        d0 = self.d0(dd, s5)
        d1 = self.d1(d0, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        outputs = self.outputs(d4)
        del s1, s2, s3, s4, s5, s6, s7, s8
        del p1, p2, p3, p4, p5, p6, p7, p8
        del db, dc, dd, d0, d1, d2, d3, d4
        
        return outputs
    
        
