import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(DeepNeuralNetwork, self).__init__()
        # Define the architecture of the network
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, 8)
        self.fc8 = nn.Linear(8, 1)  # Output layer

    def forward(self, x):
        # Define the forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)  # No activation function for the final layer
        return x

# Example usage
input_size = 250
model = DeepNeuralNetwork(input_size)

# Example input vector
x = torch.randn(1, input_size)  # Batch size of 1

# Forward pass
y_pred = model(x)
print(y_pred)
