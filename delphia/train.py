import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# Define the DeepNeuralNetwork class and the DataLoader class here (see previous responses)

# Function to load data from .npy files
def load_data(file_path):
    data = np.load(file_path)
    X = data['X']
    y = data['y']
    return X, y

# Function to create DataLoader for training and validation sets
def create_dataloader(train_file, val_file, batch_size):
    X_train, y_train = load_data(train_file)
    X_val, y_val = load_data(val_file)

    # Convert data to PyTorch tensors
    X_train, X_val = torch.Tensor(X_train), torch.Tensor(X_val)
    y_train, y_val = torch.Tensor(y_train), torch.Tensor(y_val)

    # Create TensorDataset and DataLoader for training set
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create TensorDataset and DataLoader for validation set
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Hyperparameters
input_size = 250
batch_size = 32
learning_rate = 0.001
epochs = 10

# Paths to training and validation data files
train_file_path = 'train_data.npy'
val_file_path = 'val_data.npy'

# Initialize the model
model = DeepNeuralNetwork(input_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# DataLoader for training and validation sets
train_loader, val_loader = create_dataloader(train_file_path, val_file_path, batch_size)

# Training loop
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_inputs, val_labels in val_loader:
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels).item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {val_loss}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
