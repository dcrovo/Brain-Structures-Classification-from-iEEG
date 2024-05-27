import torch
import torch.nn as nn
from typing import List

#Define Model
class Baseline_MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Baseline_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
    
class Baseline_ANN(nn.Module):

    """
    A deep artificial neural network (ANN) model for classification tasks.

    Args:
        input_size (int): Number of input features.
        num_classes (int): Number of output classes.
        hidden_layers (List[int], optional): List of neuron counts for each hidden layer. Default is [2048, 1024, 256, 64].
        dropout (float, optional): Dropout rate. Default is 0.1.
        activation (nn.Module, optional): Activation function to use. Default is nn.ReLU.
    """
    def __init__(self, input_size: int, num_classes: int, hidden_layers: List[int] = [2048, 1024, 256, 64], 
                 dropout: float = 0.1, activation: nn.Module = nn.ReLU()):
        super(Baseline_ANN, self).__init__()
        
        self.hidden_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # Create input layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_layers[0]))

        # Create hidden layers
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        # Create output layer
        self.output_layer = nn.Linear(hidden_layers[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        
        x = self.output_layer(x)
        return x