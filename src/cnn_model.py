import torch
import torch.nn as nn
from typing import List, Tuple

class CNN_Ieeg_Model(nn.Module):
    """
    A deep convolutional neural network (CNN) model for classification tasks.

    Args:
        input_size (int): Number of input features (sequence length).
        num_classes (int): Number of output classes.
        conv_filters (List[int], optional): List of filter counts for each convolutional layer. Default is [64, 128, 256, 512].
        fc_neurons (List[int], optional): List of neuron counts for each fully connected layer. Default is [1024, 256].
        dropout (float, optional): Dropout rate. Default is 0.3.
        activation (nn.Module, optional): Activation function to use. Default is nn.ReLU.
    """
    def __init__(self, input_size: int, num_classes: int, 
                 conv_filters: List[int] = [64, 128, 256, 512], 
                 fc_neurons: List[int] = [1024, 256], 
                 dropout: float = 0.3, 
                 activation: nn.Module = nn.ReLU()):
        super(CNN_Ieeg_Model, self).__init__()
        
        # Initialize lists to hold convolutional and batch normalization layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # Store the activation function and dropout layer
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Create convolutional layers
        in_channels = 1  # Initial number of input channels (1 for 1D signals)
        for filters in conv_filters:
            self.conv_layers.append(nn.Conv1d(in_channels, filters, kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm1d(filters))
            in_channels = filters  # Update the input channels for the next layer

        # Calculate the size after convolutions and pooling
        conv_output_size = input_size
        for _ in conv_filters:
            conv_output_size = conv_output_size // 2  # Each maxpool layer halves the size

        
        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(conv_filters[-1] * conv_output_size, fc_neurons[0]))
        for i in range(len(fc_neurons) - 1):
            self.fc_layers.append(nn.Linear(fc_neurons[i], fc_neurons[i + 1]))

        # Output layer
        self.output_layer = nn.Linear(fc_neurons[-1], num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Output tensor and list of feature maps from each layer.
        """
        feature_maps = []
        
        # Forward pass through convolutional layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.activation(bn(conv(x)))
            feature_maps.append(x)  # Store feature map
            x = self.maxpool(x)
            x = self.dropout(x)
        
        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Forward pass through fully connected layers
        for fc in self.fc_layers:
            x = self.activation(fc(x))
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x, feature_maps


class CNN_Head(nn.Module):
    """
    A deep convolutional neural network (CNN) model for feature extraction.

    Args:
        input_size (int): Number of input features (sequence length).
        conv_filters (List[int], optional): List of filter counts for each convolutional layer. Default is [64, 128, 256, 512].
        dropout (float, optional): Dropout rate. Default is 0.3.
        activation (nn.Module, optional): Activation function to use. Default is nn.ReLU.
    """
    def __init__(self, input_size: int, 
                 conv_filters: List[int] = [64, 128, 256, 512], 
                 dropout: float = 0.3, 
                 activation: nn.Module = nn.ReLU()):
        super(CNN_Head, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        in_channels = 1  # Initial number of input channels (1 for 1D signals)
        for filters in conv_filters:
            self.conv_layers.append(nn.Conv1d(in_channels, filters, kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm1d(filters))
            in_channels = filters  # Update the input channels for the next layer

        # Calculate the size after convolutions and pooling
        conv_output_size = input_size
        for _ in conv_filters:
            conv_output_size = conv_output_size // 2  # Each maxpool layer halves the size

        self.conv_output_size = conv_output_size
        self.output_channels = conv_filters[-1]  # Number of channels in the final convolution layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.activation(bn(conv(x)))
            x = self.maxpool(x)
            x = self.dropout(x)
        
        return x