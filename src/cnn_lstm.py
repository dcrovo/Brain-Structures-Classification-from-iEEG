import torch
import torch.nn as nn
from typing import List, Tuple

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
    
class CNN_LSTM_Model(nn.Module):
    """
    A model that combines a CNN feature extractor with an LSTM for sequence classification.

    Args:
        input_size (int): Number of input features (sequence length).
        num_classes (int): Number of output classes.
        conv_filters (List[int], optional): List of filter counts for each convolutional layer. Default is [64, 128, 256, 512].
        lstm_hidden_size (int, optional): Number of hidden units in the LSTM. Default is 128.
        lstm_num_layers (int, optional): Number of layers in the LSTM. Default is 1.
        fc_neurons (List[int], optional): List of neuron counts for each fully connected layer. Default is [1024, 256].
        dropout (float, optional): Dropout rate. Default is 0.3.
        activation (nn.Module, optional): Activation function to use. Default is nn.ReLU.
    """
    def __init__(self, input_size: int, num_classes: int, 
                 conv_filters: List[int] = [64, 128, 256, 512], 
                 lstm_hidden_size: int = 64, lstm_num_layers: int = 4,
                 fc_neurons: List[int] = [1024, 256],
                 dropout: float = 0.3, activation: nn.Module = nn.ReLU()):
        super(CNN_LSTM_Model, self).__init__()

        self.cnn_head = CNN_Head(input_size, conv_filters, dropout, activation)
        self.lstm = nn.LSTM(input_size=self.cnn_head.output_channels, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True)

        # Fully connected layers for concatenated output
        self.fc_layers = nn.ModuleList()
        lstm_output_size = lstm_hidden_size * self.cnn_head.conv_output_size
        self.fc_layers.append(nn.Linear(lstm_output_size, fc_neurons[0]))
        for i in range(len(fc_neurons) - 1):
            self.fc_layers.append(nn.Linear(fc_neurons[i], fc_neurons[i + 1]))
        self.output_layer = nn.Linear(fc_neurons[-1], num_classes)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, seq_length = x.size()
        conv_out = self.cnn_head(x)  # Pass through CNN head
        x_lstm = conv_out.permute(0, 2, 1)  # Reshape for LSTM (batch, seq_len, features)
        out_lstm, _ = self.lstm(x_lstm)  # Pass through LSTM

        x = out_lstm.contiguous().view(batch_size, -1)  # Flatten the output
        for fc in self.fc_layers:
            x = self.activation(fc(x))
            x = self.dropout(x)
        
        x = self.output_layer(x)
        return x

class ParallelCNNLSTMModel(nn.Module):
    def __init__(self, input_size: int, input_size_lstm: int, num_classes: int, 
                 conv_filters: List[int] = [64, 128, 256, 512], 
                 lstm_hidden_size: int = 64, lstm_num_layers: int = 4,
                 fc_neurons: List[int] = [1024, 256],
                 dropout: float = 0.3, activation: nn.Module = nn.ReLU()):
        super(ParallelCNNLSTMModel, self).__init__()

        # CNN Head
        self.cnn_head = CNN_Head(input_size, conv_filters, dropout, activation)
        
        # LSTM
        self.lstm = nn.LSTM(input_size=input_size_lstm, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True)
        
        # Fully connected layers for LSTM output
        self.fc_lstm = nn.Linear(lstm_hidden_size, 128)

        # Fully connected layers for concatenated output
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(128 + self.cnn_head.output_channels * self.cnn_head.conv_output_size, fc_neurons[0]))
        for i in range(len(fc_neurons) - 1):
            self.fc_layers.append(nn.Linear(fc_neurons[i], fc_neurons[i + 1]))
        self.output_layer = nn.Linear(fc_neurons[-1], num_classes)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN path
        batch_size, num_channels, seq_length = x.size()
        x_cnn = self.cnn_head(x)
        x_cnn = x_cnn.view(batch_size, -1)  # Flatten for concatenation
        
        # LSTM path
        x_lstm, _ = self.lstm(x.permute(0, 2, 1))  # Adjust shape for LSTM
        x_lstm = self.fc_lstm(x_lstm[:, -1, :])  # Take the last output of LSTM

        # Concatenate CNN and LSTM outputs
        x = torch.cat([x_cnn, x_lstm], dim=1)

        # Fully connected layers
        for fc in self.fc_layers:
            x = self.activation(fc(x))
            x = self.dropout(x)
        
        x = self.output_layer(x)
        return x
    

class LSTM_CNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, 
                 seq_length: int, 
                 conv_filters: List[int] = [64, 128, 256, 512], 
                 fc_neurons: List[int] = [1024,256],
                 dropout: float = 0.3, 
                 activation: nn.Module = nn.ReLU()):
        super(LSTM_CNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        cnn_layers = []
        in_channels = hidden_size
        for filters in conv_filters:
            cnn_layers.append(nn.Conv1d(in_channels, filters, kernel_size=3, stride=1, padding=1))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = filters
        self.cnn = nn.Sequential(*cnn_layers)
        
        self.flatten = nn.Flatten()
        
        # Calculate the size after convolutions and pooling
        cnn_output_size = seq_length // (2 ** len(conv_filters))
        fc_input_size = conv_filters[-1] * cnn_output_size
        
        fc_layers = []
        for neurons in fc_neurons:
            fc_layers.append(nn.Linear(fc_input_size, neurons))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            fc_input_size = neurons
        fc_layers.append(nn.Linear(fc_input_size, num_classes))
        
        self.fc = nn.Sequential(*fc_layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.permute(0, 2, 1)  # Adjust shape for CNN (batch, hidden_size, seq_length)
        out = self.cnn(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out