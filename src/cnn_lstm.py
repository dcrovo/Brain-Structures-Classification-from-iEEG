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
    

class TransformerStem(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=10, stride=2, bias=False, padding=4)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(out_channels // 2)

        self.conv2 = nn.Conv1d(out_channels // 2, out_channels, kernel_size=3, stride=1, bias=False, padding=1)
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, bias=False, padding=1)
        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class CNNHead(nn.Module):
    def __init__(self, input_size, conv_filters, dropout, activation):
        super(CNNHead, self).__init__()
        layers = []
        in_channels = 1
        for out_channels in conv_filters:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(activation)
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*layers)
        self.output_channels = in_channels
        self.conv_output_size = input_size // (2 ** len(conv_filters))

    def forward(self, x):
        return self.conv_layers(x)

class ParallelCNNTransformerModel(nn.Module):
    def __init__(self, input_size: int, input_size_transformer: int, num_classes: int, 
                 conv_filters: list = [64, 128, 256, 512], 
                 transformer_dim: int = 512, num_heads: int = 8, 
                 transformer_depth: int = 4,
                 fc_neurons: list = [1024, 256],
                 dropout: float = 0.3, activation: nn.Module = nn.ReLU()):
        super(ParallelCNNTransformerModel, self).__init__()

        # CNN Head
        self.cnn_head = CNNHead(input_size, conv_filters, dropout, activation)

        # Transformer
        self.transformer_stem = TransformerStem(in_channels=1, out_channels=transformer_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(dim=transformer_dim, num_heads=num_heads, dropout=dropout) for _ in range(transformer_depth)]
        )
        self.fc_transformer = nn.Linear(transformer_dim, 128)

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

        # Transformer path
        x_transformer = self.transformer_stem(x)
        x_transformer = x_transformer.permute(2, 0, 1)  # Adjust shape for Transformer
        for block in self.transformer_blocks:
            x_transformer = block(x_transformer)
        x_transformer = x_transformer[-1]  # Take the last output of Transformer
        x_transformer = self.fc_transformer(x_transformer)

        # Concatenate CNN and Transformer outputs
        x = torch.cat([x_cnn, x_transformer], dim=1)

        # Fully connected layers
        for fc in self.fc_layers:
            x = self.activation(fc(x))
            x = self.dropout(x)
        
        x = self.output_layer(x)
        return x