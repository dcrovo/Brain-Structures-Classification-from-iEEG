import torch
import torch.nn as nn
from typing import List, Tuple

import torch
import torch.nn as nn
from typing import List, Tuple

class LSTM_Ieeg_Model(nn.Module):
    def __init__(self, device: torch.device, input_size: int, num_classes: int, 
                 lstm_layers: int = 4, lstm_h_size: int = 4, 
                 fc_neurons: List[int] = [256, 128], 
                 dropout: float = 0.1, bidirectional: bool = False,
                 activation: nn.Module = nn.ReLU()):
        super(LSTM_Ieeg_Model, self).__init__()
        
        self.device = device
        self.lstm_layers = lstm_layers
        self.lstm_h_size = lstm_h_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.activation = activation
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, lstm_h_size, lstm_layers, 
                            dropout=dropout if lstm_layers > 1 else 0, 
                            bidirectional=bidirectional,
                            batch_first=True)
        
        # Calculate the dimension of LSTM output
        lstm_output_size = lstm_h_size * 2 if bidirectional else lstm_h_size
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(lstm_output_size, fc_neurons[0]))
        for i in range(len(fc_neurons) - 1):
            self.fc_layers.append(nn.Linear(fc_neurons[i], fc_neurons[i + 1]))
        
        # Output layer
        self.output_layer = nn.Linear(fc_neurons[-1], num_classes)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h_0 = torch.zeros(self.lstm_layers * (2 if self.bidirectional else 1), batch_size, self.lstm_h_size).to(self.device)
        c_0 = torch.zeros(self.lstm_layers * (2 if self.bidirectional else 1), batch_size, self.lstm_h_size).to(self.device)
        
        x, _ = self.lstm(x, (h_0, c_0))
        
        if self.bidirectional:
            # Concatenate the outputs from the last and first timesteps for bidirectional LSTM
            x = torch.cat((x[:, -1, :self.lstm_h_size], x[:, 0, self.lstm_h_size:]), dim=1)
        else:
            x = x[:, -1, :]
        
        for fc in self.fc_layers:
            x = self.activation(fc(x))
            x = self.dropout_layer(x)
        
        x = self.output_layer(x)
        
        return x


    

class LSTM_Ieeg_Model_2(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.3):
        super(LSTM_Ieeg_Model_2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

