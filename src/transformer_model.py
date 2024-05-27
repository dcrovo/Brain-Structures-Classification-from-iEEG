# transformer_model.py

import torch
import torch.nn as nn
import math
from typing import List

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, num_classes: int, num_heads: int, num_layers: int, dropout: float = 0.1, max_len: int = 5000, fc_neurons: List[int] = [128, 64]):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(model_dim, fc_neurons[0]))
        for i in range(len(fc_neurons) - 1):
            self.fc_layers.append(nn.Linear(fc_neurons[i], fc_neurons[i + 1]))
        self.fc_layers.append(nn.Linear(fc_neurons[-1], num_classes))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling

        for fc in self.fc_layers[:-1]:
            x = self.activation(fc(x))
            x = self.dropout(x)
        
        x = self.fc_layers[-1](x)
        return x
