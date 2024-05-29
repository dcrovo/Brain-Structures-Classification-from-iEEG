import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvEmbeddingStem(nn.Module):
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

class MultiheadSelfAttentionBlock(nn.Module):
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

class ConvTransformerModel(nn.Module):
    def __init__(self, input_size: int, num_classes: int, 
                 transformer_dim: int = 128, num_heads: int = 4, 
                 transformer_depth: int = 2,
                 fc_neurons: list = [512, 128],
                 dropout: float = 0.3, activation: nn.Module = nn.ReLU()):
        super(ConvTransformerModel, self).__init__()

        # Convolutional Embedding Stem
        self.conv_embedding_stem = ConvEmbeddingStem(in_channels=1, out_channels=transformer_dim)
        
        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [MultiheadSelfAttentionBlock(dim=transformer_dim, num_heads=num_heads, dropout=dropout) for _ in range(transformer_depth)]
        )
        self.fc_transformer = nn.Linear(transformer_dim, 128)

        # Fully connected layers for output
        self.transformer_output_size = 128  # Size after the fc_transformer layer

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.transformer_output_size, fc_neurons[0]))
        for i in range(len(fc_neurons) - 1):
            self.fc_layers.append(nn.Linear(fc_neurons[i], fc_neurons[i + 1]))
        self.output_layer = nn.Linear(fc_neurons[-1], num_classes)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transformer path
        x_transformer = self.conv_embedding_stem(x)
        x_transformer = x_transformer.permute(2, 0, 1)  # Adjust shape for Transformer
        for block in self.transformer_blocks:
            x_transformer = block(x_transformer)
        x_transformer = x_transformer[-1]  # Take the last output of Transformer
        x_transformer = self.fc_transformer(x_transformer)

        # Fully connected layers
        x = x_transformer
        for fc in self.fc_layers:
            x = self.activation(fc(x))
            x = self.dropout(x)
        
        x = self.output_layer(x)
        return x