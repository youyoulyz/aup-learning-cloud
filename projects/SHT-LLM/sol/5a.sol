#1.3
import torch
import torch.nn as nn

input_tensor = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [-1.0, -0.5, 0.5, 1.0],
    [5.0, 6.0, 1.0, 2.0]
], dtype=torch.float32)

class CustomRMSNorm(nn.Module):
    def __init__(self, normalized_shape, variance_epsilon=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rsqrt_variance = torch.rsqrt(variance + self.variance_epsilon)
        normalized_x = x * rsqrt_variance
        scaled_output = normalized_x * self.gamma
        return scaled_output

feature_dim = input_tensor.shape[-1]
rms_norm_layer = CustomRMSNorm(feature_dim)
layer_norm_layer = nn.LayerNorm(feature_dim)

rms_norm_output = rms_norm_layer(input_tensor)
layer_norm_output = layer_norm_layer(input_tensor)

rms_mean = rms_norm_output.mean(dim=-1)
rms_std = rms_norm_output.std(dim=-1)
layer_norm_mean = layer_norm_output.mean(dim=-1)
layer_norm_std = layer_norm_output.std(dim=-1)

#2
def __init__(self, dim=128, max_seq_len=512, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Pre-compute positional encodings
        pe = torch.zeros(max_seq_len, dim)
        
        # Position indices
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        print(f"Position tensor shape: {position.shape}")
        
        # Frequency computation
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           -(torch.log(torch.tensor(base)) / dim))
        print(f"Division term shape: {div_term.shape}")
        print(f"First few frequency values: {div_term[:5]}")
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)