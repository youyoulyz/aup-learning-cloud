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

