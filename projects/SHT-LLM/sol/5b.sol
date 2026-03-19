#2
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        RMSNorm layer as used in LLaMA and other modern models.
        
        Args:
            dim: The dimension to normalize (usually the last dimension)
            eps: Small epsilon to prevent division by zero
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        """Compute the RMS normalization"""
        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms
    
    def forward(self, x):
        """Forward pass with learnable scaling"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# 2.2
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        output = self.gamma * x_normalized + self.beta
        return output

batch_size, seq_len, hidden_dim = 4, 10, 32
input_tensor = torch.randn(batch_size, seq_len, hidden_dim)

custom_ln = CustomLayerNorm(hidden_dim)
pytorch_ln = nn.LayerNorm(hidden_dim)

with torch.no_grad():
    custom_ln.gamma.data.copy_(pytorch_ln.weight.data)
    custom_ln.beta.data.copy_(pytorch_ln.bias.data)

custom_output = custom_ln(input_tensor)
pytorch_output = pytorch_ln(input_tensor)

are_outputs_close = torch.allclose(custom_output, pytorch_output, atol=1e-5)

#3

class PostNormBlock(nn.Module):
    """Post-Norm Transformer Block (Original Transformer style)"""
    def __init__(self, dim):
        super().__init__()
        self.attention = SimpleAttention(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
        
    def forward(self, x):
        # Post-Norm: x → Attention → Add & Norm → FFN → Add & Norm
        x = self.ln1(x + self.attention(x))
        x = self.ln2(x + self.ffn(x))
        return x

class PreNormBlock(nn.Module):
    """Pre-Norm Transformer Block (Modern style)"""
    def __init__(self, dim):
        super().__init__()
        self.attention = SimpleAttention(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
        
    def forward(self, x):
        # Pre-Norm: x → Norm → Attention → Add → Norm → FFN → Add
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

#4
    class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as used in the original Transformer"""
    
    def __init__(self, dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create frequency dividers
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           (-math.log(10000.0) / dim))
        
        # Apply sine and cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: [max_len, 1, dim]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input embeddings"""
        # x shape: [batch_size, seq_len, dim]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :, :].transpose(0, 1)
        return self.dropout(x)

# Test positional encoding
print(" Advanced Positional Encoding Implementation")