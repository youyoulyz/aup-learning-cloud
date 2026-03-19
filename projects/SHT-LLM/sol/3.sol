#3

in_features, out_features = 5, 3
linear = nn.Linear(in_features, out_features).to(device)

x = torch.randn(2, 4, 8, in_features, device=device)
y_official = linear(x)

print("--- Setup ---")
print(f"Input Tensor Shape:  {x.shape}")
print(f"Linear Layer Weight Shape: {linear.weight.shape} (out_features, in_features)")
print(f"Linear Layer Bias Shape:   {linear.bias.shape} (out_features)")
print(f"Official Output Shape: {y_official.shape}")
print("-" * 30)

print("\n--- Implementing Manually (Matmul + Bias) ---")
y_matmul = torch.matmul(x, linear.weight.t())
y_manual = y_matmul + linear.bias

print(f"Shape after manual matmul: {y_matmul.shape}")
print(f"Shape after adding bias:   {y_manual.shape}")

print("\n--- Implementing with Einstein Summation (einsum) ---")
einsum_string = "...ij,kj->...ik"
y_einsum_matmul = torch.einsum(einsum_string, x, linear.weight)
y_einsum = y_einsum_matmul + linear.bias

print(f"Einsum string used: '{einsum_string}'")
print(f"Shape after einsum matmul: {y_einsum_matmul.shape}")
print(f"Shape after adding bias:   {y_einsum.shape}")

print("\n--- Verification ---")
manual_matches = torch.allclose(y_official, y_manual)
einsum_matches = torch.allclose(y_official, y_einsum)

print(f"Manual implementation matches official output: {manual_matches}")
print(f"Einsum implementation matches official output: {einsum_matches}")

assert manual_matches and einsum_matches, "Verification failed! Your implementations are incorrect."
print("\n🎉 Verification successful! All methods produce identical results.")

#4

batch_size, channels, height, width = 8, 3, 64, 64
image_batch = torch.randn(batch_size, channels, height, width)

print("--- Data Normalization Exercise ---")
print(f"Input image batch shape: {image_batch.shape}")

channel_mean = torch.mean(image_batch, dim=(0, 2, 3))
channel_std = torch.std(image_batch, dim=(0, 2, 3))

print(f"\nCalculated channel mean shape: {channel_mean.shape}")
print(f"Calculated channel std shape:  {channel_std.shape}")
assert channel_mean.shape == (channels,)

mean_reshaped = channel_mean.view(1, channels, 1, 1)
std_reshaped = channel_std.view(1, channels, 1, 1)

print(f"\nReshaped mean for broadcasting: {mean_reshaped.shape}")
print(f"Reshaped std for broadcasting:  {std_reshaped.shape}")
assert mean_reshaped.shape == (1, channels, 1, 1)

normalized_batch = (image_batch - mean_reshaped) / (std_reshaped + 1e-6) # Add epsilon for stability

print(f"\nShape after normalization: {normalized_batch.shape}")
assert normalized_batch.shape == image_batch.shape
print("Normalization successful!")

row_mask = torch.randn(batch_size, 1, height, 1)
masked_batch = normalized_batch * row_mask

print(f"\nChallenge: Mask shape: {row_mask.shape}")
print(f"Shape after applying mask: {masked_batch.shape}")
assert masked_batch.shape == image_batch.shape
print("Masking successful!")

print("\n--- Debugging Challenge ---")
a = torch.randn(2, 4)
b = torch.randn(4) # Corrected to be compatible for a simpler example
print(f"Shape of a: {a.shape}")
print(f"Shape of b: {b.shape}")
print("Operation `a + b` with corrected b shape.")

# To make the original 'b' of shape (3) compatible with 'a' of (2,4) is more complex.
# A more direct fix for a common case is making the trailing dimension match or be 1.
# Let's reshape b from (4) to (4) -> this is trivial, or let's assume b was (2,1)
b_compatible = torch.randn(2,1)
c = a + b_compatible
print(f"Reshaped b to: {b_compatible.shape}")
print(f"Successfully performed `a + b_compatible`. Output shape: {c.shape}")
assert c.shape == (2, 4)

#5
import math
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        qkv = self.qkv_proj(x)
        
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_transposed = k.transpose(-2, -1)
        scores = torch.matmul(q, k_transposed) / math.sqrt(self.head_dim)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        attention_output = self.out_proj(attention_output)

        x = self.norm1(x + self.dropout(attention_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x