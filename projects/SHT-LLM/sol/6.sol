#3


batch_size = 2
seq_len = 4
hidden_dim = 8
head_dim = 8

q = torch.randn(batch_size, seq_len, head_dim)
k = torch.randn(batch_size, seq_len, head_dim)
v = torch.randn(batch_size, seq_len, head_dim)

mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
mask = mask.masked_fill(mask == 1, -torch.inf)

def calculate_attention_scores(query, key, mask=None):
    d_k = key.size(-1)

    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    scaled_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scaled_scores = scaled_scores + mask

    attention_weights = F.softmax(scaled_scores, dim=-1)

    return attention_weights

attention_weights = calculate_attention_scores(q, k, mask)


#4

# Implement Multi-Head Attention
print("Implementing multi-head attention mechanism")

# Configuration for multi-head attention
num_heads = 2
head_dim = hidden_dim // num_heads
print(f"Number of heads: {num_heads}")
print(f"Head dimension: {head_dim}")
print(f"Total dimension: {num_heads * head_dim}")

# Step 1: Reshape Q, K, V for multi-head processing
print(f"\nReshaping for multi-head attention:")
print(f"Original shape: {q.shape}")

# Reshape: [batch, seq_len, hidden_dim] -> [batch, seq_len, num_heads, head_dim]
q_heads = q.view(batch_size, seq_len, num_heads, head_dim)
k_heads = k.view(batch_size, seq_len, num_heads, head_dim)
v_heads = v.view(batch_size, seq_len, num_heads, head_dim)

# Transpose: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
q_heads = q_heads.transpose(1, 2)
k_heads = k_heads.transpose(1, 2)
v_heads = v_heads.transpose(1, 2)

print(f"Multi-head shape: {q_heads.shape}")
print(f"Shape interpretation: [batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}]")

# Step 2: Verify head separation
print(f"\nHead separation verification:")
print(f"Head 0 query (batch 0, position 0): {q_heads[0, 0, 0, :]}")
print(f"Head 1 query (batch 0, position 0): {q_heads[0, 1, 0, :]}")

# Step 3: Analyze head independence
head0_norm = q_heads[0, 0, :, :].norm(dim=-1)
head1_norm = q_heads[0, 1, :, :].norm(dim=-1)
print(f"\nHead 0 query norms: {head0_norm}")
print(f"Head 1 query norms: {head1_norm}")

# Step 4: Demonstrate that each head can learn different patterns
correlation = F.cosine_similarity(
    q_heads[0, 0, 0, :].unsqueeze(0),
    q_heads[0, 1, 0, :].unsqueeze(0)
)
print(f"Cosine similarity between head 0 and head 1 queries: {correlation.item():.4f}")
print("Low correlation indicates heads can learn different patterns")
