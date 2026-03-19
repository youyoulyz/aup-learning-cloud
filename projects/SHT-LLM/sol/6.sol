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