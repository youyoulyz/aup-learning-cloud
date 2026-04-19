# LLM08: Mixture of Experts (MoE) & Numerical Precision — Solutions

## Solution Code for All TODOs

### 1. Combining Layer - Concentrating Expert Outputs (New Cell)

```python
def combine_expert_outputs(expert_outputs, gate_probs, topk_idx):
    """Combine expert outputs using gate weights.
    
    This is the key operation that merges results from multiple experts.
    Formula: output = Σᵢ gate_weight[i] × expert_output[i]
    
    Args:
        expert_outputs: List of tensors, each (batch, seq_len, hidden_dim)
        gate_probs: (batch, seq_len, num_experts) — sparse gate weights
        topk_idx: (batch, seq_len, k) — indices of selected experts
    
    Returns:
        combined_output: (batch, seq_len, hidden_dim)
        combine_ops: Number of multiply-add operations performed
    """
    B, S, D = expert_outputs[0].shape
    num_experts = len(expert_outputs)
    k = topk_idx.shape[-1]
    
    # Initialize output tensor
    output = torch.zeros(B, S, D, device=expert_outputs[0].device)
    
    # Count operations for analysis
    multiply_ops = 0
    add_ops = 0
    
    # Combine expert outputs weighted by gate probabilities
    for i in range(num_experts):
        # Get gate weight for this expert (B, S, 1)
        expert_weight = gate_probs[:, :, i].unsqueeze(-1)
        
        # Only compute if some tokens selected this expert
        if expert_weight.sum() > 0:
            # Multiply expert output by gate weight
            weighted_output = expert_outputs[i] * expert_weight
            
            # Accumulate to output
            output += weighted_output
            
            # Count operations: D multiplies + D adds per token
            multiply_ops += B * S * D
            add_ops += B * S * D
    
    total_ops = multiply_ops + add_ops
    return output, total_ops
```

**Explanation:**
- `expert_weight = gate_probs[:, :, i].unsqueeze(-1)`: Get gate probability for expert i, reshape for broadcasting
- `weighted_output = expert_outputs[i] * expert_weight`: Scale expert output by gate weight
- `output += weighted_output`: Accumulate weighted outputs from all experts
- **Compute cost**: $O(k \cdot B \cdot S \cdot D)$ where $k$ is top-k, $B$ is batch, $S$ is seq_len, $D$ is hidden dim

---

### 2. Router Compute Cost Analysis (New Cell)

```python
def analyze_router_compute(hidden_dim, num_experts, k, batch_size, seq_len):
    """Analyze compute cost of the router (gating network).
    
    Router computes: G(x) = softmax(TopK(x @ W_g))
    
    Args:
        hidden_dim: Input hidden dimension (d)
        num_experts: Number of experts (E)
        k: Top-k selection
        batch_size: Batch size (B)
        seq_len: Sequence length (S)
    
    Returns:
        dict with compute breakdown
    """
    total_tokens = batch_size * seq_len
    
    # Step 1: Linear projection x @ W_g
    # W_g shape: (hidden_dim, num_experts)
    # For each token: hidden_dim * num_experts multiply-adds
    linear_ops = total_tokens * hidden_dim * num_experts * 2  # multiply + add
    
    # Step 2: Top-K selection
    # Using naive sorting: O(E log E) per token
    # For small E, this is negligible
    import math
    topk_ops = total_tokens * num_experts * math.log2(num_experts) if num_experts > 1 else 0
    
    # Step 3: Softmax over k values
    # exp, sum, divide for each of k values
    softmax_ops = total_tokens * k * 3  # exp + partial sum + divide
    
    # Total router ops
    total_router_ops = linear_ops + topk_ops + softmax_ops
    
    # Compare with expert FFN cost
    # FFN: gate_proj + up_proj + down_proj
    intermediate_dim = hidden_dim * 4  # Typical ratio
    expert_ops = total_tokens * (hidden_dim * intermediate_dim * 2 * 2 + intermediate_dim * hidden_dim * 2)
    
    return {
        'linear_proj_ops': linear_ops,
        'topk_ops': topk_ops,
        'softmax_ops': softmax_ops,
        'total_router_ops': total_router_ops,
        'single_expert_ffn_ops': expert_ops,
        'router_overhead_pct': total_router_ops / expert_ops * 100
    }
```

**Explanation:**
- **Linear projection**: $O(B \cdot S \cdot d \cdot E \cdot 2)$ — dominant cost, matrix multiplication
- **Top-K selection**: $O(B \cdot S \cdot E \log E)$ — negligible for small $E$
- **Softmax**: $O(B \cdot S \cdot k \cdot 3)$ — negligible for small $k$
- **Total router cost**: $O(d \cdot E)$ per token
- **Router overhead**: ~0.02% of expert compute (negligible!)

---

### 3. Top-K Gating Network (Cell 3)

```python
class TopKGating(nn.Module):
    """Router that selects top-k experts for each token.
    
    The gating network:
    1. Projects input to expert logits via linear layer
    2. Selects top-k experts per token
    3. Normalizes selected scores with softmax
    4. Creates sparse gate with only top-k nonzero values
    """

    def __init__(self, hidden_dim: int, num_experts: int, k: int = 2):
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.k = k
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            gate_probs: (batch, seq_len, num_experts) — sparse, only top-k nonzero
            topk_idx:   (batch, seq_len, k) — indices of selected experts
        """
        # Compute router scores (logits for each expert)
        scores = self.router(x)  # (B, S, E)
        
        # Select top-k scores and their indices
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)  # (B, S, k)
        
        # Normalize selected scores with softmax
        probs = torch.softmax(topk_scores, dim=-1)  # normalize among selected

        # Build sparse gate: only top-k positions are nonzero
        # Use scatter_ to place probabilities at selected indices
        gate = torch.zeros_like(scores)
        gate.scatter_(-1, topk_idx, probs)

        return gate, topk_idx
```

**Explanation:**
- `scores = self.router(x)`: Projects hidden states to expert logits
- `torch.topk(scores, self.k, dim=-1)`: Selects top-k experts per token
- `torch.softmax(topk_scores, dim=-1)`: Normalizes scores among selected experts
- `gate.scatter_(-1, topk_idx, probs)`: Places probabilities at selected expert indices, creating sparse gate

---

### 2. Expert FFN (Cell 4)

```python
class Expert(nn.Module):
    """A single FFN expert — identical structure to LLaMA's MLP.
    
    Uses SwiGLU activation: SiLU(gate) * up
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        # Initialize projection layers
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act_fn    = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation.
        
        Formula: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        """
        # Implement SwiGLU forward pass
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**Explanation:**
- `gate_proj`, `up_proj`, `down_proj`: Three linear layers for SwiGLU
- `self.act_fn(self.gate_proj(x)) * self.up_proj(x)`: SwiGLU gating mechanism
- This structure matches LLaMA's MLP with gated activation

---

### 3. MoE Layer Forward Pass (Cell 5)

```python
class MoELayer(nn.Module):
    """Mixture of Experts layer that replaces the standard FFN.
    
    The MoE layer:
    1. Routes each token to top-k experts via gating
    2. Computes each expert's output
    3. Combines outputs weighted by gate probabilities
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int,
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.gate = TopKGating(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList(
            [Expert(hidden_dim, intermediate_dim) for _ in range(num_experts)]
        )
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, hidden_dim)"""
        B, S, D = x.shape
        gate_probs, topk_idx = self.gate(x)  # (B,S,E), (B,S,k)

        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Compute each expert's output weighted by gate probability
        for i, expert in enumerate(self.experts):
            # Get gate weight for this expert
            expert_weight = gate_probs[:, :, i].unsqueeze(-1)  # (B, S, 1)
            
            # Only compute if some tokens selected this expert
            if expert_weight.sum() > 0:
                expert_out = expert(x)  # (B, S, D)
                # Accumulate weighted expert output
                output += expert_weight * expert_out

        return output
```

**Explanation:**
- `output = torch.zeros_like(x)`: Initialize output with same shape as input
- `expert_weight = gate_probs[:, :, i].unsqueeze(-1)`: Get probability for expert i, add dimension for broadcasting
- `if expert_weight.sum() > 0`: Skip computation if no tokens selected this expert
- `output += expert_weight * expert_out`: Accumulate weighted expert outputs

---

### 4. Load Balancing Loss (Cell 6)

```python
def load_balancing_loss(gate_probs: torch.Tensor, topk_idx: torch.Tensor,
                        num_experts: int) -> torch.Tensor:
    """
    Compute Switch Transformer load-balancing loss.
    
    The loss encourages uniform expert utilization by penalizing:
    - Experts that receive too many tokens (high f_i)
    - Experts with high average probabilities (high P_i)
    
    Args:
        gate_probs: (B, S, E) — sparse gate probabilities
        topk_idx:   (B, S, k) — indices of selected experts
        num_experts: total number of experts
    
    Returns:
        loss: scalar loss value
    """
    B, S, E = gate_probs.shape
    total_tokens = B * S

    # f_i — fraction of tokens routed to expert i
    expert_counts = torch.zeros(E, device=gate_probs.device)
    for i in range(E):
        expert_counts[i] = (topk_idx == i).float().sum()  # Count tokens routed to expert i
    f = expert_counts / total_tokens  # Normalize by total_tokens

    # P_i — average probability assigned to expert i
    P = gate_probs.mean(dim=(0, 1))  # Mean over batch and sequence dimensions

    # Compute loss = N_E * sum(f_i * P_i)
    loss = E * (f * P).sum()
    return loss
```

**Explanation:**
- `expert_counts[i] = (topk_idx == i).float().sum()`: Count how many tokens selected expert i
- `f = expert_counts / total_tokens`: Fraction of tokens routed to each expert
- `P = gate_probs.mean(dim=(0, 1))`: Average probability assigned to each expert
- `loss = E * (f * P).sum()`: Load balancing loss encourages uniform distribution

---

### 5. Model Memory Calculation (Cell 8)

```python
def model_memory(model, dtype=None):
    """Calculate model memory footprint for a given dtype.
    
    Args:
        model: PyTorch model
        dtype: target dtype (or None for original)
    
    Returns:
        total bytes required for model parameters
    """
    total = 0
    for p in model.parameters():
        if dtype:
            # Calculate memory with target dtype
            total += p.numel() * torch.tensor(0, dtype=dtype).element_size()
        else:
            # Calculate memory with original dtype
            total += p.numel() * p.element_size()
    return total
```

**Explanation:**
- `p.numel()`: Number of elements in the parameter tensor
- `torch.tensor(0, dtype=dtype).element_size()`: Size in bytes of the target dtype
- `p.element_size()`: Size in bytes of the parameter's current dtype

---

### 6. Quantization Functions (Cell 9)

```python
def quantize_tensor(x: torch.Tensor, num_bits: int = 8):
    """Symmetric quantization of a float tensor to num_bits integers.
    
    Symmetric quantization uses:
    - qmin = -(2^(num_bits-1))
    - qmax = 2^(num_bits-1) - 1
    - scale = max(|x|) / qmax
    - x_q = round(x / scale), clipped to [qmin, qmax]

    Returns:
        x_q:   quantized integer tensor
        scale: float scale factor for dequantization
    """
    # Calculate quantization range
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    # Calculate scale factor
    x_max = x.abs().max()  # Maximum absolute value
    scale = x_max / qmax

    # Quantize and clip to range
    x_q = torch.clamp(torch.round(x / scale), qmin, qmax).to(torch.int8)
    return x_q, scale


def dequantize_tensor(x_q: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize: int tensor -> float tensor.
    
    Formula: x_dequantized = x_q * scale
    """
    # Dequantize by multiplying with scale
    return x_q.float() * scale
```

**Explanation:**
- `qmin = -(2 ** (num_bits - 1))`: Minimum quantized value (e.g., -128 for INT8)
- `qmax = 2 ** (num_bits - 1) - 1`: Maximum quantized value (e.g., 127 for INT8)
- `scale = x.abs().max() / qmax`: Scale factor to map float range to int range
- `torch.clamp(torch.round(x / scale), qmin, qmax)`: Quantize and clip to valid range
- `x_q.float() * scale`: Dequantize by multiplying with scale

---

### 7. End-to-End Quantization (Cell 10)

```python
# Impact of quantization on model output
print("=== End-to-End Quantization Impact ===")

# Replace fc1 weight with dequantized INT8 weight
model_q8 = SimpleMLP(dim).to(device)
model_q8.load_state_dict(model_fp32.state_dict())

with torch.no_grad():
    # Quantize fc1 weight
    w = model_q8.fc1.weight.data.cpu()
    w_q, s = quantize_tensor(w, 8)
    model_q8.fc1.weight.data = dequantize_tensor(w_q, s).to(device)

    # Similarly quantize fc2 weight
    w2 = model_q8.fc2.weight.data.cpu()  # Get fc2 weight
    w2_q, s2 = quantize_tensor(w2, 8)    # Quantize
    model_q8.fc2.weight.data = dequantize_tensor(w2_q, s2).to(device)  # Dequantize and assign

    out_q8 = model_q8(x_test)
    out_orig = model_fp32(x_test)

output_error = (out_orig - out_q8).abs().mean().item()
relative_error = output_error / out_orig.abs().mean().item() * 100
print(f"Output mean abs error: {output_error:.6f}")
print(f"Relative error: {relative_error:.2f}%")
```

**Explanation:**
- `w2 = model_q8.fc2.weight.data.cpu()`: Get fc2 weight tensor
- `w2_q, s2 = quantize_tensor(w2, 8)`: Quantize to INT8
- `dequantize_tensor(w2_q, s2).to(device)`: Dequantize and move back to device

---

## Key Concepts Summary

### MoE Architecture

| Component | Purpose | Key Operation |
|-----------|---------|---------------|
| Router/Gate | Select experts | `Linear + TopK + Softmax` |
| Experts | Process tokens | `SwiGLU FFN` |
| Load Balancing | Prevent expert collapse | `L = N_E * Σ(f_i * P_i)` |

### Numerical Precision

| Format | Bits | Exponent | Mantissa | Use Case |
|--------|------|----------|----------|----------|
| FP32 | 32 | 8 | 23 | Full precision |
| BF16 | 16 | 8 | 7 | Training (wide range) |
| FP16 | 16 | 5 | 10 | Inference (narrow range) |
| INT8 | 8 | N/A | N/A | Quantized inference |
| INT4 | 4 | N/A | N/A | Highly compressed |

### Quantization Formulas

- **Scale**: `scale = max(|x|) / (2^(bits-1) - 1)`
- **Quantize**: `x_q = clamp(round(x / scale))`
- **Dequantize**: `x_deq = x_q * scale`
