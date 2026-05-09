# LLM09: LoRA Fine-Tuning — Solutions

## Solution Code for All TODOs

---

### Cell 2: LoRALayer — `__init__` (TODO 1: Initialize A matrix)

```python
self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
```

**Explanation:**
- `torch.randn(in_dim, rank)`: Creates a matrix of shape `(in_dim, rank)` filled with values drawn from a standard normal distribution N(0, 1).
- `* std_dev`: Scales the values so that the standard deviation becomes `1 / sqrt(rank)`. This is critical for training stability — without this scaling, the product `x @ A @ B` could have very large or very small values when rank is large.
- `nn.Parameter(...)`: Wraps the tensor as a learnable parameter that will be registered with the module and included in `model.parameters()`.
- **Why Gaussian?** A is the "projection down" matrix. Random initialization ensures the low-rank subspace is explored uniformly. The small variance (1/sqrt(rank)) keeps the initial LoRA output near zero (since B starts at zero anyway).
- **Shape:** `(in_dim, rank)` — projects input from `in_dim` dimensions down to `rank` dimensions.

---

### Cell 2: LoRALayer — `__init__` (TODO 2: Create dropout layer)

```python
self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
```

**Explanation:**
- `nn.Dropout(dropout)`: During training, randomly zeros out elements with probability `dropout`. This acts as regularization to prevent overfitting.
- `nn.Identity()`: A no-op layer (passes input through unchanged). Used when `dropout=0.0` so we don't need conditional logic in `forward()`.
- **Calling convention:** `nn.Dropout(p)` where `p` is the probability of zeroing an element. `p=0.05` means 5% of elements are zeroed.
- **Why on input?** Applying dropout to the input `x` before the LoRA computation helps prevent the adapter from overfitting to specific input patterns.

---

### Cell 2: LoRALayer — `forward` (TODO 3: First matrix multiplication)

```python
intermediate = x_dropped @ self.A  # Shape: (..., rank)
```

**Explanation:**
- `@` is the matrix multiplication operator in Python (equivalent to `torch.matmul`).
- `x_dropped`: Shape `(..., in_dim)` where `...` can be any batch/sequence dimensions.
- `self.A`: Shape `(in_dim, rank)`.
- Result `intermediate`: Shape `(..., rank)` — the input projected into the low-rank space.
- **Why first multiply by A?** This reduces the dimensionality from `in_dim` to `rank`. The intermediate result is much smaller than the full weight matrix, saving memory and compute.
- **Broadcasting:** PyTorch automatically broadcasts the batch dimensions. For example, if `x_dropped` is `(batch, seq_len, in_dim)` and `A` is `(in_dim, rank)`, the result is `(batch, seq_len, rank)`.

---

### Cell 2: LoRALayer — `forward` (TODO 4: Second matrix multiplication)

```python
output = intermediate @ self.B  # Shape: (..., out_dim)
```

**Explanation:**
- `intermediate`: Shape `(..., rank)`.
- `self.B`: Shape `(rank, out_dim)`.
- Result `output`: Shape `(..., out_dim)` — the low-rank adaptation projected back to the output space.
- **Two-step vs. one-step:** We could compute `(x @ A @ B)` as `x @ (A @ B)`, but that would create a full `(in_dim, out_dim)` matrix first, defeating the purpose of LoRA. The two-step approach keeps memory usage at `O(rank)` instead of `O(in_dim * out_dim)`.
- **Final step:** After this line, the result is multiplied by `self.scaling` (i.e., `alpha / rank`) to control adaptation strength.

---

### Cell 3: LinearWithLoRA — `freeze_original_parameters` (TODO 5)

```python
param.requires_grad = False
```

**Explanation:**
- `requires_grad`: A PyTorch flag that determines whether a tensor should track gradients for backpropagation.
- Setting it to `False` means:
  - The parameter's values will NOT be updated during `optimizer.step()`.
  - No gradient will be computed for this parameter during `loss.backward()`.
  - This saves memory (no gradient stored) and compute.
- **Why freeze?** The original linear layer contains pre-trained knowledge. We want to preserve this and only train the LoRA adapter (A and B matrices).
- **Calling convention:** `param.requires_grad = False` must be set BEFORE the training loop starts.

---

### Cell 3: LinearWithLoRA — `unfreeze_original_parameters` (TODO 6)

```python
param.requires_grad = True
```

**Explanation:**
- Re-enables gradient tracking for the original linear layer parameters.
- **When to use:** This is for comparison experiments only (e.g., comparing LoRA vs. full fine-tuning). In normal LoRA training, you would NOT call this.
- **Note:** After unfreezing, these parameters will be updated by the optimizer. If you want to do full fine-tuning alongside LoRA, you'd need to add these params to the optimizer as well.

---

### Cell 6: `apply_lora_to_network` — Strategy "all" (TODO 7)

```python
indices_to_modify = [i for i, _ in linear_layers]
```

**Explanation:**
- `linear_layers` is a list of `(index, layer)` tuples from `model.get_linear_layers()`.
- `[i for i, _ in linear_layers]` is a list comprehension that extracts just the indices.
- The `_` is a Python convention for "I don't need this value" — it's the `nn.Linear` layer object.
- **Result:** A list like `[0, 2, 4]` — the indices of ALL linear layers in the model's `nn.Sequential`.
- **Effect:** Every linear layer will get a LoRA adapter. This gives maximum adaptation but uses the most parameters.

---

### Cell 6: `apply_lora_to_network` — Strategy "output_only" (TODO 8)

```python
indices_to_modify = [linear_layers[-1][0]]
```

**Explanation:**
- `linear_layers[-1]`: Gets the LAST `(index, layer)` tuple — the output layer.
- `[0]`: Extracts the index from the tuple.
- `[...]`: Wraps it in a list (the loop below expects a list).
- **Result:** A list with a single index, e.g., `[4]` — only the output layer.
- **Rationale:** In many tasks, the output layer is most task-specific and benefits most from adaptation. The hidden layers may already extract useful features.

---

### Cell 6: `apply_lora_to_network` — Strategy "input_output" (TODO 9)

```python
indices_to_modify = [linear_layers[0][0], linear_layers[-1][0]]
```

**Explanation:**
- `linear_layers[0][0]`: Index of the FIRST linear layer (input projection).
- `linear_layers[-1][0]`: Index of the LAST linear layer (output projection).
- **Result:** A list like `[0, 4]` — the first and last linear layers.
- **Rationale:** A balanced approach — adapt the input (how the model reads the data) and output (how the model produces predictions), but leave the hidden layers frozen. This is a good middle ground between "all" and "output_only".

---

### Cell 7: `freeze_linear_layers` — Keep LoRA trainable (TODO 10)

```python
param.requires_grad = True
```

**Explanation:**
- This line is inside the `if exclude_lora and "lora" in name.lower():` block.
- It ensures that LoRA parameters (A and B matrices) remain trainable.
- **Name matching:** `model.named_parameters()` yields names like `"layers.0.lora.A"`, `"layers.0.lora.B"`. The `"lora" in name.lower()` check identifies these.
- **Why explicit?** Even though LoRA params are created with `requires_grad=True` by default, this function explicitly sets it to be safe. Some model loading/conversion code might accidentally freeze everything.

---

### Cell 7: `freeze_linear_layers` — Freeze other params (TODO 11)

```python
param.requires_grad = False
```

**Explanation:**
- This is the `else` branch — applies to ALL parameters that are NOT LoRA.
- This includes:
  - Base model weights (`linear.weight`, `linear.bias`)
  - Embedding layers
  - Layer normalization
  - Any other model parameters
- **Effect:** Only LoRA A,B matrices will receive gradients and be updated by the optimizer. This is the core of parameter-efficient fine-tuning.
- **Memory savings:** Frozen params don't store gradients, saving ~50% of training memory for those parameters.

---

## Complete LoRALayer Class (for reference)

```python
class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer implementing ΔW = A × B^T decomposition.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        rank: Rank of the decomposition (r)
        alpha: Scaling factor for LoRA adaptation
        dropout: Dropout probability for regularization
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Initialize A matrix with Gaussian distribution
        # Standard deviation based on rank for stable initialization
        std_dev = 1 / math.sqrt(rank)
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)  # TODO 1

        # Initialize B matrix with zeros (important for stable training start)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()  # TODO 2

        # Scaling factor - controls adaptation strength
        self.scaling = alpha / rank

    def forward(self, x):
        """
        Forward pass: x @ A @ B^T * scaling
        Efficient computation: (x @ A) @ B^T
        """
        # Apply dropout to input if specified
        x_dropped = self.dropout(x)

        # Efficient computation: (x @ A) @ B^T
        # This avoids creating the full A @ B^T matrix
        intermediate = x_dropped @ self.A  # Shape: (..., rank)  # TODO 3
        output = intermediate @ self.B  # Shape: (..., out_dim)  # TODO 4

        return output * self.scaling
```

## Complete LinearWithLoRA Class (for reference)

```python
class LinearWithLoRA(nn.Module):
    """
    Linear layer enhanced with LoRA adaptation.
    Combines frozen pre-trained weights with trainable low-rank adaptation.
    """

    def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            in_dim=linear_layer.in_features, out_dim=linear_layer.out_features, rank=rank, alpha=alpha, dropout=dropout
        )
        self.freeze_original_parameters()

    def freeze_original_parameters(self):
        """Freeze original linear layer parameters"""
        for param in self.linear.parameters():
            param.requires_grad = False  # TODO 5

    def unfreeze_original_parameters(self):
        """Unfreeze original linear layer parameters (for comparison)"""
        for param in self.linear.parameters():
            param.requires_grad = True  # TODO 6

    def forward(self, x):
        """
        Forward pass combining original and LoRA outputs
        """
        original_output = self.linear(x)
        lora_output = self.lora(x)
        return original_output + lora_output
```

## Complete `apply_lora_to_network` (for reference)

```python
def apply_lora_to_network(model, strategy="all", rank=4, alpha=8.0):
    """
    Apply LoRA to network layers based on different strategies

    Args:
        model: The neural network model
        strategy: 'all', 'output_only', 'input_output', or 'selective'
        rank: LoRA rank parameter
        alpha: LoRA alpha parameter
    """
    linear_layers = model.get_linear_layers()

    if strategy == "all":
        indices_to_modify = [i for i, _ in linear_layers]  # TODO 7
    elif strategy == "output_only":
        indices_to_modify = [linear_layers[-1][0]]  # TODO 8
    elif strategy == "input_output":
        indices_to_modify = [linear_layers[0][0], linear_layers[-1][0]]  # TODO 9
    elif strategy == "selective":
        indices_to_modify = []
        for i, layer in linear_layers:
            layer_size = layer.in_features * layer.out_features
            if layer_size > 10000:
                layer_rank = rank * 2
            else:
                layer_rank = rank
            indices_to_modify.append((i, layer_rank))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    modified_count = 0
    for item in indices_to_modify:
        if isinstance(item, tuple):
            i, layer_rank = item
        else:
            i, layer_rank = item, rank

        original_layer = model.layers[i]
        lora_layer = LinearWithLoRA(original_layer, rank=layer_rank, alpha=alpha).to(device)
        model.layers[i] = lora_layer
        modified_count += 1

    return modified_count
```

## Complete `freeze_linear_layers` (for reference)

```python
def freeze_linear_layers(model, exclude_lora=True):
    """
    Freeze linear layer parameters while optionally preserving LoRA trainability

    Args:
        model: Neural network model
        exclude_lora: If True, keep LoRA parameters trainable
    """
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        if exclude_lora and "lora" in name.lower():
            param.requires_grad = True  # TODO 10
            trainable_params += param.numel()
        else:
            param.requires_grad = False  # TODO 11
            frozen_params += param.numel()

    return frozen_params, trainable_params
```

## Summary Table

| TODO | Location | Answer | Key Concept |
|------|----------|--------|-------------|
| 1 | `LoRALayer.__init__` | `torch.randn(in_dim, rank) * std_dev` | Gaussian init with variance 1/rank |
| 2 | `LoRALayer.__init__` | `nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()` | Conditional dropout layer |
| 3 | `LoRALayer.forward` | `x_dropped @ self.A` | Project to low-rank space |
| 4 | `LoRALayer.forward` | `intermediate @ self.B` | Project back to output space |
| 5 | `freeze_original_parameters` | `param.requires_grad = False` | Freeze base weights |
| 6 | `unfreeze_original_parameters` | `param.requires_grad = True` | Unfreeze for comparison |
| 7 | `apply_lora_to_network` | `[i for i, _ in linear_layers]` | All layers |
| 8 | `apply_lora_to_network` | `[linear_layers[-1][0]]` | Last layer only |
| 9 | `apply_lora_to_network` | `[linear_layers[0][0], linear_layers[-1][0]]` | First + last |
| 10 | `freeze_linear_layers` | `param.requires_grad = True` | Keep LoRA trainable |
| 11 | `freeze_linear_layers` | `param.requires_grad = False` | Freeze non-LoRA |
