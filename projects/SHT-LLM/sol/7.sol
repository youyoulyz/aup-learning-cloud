# LLM07: FlashAttention — Solutions

## Solution Code for All TODOs

### 1. Standard Attention (Cell 3)

```python
def standard_attention(Q, K, V):
    """Naive scaled dot-product attention.
    Args: Q, K, V — (batch, num_heads, seq_len, head_dim)
    Returns: output — same shape as V

    Steps:
    1. Compute attention scores: Q @ K^T / sqrt(d_k)
    2. Apply softmax to get attention weights
    3. Apply attention weights to values
    """
    # Get head dimension from Q shape
    d_k = Q.size(-1)

    # Compute scaled attention scores
    # Hint: Use torch.matmul and transpose the last two dimensions of K
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply softmax along the last dimension
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply attention weights to values
    output = torch.matmul(attn_weights, V)

    return output
```

**Explanation:**
- `d_k = Q.size(-1)`: Gets the head dimension from the last dimension of Q
- `torch.matmul(Q, K.transpose(-2, -1))`: Computes Q @ K^T for attention scores
- `torch.softmax(scores, dim=-1)`: Normalizes scores to probabilities
- `torch.matmul(attn_weights, V)`: Applies attention weights to values


---

### 3. Tiled Matrix Multiplication (Cell 4)

```python
def tiled_matmul(A, B, tile_size=4):
    """Tile-based matmul — tiles are loaded into 'shared memory' once.

    Args:
        A: (M, K) input matrix
        B: (K, N) input matrix
        tile_size: size of each tile block

    Returns:
        C: (M, N) output matrix
        global_reads: number of global memory reads
    """
    M, K_ = A.shape
    K2, N = B.shape
    C = torch.zeros(M, N, device=A.device)
    global_reads = 0

    # Loop over tiles of C (ti, tj)
    for ti in range(0, M, tile_size):
        for tj in range(0, N, tile_size):
            # C_tile accumulates in "registers"
            # Loop over tiles of K (tk)
            for tk in range(0, K_, tile_size):
                # Load A-tile and B-tile into "shared memory"
                # Each tile is read once from global memory
                A_tile = A[ti:ti+tile_size, tk:tk+tile_size]
                B_tile = B[tk:tk+tile_size, tj:tj+tile_size]

                # Count global memory reads (2 values per element in each tile)
                global_reads += 2 * tile_size * tile_size

                # Accumulate tile multiplication into C
                C[ti:ti+tile_size, tj:tj+tile_size] += A_tile @ B_tile

    return C, global_reads
```

**Explanation:**
- Three nested loops: outer two iterate over output tiles (ti, tj), inner loop iterates over K dimension (tk)
- Each tile is loaded once from global memory and reused for multiple computations
- `A_tile = A[ti:ti+tile_size, tk:tk+tile_size]`: Load A tile with proper slicing
- `B_tile = B[tk:tk+tile_size, tj:tj+tile_size]`: Load B tile with proper slicing
- `global_reads += 2 * tile_size * tile_size`: Each element in both tiles is read once
- `C[ti:ti+tile_size, tj:tj+tile_size] += A_tile @ B_tile`: Accumulate tile multiplication

---



### 5. Online Softmax (Cell 5)

```python
def online_softmax(x: torch.Tensor, block_size: int = 4) -> torch.Tensor:
    """Compute softmax of 1-D tensor x using online (incremental) algorithm.

    Processes x in blocks of `block_size`, maintaining running max and sum.

    Algorithm:
    1. Initialize running max m = -inf and running sum d = 0
    2. For each block:
       - Find block max m_block
       - Update running max: m_new = max(m, m_block)
       - Rescale old sum: d = d * exp(m - m_new)
       - Add block contribution: d += sum(exp(block - m_new))
    3. Return: exp(x - m) / d
    """
    N = x.shape[0]

    # Initialize running statistics
    m = torch.tensor(float("-inf"))  # running max, start with -inf
    d = torch.tensor(0.0)            # running sum of exp, start with 0

    # Process each block
    for start in range(0, N, block_size):
        block = x[start : start + block_size]

        # Find block maximum
        # Hint: Use block.max() to find the maximum value in the current block
        m_block = block.max()

        # Update running maximum
        # Hint: m_new = max(m, m_block)
        m_new = torch.max(m, m_block)

        # Re-scale old sum and add new block contribution
        # Hint: d_new = d_old * exp(m_old - m_new) + sum(exp(block - m_new))
        d = d * torch.exp(m - m_new) + torch.exp(block - m_new).sum()

        m = m_new

    # Final softmax values
    # Hint: exp(x - m) / d
    return torch.exp(x - m) / d
```

**Explanation:**
- `m = torch.tensor(float("-inf"))`: Initialize running max to negative infinity
- `d = torch.tensor(0.0)`: Initialize running sum to zero
- `m_block = block.max()`: Find maximum in current block
- `m_new = torch.max(m, m_block)`: Update running maximum
- `d = d * torch.exp(m - m_new) + torch.exp(block - m_new).sum()`: Rescale old sum and add new contribution
- `torch.exp(x - m) / d`: Final normalization

---

### 6. FlashAttention V2 Simulation (Cell 7)

```python
def flash_attention_sim(Q, K, V, block_size=4):
    """Simplified FlashAttention V2 (Python simulation).

    This mirrors the real algorithm's logic but runs on CPU/GPU tensors
    without custom CUDA kernels. The purpose is educational.

    Args:
        Q, K, V: (seq_len, head_dim)
        block_size: tile size for K/V blocks
    Returns:
        O: (seq_len, head_dim)
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # Initialize output and running statistics
    O = torch.zeros_like(Q)  # Output accumulator
    l = torch.zeros(N, 1, device=Q.device)   # Running sum of exp (denominator)
    m = torch.full((N, 1), float("-inf"), device=Q.device)  # Running max

    # Outer loop over Q blocks (V2 style)
    for j in range(0, N, block_size):
        # Load K and V blocks into "SRAM"
        Kj = K[j : j + block_size]  # (block_size, d)
        Vj = V[j : j + block_size]  # (block_size, d)

        # Compute local attention scores for ALL query positions vs. this K-block
        S_block = (Q @ Kj.T) * scale  # (N, block_size)

        # Online softmax update
        m_block = S_block.max(dim=-1, keepdim=True).values  # (N, 1)
        m_new   = torch.max(m, m_block)

        # Correction factor for previously accumulated values
        correction = torch.exp(m - m_new)

        # New exponentials for this block
        p_block = torch.exp(S_block - m_new)  # (N, block_size)

        # Update running statistics
        l_new = l * correction + p_block.sum(dim=-1, keepdim=True)

        # Update output accumulator
        # Hint: rescale old O and add new contribution
        O = O * correction + p_block @ Vj

        m = m_new
        l = l_new

    # Final normalization
    O = O / l
    return O
```

**Explanation:**
- `O = torch.zeros_like(Q)`: Initialize output accumulator
- `l = torch.zeros(N, 1, ...)`: Running denominator (sum of exponentials)
- `m = torch.full((N, 1), float("-inf"), ...)`: Running maximum for numerical stability
- `Kj = K[j : j + block_size]`: Load K block
- `Vj = V[j : j + block_size]`: Load V block
- `S_block = (Q @ Kj.T) * scale`: Compute attention scores for all queries vs. current K block
- `m_block = S_block.max(dim=-1, keepdim=True).values`: Block maximum for each row
- `m_new = torch.max(m, m_block)`: Update running maximum
- `correction = torch.exp(m - m_new)`: Rescale factor for previously accumulated values
- `p_block = torch.exp(S_block - m_new)`: New exponentials for this block
- `l_new = l * correction + p_block.sum(dim=-1, keepdim=True)`: Update running sum
- `O = O * correction + p_block @ Vj`: Rescale old output and add new contribution
- `O = O / l`: Final normalization

---

### 7. Memory Analysis (Cell 9)

```python
def memory_analysis(N, d, dtype_bytes=2):
    """Compare memory footprint of standard vs flash attention.

    Args:
        N: sequence length
        d: head dimension
        dtype_bytes: bytes per element (2 for FP16)

    Returns:
        std_intermediate: memory for standard attention intermediates
        flash_intermediate: memory for flash attention intermediates
    """
    # Standard attention stores S (N×N) and P (N×N)
    std_intermediate = 2 * N * N * dtype_bytes  # bytes

    # Flash attention only stores running max, sum, and output per row
    # Hint: m (1 per row), l (1 per row), O (d per row)
    flash_intermediate = N * (1 + 1 + d) * dtype_bytes  # m, l, O per row

    return std_intermediate, flash_intermediate
```

**Explanation:**
- `std_intermediate = 2 * N * N * dtype_bytes`: Standard attention stores two N×N matrices (scores S and probabilities P)
- `flash_intermediate = N * (1 + 1 + d) * dtype_bytes`: Flash attention stores per-row statistics: m (1 value), l (1 value), and O (d values)

---

## Key Concepts Summary

| Concept | Key Formula/Insight |
|---------|---------------------|
| Standard Attention IO | $O(Nd + N^2)$ - stores full N×N score matrix |
| Tiled Matmul Reduction | $\frac{\text{naive reads}}{\text{tile size}}$ reduction |
| Online Softmax Update | $m_{new} = \max(m, m_{block})$, $d_{new} = d \cdot e^{m - m_{new}} + \sum e^{x - m_{new}}$ |
| FlashAttention IO | $O(N^2 d^2 / M)$ where M is SRAM size |
| Memory Savings | Standard: $2N^2$, Flash: $N(2 + d)$ |
| V1 vs V2 | V1: $O(N^2)$ O writes, V2: $O(N)$ O writes (deferred rescaling) |

---

## HIP Kernel Key Points

```hip
// Shared memory declaration
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// Thread synchronization
__syncthreads();  // Barrier within block

// Thread indices
int tx = threadIdx.x;
int ty = threadIdx.y;
int row = blockIdx.y * TILE_SIZE + ty;
int col = blockIdx.x * TILE_SIZE + tx;
```

**Tiling Strategy:**
1. Load tiles from global memory (HBM) to shared memory (SRAM)
2. Synchronize all threads in block
3. Compute using shared memory (fast!)
4. Repeat for all K-tiles
5. Write result to global memory
