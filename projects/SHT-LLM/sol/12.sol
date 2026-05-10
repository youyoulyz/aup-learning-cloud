# LLM12: LLM Inference & KV-Cache — Solutions

## Solution Code for All TODOs

---

### TODO 1: `greedy_decode` — return argmax index

```python
    return logits.argmax(dim=-1).item()
```

- `logits.argmax(dim=-1)` returns the index of the largest logit along the vocabulary dimension.
- `.item()` converts the 0-d tensor to a Python int.

---

### TODO 2: `temperature_scale` — divide logits by T

```python
    return logits / temperature
```

- Dividing by T < 1 amplifies differences (sharper); T > 1 reduces differences (flatter).

---

### TODO 3: `top_k_filter` — mask-fill below threshold

```python
    return logits.masked_fill(logits < threshold, float("-inf"))
```

- `torch.topk(logits, k)` finds the k largest values; `topk_vals[..., -1]` is the k-th largest.
- `masked_fill` sets every logit below that threshold to `-inf`, effectively removing them from the distribution.

---

### TODO 4: `top_p_filter` — scatter sorted logits back

```python
    output.scatter_(-1, sorted_idx, sorted_logits)
```

- After masking low-probability tokens to `-inf` in `sorted_logits`, we must restore the original order.
- `scatter_` places values from `sorted_logits` back into `output` at the positions indicated by `sorted_idx`.

---

### TODO 5: `sample_token` — sample from distribution

```python
    return torch.multinomial(probs, num_samples=1).item()
```

- After applying temperature → top-k → top-p filtering, convert logits to probabilities with `torch.softmax`.
- `torch.multinomial` draws one sample according to the probability distribution.

---

### TODO 6: `apply_repetition_penalty` — divide positive logit

```python
            logits[token_id] /= penalty
```

- For each unique previously-generated token: if its logit is positive, divide by `penalty` (> 1 reduces it); if negative, multiply by `penalty` (makes it more negative). This pushes repeated tokens away from being selected.

---

### TODO 7: `apply_frequency_presence_penalty` — subtract frequency term

```python
        logits[token_id] -= frequency_penalty * count
```

- Use `collections.Counter` to count how many times each token appeared. Subtract `frequency_penalty * count` (scales with usage) and `presence_penalty` (flat cost per unique token). This is OpenAI's approach.

---

### TODO 8: `TinyDecoderLayer.forward` — Q projection

```python
        Q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
```

- Project normalized input through Q/K/V linear layers, reshape from `(B, T, hidden_dim)` to `(B, T, num_heads, head_dim)`, then transpose to `(B, num_heads, T, head_dim)` for attention.

---

### TODO 9: `TinyDecoderLayer.forward` — cache concatenation

```python
            K = torch.cat([kv_cache[0], K_new], dim=2)
```

- During decode, append the newly projected K/V to the cached past K/V along the sequence dimension (`dim=2`). This avoids re-projecting past tokens — the core KV-Cache optimization.

---

### TODO 10: `TinyDecoderLayer.forward` — attention call

```python
        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=(kv_cache is None and T > 1))
```

- Use `is_causal=True` only during prefill (when `kv_cache is None and T > 1`). During decode, `T=1` so a single query attends to all past tokens without needing a mask.

---

### TODO 11: `generate()` — prefill model call

```python
    logits, kv_caches = model(input_ids, kv_caches=None)
```

- Run the model once on the full prompt with no cache. This produces logits for all positions and initializes the KV caches for each layer.

---

### TODO 12: `generate()` — decode model call

```python
        logits, kv_caches = model(new_input, kv_caches=kv_caches)
```

- For each decode step, pass the single new token through the model, reusing the accumulated KV caches. This is the critical optimization — only 1 new token goes through K/V projection per step.

---

### TODO 13: `generate_no_cache()` — full recompute

```python
        logits, _ = model(input_ids, kv_caches=None)  # no cache — full recompute!
```

- Build the input from ALL generated tokens so far and run the model with `kv_caches=None`. This recomputes K/V projections on the entire sequence every step, making it $O(s^2)$ per token instead of $O(s)$.

---

## Summary Table

| TODO | Location | Answer | Key Concept |
|------|----------|--------|-------------|
| 1 | `greedy_decode` | `logits.argmax(dim=-1).item()` | Select highest-probability token |
| 2 | `temperature_scale` | `logits / temperature` | Sharpen (T<1) or flatten (T>1) distribution |
| 3 | `top_k_filter` | `logits.masked_fill(logits < threshold, float("-inf"))` | Keep only k largest logits |
| 4 | `top_p_filter` | `output.scatter_(-1, sorted_idx, sorted_logits)` | Restore original order after sorting+masking |
| 5 | `sample_token` | `torch.multinomial(probs, num_samples=1).item()` | Sample from filtered distribution |
| 6 | `apply_repetition_penalty` | `logits[token_id] /= penalty` | HuggingFace-style repetition avoidance |
| 7 | `apply_frequency_presence_penalty` | `logits[token_id] -= frequency_penalty * count` | OpenAI-style frequency penalty |
| 8 | `TinyDecoderLayer.forward` Q | `self.q_proj(x_norm).view(...).transpose(1, 2)` | Project and reshape Q tensor |
| 9 | `TinyDecoderLayer.forward` cache | `torch.cat([kv_cache[0], K_new], dim=2)` | KV-Cache concatenation |
| 10 | `TinyDecoderLayer.forward` attention | `F.scaled_dot_product_attention(Q, K, V, is_causal=...)` | Causal mask only during prefill |
| 11 | `generate()` prefill | `model(input_ids, kv_caches=None)` | One-shot full prompt processing |
| 12 | `generate()` decode | `model(new_input, kv_caches=kv_caches)` | Single-token decode with cache reuse |
| 13 | `generate_no_cache()` | `model(input_ids, kv_caches=None)` | Full recompute baseline |
