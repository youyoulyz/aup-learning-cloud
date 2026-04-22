# LLM15: LLM Quantization and Evaluation - Solutions

## Section 2.1: Quantization Fundamentals

### TODO 1: quantize_tensor function
```python
def quantize_tensor(x: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, float]:
    """Quantize a tensor to specified bit-width."""
    # Calculate quantization parameters
    min_val = x.min()
    max_val = x.max()

    # Calculate scale and zero point
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = -min_val / scale + qmin

    # Quantize
    x_scaled = (x - min_val) / scale + qmin
    x_quantized = torch.clamp(torch.round(x_scaled), qmin, qmax)

    # Dequantize for comparison
    x_dequantized = (x_quantized - qmin) * scale + min_val

    return x_dequantized, scale.item(), zero_point.item()
```

### TODO 2: calculate_mse function
```python
def calculate_mse(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Calculate Mean Squared Error between original and quantized tensors."""
    return ((original - quantized) ** 2).mean().item()
```

### TODO 3: calculate_snr function
```python
def calculate_snr(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Calculate Signal-to-Noise Ratio in dB."""
    signal_power = (original ** 2).mean()
    noise_power = ((original - quantized) ** 2).mean()
    return 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()
```

## Section 3.3: AWQ Quantization

### TODO 4: awq_quantize function
```python
def awq_quantize(weight: torch.Tensor,
                 activation: torch.Tensor,
                 bits: int = 4,
                 salient_ratio: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simplified AWQ quantization with salient weight protection."""
    out_features, in_features = weight.shape

    # Calculate channel importance from activation magnitude
    channel_importance = activation.abs().mean(dim=0)

    # Identify salient channels (top salient_ratio%)
    n_salient = int(in_features * salient_ratio)
    _, salient_indices = torch.topk(channel_importance, n_salient)

    # Create scaling factors
    scale = torch.ones(in_features, device=weight.device)

    # Grid search for optimal scaling (simplified)
    best_scale = 0.5
    best_mse = float('inf')

    for s in [0.3, 0.4, 0.5, 0.6, 0.7]:
        scale_test = scale.clone()
        scale_test[salient_indices] = s

        weight_scaled = weight * scale_test.unsqueeze(0)
        weight_quant, _, _ = quantize_tensor(weight_scaled, bits)
        weight_unscaled = weight_quant / scale_test.unsqueeze(0)

        mse = calculate_mse(weight, weight_unscaled)

        if mse < best_mse:
            best_mse = mse
            best_scale = s

    # Apply best scaling
    scale[salient_indices] = best_scale
    weight_scaled = weight * scale.unsqueeze(0)
    weight_quant, _, _ = quantize_tensor(weight_scaled, bits)
    weight_final = weight_quant / scale.unsqueeze(0)

    return weight_final, salient_indices
```

## Section 5.3: LLM-Compressor Quantization

### TODO 5: quantize_with_llm_compressor function
```python
def quantize_with_llm_compressor(model_id: str, output_dir: str):
    """Quantize a model using LLM-Compressor with GPTQ."""
    if not COMPRESSOR_AVAILABLE:
        return

    print(f"🚀 Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )

    NUM_CALIBRATION_SAMPLES = 2048
    MAX_SEQUENCE_LENGTH = 1024
    # 1. Load calibration data
    print("📊 Preparing calibration data...")
    raw_dataset = load_dataset("allenai/c4", "default", split="train", streaming=True)

    # 2. Manually extract data and process (fix column name issues)
    calibration_data = []
    iterator = iter(raw_dataset)

    print(f"📥 Collecting {NUM_CALIBRATION_SAMPLES} samples...")
    while len(calibration_data) < NUM_CALIBRATION_SAMPLES:
        try:
            example = next(iterator)
            # C4 dataset uses example["text"]
            text = example["text"]

            # Tokenize and truncate the text
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=True
            )

            # Transformers expects a list of dicts with "input_ids" and "attention_mask"
            calibration_data.append({
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0]
            })
        except StopIteration:
            break
    final_dataset = Dataset.from_list(calibration_data)
    # 3. Prepare quantization recipe
    print("🚀 Preparing quantization recipe...")
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"]
    )

    # 4. Conduct quantization
    print(f"⚡ Starting quantization (expected < 2 minutes)...")
    oneshot(
        model=model,
        dataset=final_dataset,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    print(f"💾 Saving to: {output_dir}")
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print("✅ Quantization complete!")
```

## Section 6.2: MMLU Evaluation

### TODO 6: evaluate_mmlu_subset function
```python
def evaluate_mmlu_subset(model, tokenizer, subject: str = "computer_security",
                         n_samples: int = 100) -> float:
    """Evaluate on MMLU subset using logprob scoring."""
    if not DATASETS_AVAILABLE:
        print("datasets library not available.")
        return 0.0

    # Load MMLU subset
    dataset = load_dataset("cais/mmlu", name=subject, split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    correct = 0
    total = len(dataset)

    model.eval()

    for example in tqdm(dataset, desc=f"Evaluating {subject}"):
        question = example['question']
        choices = example['choices']
        answer = example['answer']

        # Format prompt
        prompt = f"Question: {question}\n\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "\nAnswer:"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        # Get logits and score each choice
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Logits for next token
            softmax = torch.softmax(logits, dim=-1)

            # Score each choice (A, B, C, D)
            choice_probs = []
            for i in range(len(choices)):
                choice_letter = chr(65+i)
                # Encode the choice letter with leading space (common tokenization)
                choice_token = tokenizer.encode(f" {choice_letter}", add_special_tokens=False)
                if choice_token:
                    token_id = choice_token[0]
                    prob = softmax[token_id].item()
                    choice_probs.append((i, prob))

            # Pick the choice with highest probability
            pred_idx = max(choice_probs, key=lambda x: x[1])[0]

            if pred_idx == answer:
                correct += 1

    accuracy = correct / total * 100
    return accuracy
```
