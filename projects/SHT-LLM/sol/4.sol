#1.3
print("=== Working with Token Tensors ===")

sample_texts = [
    "Artificial intelligence is transforming the world.",
    "Machine learning models need large datasets for training.",
    "Deep learning uses neural networks."
]

print("Running batch processing...")
batch_encoded = tokenizer(
    sample_texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

batch_input_ids = batch_encoded["input_ids"]
batch_attention_mask = batch_encoded["attention_mask"]

print(f"Batch input IDs shape: {batch_input_ids.shape}")
print(f"Batch attention mask shape: {batch_attention_mask.shape}")

if torch.cuda.is_available():
    batch_input_ids = batch_input_ids.to(device)
    batch_attention_mask = batch_attention_mask.to(device)
    print(f"Batch tensors successfully moved to device: {batch_input_ids.device}")
else:
    print("CUDA not available. Tensors remain on CPU.")

print("\nAnalyzing padding per sample:")
for i in range(len(sample_texts)):
    sample_mask = batch_attention_mask[i]

    actual_tokens_count = sample_mask.sum().item()
    total_tokens_count = len(sample_mask)
    padded_tokens_count = total_tokens_count - actual_tokens_count

    print(f"  Sample {i+1}: {actual_tokens_count} real tokens, {padded_tokens_count} padding tokens")

#2.1


vocab_size = len(tokenizer)
print(f"Total vocabulary size: {vocab_size:,} tokens")

# Sample vocabulary entries
print(f"\nSample vocabulary entries:")
sample_ids = [0, 1, 2, 100, 1000, vocab_size-1]
for token_id in sample_ids:
    if token_id < vocab_size:
        token = tokenizer.decode([token_id])
        print(f"  ID {token_id:>6}: '{token}'")

# Most common tokens (usually shorter, more frequent)
print(f"\nFirst 20 tokens (typically most common):")
first_tokens = [(i, tokenizer.decode([i])) for i in range(min(20, vocab_size))]
for token_id, token in first_tokens:
    print(f"  {token_id:>2}: '{token}'")

# Token length analysis
print(f"\nToken length analysis:")
sample_size = min(1000, vocab_size)
token_lengths = []
for i in range(sample_size):
    token_text = tokenizer.decode([i])
    token_lengths.append(len(token_text))

if token_lengths:
    avg_length = sum(token_lengths) / len(token_lengths)
    max_length = max(token_lengths)
    min_length = min(token_lengths)
    
    print(f"  Sample size: {sample_size} tokens")
    print(f"  Average token length: {avg_length:.2f} characters")
    print(f"  Token length range: {min_length} - {max_length} characters")

# Special character handling
print(f"\nSpecial character analysis:")
special_chars = ['!', '@', '#', '$', '%', '&', '*', '(', ')', '-', '_', '=', '+']
for char in special_chars:
    tokens = tokenizer.tokenize(char)
    ids = tokenizer.encode(char, add_special_tokens=False)
    print(f"  '{char}' -> tokens: {tokens} -> IDs: {ids}")

# Number handling
print(f"\nNumber tokenization:")
numbers = ['0', '123', '2023', '3.14', '1,000,000']
for num in numbers:
    tokens = tokenizer.tokenize(num)
    print(f"  '{num}' -> {len(tokens)} tokens: {tokens}")

#3.1


def analyze_tokenization(word_list, tokenizer):
    categorized_words = {}
    most_fragmented_word = {"word": "", "count": 0}

    for word in word_list:
        tokens = tokenizer.tokenize(word)
        num_tokens = len(tokens)

        if num_tokens not in categorized_words:
            categorized_words[num_tokens] = []
        categorized_words[num_tokens].append(word)

        if num_tokens > most_fragmented_word["count"]:
            most_fragmented_word["word"] = word
            most_fragmented_word["count"] = num_tokens

    return categorized_words, most_fragmented_word

analysis_results, most_split_word = analyze_tokenization(test_words, tokenizer)

#4.1

special_tokens_dict = {'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

system_message = "You are a helpful assistant that summarizes technical documents."
user_query = "Based on the provided text, what is the main innovation described?"
long_context = "Tokenization is the process of breaking down a stream of text into smaller units called tokens. These tokens can be words, subwords, or characters. For Large Language Models (LLMs), subword tokenization, like Byte-Pair Encoding (BPE), is common. It balances vocabulary size and the ability to handle unknown words. " * 8
max_window_size = 256

def create_llm_prompt(system_message, user_query, context, tokenizer, max_context_length):
    final_prompt = f"<|system|>{system_message}<|endoftext|><|user|>{context}\n\n{user_query}<|endoftext|><|assistant|>"

    prompt_tokens = tokenizer.encode(final_prompt)
    system_tokens = tokenizer.encode(system_message, add_special_tokens=False)
    query_tokens = tokenizer.encode(user_query, add_special_tokens=False)
    context_tokens = tokenizer.encode(context, add_special_tokens=False)

    total_token_count = len(prompt_tokens)
    system_token_count = len(system_tokens)
    query_token_count = len(query_tokens)
    context_token_count = len(context_tokens)

    is_within_limit = total_token_count <= max_context_length

    system_percentage = (system_token_count / total_token_count * 100) if total_token_count > 0 else 0
    query_percentage = (query_token_count / total_token_count * 100) if total_token_count > 0 else 0
    context_percentage = (context_token_count / total_token_count * 100) if total_token_count > 0 else 0
    
    analysis = {
        "final_prompt": final_prompt,
        "total_tokens": total_token_count,
        "is_within_limit": is_within_limit,
        "component_analysis": {
            "system": {"tokens": system_token_count, "percentage": system_percentage},
            "user_query": {"tokens": query_token_count, "percentage": query_percentage},
            "context": {"tokens": context_token_count, "percentage": context_percentage},
        }
    }
    
    return analysis

prompt_analysis = create_llm_prompt(system_message, user_query, long_context, tokenizer, max_window_size)