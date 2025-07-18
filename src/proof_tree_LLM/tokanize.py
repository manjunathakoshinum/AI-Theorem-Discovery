from transformers import AutoTokenizer
from tqdm import tqdm
import json
import os

# Configuration
MODEL_PATH = "./llama-3.2-3b-instruct"  # Local path to tokenizer files
INPUT_FILE = "proof_tree_LLM/llama_prompt_dataset.jsonl"
OUTPUT_FILE = "proof_tree_LLM/tokenized_prompt_dataset.jsonl"
MAX_LENGTH = 512  # Truncate long sequences for memory efficiency

# Load tokenizer
print(f"🔍 Loading tokenizer from: {MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
except Exception as e:
    print("❌ Failed to load tokenizer. Check that tokenizer.json and tokenizer_config.json exist.")
    raise e

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_example(example):
    prompt = example["prompt"]
    target = example["target"]

    # Tokenize prompt and target separately
    prompt_tokens = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH // 2)
    target_tokens = tokenizer(target, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH // 2)

    input_ids = prompt_tokens["input_ids"] + target_tokens["input_ids"]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_tokens["input_ids"]) + target_tokens["input_ids"]

    # Final truncation to MAX_LENGTH
    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# Load JSONL file
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Save to JSONL file
def save_tokenized(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")

# Main pipeline
def main():
    print(f"📂 Loading dataset from: {INPUT_FILE}")
    examples = load_jsonl(INPUT_FILE)

    print("🧠 Tokenizing examples...")
    tokenized_dataset = []
    for ex in tqdm(examples):
        tokenized = tokenize_example(ex)
        tokenized_dataset.append(tokenized)

    print(f"💾 Saving tokenized dataset to: {OUTPUT_FILE}")
    save_tokenized(tokenized_dataset, OUTPUT_FILE)
    print("✅ Tokenization complete.")

if __name__ == "__main__":
    main()
