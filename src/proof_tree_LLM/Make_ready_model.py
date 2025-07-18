import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import os

# Hugging Face Token (required for gated models like LLaMA)
hf_token = os.environ.get("HF_TOKEN") or "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"

# Base model name (you can change this to another compatible model)
base_model = "meta-llama/Llama-2-7b-hf"

# Set 4-bit quantization config (bnb)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    use_fast=True,
    token=hf_token
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization and memory constraints
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={"": "auto"},
    token=hf_token,
    offload_folder="./offload"
)

# Safety for low memory GPUs
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for PEFT training
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Save tokenizer and config for reuse
model.save_pretrained("./qlora_model")
tokenizer.save_pretrained("./qlora_model")
