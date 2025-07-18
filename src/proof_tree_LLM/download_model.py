import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from accelerate import Accelerator
import os

# Hugging Face token for authentication (replace with your token)
hf_token = "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"

# Model you want to download (change to your desired model)
model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # You can change this to the 7B model or smaller ones

# Define the configuration for 4-bit quantization (for saving memory)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Select optimal quantization
    bnb_4bit_use_double_quant=True,  # Enables double quantization to optimize space
    bnb_4bit_compute_dtype=torch.float16  # Uses mixed precision for GPU operations
)

# Function to load model and tokenizer with offloading
def load_model(model_id, hf_token, bnb_config):
    try:
        # Load the model and tokenizer
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
        
        # Load model with offloading configurations
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",  # Auto-distribute across available resources (GPU/CPU)
            torch_dtype=torch.float16,  # For GPU usage optimization
            low_cpu_mem_usage=True,  # Low memory usage on CPU
            token=hf_token
        )
        print("Model loaded successfully.")
        
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

# Download and load the model
model, tokenizer = load_model(model_id, hf_token, bnb_config)

if model and tokenizer:
    print(f"Model {model_id} and tokenizer loaded.")
else:
    print("Failed to load the model.")

# Example of using the model for inference
def generate_text(prompt, model, tokenizer):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to the appropriate device (CPU/GPU)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Generate output using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,  # Adjust length as needed
            num_return_sequences=1,  # Number of sequences to generate
            temperature=0.7,  # Adjust temperature for creativity
        )

    # Decode and return the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage: generate text from a prompt
prompt = "In the field of hyperbolic geometry, prove that"
generated_text = generate_text(prompt, model, tokenizer)
print(f"Generated Text: {generated_text}")
