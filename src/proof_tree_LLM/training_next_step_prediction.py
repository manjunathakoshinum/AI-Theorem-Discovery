import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from torch.optim import AdamW
from tqdm import tqdm
import os

# === Configurations ===
model_dir = "./llama-3.2-3b-instruct"
data_path = r"proof_tree_LLM\tokenized_prompt_dataset.jsonl"
save_dir = "D:/we_light_the_fire/trained_llama3_model"
batch_size = 1
num_epochs = 3
max_length = 512  # Truncate for memory efficiency
learning_rate = 5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_dir, legacy=True)
tokenizer.model_max_length = max_length
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Dataset ===
class TokenizedDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    "input_ids": item["input_ids"][:max_length],
                    "attention_mask": item["attention_mask"][:max_length],
                    "labels": item["labels"][:max_length]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx]["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(self.data[idx]["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.data[idx]["labels"], dtype=torch.long)
        }

# === Model Loading with 4-bit Quantization ===
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    quantization_config=bnb_config
)
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()

# === Data Loader ===
dataset = TokenizedDataset(data_path)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

# === Optimizer and Scheduler ===
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=50, num_training_steps=total_steps
)

# === Training Loop ===
model.train()
for epoch in range(num_epochs):
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

# === Save Final Model ===
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model and tokenizer saved to: {save_dir}")
