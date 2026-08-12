import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from src.tokenizer import get_tokenizer

class ChatMLDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=1024):
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format using chat template
        prompt = self.tokenizer.apply_chat_template(item["messages"], tokenize=False)
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        
        # Labels are same as input_ids for Causal LM, we ignore pad tokens
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": input_ids, "labels": labels}

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")
    
    tokenizer = get_tokenizer()
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Configure LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    
    # Check parameters
    model.print_trainable_parameters()
    
    dataset = ChatMLDataset("data/train.jsonl", tokenizer)
    
    # Configure Trainer
    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir="./hf_export/checkpoints",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Silences the MPS pin_memory warning
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Training complete! Merging LoRA weights...")
    model = model.to("cpu")
    model = model.merge_and_unload()  # type: ignore
    
    print("Saving fully merged model to hf_export...")
    os.makedirs("hf_export", exist_ok=True)
    
    assert model is not None, "Model should not be None after merge"
    assert tokenizer is not None, "Tokenizer should not be None"
    
    model.save_pretrained("hf_export") # type: ignore
    tokenizer.save_pretrained("hf_export")
    print("Saved to hf_export! You can now use chat_app.py or inference.py.")

if __name__ == "__main__":
    train()
