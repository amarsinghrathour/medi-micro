import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import get_medimicro_300m_model
from tokenizer import get_tokenizer

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
        
        return input_ids, labels

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")
    
    tokenizer = get_tokenizer()
    model = get_medimicro_300m_model().to(device)
    
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    
    dataset = ChatMLDataset("data/train.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    epochs = 3
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (input_ids, labels) in enumerate(pbar):
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision for MPS
            with torch.autocast(device_type="mps", dtype=torch.bfloat16) if str(device) == "mps" else torch.autocast(device_type="cpu"):
                logits = model(input_ids)
                
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1),
                    ignore_index=-100
                )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")
        
    print("Training complete! Saving model...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/medi_micro_custom.pt")
    print("Saved to checkpoints/medi_micro_custom.pt")

if __name__ == "__main__":
    train()
