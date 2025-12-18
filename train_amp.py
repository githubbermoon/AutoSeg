import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
import wandb
import random
import numpy as np
from tqdm import tqdm

# --- Isolated Training Logic ---
# Does NOT import model_utils or app to avoid circular deps or accidental inference triggers.

class DummyDataset(Dataset):
    """
    Placeholder dataset for demonstration.
    Real usage: Load images/masks from 'data/' folder.
    """
    def __init__(self, length=10):
        self.length = length
        self.size = (3, 512, 512)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # random image
        pixel_values = torch.rand(self.size)
        # random labels 0..149
        labels = torch.randint(0, 150, (512, 512))
        return {"pixel_values": pixel_values, "labels": labels}

def train(args):
    # 1. Setup W&B
    wandb.init(project="terrain-safety-v1", job_type="training", config=args)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")
    
    # 2. Model
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512" 
    # In real fine-tuning, usually start from pre-trained generic
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    model.train()
    
    # 3. Data
    # For demo, use dummy. In real Colab, user mounts Drive.
    train_dataset = DummyDataset(length=20) 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 4. AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    
    # 5. Loop
    global_step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0
        
        for batch in tqdm(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            global_step += 1
            
            wandb.log({"train_loss": loss.item(), "epoch": epoch})
            
        print(f"Average Loss: {epoch_loss / len(train_loader)}")
        
        # Save Checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        
        # Log artifact
        artifact = wandb.Artifact(f"model-epoch-{epoch}", type="model")
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)
        
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    
    args = parser.parse_args()
    train(args)
