import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from torch.cuda.amp import GradScaler
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader


from models.alex_net import SimpleCNN
from models.res_net import TinyResNet
from utils.constants import TINY_IMAGENET_MEAN, TINY_IMAGENET_STD

def get_transforms(aug_type="standard"):
    """
    Returns specific transformation pipelines based on the experimental setup.
    
    Args:
        aug_type (str): 'baseline', 'standard', or 'heavy'
    """
    from utils.constants import TINY_IMAGENET_MEAN, TINY_IMAGENET_STD
    
    # Common ops for all validation and test sets
    base_ops = [
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=TINY_IMAGENET_MEAN, std=TINY_IMAGENET_STD)
    ]

    if aug_type == "baseline":
        # No augmentation: Used to establish the "Overfitting Baseline"
        train_ops = base_ops
        
    elif aug_type == "standard":
        # Standard geometric augmentations
        train_ops = [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=TINY_IMAGENET_MEAN, std=TINY_IMAGENET_STD)
        ]
        
    elif aug_type == "heavy":
        # Aggressive augmentations: Color, Scale, and Erasing
        train_ops = [
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=TINY_IMAGENET_MEAN, std=TINY_IMAGENET_STD),
            transforms.RandomErasing(p=0.2) # Forces the model to look at multiple features
        ]
    
    return transforms.Compose(train_ops), transforms.Compose(base_ops)

def prepare_dataloaders(batch_size=128, aug_level="standard"):
    """Loads, splits, and prepares DataLoaders."""
    full_dataset = load_dataset("zh-plus/tiny-imagenet")
    
    # Split strategy: Orig Valid -> Test; Part of Orig Train -> Val
    test_ds = full_dataset["valid"]
    train_val_split = full_dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = train_val_split["train"], train_val_split["test"]

    train_tf, val_tf = get_transforms(aug_type=aug_level)

    # Mapping transforms to datasets
    train_ds.set_transform(lambda e: {"pixel_values": [train_tf(i) for i in e["image"]], "label": e["label"]})
    val_ds.set_transform(lambda e: {"pixel_values": [val_tf(i) for i in e["image"]], "label": e["label"]})
    test_ds.set_transform(lambda e: {"pixel_values": [val_tf(i) for i in e["image"]], "label": e["label"]})

    def collate_fn(batch):
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['label'] for x in batch])
        }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

def evaluate(model, loader, device):
    """Calculates Top-1 and Top-5 Accuracy."""
    model.eval()
    correct_1, correct_5, total = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            
            _, pred1 = outputs.topk(1, 1, True, True)
            correct_1 += pred1.eq(labels.view(-1, 1).expand_as(pred1)).sum().item()
            
            _, pred5 = outputs.topk(5, 1, True, True)
            correct_5 += pred5.eq(labels.view(-1, 1).expand_as(pred5)).sum().item()
            
            total += labels.size(0)
    return (100. * correct_1 / total), (100. * correct_5 / total)


def evaluate(model, loader, criterion, device):
    """Calculates Loss, Top-1, and Top-5 Accuracy."""
    model.eval()
    running_loss = 0.0
    correct_1, correct_5, total = 0, 0, 0
    
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            
            # Calculate Loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Top-1 and Top-5 logic
            _, pred1 = outputs.topk(1, 1, True, True)
            correct_1 += pred1.eq(labels.view(-1, 1).expand_as(pred1)).sum().item()
            
            _, pred5 = outputs.topk(5, 1, True, True)
            correct_5 += pred5.eq(labels.view(-1, 1).expand_as(pred5)).sum().item()
            
            total += labels.size(0)
            
    val_loss = running_loss / len(loader)
    acc1 = 100. * correct_1 / total
    acc5 = 100. * correct_5 / total
    
    return val_loss, acc1, acc5


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    
    for batch in loader:
        inputs, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Modern Mixed Precision Syntax ('cuda' is explicit)
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # 1. Scale the LOSS
        scaler.scale(loss).backward()
        
        # 2. Step the scaler (it unscales the gradients and calls optimizer.step())
        scaler.step(optimizer)
        
        # 3. Update the scaler for the next iteration
        scaler.update()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_1, correct_5, total = 0, 0, 0
    
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, pred1 = outputs.topk(1, 1, True, True)
            correct_1 += pred1.eq(labels.view(-1, 1).expand_as(pred1)).sum().item()
            
            _, pred5 = outputs.topk(5, 1, True, True)
            correct_5 += pred5.eq(labels.view(-1, 1).expand_as(pred5)).sum().item()
            
            total += labels.size(0)
            
    return running_loss / len(loader), (100. * correct_1 / total), (100. * correct_5 / total)

def run_experiment(aug_level="standard", model_name="SimpleCNN"):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Experiment: {aug_level} with model {model_name}")
    
    train_loader, val_loader, _ = prepare_dataloaders(batch_size=512, aug_level=aug_level)
    
    if model_name == "SimpleCNN":
        model = SimpleCNN(num_classes=200).to(device)

    elif model_name == "TinyResNet":
        model = TinyResNet(num_classes=200).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda')
    
    # 2. Logging & Early Stopping Config
    history = []
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    max_epochs = 100
    
    # 3. Training Loop
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, v_acc1, v_acc5 = evaluate(model, val_loader, criterion, device)
        
        # Log entry
        stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc1': v_acc1,
            'val_acc5': v_acc5
        }
        history.append(stats)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {v_acc1:.2f}%")

        # 4. Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f"best_model_{aug_level}_{model_name}.pth")
            print(f"New best model saved (Val Loss: {val_loss:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 5. Save Final Logs

    df = pd.DataFrame(history)
    df.to_csv(f"outputs/results_{aug_level}_{model_name}.csv", index=False)
    print(f"Experiment {aug_level}_{model_name} complete. Logs saved.")

if __name__ == "__main__":


    models = ["SimpleCNN", "TinyResNet"]
    aug_levels = ["baseline", "standard", "heavy"]
    for model_name in models:
        for aug_level in aug_levels:
            run_experiment(aug_level=aug_level, model_name=model_name)

