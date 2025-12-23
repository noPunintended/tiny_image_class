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
from models.hierarchy_resnet import HierarchicalResNet
from utils.losses import CenterLoss
from utils.constants import TINY_IMAGENET_MEAN, TINY_IMAGENET_STD


def get_hierarchy_map():
    # Generate by LLM with human oversight
    mapping = {}

    for i in range(200):
        # --- ANIMALS (0-53, 170, 184, 187, 198) ---
        if i <= 53 or i in [170, 184, 187, 198]:
            coarse = "Animal"
            if i in [22, 23, 24, 25, 26, 187]: mid = "Canine"
            elif i in [27, 28, 29, 30, 31]: mid = "Feline"
            elif i in [33, 34, 35, 36, 37, 38, 39, 40, 184]: mid = "Insect"
            elif i in [49, 50, 51]: mid = "Primate"
            elif i in [0, 12, 13, 15, 16, 17, 41, 190]: mid = "Aquatic"
            else: mid = "Other_Animal"

        # --- ARTIFACTS / OBJECTS (54-166, 185, 186, 194, 196, 197) ---
        elif 54 <= i <= 166 or i in [185, 186, 194, 196, 197]:
            coarse = "Artifact"
            if i in [64, 74, 86, 95, 99, 100, 103, 107, 108, 113, 123, 134, 142, 153, 155]:
                mid = "Vehicle"
            elif i in [55, 68, 71, 78, 97, 104, 111, 112, 124, 133, 138, 139, 145, 147, 158, 185]:
                mid = "Clothing"
            elif i in [60, 83, 88, 115, 143, 146, 151, 154, 159, 162]:
                mid = "Structure"
            else:
                mid = "Object_Tool"

        # --- FOOD / DRINK (167-169, 171-183, 193, 195, 199) ---
        elif 167 <= i <= 183 or i in [169, 193, 195, 199]:
            coarse = "Food"
            if i in [175, 176, 177, 178, 179, 193, 195, 199]: mid = "Produce"
            else: mid = "Prepared_Food"

        # --- NATURE / SCENERY (188-192) ---
        elif 188 <= i <= 192:
            coarse = "Nature"
            mid = "Landscape"
            
        else:
            coarse, mid = "Misc", "Misc"
            
        mapping[i] = (mid, coarse)
    
    return mapping


def get_hierarchical_metadata():
    # Reuse the mapping logic we just finalized
    h_map = get_hierarchy_map() 
    
    # Generate unique integer IDs for the new levels
    mid_cats = sorted(list(set([m for m, c in h_map.values()])))
    coarse_cats = sorted(list(set([c for m, c in h_map.values()])))
    
    mid_to_idx = {name: i for i, name in enumerate(mid_cats)}
    coarse_to_idx = {name: i for i, name in enumerate(coarse_cats)}
    
    # Final lookup: fine_idx -> [coarse_idx, mid_idx]
    lookup = {
        k: [coarse_to_idx[v[1]], mid_to_idx[v[0]]] 
        for k, v in h_map.items()
    }
    
    return lookup, len(coarse_cats), len(mid_cats)


def add_hierarchy_labels(sample, lookup):
    """
    Hugging Face map function to add hierarchical targets.
    """
    fine_label = sample['label']
    coarse_label, mid_label = lookup[fine_label]
    
    sample['hierarchical_label'] = [coarse_label, mid_label, fine_label]
    return sample


def get_transforms(aug_type="standard"):
    """
    Returns specific transformation pipelines based on the experimental setup.
    
    Args:
        aug_type (str): 'baseline', 'standard', or 'heavy'
    """
    
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


def prepare_dataloaders(batch_size=128, aug_level="standard", use_hierarchy=True):
    """Loads, splits, and prepares DataLoaders."""
    full_dataset = load_dataset("zh-plus/tiny-imagenet")
    
    hierarchy_counts = (200,)
    if use_hierarchy:
        mapper, n_coarse, n_mid = get_hierarchical_metadata()
        full_dataset = full_dataset.map(lambda x: add_hierarchy_labels(x, mapper))
        hierarchy_counts = (n_coarse, n_mid, 200)

    test_ds = full_dataset["valid"]
    train_val_split = full_dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = train_val_split["train"], train_val_split["test"]

    train_tf, val_tf = get_transforms(aug_type=aug_level)

    def transform_wrapper(examples, transform_func):
        output = {
            "pixel_values": [transform_func(i.convert("RGB")) for i in examples["image"]],
            "labels": examples["label"]
        }
        if use_hierarchy:
            output["hierarchical_label"] = examples["hierarchical_label"]
        return output

    train_ds.set_transform(lambda e: transform_wrapper(e, train_tf))
    val_ds.set_transform(lambda e: transform_wrapper(e, val_tf))
    test_ds.set_transform(lambda e: transform_wrapper(e, val_tf))

    def collate_fn(batch):
        data = {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }
        if use_hierarchy:
            data['hierarchical_label'] = torch.tensor([x['hierarchical_label'] for x in batch])
        return data

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, hierarchy_counts


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


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, 
                    use_hierarchy=False, lambdas=(0.5, 1.0, 1.0, 1.0)):
    model.train()
    running_loss = 0.0
    
    for batch in loader:
        inputs = batch['pixel_values'].to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            if use_hierarchy:
                # 1. Unpack labels correctly
                hierarchical_labels = batch['hierarchical_label']
                if isinstance(hierarchical_labels, list):
                    hierarchical_labels = torch.stack(hierarchical_labels, dim=1)
                
                targets = hierarchical_labels.to(device)
                t_coarse, t_mid, t_fine = targets[:, 0], targets[:, 1], targets[:, 2]
                
                # 2. Forward pass (features for Center Loss, tuple for CE losses)
                features, (logits_c, logits_m, logits_f) = model(inputs)
                
                # 3. Use the criterion dictionary (much faster than re-initializing)
                ce_loss_func = criterion['ce']
                center_loss_func = criterion['center']
                
                l_coarse = ce_loss_func(logits_c, t_coarse)
                l_mid    = ce_loss_func(logits_m, t_mid)
                l_fine   = ce_loss_func(logits_f, t_fine)
                l_center = center_loss_func(features, t_fine)
                
                # Total Loss (Equation 4)
                loss = (lambdas[0] * l_center + 
                        lambdas[1] * l_coarse + 
                        lambdas[2] * l_mid + 
                        lambdas[3] * l_fine)
            else:
                # Standard Mode
                labels = batch['labels'].to(device)
                # In standard mode, criterion is just a single CrossEntropyLoss object
                logits = model(inputs) 
                loss = criterion(logits, labels)
        
        # Backprop (Same for both modes)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)


def evaluate(model, loader, criterion, device, use_hierarchy=False, lambdas=(0.01, 1.0, 1.0, 1.0)):
    model.eval()
    
    # Loss accumulators
    total_loss, total_c_loss, total_m_loss, total_f_loss, total_cent_loss = 0, 0, 0, 0, 0
    
    # Accuracy accumulators
    correct_f1, correct_f5 = 0, 0 
    correct_c1, correct_m1 = 0, 0 
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['pixel_values'].to(device)
            batch_size = inputs.size(0)
            
            if use_hierarchy:
                h_labels = batch['hierarchical_label'].to(device)
                t_coarse, t_mid, t_fine = h_labels[:, 0], h_labels[:, 1], h_labels[:, 2]
                
                features, (logits_c, logits_m, logits_f) = model(inputs)
                
                # Component Losses
                ce_fn = criterion['ce']
                l_coarse = ce_fn(logits_c, t_coarse)
                l_mid    = ce_fn(logits_m, t_mid)
                l_fine   = ce_fn(logits_f, t_fine)
                l_center = criterion['center'](features, t_fine)
                
                batch_loss = (lambdas[0] * l_center + lambdas[1] * l_coarse + 
                              lambdas[2] * l_mid + lambdas[3] * l_fine)
                
                # Accumulate component losses (unweighted for clear reporting)
                total_c_loss += l_coarse.item()
                total_m_loss += l_mid.item()
                total_f_loss += l_fine.item()
                total_cent_loss += l_center.item()
                
                # Accuracies
                correct_c1 += logits_c.argmax(dim=1).eq(t_coarse).sum().item()
                correct_m1 += logits_m.argmax(dim=1).eq(t_mid).sum().item()
                
                labels = t_fine
                outputs = logits_f
            else:
                labels = batch['labels'].to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
            
            total_loss += batch_loss.item()
            
            # Top-1 and Top-5 Fine Accuracy
            _, pred1 = outputs.topk(1, 1, True, True)
            correct_f1 += pred1.eq(labels.view(-1, 1).expand_as(pred1)).sum().item()
            _, pred5 = outputs.topk(5, 1, True, True)
            correct_f5 += pred5.eq(labels.view(-1, 1).expand_as(pred5)).sum().item()
            
            total_samples += batch_size
            
    # Calculate final averages
    num_batches = len(loader)
    avg_loss = total_loss / num_batches
    acc1 = 100. * correct_f1 / total_samples
    acc5 = 100. * correct_f5 / total_samples
    
    if use_hierarchy:
        acc_c = 100. * correct_c1 / total_samples
        acc_m = 100. * correct_m1 / total_samples
        
        print(f"\n--- Hierarchical Validation Report ---")
        print(f"Accuracy | Coarse: {acc_c:.2f}% | Mid: {acc_m:.2f}% | Fine: {acc1:.2f}%")
        print(f"Avg Loss | L1: {total_c_loss/num_batches:.4f} | L2: {total_m_loss/num_batches:.4f} | "
              f"L3: {total_f_loss/num_batches:.4f} | Cent: {total_cent_loss/num_batches:.4f}")
        
    return avg_loss, acc1, acc5


def get_per_class_accuracy(model, loader, device, num_classes=200):
    model.eval()
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch['pixel_values'].to(device), batch['labels'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    per_class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            per_class_acc[i] = accuracy
            
    return per_class_acc



def run_experiment(aug_level="standard", model_name="SimpleCNN", use_hierarchy=False):
    # 1. Setup
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Experiment: {aug_level} with model {model_name}")
    
    if use_hierarchy:
        print("Using Hierarchical Labels for Training")
        train_loader, val_loader, _, hierarchy_counts = prepare_dataloaders(batch_size=512, aug_level=aug_level, use_hierarchy=True)

        if model_name == "HierarchicalResNet":
            n_coarse, n_mid, n_fine = hierarchy_counts
            model = HierarchicalResNet(num_l1=n_coarse, num_l2=n_mid, num_l3=n_fine).to(device)

        center_loss_fn = CenterLoss(num_classes=200, feat_dim=512).to(device)
    
        criterion = {
            'ce': nn.CrossEntropyLoss(),
            'center': center_loss_fn
        }
    
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'weight_decay': 5e-4},
            {'params': center_loss_fn.parameters(), 'lr': 0.5} 
        ], lr=learning_rate)
        

    else:
        print("Using Standard Flat Labels for Training")
        train_loader, val_loader, _, _ = prepare_dataloaders(batch_size=512, aug_level=aug_level)
    
        if model_name == "SimpleCNN":
            model = SimpleCNN(num_classes=200).to(device)

        elif model_name == "TinyResNet":
            model = TinyResNet(num_classes=200).to(device)

    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler = torch.amp.GradScaler('cuda')
    
    # 2. Logging & Early Stopping Config
    history = []
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    max_epochs = 100
    
    # 3. Training Loop
    for epoch in range(max_epochs):

        if use_hierarchy:
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_hierarchy=True, lambdas=(0.01, 1.0, 1.0, 1.0))

        else:
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_hierarchy=False)
        val_loss, v_acc1, v_acc5 = evaluate(model, val_loader, criterion, device, use_hierarchy=use_hierarchy)
        
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

    print("\n--- Performing Error Analysis on Validation Set ---")
    model.load_state_dict(torch.load(f"best_model_{aug_level}_{model_name}.pth"))

    # # Run the per-class check on VAL
    # val_per_class = get_per_class_accuracy(model, val_loader, device)

    # # Logic: Find the worst 5 classes to see where the model is struggling
    # worst_classes = sorted(val_per_class.items(), key=lambda x: x[1])[:5]
    # print(f"Hardest classes in Val: {worst_classes}")

if __name__ == "__main__":


    # models = ["SimpleCNN", "TinyResNet"]
    # aug_levels = ["baseline", "standard", "heavy"]
    # for model_name in models:
    #     for aug_level in aug_levels:
    #         run_experiment(aug_level=aug_level, model_name=model_name)

    # run_experiment(aug_level="standard", model_name="TinyResNet", use_hierarchy=False)
    run_experiment(aug_level="standard", model_name="HierarchicalResNet", use_hierarchy=True)
