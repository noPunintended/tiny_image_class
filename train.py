import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from models.alex_net import SimpleCNN
from models.res_net import TinyResNet
from models.hierarchy_resnet import HierarchicalResNet
from utils.losses import CenterLoss
from utils.helper import prepare_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, 
                    use_hierarchy=False, lambdas=(0.5, 1.0, 1.0, 1.0)):
    """
        Performs a single training epoch over the dataset.

        Supports both standard classification and hierarchical multi-task learning 
        with Center Loss. Uses Automatic Mixed Precision (AMP) for optimized 
        throughput on modern GPUs.

        :param model: The neural network model (e.g., TinyResNet or HierarchicalResNet).
        :param loader: DataLoader providing training batches.
        :param criterion: Loss function(s). Either a single loss or a dict 
                        containing 'ce' and 'center' for hierarchical mode.
        :param optimizer: PyTorch optimizer (handles both model and center parameters).
        :param scaler: GradScaler for AMP training.
        :param device: torch.device ('cuda' or 'cpu').
        :param use_hierarchy: Bool; if True, computes losses for three hierarchical 
                            levels plus Center Loss.
        :param lambdas: Tuple of weights (lambda_center, lambda_L1, lambda_L2, lambda_L3) 
                        to balance the total hierarchical loss.
        :return: Average training loss for the epoch.
    """

    model.train()
    running_loss = 0.0
    
    for batch in loader:
        inputs = batch['pixel_values'].to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            if use_hierarchy:
                # 1. Unpack labels
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
    """
    Evaluates the model on the validation or test set.

    In hierarchical mode, it tracks and prints accuracy for Coarse, Mid, and 
    Fine levels, as well as the raw average values of all four loss components.

    :param model: The neural network to evaluate.
    :param loader: DataLoader for the validation/test data.
    :param criterion: Criterion dictionary or single loss function.
    :param device: Device to perform computation on.
    :param use_hierarchy: Bool; if True, performs multi-level hierarchical evaluation.
    :param lambdas: Weights used for total loss calculation in hierarchical mode.
    :return: Tuple of (average_loss, top1_fine_accuracy, top5_fine_accuracy).
    """

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
    """
    Orchestrates the full training pipeline for a specific experimental setup.

    Handles data loading, model initialization (Standard vs. Hierarchical), 
    optimizing configurations, and training with early stopping. Saves the 
    best model weights and training history logs.

    :param aug_level: Type of data augmentation ('baseline', 'standard', 'heavy').
    :param model_name: Name of the architecture ('SimpleCNN', 'TinyResNet', 'HierarchicalResNet').
    :param use_hierarchy: Whether to utilize hierarchical labels and Center Loss.
    :return: None. Outputs saved weights and .csv logs.
    """

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
            {'params': model.parameters()},
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
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_hierarchy=True, lambdas=(0.001, 1.0, 1.0, 1.0))

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

if __name__ == "__main__":


    models = ["SimpleCNN", "TinyResNet", "HierarchicalResNet"]
    aug_levels = ["baseline", "standard", "heavy"]

    for model_name in models:
        for aug_level in aug_levels:
            use_hierarchy = False
            if model_name == "HierarchicalResNet":
                use_hierarchy = True
            run_experiment(aug_level=aug_level, model_name=model_name, use_hierarchy=use_hierarchy)
