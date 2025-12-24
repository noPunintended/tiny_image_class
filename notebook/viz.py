import os
os.chdir("..")
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# Reading into DataFrames
def alex_res_viz():
    df_alex_base = pd.read_csv('outputs/results_baseline_SimpleCNN.csv')
    df_alex_std = pd.read_csv('outputs/results_standard_SimpleCNN.csv')
    df_alex_heavy = pd.read_csv('outputs/results_heavy_SimpleCNN.csv')

    df_resnet_base = pd.read_csv('outputs/results_baseline_TinyResNet.csv')
    df_resnet_std = pd.read_csv('outputs/results_standard_TinyResNet.csv')
    df_resnet_heavy = pd.read_csv('outputs/results_heavy_TinyResNet.csv')

    # Plotting
    fig, axes = plt.subplots(5, 1, figsize=(10, 18), constrained_layout=True)

    # Plot 1: Validation Accuracy Comparison
    axes[0].plot(df_alex_base['epoch'], df_alex_base['val_acc1'], label='AlexNet Baseline', linestyle='--')
    axes[0].plot(df_alex_std['epoch'], df_alex_std['val_acc1'], label='AlexNet Standard', linestyle='--')
    axes[0].plot(df_alex_heavy['epoch'], df_alex_heavy['val_acc1'], label='AlexNet Heavy', linestyle='--')
    axes[0].plot(df_resnet_base['epoch'], df_resnet_base['val_acc1'], label='ResNet Baseline', linewidth=2)
    axes[0].plot(df_resnet_std['epoch'], df_resnet_std['val_acc1'], label='ResNet Standard', linewidth=2)
    axes[0].plot(df_resnet_heavy['epoch'], df_resnet_heavy['val_acc1'], label='ResNet Heavy', linewidth=2)
    axes[0].set_title('Validation Top-1 Accuracy across Experiments', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: AlexNet Heavy Overfitting Analysis
    axes[1].plot(df_alex_heavy['epoch'], df_alex_heavy['train_loss'], label='Train Loss', color='blue')
    axes[1].plot(df_alex_heavy['epoch'], df_alex_heavy['val_loss'], label='Val Loss', color='red')
    axes[1].set_title('AlexNet (Heavy Augment) Loss: Training vs Validation', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_xlim(0, 60)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: ResNet Heavy Overfitting Analysis
    axes[2].plot(df_resnet_heavy['epoch'], df_resnet_heavy['train_loss'], label='Train Loss', color='blue')
    axes[2].plot(df_resnet_heavy['epoch'], df_resnet_heavy['val_loss'], label='Val Loss', color='red')
    axes[2].set_title('ResNet (Heavy Augment) Loss: Training vs Validation', fontsize=14)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_xlim(0, 60)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    # Plot 3: AlexNet Standard Overfitting Analysis
    axes[3].plot(df_alex_std['epoch'], df_alex_std['train_loss'], label='Train Loss', color='blue')
    axes[3].plot(df_alex_std['epoch'], df_alex_std['val_loss'], label='Val Loss', color='red')
    axes[3].set_title('AlexNet (Standard Augment) Loss: Training vs Validation', fontsize=14)
    axes[3].set_xlabel('Epoch', fontsize=12)
    axes[3].set_xlim(0, 60)
    axes[3].set_ylabel('Loss', fontsize=12)
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)

    # Plot 4: ResNet Standard Overfitting Analysis
    axes[4].plot(df_resnet_std['epoch'], df_resnet_std['train_loss'], label='Train Loss', color='blue')
    axes[4].plot(df_resnet_std['epoch'], df_resnet_std['val_loss'], label='Val Loss', color='red')
    axes[4].set_title('ResNet (Standard Augment) Loss: Training vs Validation', fontsize=14)
    axes[4].set_xlabel('Epoch', fontsize=12)
    axes[4].set_xlim(0, 60)
    axes[4].set_ylabel('Loss', fontsize=12)
    axes[4].legend(fontsize=10)
    axes[4].grid(True, alpha=0.3)

    plt.savefig('experiment_comparison_plots.png', dpi=300)

    print("Best Validation Loss:")
    print(f"AlexNet Base: {df_alex_base['val_loss'].max()}")
    print(f"AlexNet Standard: {df_alex_std['val_loss'].max()}")
    print(f"AlexNet Heavy: {df_alex_heavy['val_loss'].max()}")
    print(f"ResNet Base: {df_resnet_base['val_loss'].max()}")
    print(f"ResNet Standard: {df_resnet_std['val_loss'].max()}")
    print(f"ResNet Heavy: {df_resnet_heavy['val_loss'].max()}")

    print("Best Validation Accuracies:")
    print(f"AlexNet Base: {df_alex_base['val_acc1'].max()}%")
    print(f"AlexNet Standard: {df_alex_std['val_acc1'].max()}%")
    print(f"AlexNet Heavy: {df_alex_heavy['val_acc1'].max()}%")
    print(f"ResNet Base: {df_resnet_base['val_acc1'].max()}%")
    print(f"ResNet Standard: {df_resnet_std['val_acc1'].max()}%")
    print(f"ResNet Heavy: {df_resnet_heavy['val_acc1'].max()}%")


def heirarchy_viz():

    df_resnet_std = pd.read_csv('outputs/results_standard_TinyResNet.csv')
    df_hier_std = pd.read_csv('outputs/results_standard_HierarchicalResNet.csv')

    fig, axes = plt.subplots(2, 1, figsize=(10, 18), constrained_layout=True)

    # Plot 1: Validation Accuracy Comparison
    axes[0].plot(df_resnet_std['epoch'], df_resnet_std['val_acc1'], label='ResNet Standard', linestyle='--')
    axes[0].plot(df_hier_std['epoch'], df_hier_std['val_acc1'], label='Hierarchical ResNet Standard')
    axes[0].set_title('Validation Top-1 Accuracy across Experiments', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Hierarchical ResNet Standard Overfitting Analysis
    axes[1].plot(df_hier_std['epoch'], df_hier_std['train_loss'], label='Train Loss', color='blue')
    axes[1].plot(df_hier_std['epoch'], df_hier_std['val_loss'], label='Val Loss', color='red')
    axes[1].set_title('Hierarchical ResNet (Standard Augment) Loss: Training vs Validation', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_xlim(0, 60)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)




def get_embeddings(model, loader, device, limit=1000):
    model.eval()
    features_list = []
    coarse_labels = []
    fine_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i * loader.batch_size >= limit: break
            
            inputs = batch['pixel_values'].to(device)
            # Use 'hierarchical_label' if available, otherwise 'labels'
            if 'hierarchical_label' in batch:
                h_labels = batch['hierarchical_label']
                c_val = h_labels[:, 0].numpy()
                f_val = h_labels[:, 2].numpy()
            else:
                f_val = batch['labels'].numpy()
                c_val = f_val # Fallback
            
            output = model(inputs)
            
            # --- THE FIX: Smart Unpacking ---
            if isinstance(output, tuple):
                # This is the Hierarchical model: (features, (logits_c, m, f))
                features = output[0]
            else:
                # This is the Standard model: just logits
                # NOTE: If TinyResNet doesn't return features, 
                # you are currently plotting LOGITS in t-SNE.
                features = output
            
            features_list.append(features.cpu().numpy())
            coarse_labels.append(c_val)
            fine_labels.append(f_val)
            
    return (np.concatenate(features_list), 
            np.concatenate(coarse_labels), 
            np.concatenate(fine_labels))


def heirarchical_tsne_viz(model_h, val_loader, device):

    # 1. Extract features from your best Hierarchical Model
    # Assuming 'model_h' is your loaded HierarchicalResNet
    features, c_labels, f_labels = get_embeddings(model_h, val_loader, device, limit=1200)

    # 2. Run T-SNE (This might take a minute even on a 5080)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embeddings_2d = tsne.fit_transform(features)

    # 3. Visualize
    plt.figure(figsize=(16, 10))

    # Plot Colored by Coarse Class (The 4 big groups)
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], 
                    hue=c_labels, palette='viridis', s=30, alpha=0.7)
    plt.title("Hierarchical Space: Colored by Coarse Categories\n(Animals vs. Artifacts etc.)", fontsize=14)
    plt.legend(title="Coarse ID", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot Colored by Fine Class (Shows how tight the 200 species are)
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], 
                    hue=f_labels, palette='tab20', s=15, alpha=0.5, legend=False)
    plt.title("Hierarchical Space: Colored by 200 Fine Classes\n(Shows Cluster Compactness)", fontsize=14)

    plt.tight_layout()
    plt.savefig('hierarchical_tsne_comparison.png', dpi=300)
    plt.show()


def plot_tsne_comparison(model_standard, model_hier, loader, device):
    # Extract embeddings for both
    feat_s, _, _ = get_embeddings(model_standard, loader, device, limit=1000)
    feat_h, c_labels, _ = get_embeddings(model_hier, loader, device, limit=1000)
    
    # Run T-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embed_s = tsne.fit_transform(feat_s)
    embed_h = tsne.fit_transform(feat_h)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot Standard ResNet
    sns.scatterplot(x=embed_s[:, 0], y=embed_s[:, 1], hue=c_labels, 
                    palette='viridis', ax=axes[0], s=30, alpha=0.6)
    axes[0].set_title("Standard ResNet-18 \nFeatures are scattered and overlapping", fontsize=15)
    
    # Plot Hierarchical ResNet
    sns.scatterplot(x=embed_h[:, 0], y=embed_h[:, 1], hue=c_labels, 
                    palette='viridis', ax=axes[1], s=30, alpha=0.6)
    axes[1].set_title("Hierarchical ResNet\nFeatures form semantically meaningful islands", fontsize=15)
    
    plt.savefig('resnet_vs_hierarchical_tsne.png', dpi=300)
    plt.show()