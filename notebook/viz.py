import os
os.chdir("..")
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
    fig, axes = plt.subplots(3, 1, figsize=(10, 18), constrained_layout=True)

    # Plot 1: Validation Accuracy Comparison
    axes[0].plot(df_alex_base['epoch'], df_alex_base['val_acc1'], label='SimpleCNN Baseline', linestyle='--')
    axes[0].plot(df_alex_std['epoch'], df_alex_std['val_acc1'], label='SimpleCNN Standard', linestyle='--')
    axes[0].plot(df_alex_heavy['epoch'], df_alex_heavy['val_acc1'], label='SimpleCNN Heavy', linestyle='--')
    axes[0].plot(df_resnet_base['epoch'], df_resnet_base['val_acc1'], label='ResNet18 Baseline', linewidth=2)
    axes[0].plot(df_resnet_std['epoch'], df_resnet_std['val_acc1'], label='ResNet18 Standard', linewidth=2)
    axes[0].plot(df_resnet_heavy['epoch'], df_resnet_heavy['val_acc1'], label='ResNet18 Heavy', linewidth=2)
    axes[0].set_title('Validation Top-1 Accuracy across Experiments', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: SimpleCNN Heavy Overfitting Analysis
    axes[1].plot(df_alex_heavy['epoch'], df_alex_heavy['train_loss'], label='Train Loss', color='blue')
    axes[1].plot(df_alex_heavy['epoch'], df_alex_heavy['val_loss'], label='Val Loss', color='red')
    axes[1].set_title('SimpleCNN (Heavy Augment) Loss: Training vs Validation', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: ResNet Heavy Overfitting Analysis
    axes[2].plot(df_resnet_heavy['epoch'], df_resnet_heavy['train_loss'], label='Train Loss', color='blue')
    axes[2].plot(df_resnet_heavy['epoch'], df_resnet_heavy['val_loss'], label='Val Loss', color='red')
    axes[2].set_title('ResNet18 (Heavy Augment) Loss: Training vs Validation', fontsize=14)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.savefig('experiment_comparison_plots.png', dpi=300)

    print("Best Validation Accuracies:")
    print(f"SimpleCNN Base: {df_alex_base['val_acc1'].max()}%")
    print(f"SimpleCNN Standard: {df_alex_std['val_acc1'].max()}%")
    print(f"SimpleCNN Heavy: {df_alex_heavy['val_acc1'].max()}%")
    print(f"ResNet Base: {df_resnet_base['val_acc1'].max()}%")
    print(f"ResNet Standard: {df_resnet_std['val_acc1'].max()}%")
    print(f"ResNet Heavy: {df_resnet_heavy['val_acc1'].max()}%")