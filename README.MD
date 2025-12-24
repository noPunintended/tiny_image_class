# Hierarchical ResNet for Tiny ImageNet

# For detailed explaination read in notebook/explaination.ipynb
# For training run train.py

A computer vision pipeline using **Multi-Task Learning (MTL)** and **Center Loss** to classify 200 categories from the Tiny ImageNet dataset. This project explores the transition from flat classification to a hierarchical feature space.

## Key Features
- **Hierarchical Architecture**: Multi-head ResNet-18 capturing Coarse, Mid, and Fine-grained semantics.
- **Advanced Loss Functions**: Hybrid loss combining **Cross-Entropy** (for separation) and **Center Loss** (for intra-class compactness).
- **Latent Space Optimization**: Features projected onto a unit hypersphere with temperature scaling ($s=20$).
- **Data Augmentation**: Robust pipeline including RandomErasing, ColorJitter, and RandomResizedCrops.

## 📊 Results Summary
| Model | Top-1 Accuracy | Top-5 Accuracy | Semantic Consistency (Coarse) |
| :--- | :--- | :--- | :--- |
| Standard ResNet-18 | 50.69% | 75.71% | N/A |
| **Hierarchical ResNet** | **49.95%** | **75.78%** | **86.50%** |

> **Note**: While raw accuracy is similar, the Hierarchical model provides a semantically structured latent space, making it superior for downstream tasks like **Image Search**.

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- Prefer NVIDIA GPU 

### Environment Setup
```bash
# Clone the repository
git clone [https://github.com/noPunintended/tiny-image-hierarchical.git](https://github.com/noPunintended/tiny_image_class.git)
cd tiny-image-hierarchical

# Recreate the Conda environment
conda env create -f environment.yml
conda activate your_env_name

# Install on PIP
pip install $(python -c "import yaml; print(' '.join(yaml.safe_load(open('environment.yml'))['dependencies'][-1]['pip']))")



### This project were developed on WSL2 with a NVIDIA GPU and Anaconda