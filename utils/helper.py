import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils.constants import TINY_IMAGENET_MEAN, TINY_IMAGENET_STD


def convert_to_RGB(image):
    """Convert grayscale image to RGB by duplicating channels."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def get_hierarchy_map():
    """Generates a mapping from fine labels to (mid, coarse) labels."""
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
    """Generates hierarchical label mappings and counts."""
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