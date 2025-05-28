import kagglehub
import os
import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

def data_preprocessing_tumor_stratified():
    """
    Downloads a brain tumor MRI dataset from Kaggle, merges training and testing folders
    into a single directory, and performs a stratified split into train, validation, and test sets.
    Returns DataLoader objects for each split.
    """
    # Download the dataset from Kaggle and return the local path
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

    # Define paths to the original Training and Testing folders
    train_path = os.path.join(dataset_path, "Training")
    test_path = os.path.join(dataset_path, "Testing")
    
    # Create a new directory to hold all images together (General_Dataset)
    general_dataset_path = os.path.join(dataset_path, "General_Dataset")
    os.makedirs(general_dataset_path, exist_ok=True)
    
    # Move images from Training and Testing into the General_Dataset directory,
    # preserving class subfolders. This unifies the dataset for consistent splitting.
    for source_path in [train_path, test_path]:
        for class_name in os.listdir(source_path):
            class_path = os.path.join(source_path, class_name)
            general_class_path = os.path.join(general_dataset_path, class_name)
            os.makedirs(general_class_path, exist_ok=True)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                shutil.move(img_path, os.path.join(general_class_path, img_name))
    
    # Define image transformations: 
    # 1) Center crop to 400x400, 2) Resize to 200x200, 3) Convert to PyTorch tensor
    transform = transforms.Compose([
        transforms.CenterCrop((400, 400)),  # Crop the central 400x400 region
        transforms.Resize((200, 200)),      # Resize image to 200x200
        transforms.ToTensor(),              # Convert PIL image to PyTorch tensor
    ])

    # Load the combined dataset using ImageFolder, which expects subdirectories per class
    general_dataset = ImageFolder(root=general_dataset_path, transform=transform)
    
    # Extract class labels (targets) for all samples
    targets = general_dataset.targets
    # Identify unique class indices
    classes = list(set(targets))
    
    # Initialize lists to store indices for stratified splits
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Define split ratios for train, validation, and test sets
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    # Perform stratified splitting: for each class, split its indices accordingly
    for class_label in classes:
        # Get the list of indices for samples belonging to this class
        class_indices = [i for i, target in enumerate(targets) if target == class_label]
        
        # Compute how many samples go to train, validation, and test
        class_size = len(class_indices)
        train_size = int(train_ratio * class_size)
        val_size = int(val_ratio * class_size)
        test_size = class_size - train_size - val_size  # Remainder goes to test
        
        # Assign the first train_size indices to training
        train_indices.extend(class_indices[:train_size])
        # Assign the next val_size indices to validation
        val_indices.extend(class_indices[train_size:train_size + val_size])
        # Assign the rest to testing
        test_indices.extend(class_indices[train_size + val_size:])
    
    # Create Subset objects for train, validation, and test based on the collected indices
    train_set = Subset(general_dataset, train_indices)
    val_set = Subset(general_dataset, val_indices)
    test_set = Subset(general_dataset, test_indices)
    
    # Wrap each subset in a DataLoader for batch processing
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader
