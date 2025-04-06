""" import os
import pandas as pd
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import shutil
import kagglehub
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import random
import skimage
from PIL import Image
from torchvision.models import ResNet18_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader


class ChestXRayDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = {'positive': 1, 'negative': 0}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.label_map[self.df.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def make_loaders(data_dir):
    BATCH_SIZE = 64
    IMAGE_H, IMAGE_W = 224, 224
    NUM_WORKERS = 4
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    train_df = pd.read_csv(os.path.join(data_dir, 'train.txt'), sep=" ", header=None)
    test_df = pd.read_csv(os.path.join(data_dir, 'test.txt'), sep=" ", header=None)
    val_df = pd.read_csv(os.path.join(data_dir, 'val.txt'), sep=" ", header=None)
    
    train_df = train_df.rename(columns={1: 'image_name', 2: 'diagnosis'})
    test_df = test_df.rename(columns={1: 'image_name', 2: 'diagnosis'})
    val_df = val_df.rename(columns={1: 'image_name', 2: 'diagnosis'})
    
    train_df.drop(columns=[0, 3], axis=1, inplace=True)
    test_df.drop(columns=[0, 3], axis=1, inplace=True)
    val_df.drop(columns=[0, 3], axis=1, inplace=True)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((IMAGE_H, IMAGE_W), scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.RandomVerticalFlip(p=0.7),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))], p=0.7),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_H, IMAGE_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_root = os.path.join(data_dir, 'train')
    test_root = os.path.join(data_dir, 'test')
    val_root = os.path.join(data_dir, 'val')
    
    train_dataset = ChestXRayDataset(train_df, train_root, transform=train_transform)
    test_dataset = ChestXRayDataset(test_df, test_root, transform=val_test_transform)
    val_dataset = ChestXRayDataset(val_df, val_root, transform=val_test_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    return train_loader, test_loader, val_loader """




import kagglehub
import os
import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split as randspl

from torchvision.datasets import ImageFolder


def data_preprocessing_tumor():
    # Download the dataset from KaggleHub
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

    # Define paths to training and testing directories
    train_path = os.path.join(dataset_path, "Training")
    test_path = os.path.join(dataset_path, "Testing")
    
    # Making new data split by aggregating all the data together and splitting again
    general_dataset_path = os.path.join(dataset_path, "General_Dataset")
    os.makedirs(general_dataset_path, exist_ok=True)
    
    
    for source_path in [train_path, test_path]:
        
        for class_name in os.listdir(source_path):
            class_path = os.path.join(source_path, class_name)
            general_class_path = os.path.join(general_dataset_path, class_name)
            os.makedirs(general_class_path, exist_ok=True)
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                shutil.move(img_path, os.path.join(general_class_path, img_name))
    

    # Define image transformations
    transform = transforms.Compose([
        transforms.CenterCrop((400,400)), #from input 512x512 to 400x400
        transforms.Resize((224, 224)),  # Resize images to 200х200 pixels
        transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    # Load datasets with ImageFolder
    general_dataset = ImageFolder(root=general_dataset_path, transform=transform)
    
    total_size = len(general_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_set, val_set, test_set = randspl(
        general_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
        
    )
        
    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader
