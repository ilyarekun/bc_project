import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import numpy as np
import kagglehub
import os
import shutil
from sklearn.metrics import precision_recall_fscore_support

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model definition
class BrainCNN(nn.Module):
    def __init__(self):
        super(BrainCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.45),
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.45),
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.45),
            nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.45),
            nn.Conv2d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.4),
            nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 4),
        )
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc_layers(out)
        return out

# Data preprocessing function
def data_preprocessing_tumor_stratified(num_clients=4):
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    train_path = os.path.join(dataset_path, "Training")
    test_path = os.path.join(dataset_path, "Testing")
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
    
    transform = transforms.Compose([
        transforms.CenterCrop((400, 400)),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
    ])
    
    general_dataset = ImageFolder(root=general_dataset_path, transform=transform)
    targets = general_dataset.targets
    classes = list(set(targets))
    
    train_indices = []
    val_indices = []
    test_indices = []
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    for class_label in classes:
        class_indices = [i for i, target in enumerate(targets) if target == class_label]
        class_size = len(class_indices)
        train_size = int(train_ratio * class_size)
        val_size = int(val_ratio * class_size)
        test_size = class_size - train_size - val_size
        train_indices.extend(class_indices[:train_size])
        val_indices.extend(class_indices[train_size:train_size + val_size])
        test_indices.extend(class_indices[train_size + val_size:])
    
    train_set = Subset(general_dataset, train_indices)
    val_set = Subset(general_dataset, val_indices)
    test_set = Subset(general_dataset, test_indices)
    
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    client_indices = {client: [] for client in range(num_clients)}
    for class_label in classes:
        class_train_indices = [idx for idx in train_indices if general_dataset.targets[idx] == class_label]
        np.random.shuffle(class_train_indices)
        splits = np.array_split(class_train_indices, num_clients)
        for client in range(num_clients):
            client_indices[client].extend(splits[client].tolist())
    
    client_train_loaders = []
    for client in range(num_clients):
        subset = Subset(general_dataset, client_indices[client])
        loader = DataLoader(subset, batch_size=64, shuffle=True)
        client_train_loaders.append(loader)
    
    return client_train_loaders, val_loader, test_loader

# Compute data loaders for 4 clients
client_train_loaders, val_loader, test_loader = data_preprocessing_tumor_stratified(num_clients=4)

# Utility functions
def get_model():
    return BrainCNN().to(device)

def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

def get_loss_function():
    return nn.CrossEntropyLoss()

def fit_config(server_round: int):
    return {"local_epochs": 5}  # 5 local epochs per client per round

def evaluate_fn(server_round, parameters, config):
    model = get_model()
    state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    model.load_state_dict(state_dict)
    model.eval()
    
    criterion = get_loss_function()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    val_loss /= len(val_loader.dataset)
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    
    return {
        "val_loss": val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }