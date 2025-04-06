import kagglehub
import os
import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

def data_preprocessing_tumor_stratified():
    # Загружаем датасет с Kaggle
    dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

    # Определяем пути к тренировочным и тестовым данным
    train_path = os.path.join(dataset_path, "Training")
    test_path = os.path.join(dataset_path, "Testing")
    
    # Создаем общий датасет, объединяя тренировочные и тестовые данные
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
    
    # Определяем трансформации изображений
    transform = transforms.Compose([
        transforms.CenterCrop((400, 400)),  # Обрезаем изображения до 400x400
        transforms.Resize((200, 200)),      # Изменяем размер до 200x200
        transforms.ToTensor(),              # Преобразуем в тензоры PyTorch
    ])

    # Загружаем общий датасет с помощью ImageFolder
    general_dataset = ImageFolder(root=general_dataset_path, transform=transform)
    
    # Получаем метки классов
    targets = general_dataset.targets
    classes = list(set(targets))
    
    # Инициализируем списки индексов для train, val, test
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Определяем пропорции разделения
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    # Стратифицированное разделение по классам
    for class_label in classes:
        # Собираем индексы образцов для данного класса
        class_indices = [i for i, target in enumerate(targets) if target == class_label]
        
        # Вычисляем размеры для train, val, test
        class_size = len(class_indices)
        train_size = int(train_ratio * class_size)
        val_size = int(val_ratio * class_size)
        test_size = class_size - train_size - val_size  # Остаток для test, чтобы не терять данные
        
        # Разделяем индексы на train, val и test
        train_indices.extend(class_indices[:train_size])
        val_indices.extend(class_indices[train_size:train_size + val_size])
        test_indices.extend(class_indices[train_size + val_size:])
    
    # Создаем подмножества для train, val, test
    train_set = Subset(general_dataset, train_indices)
    val_set = Subset(general_dataset, val_indices)
    test_set = Subset(general_dataset, test_indices)
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader