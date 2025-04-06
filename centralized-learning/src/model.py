from torch import nn
import torch.optim as optim
import torch
import time
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


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
        out = out.view(out.size(0), -1)     # Flatten
        out = self.fc_layers(out)
        return out

    def train_model(self, train_loader, valid_loader, num_epochs=50, patience=6, delta=0.001, learning_rate=0.001, save_path="./braincnn_prototype.weights"):
        """
        Метод для обучения модели BrainCNN.
        
        Аргументы:
            train_loader (DataLoader): Загрузчик данных для обучения.
            valid_loader (DataLoader): Загрузчик данных для валидации.
            num_epochs (int): Количество эпох обучения. По умолчанию 50.
            patience (int): Терпение для ранней остановки. По умолчанию 3.
            delta (float): Минимальное улучшение для ранней остановки. По умолчанию 0.001.
            learning_rate (float): Скорость обучения. По умолчанию 0.001.
            save_path (str): Путь для сохранения модели. По умолчанию "./braincnn_prototype.weights".
        
        Возвращает:
            tuple: Списки метрик (train_loss_metr, val_loss_metr, train_acc_metr, val_acc_metr).
        """
        start_time = time.time()

        # Определение устройства
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(device)
        
        # Поддержка нескольких GPU
        if torch.cuda.device_count() > 1:
            print("Using DataParallel for multi-GPU training")
            self = nn.DataParallel(self)
        
        # Инициализация критерия, оптимизатора и ранней остановки
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)
        early_stopping = EarlyStopping(patience=patience, delta=delta)
        
        # Метрики
        train_loss_metr = []
        val_loss_metr = []
        train_acc_metr = []
        val_acc_metr = []
        
        # Цикл обучения
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            # Обучение
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                
                _, predicted = torch.max(output, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()
            
            train_loss /= len(train_loader.dataset)
            train_acc = correct_train / total_train
            
            # Валидация
            self.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            val_targets = []
            val_preds = []
            
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = self(data)
                    loss = criterion(output, target)
                    val_loss += loss.item() * data.size(0)
                    
                    _, predicted = torch.max(output, 1)
                    total_val += target.size(0)
                    correct_val += (predicted == target).sum().item()
                    
                    val_targets.extend(target.cpu().numpy())
                    val_preds.extend(predicted.cpu().numpy())
            
            val_loss /= len(valid_loader.dataset)
            val_acc = correct_val / total_val
            
            # Сохранение метрик
            train_loss_metr.append(train_loss)
            val_loss_metr.append(val_loss)
            train_acc_metr.append(train_acc)
            val_acc_metr.append(val_acc)
            
            # Вывод прогресса
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Epoch {epoch+1}: Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
            elapsed_time = time.time() - start_time
            print(f"Training completed in {elapsed_time:.2f} seconds")

            print('\n')
            
            # Проверка ранней остановки
            early_stopping(val_loss, self)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # Загрузка лучшей модели и сохранение
        early_stopping.load_best_model(self)
        torch.save(self.state_dict(), save_path)
        
        return train_loss_metr, val_loss_metr, train_acc_metr, val_acc_metr, early_stopping


# Класс EarlyStopping (без изменений)
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)