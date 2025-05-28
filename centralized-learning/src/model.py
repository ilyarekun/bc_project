from torch import nn
import torch.optim as optim
import torch
import time
import random
import numpy as np

# Set a fixed random seed for reproducibility across CPU, GPU, and NumPy
seed = 42
torch.manual_seed(seed)                # Seed CPU RNG
torch.cuda.manual_seed(seed)           # Seed current GPU
torch.cuda.manual_seed_all(seed)       # Seed all GPUs if using DataParallel
random.seed(seed)                      # Seed Python built-in RNG
np.random.seed(seed)                   # Seed NumPy RNG
torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CuDNN
torch.backends.cudnn.benchmark = False     # Disable benchmark mode for consistency


class BrainCNN(nn.Module):
    """
    Convolutional Neural Network for brain tumor classification.
    """

    def __init__(self):
        super(BrainCNN, self).__init__()

        # Convolutional blocks: each block has Conv2d -> ReLU -> BatchNorm -> MaxPool -> Dropout
        self.conv_layers = nn.Sequential(
            # First conv block: input channels=3 (RGB), output channels=64, kernel size=7, padding=3 to preserve spatial dims
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),            # Downsample by a factor of 2
            nn.Dropout2d(0.45),         # Dropout 45% of feature maps

            # Second conv block: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.45),

            # Third conv block: 128 -> 128 channels
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.45),

            # Fourth conv block: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.45),

            # Fifth conv block: 256 -> 256 channels
            nn.Conv2d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.4),

            # Sixth conv block: 256 -> 512 channels
            nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4)
        )

        # Fully connected layers: flatten features then pass through three linear layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),  # Assuming final feature map is 3x3 spatially
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(512, 4)  # Final output: 4 classes
        )

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, H, W).
        Returns:
            out (Tensor): Logits for each class, shape (batch_size, 4).
        """
        out = self.conv_layers(x)                   # Apply convolutional blocks
        out = out.view(out.size(0), -1)             # Flatten to (batch_size, num_features)
        out = self.fc_layers(out)                   # Apply fully connected layers
        return out

    def train_model(
        self,
        train_loader,
        valid_loader,
        num_epochs=50,
        patience=6,
        delta=0.001,
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0.001,
        save_path="./braincnn_prototype.weights"
    ):
        """
        Train the BrainCNN model with early stopping.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            valid_loader (DataLoader): DataLoader for validation data.
            num_epochs (int): Maximum number of epochs to train (default: 50).
            patience (int): Number of epochs with no improvement to wait before stopping early (default: 6).
            delta (float): Minimum change in validation loss to qualify as improvement (default: 0.001).
            learning_rate (float): Learning rate for SGD optimizer (default: 0.001).
            momentum (float): Momentum for SGD optimizer (default: 0.9).
            weight_decay (float): Weight decay (L2 regularization) for optimizer (default: 0.001).
            save_path (str): File path to save the best model weights (default: "./braincnn_prototype.weights").
        
        Returns:
            tuple: (
                train_loss_metr (list of float),
                val_loss_metr (list of float),
                train_acc_metr (list of float),
                val_acc_metr (list of float),
                early_stopping (EarlyStopping object)
            )
        """
        start_time = time.time()

        # Determine computation device (GPU if available, else CPU)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(device)

        # If multiple GPUs are available, wrap the model in DataParallel
        if torch.cuda.device_count() > 1:
            print("Using DataParallel for multi-GPU training")
            self = nn.DataParallel(self)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Cross-entropy for multi-class classification
        optimizer = optim.SGD(
            self.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # Initialize early stopping object
        early_stopping = EarlyStopping(patience=patience, delta=delta)

        # Lists to store per-epoch metrics
        train_loss_metr = []
        val_loss_metr = []
        train_acc_metr = []
        val_acc_metr = []

        # Training loop
        for epoch in range(num_epochs):
            self.train()  # Set model to training mode
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            # Iterate over training batches
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()           # Clear previous gradients
                output = self(data)             # Forward pass
                loss = criterion(output, target)  # Compute loss
                loss.backward()                 # Backpropagate errors
                optimizer.step()                # Update model parameters
                train_loss += loss.item() * data.size(0)

                # Compute training accuracy
                _, predicted = torch.max(output, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()

            # Average training loss over all samples
            train_loss /= len(train_loader.dataset)
            train_acc = correct_train / total_train

            # Validation phase
            self.eval()  # Set model to evaluation mode
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            val_targets = []
            val_preds = []

            with torch.no_grad():  # Disable gradient computation
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = self(data)          # Forward pass
                    loss = criterion(output, target)
                    val_loss += loss.item() * data.size(0)

                    # Compute validation accuracy
                    _, predicted = torch.max(output, 1)
                    total_val += target.size(0)
                    correct_val += (predicted == target).sum().item()

                    # Store targets and predictions for potential further metrics
                    val_targets.extend(target.cpu().numpy())
                    val_preds.extend(predicted.cpu().numpy())

            # Average validation loss and accuracy
            val_loss /= len(valid_loader.dataset)
            val_acc = correct_val / total_val

            # Append metrics for this epoch
            train_loss_metr.append(train_loss)
            val_loss_metr.append(val_loss)
            train_acc_metr.append(train_acc)
            val_acc_metr.append(val_acc)

            # Print progress for this epoch
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            elapsed_time = time.time() - start_time
            print(f"Elapsed Time: {elapsed_time:.2f} seconds\n")

            # Check early stopping condition
            early_stopping(val_loss, self)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

        # After training loop, load the best saved model state
        early_stopping.load_best_model(self)

        # Save the best model weights to disk
        torch.save(self.state_dict(), save_path)

        return train_loss_metr, val_loss_metr, train_acc_metr, val_acc_metr, early_stopping


class EarlyStopping:
    """
    Utility for stopping training when validation loss does not improve.
    """

    def __init__(self, patience=5, delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """
        Call method to update early stopping state.

        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): Model being trained.
        """
        # We use negative validation loss as the score to maximize
        score = -val_loss

        # First time setting best_score
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        # If no improvement beyond delta, increment counter
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement found: update best_score and reset counter
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        """
        Load the best saved model state into the provided model.

        Args:
            model (nn.Module): Model into which to load the best state.
        """
        model.load_state_dict(self.best_model_state)
