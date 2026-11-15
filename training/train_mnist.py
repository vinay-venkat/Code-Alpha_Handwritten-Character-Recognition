import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure root directory in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.cnn_models import CharacterCNN

class MNISTCSVDataset(Dataset):
    def __init__(self, csv_path: str, max_samples: int = None):
        print(f"Loading MNIST from {csv_path}...")
        df = pd.read_csv(csv_path, nrows=max_samples)
        self.labels = df.iloc[:, 0].values.astype(np.int64)
        # Reshape to (N, 1, 28, 28) and normalize to [0, 1]
        raw_pixels = df.iloc[:, 1:].values.astype(np.float32) / 255.0
        self.images = raw_pixels.reshape(-1, 1, 28, 28)
        print(f"Loaded {len(self.labels)} samples.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

def train_mnist_model(epochs: int = 3, batch_size: int = 128, max_train_samples: int = 30000):
    device = config.DEVICE
    print(f"Using device: {device}")
    
    train_dataset = MNISTCSVDataset(config.MNIST_TRAIN_CSV, max_samples=max_train_samples)
    test_dataset = MNISTCSVDataset(config.MNIST_TEST_CSV, max_samples=5000)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = CharacterCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\n--- Starting MNIST Model Training ---")
    start_time = time.time()
    
    history = {"train_loss": [], "test_acc": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total * 100.0
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == targets).sum().item()
                test_total += targets.size(0)
                
        test_acc = test_correct / test_total * 100.0
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        
        print(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.1f}s. Final Test Accuracy: {test_acc:.2f}%")
    
    # Save checkpoint
    os.makedirs(os.path.dirname(config.MNIST_MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": 10,
        "classes": config.MNIST_CLASSES,
        "test_acc": test_acc,
        "history": history
    }, config.MNIST_MODEL_PATH)
    print(f"Saved MNIST model to: {config.MNIST_MODEL_PATH}")
    return model, test_acc

if __name__ == "__main__":
    train_mnist_model()

