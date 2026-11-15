import os
import sys
import io
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.cnn_models import CharacterCNN

class EMNISTDataset(Dataset):
    def __init__(self, parquet_path: str, samples_per_class: int = 600):
        print(f"Loading EMNIST from {parquet_path}...", flush=True)
        df = pd.read_parquet(parquet_path, engine="fastparquet")
        
        # Balance across 26 classes
        balanced_dfs = []
        for c in range(26):
            cdf = df[df["label"] == c]
            if len(cdf) > samples_per_class:
                cdf = cdf.sample(samples_per_class, random_state=42)
            balanced_dfs.append(cdf)
        df_balanced = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=42)
        
        self.labels = df_balanced["label"].values.astype(np.int64)
        print(f"Decoding {len(self.labels)} images...", flush=True)
        
        t0 = time.time()
        decoded_images = []
        for raw_bytes in df_balanced["image.bytes"]:
            # Transpose .T to make standard NIST upright orientation
            arr = np.array(Image.open(io.BytesIO(raw_bytes))).T.astype(np.float32) / 255.0
            decoded_images.append(arr)
            
        self.images = np.stack(decoded_images, axis=0)[:, np.newaxis, :, :]
        print(f"Loaded and decoded {len(self.images)} images in {time.time() - t0:.2f}s.", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

def train_emnist_model(epochs: int = 3, batch_size: int = 128, samples_per_class_train: int = 600, samples_per_class_test: int = 100):
    device = config.DEVICE
    print(f"Using device: {device}", flush=True)
    
    train_dataset = EMNISTDataset(config.EMNIST_TRAIN_PARQUET, samples_per_class=samples_per_class_train)
    test_dataset = EMNISTDataset(config.EMNIST_TEST_PARQUET, samples_per_class=samples_per_class_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = CharacterCNN(num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\n--- Starting EMNIST Character Model Training ---", flush=True)
    start_time = time.time()
    
    history = {"train_loss": [], "test_acc": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, targets in train_loader:
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
        
        # Test evaluation
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
        
        print(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%", flush=True)
        
    duration = time.time() - start_time
    print(f"\nEMNIST Training completed in {duration:.1f}s. Final Test Accuracy: {test_acc:.2f}%", flush=True)
    
    os.makedirs(os.path.dirname(config.EMNIST_MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": 26,
        "classes": config.EMNIST_CLASSES,
        "test_acc": test_acc,
        "history": history
    }, config.EMNIST_MODEL_PATH)
    print(f"Saved EMNIST model to: {config.EMNIST_MODEL_PATH}", flush=True)
    return model, test_acc

if __name__ == "__main__":
    train_emnist_model()
