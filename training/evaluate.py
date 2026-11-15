import os
import sys
import io
import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.cnn_models import CharacterCNN

device = config.DEVICE

def evaluate_mnist():
    if not os.path.exists(config.MNIST_MODEL_PATH):
        print("MNIST model not found.")
        return
        
    print("Evaluating MNIST Digit CNN...")
    ckpt = torch.load(config.MNIST_MODEL_PATH, map_location=device)
    model = CharacterCNN(num_classes=10).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Load 5000 test samples
    df = pd.read_csv(config.MNIST_TEST_CSV, nrows=5000)
    y_true = df.iloc[:, 0].values.astype(int)
    X = df.iloc[:, 1:].values.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    
    tensor_X = torch.from_numpy(X).to(device)
    with torch.no_grad():
        outputs = model(tensor_X)
        y_pred = outputs.argmax(dim=1).cpu().numpy()
        
    acc = (y_true == y_pred).mean() * 100.0
    print(f"MNIST Test Accuracy: {acc:.2f}%")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=config.MNIST_CLASSES))
    
    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.MNIST_CLASSES,
                yticklabels=config.MNIST_CLASSES)
    plt.title(f"MNIST Test Confusion Matrix (Acc: {acc:.2f}%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    out_path = os.path.join(config.STATIC_DIR, "mnist_confusion_matrix.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")

def evaluate_emnist():
    if not os.path.exists(config.EMNIST_MODEL_PATH):
        print("EMNIST model not found.")
        return
        
    print("Evaluating EMNIST Character CNN...")
    ckpt = torch.load(config.EMNIST_MODEL_PATH, map_location=device)
    model = CharacterCNN(num_classes=26).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    df = pd.read_parquet(config.EMNIST_TEST_PARQUET, engine="fastparquet").head(5200)
    y_true = df["label"].values.astype(int)
    
    decoded = []
    for raw in df["image.bytes"]:
        arr = np.array(Image.open(io.BytesIO(raw))).T.astype(np.float32) / 255.0
        decoded.append(arr)
    X = np.stack(decoded)[:, np.newaxis, :, :]
    
    tensor_X = torch.from_numpy(X).to(device)
    with torch.no_grad():
        outputs = model(tensor_X)
        y_pred = outputs.argmax(dim=1).cpu().numpy()
        
    acc = (y_true == y_pred).mean() * 100.0
    print(f"EMNIST Letters Test Accuracy: {acc:.2f}%")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Purples',
                xticklabels=config.EMNIST_CLASSES,
                yticklabels=config.EMNIST_CLASSES)
    plt.title(f"EMNIST Letters Confusion Matrix (Acc: {acc:.2f}%)")
    plt.xlabel("Predicted Character")
    plt.ylabel("True Character")
    plt.tight_layout()
    out_path = os.path.join(config.STATIC_DIR, "emnist_confusion_matrix.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")

if __name__ == "__main__":
    evaluate_mnist()
    evaluate_emnist()

