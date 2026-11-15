import os
import sys
import io
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.crnn_model import CRNNWordRecognizer

VOCAB_CHAR2IDX = config.CRNN_CHAR2IDX
VOCAB_IDX2CHAR = config.CRNN_IDX2CHAR

class SyntheticHandwrittenWordDataset(Dataset):
    """
    Generates synthetic handwritten word images (32x128) using real handwritten
    characters from MNIST and EMNIST, supporting end-to-end sequence modeling with CTC.
    """
    def __init__(self, num_samples: int = 2500):
        self.num_samples = num_samples
        print("Gathering character glyphs from MNIST and EMNIST for CRNN sequence training...", flush=True)
        
        # Load sample digits from MNIST
        df_mnist = pd.read_csv(config.MNIST_TEST_CSV, nrows=2000)
        self.digit_glyphs = {str(i): [] for i in range(10)}
        for _, row in df_mnist.iterrows():
            lbl = str(int(row.iloc[0]))
            pix = row.iloc[1:].values.reshape(28, 28).astype(np.uint8)
            if len(self.digit_glyphs[lbl]) < 100:
                self.digit_glyphs[lbl].append(pix)
                
        # Load sample letters from EMNIST
        df_emnist = pd.read_parquet(config.EMNIST_TEST_PARQUET, engine="fastparquet")
        self.letter_glyphs = {chr(ord('A') + i): [] for i in range(26)}
        for _, row in df_emnist.iterrows():
            lbl = chr(ord('A') + int(row["label"]))
            if len(self.letter_glyphs[lbl]) < 60:
                arr = np.array(Image.open(io.BytesIO(row["image.bytes"]))).T.astype(np.uint8)
                self.letter_glyphs[lbl].append(arr)
                
        self.available_chars = list(self.digit_glyphs.keys()) + list(self.letter_glyphs.keys())
        
        # Predefined vocabulary words + random words
        self.word_corpus = [
            "AI", "ML", "CNN", "DEEP", "DATA", "TEST", "CODE", "NOTE", "READ", "MATH",
            "GOAL", "TEXT", "BYTE", "SCAN", "OCR", "BEST", "FAST", "PAGE", "LINE", "WORD",
            "123", "2026", "456", "789", "A1", "B2", "C3", "X9", "Y8", "Z7", "HELLO", "WORLD"
        ]

    def _render_word_image(self, word: str) -> np.ndarray:
        canvas = np.zeros((config.CRNN_IMG_HEIGHT, config.CRNN_IMG_WIDTH), dtype=np.float32)
        
        target_h = 24
        scale = target_h / 28.0
        
        cur_x = random.randint(2, 8)
        y_offset_base = (config.CRNN_IMG_HEIGHT - target_h) // 2
        
        for char in word:
            if char in self.digit_glyphs and self.digit_glyphs[char]:
                glyph = random.choice(self.digit_glyphs[char])
            elif char in self.letter_glyphs and self.letter_glyphs[char]:
                glyph = random.choice(self.letter_glyphs[char])
            else:
                continue
                
            pil_glyph = Image.fromarray(glyph).resize((int(28 * scale), target_h), Image.Resampling.BILINEAR)
            g_arr = np.array(pil_glyph, dtype=np.float32) / 255.0
            
            gw = g_arr.shape[1]
            if cur_x + gw >= config.CRNN_IMG_WIDTH - 2:
                break
                
            y_jitter = random.randint(-2, 2)
            y_start = max(0, min(config.CRNN_IMG_HEIGHT - target_h, y_offset_base + y_jitter))
            
            canvas[y_start:y_start+target_h, cur_x:cur_x+gw] = np.maximum(
                canvas[y_start:y_start+target_h, cur_x:cur_x+gw],
                g_arr
            )
            cur_x += gw + random.randint(1, 4)
            
        return canvas

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < len(self.word_corpus) * 30:
            word = random.choice(self.word_corpus)
        else:
            length = random.randint(2, 4)
            word = "".join(random.choices(self.available_chars, k=length))
            
        img_arr = self._render_word_image(word)
        tensor = torch.from_numpy(img_arr).unsqueeze(0)  # (1, 32, 128)
        targets = [VOCAB_CHAR2IDX[c] for c in word if c in VOCAB_CHAR2IDX]
        return tensor, torch.tensor(targets, dtype=torch.long), word

def collate_fn_crnn(batch):
    images, targets, words = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat_targets = torch.cat(targets)
    return images, flat_targets, target_lengths, words

def train_crnn_model(epochs: int = 4, batch_size: int = 64, num_samples: int = 2500):
    device = config.DEVICE
    print(f"Using device for CRNN training: {device}", flush=True)
    
    dataset = SyntheticHandwrittenWordDataset(num_samples=num_samples)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_crnn)
    
    num_classes = len(config.CRNN_VOCAB)  # 37
    model = CRNNWordRecognizer(num_classes=num_classes).to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\n--- Starting CRNN Sequence Model Training ({epochs} epochs) ---", flush=True)
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_words = 0
        total_words = 0
        
        for images, targets, target_lengths, words in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            logits = model(images)  # (B, T, C)
            
            # CTCLoss expects log_probs shape: (T, N, C)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            batch_size_cur = images.size(0)
            seq_len = logits.size(1)  # 32
            input_lengths = torch.full(size=(batch_size_cur,), fill_value=seq_len, dtype=torch.long, device=device)
            
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item() * batch_size_cur
            
            with torch.no_grad():
                decoded = model.decode_greedy(logits, VOCAB_IDX2CHAR)
                for pred_w, true_w in zip(decoded, words):
                    if pred_w == true_w:
                        correct_words += 1
                    total_words += 1
                    
        scheduler.step()
        epoch_loss = total_loss / total_words
        word_acc = correct_words / total_words * 100.0
        print(f"Epoch [{epoch}/{epochs}] - CTC Loss: {epoch_loss:.4f} | Word Accuracy: {word_acc:.2f}%", flush=True)
        
    duration = time.time() - start_time
    print(f"\nCRNN Training completed in {duration:.1f}s.", flush=True)
    
    os.makedirs(os.path.dirname(config.CRNN_MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "vocab": config.CRNN_VOCAB,
        "char2idx": VOCAB_CHAR2IDX,
        "idx2char": VOCAB_IDX2CHAR
    }, config.CRNN_MODEL_PATH)
    print(f"Saved CRNN model to: {config.CRNN_MODEL_PATH}", flush=True)
    return model

if __name__ == "__main__":
    train_crnn_model()
