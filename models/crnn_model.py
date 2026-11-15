import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNNWordRecognizer(nn.Module):
    """
    CRNN (CNN + Bidirectional GRU + CTC) for handwritten word and sequence recognition.
    Input image dimension: (B, 1, 32, 128) -> output character sequence without segmentation.
    """
    def __init__(self, num_classes: int = 37):
        # num_classes includes CTC blank token at index 0 + vocabulary tokens
        super(CRNNWordRecognizer, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Layer 1: 32x128 -> 16x64
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            # Layer 2: 16x64 -> 8x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            # Layer 3: 8x32 -> 4x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Layer 4: 4x32 -> 2x32
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Layer 5: 2x32 -> 1x32
            nn.Conv2d(128, 256, kernel_size=(2, 1), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        
        # Recurrent Sequence Modeling (Bidirectional GRU)
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # Final classification per time step
        self.fc = nn.Linear(128 * 2, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, W)
        features = self.cnn(x)  # (B, 256, 1, W')
        features = features.squeeze(2)  # (B, 256, W')
        features = features.permute(0, 2, 1)  # (B, W', 256)
        
        recurrent_out, _ = self.rnn(features)  # (B, W', 256)
        logits = self.fc(recurrent_out)  # (B, W', num_classes)
        return logits
    
    def decode_greedy(self, logits: torch.Tensor, idx2char: dict) -> list[str]:
        """
        Greedy Best-Path decoding for CTC output.
        Collapses repeated characters and removes blanks (index 0).
        """
        # logits: (B, T, C)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)  # (B, T)
        decoded_strings = []
        
        for b in range(preds.size(0)):
            seq = preds[b].tolist()
            chars = []
            prev_token = None
            for token in seq:
                if token != prev_token:
                    if token != 0:  # 0 is blank
                        chars.append(idx2char.get(token, ''))
                    prev_token = token
            decoded_strings.append("".join(chars))
            
        return decoded_strings

