import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

def create_handwritten_word_image(text: str, filename: str, width: int = 380, height: int = 120):
    img = np.ones((height, width), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    font_scale = 1.3
    thickness = 3
    
    # Calculate text size to center
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(15, (width - text_w) // 2)
    y = max(height - 20, (height + text_h) // 2)
    
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # Slight blur to mimic natural ink flow
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    out_path = os.path.join(config.SAMPLES_DIR, filename)
    cv2.imwrite(out_path, img)
    print(f"Generated sample: {out_path}")

def generate_all_samples():
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    samples = [
        ("HELLO", "hello.png"),
        ("AI 2026", "ai_2026.png"),
        ("DATA", "data.png"),
        ("TEST 42", "test_42.png"),
        ("OCR", "ocr.png"),
        ("ABC 123", "abc_123.png")
    ]
    for text, fname in samples:
        create_handwritten_word_image(text, fname)

if __name__ == "__main__":
    generate_all_samples()

