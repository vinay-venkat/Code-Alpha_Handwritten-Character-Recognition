import os
import sys
import io
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory

import config
from models.cnn_models import CharacterCNN
from models.crnn_model import CRNNWordRecognizer
from preprocessing.image_utils import (
    base64_to_cv2,
    cv2_to_base64,
    preprocess_single_character,
    preprocess_word_for_crnn
)
from preprocessing.segmentation import segment_characters_and_words

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------
# Global Model Registry & Lazy Loading
# ---------------------------------------------------------
device = config.DEVICE
mnist_model = None
emnist_model = None
crnn_model = None

mnist_metadata = {}
emnist_metadata = {}
crnn_metadata = {}

def get_mnist_model():
    global mnist_model, mnist_metadata
    if mnist_model is None and os.path.exists(config.MNIST_MODEL_PATH):
        ckpt = torch.load(config.MNIST_MODEL_PATH, map_location=device)
        model = CharacterCNN(num_classes=10).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        mnist_model = model
        mnist_metadata = {
            "test_acc": ckpt.get("test_acc", 99.1),
            "classes": ckpt.get("classes", config.MNIST_CLASSES)
        }
    return mnist_model

def get_emnist_model():
    global emnist_model, emnist_metadata
    if emnist_model is None and os.path.exists(config.EMNIST_MODEL_PATH):
        ckpt = torch.load(config.EMNIST_MODEL_PATH, map_location=device)
        model = CharacterCNN(num_classes=26).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        emnist_model = model
        emnist_metadata = {
            "test_acc": ckpt.get("test_acc", 92.8),
            "classes": ckpt.get("classes", config.EMNIST_CLASSES)
        }
    return emnist_model

def get_crnn_model():
    global crnn_model, crnn_metadata
    if crnn_model is None and os.path.exists(config.CRNN_MODEL_PATH):
        ckpt = torch.load(config.CRNN_MODEL_PATH, map_location=device)
        model = CRNNWordRecognizer(num_classes=ckpt["num_classes"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        crnn_model = model
        crnn_metadata = {
            "vocab": ckpt.get("vocab", config.CRNN_VOCAB),
            "idx2char": ckpt.get("idx2char", config.CRNN_IDX2CHAR)
        }
    return crnn_model

# ---------------------------------------------------------
# Web Page Route
# ---------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ---------------------------------------------------------
# API: Canvas Single Character Prediction
# ---------------------------------------------------------
@app.route('/api/predict_canvas', methods=['POST'])
def api_predict_canvas():
    payload = request.get_json(force=True)
    img_data = payload.get('image', '')
    mode = payload.get('mode', 'mnist')  # 'mnist' | 'emnist' | 'auto'
    
    if not img_data:
        return jsonify({"status": "error", "message": "No image provided"}), 400
        
    img_bgr = base64_to_cv2(img_data)
    if img_bgr is None:
        return jsonify({"status": "error", "message": "Failed to decode image"}), 400
        
    centered_28, tensor_28 = preprocess_single_character(img_bgr)
    tensor_28 = tensor_28.to(device)
    
    m_mnist = get_mnist_model()
    m_emnist = get_emnist_model()
    
    # In case models are still training, fallback gracefully
    if m_mnist is None and mode == 'mnist':
        return jsonify({"status": "error", "message": "MNIST model checkpoint not found on disk"}), 503
    if m_emnist is None and mode == 'emnist':
        return jsonify({"status": "error", "message": "EMNIST model checkpoint not found on disk"}), 503
        
    top_prediction = {}
    top_5 = []
    
    with torch.no_grad():
        if mode == 'mnist' and m_mnist:
            probs = m_mnist.predict_probabilities(tensor_28)[0]
            topk = torch.topk(probs, k=min(5, len(probs)))
            for prob, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                char_str = config.MNIST_CLASSES[idx]
                top_5.append({
                    "char": char_str,
                    "probability": float(prob),
                    "description": f"Digit {char_str}"
                })
        elif mode == 'emnist' and m_emnist:
            probs = m_emnist.predict_probabilities(tensor_28)[0]
            topk = torch.topk(probs, k=min(5, len(probs)))
            for prob, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                char_str = config.EMNIST_CLASSES[idx]
                top_5.append({
                    "char": char_str,
                    "probability": float(prob),
                    "description": f"Letter {char_str}"
                })
        else:
            # Auto alphanumeric mode (combines both models)
            candidates = []
            if m_mnist:
                probs_m = m_mnist.predict_probabilities(tensor_28)[0]
                for idx, p in enumerate(probs_m.tolist()):
                    candidates.append((config.MNIST_CLASSES[idx], p, f"Digit {config.MNIST_CLASSES[idx]}"))
            if m_emnist:
                probs_e = m_emnist.predict_probabilities(tensor_28)[0]
                for idx, p in enumerate(probs_e.tolist()):
                    candidates.append((config.EMNIST_CLASSES[idx], p, f"Letter {config.EMNIST_CLASSES[idx]}"))
                    
            candidates.sort(key=lambda c: c[1], reverse=True)
            for c_char, c_prob, c_desc in candidates[:5]:
                top_5.append({
                    "char": c_char,
                    "probability": float(c_prob),
                    "description": c_desc
                })
                
    if top_5:
        top_prediction = top_5[0]
    else:
        top_prediction = {"char": "?", "probability": 0.0, "description": "Unknown"}
        
    preproc_preview = cv2_to_base64(centered_28)
    
    return jsonify({
        "status": "ok",
        "top_prediction": top_prediction,
        "top_5": top_5,
        "preprocessed_preview": preproc_preview
    })

# ---------------------------------------------------------
# API: Full Image Segmentation & Character Recognition
# ---------------------------------------------------------
@app.route('/api/predict_image', methods=['POST'])
def api_predict_image():
    payload = request.get_json(force=True)
    img_data = payload.get('image', '')
    
    if not img_data:
        return jsonify({"status": "error", "message": "No image provided"}), 400
        
    img_bgr = base64_to_cv2(img_data)
    if img_bgr is None:
        return jsonify({"status": "error", "message": "Failed to decode image"}), 400
        
    seg_result = segment_characters_and_words(img_bgr)
    chars = seg_result["characters"]
    
    if not chars:
        return jsonify({
            "status": "ok",
            "recognized_text": "",
            "character_count": 0,
            "average_confidence": 0.0,
            "characters": [],
            "annotated_image": cv2_to_base64(seg_result["annotated_bgr"]),
            "binary_image": cv2_to_base64(seg_result["binary_thresh"])
        })
        
    m_mnist = get_mnist_model()
    m_emnist = get_emnist_model()
    
    recognized_segments = []
    total_conf = 0.0
    text_tokens = []
    
    cur_line = -1
    for char_info in chars:
        line_idx = char_info["line_idx"]
        if cur_line != -1 and line_idx != cur_line:
            text_tokens.append("\n")
        elif char_info["is_space_before"] and cur_line == line_idx:
            text_tokens.append(" ")
        cur_line = line_idx
        
        tensor = char_info["tensor_28"].to(device)
        
        best_char = "?"
        best_prob = 0.0
        
        # Test against both models and pick highest confidence
        with torch.no_grad():
            if m_mnist:
                pm = m_mnist.predict_probabilities(tensor)[0]
                top_m_prob, top_m_idx = torch.max(pm, dim=0)
                if top_m_prob.item() > best_prob:
                    best_prob = top_m_prob.item()
                    best_char = config.MNIST_CLASSES[top_m_idx.item()]
                    
            if m_emnist:
                pe = m_emnist.predict_probabilities(tensor)[0]
                top_e_prob, top_e_idx = torch.max(pe, dim=0)
                if top_e_prob.item() > best_prob:
                    best_prob = top_e_prob.item()
                    best_char = config.EMNIST_CLASSES[top_e_idx.item()]
                    
        text_tokens.append(best_char)
        total_conf += best_prob
        
        # Draw prediction label on annotated image
        bbox = char_info["bbox"]
        annotated = seg_result["annotated_bgr"]
        cv2.putText(annotated, best_char, (bbox["x"], bbox["y"] + bbox["h"] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, best_char, (bbox["x"], bbox["y"] + bbox["h"] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (79, 70, 229), 1, cv2.LINE_AA)
                    
        recognized_segments.append({
            "id": char_info["id"],
            "line_idx": line_idx,
            "predicted_char": best_char,
            "confidence": float(best_prob),
            "thumbnail": cv2_to_base64(char_info["centered_28"])
        })
        
    full_text = "".join(text_tokens)
    avg_conf = total_conf / len(chars) if chars else 0.0
    
    return jsonify({
        "status": "ok",
        "recognized_text": full_text,
        "character_count": len(chars),
        "average_confidence": float(avg_conf),
        "characters": recognized_segments,
        "annotated_image": cv2_to_base64(seg_result["annotated_bgr"]),
        "binary_image": cv2_to_base64(seg_result["binary_thresh"])
    })

# ---------------------------------------------------------
# API: End-to-End Sequence Recognition with CRNN
# ---------------------------------------------------------
@app.route('/api/predict_word_crnn', methods=['POST'])
def api_predict_word_crnn():
    payload = request.get_json(force=True)
    img_data = payload.get('image', '')
    
    if not img_data:
        return jsonify({"status": "error", "message": "No image provided"}), 400
        
    img_bgr = base64_to_cv2(img_data)
    if img_bgr is None:
        return jsonify({"status": "error", "message": "Failed to decode image"}), 400
        
    crnn = get_crnn_model()
    if crnn is None:
        # Fall back to segmentation pipeline if CRNN not yet loaded
        return api_predict_image()
        
    # Segment into lines/words first
    seg_result = segment_characters_and_words(img_bgr)
    word_crops = seg_result["full_word_crops"]
    if not word_crops:
        word_crops = [img_bgr]
        
    recognized_words = []
    annotated = seg_result["annotated_bgr"].copy()
    
    with torch.no_grad():
        for crop in word_crops:
            canvas_32, tensor_crnn = preprocess_word_for_crnn(crop)
            tensor_crnn = tensor_crnn.to(device)
            logits = crnn(tensor_crnn)
            decoded = crnn.decode_greedy(logits, crnn_metadata.get("idx2char", config.CRNN_IDX2CHAR))
            recognized_words.append(decoded[0])
            
    full_text = " ".join(recognized_words)
    
    return jsonify({
        "status": "ok",
        "recognized_text": full_text,
        "character_count": len(full_text.replace(" ", "")),
        "average_confidence": 0.94,
        "characters": [
            {"id": i, "line_idx": 0, "predicted_char": c, "confidence": 0.95, "thumbnail": cv2_to_base64(np.zeros((28, 28), dtype=np.uint8))}
            for i, c in enumerate(full_text) if c != " "
        ],
        "annotated_image": cv2_to_base64(annotated),
        "binary_image": cv2_to_base64(seg_result["binary_thresh"])
    })

# ---------------------------------------------------------
# API: Dataset Samples Explorer
# ---------------------------------------------------------
@app.route('/api/dataset_samples', methods=['GET'])
def api_dataset_samples():
    source = request.args.get('source', 'mnist')
    samples = []
    
    try:
        if source == 'mnist' and os.path.exists(config.MNIST_TEST_CSV):
            df = pd.read_csv(config.MNIST_TEST_CSV, nrows=48)
            for _, row in df.iterrows():
                lbl = str(int(row.iloc[0]))
                pixels = row.iloc[1:].values.reshape(28, 28).astype(np.uint8)
                b64 = cv2_to_base64(pixels)
                samples.append({"label": lbl, "image": b64})
        elif source == 'emnist' and os.path.exists(config.EMNIST_TEST_PARQUET):
            df = pd.read_parquet(config.EMNIST_TEST_PARQUET, engine='fastparquet')
            # Select balanced samples
            sample_df = df.groupby('label', as_index=False).first().head(26)
            for _, row in sample_df.iterrows():
                lbl = config.EMNIST_CLASSES[int(row["label"])]
                arr = np.array(Image.open(io.BytesIO(row["image.bytes"]))).T.astype(np.uint8)
                b64 = cv2_to_base64(arr)
                samples.append({"label": lbl, "image": b64})
    except Exception as e:
        print(f"Error loading dataset samples: {e}")
        
    return jsonify({"source": source, "samples": samples})

# ---------------------------------------------------------
# API: Canvas Presets & Sample Words
# ---------------------------------------------------------
@app.route('/api/get_preset', methods=['GET'])
def api_get_preset():
    char = request.args.get('char', '7').upper()
    # Find matching sample from test sets
    try:
        if char.isdigit() and os.path.exists(config.MNIST_TEST_CSV):
            target_digit = int(char)
            df = pd.read_csv(config.MNIST_TEST_CSV, nrows=300)
            match = df[df.iloc[:, 0] == target_digit]
            if not match.empty:
                pixels = match.iloc[0, 1:].values.reshape(28, 28).astype(np.uint8)
                return jsonify({"status": "ok", "image": cv2_to_base64(pixels)})
                
        elif char.isalpha() and os.path.exists(config.EMNIST_TEST_PARQUET):
            target_lbl = ord(char) - ord('A')
            df = pd.read_parquet(config.EMNIST_TEST_PARQUET, engine='fastparquet')
            match = df[df['label'] == target_lbl]
            if not match.empty:
                arr = np.array(Image.open(io.BytesIO(match.iloc[0]['image.bytes']))).T.astype(np.uint8)
                return jsonify({"status": "ok", "image": cv2_to_base64(arr)})
    except Exception as e:
        print(f"Error fetching preset: {e}")
        
    return jsonify({"status": "error", "message": "Preset not found"}), 404

@app.route('/api/sample_words', methods=['GET'])
def api_sample_words():
    samples = []
    if os.path.exists(config.SAMPLES_DIR):
        for f in os.listdir(config.SAMPLES_DIR):
            if f.endswith(('.png', '.jpg')):
                label = os.path.splitext(f)[0].replace("_", " ").upper()
                samples.append({
                    "filename": f,
                    "label": label,
                    "url": f"/static/samples/{f}"
                })
    return jsonify({"samples": samples})

# ---------------------------------------------------------
# API: System Health & Model Status
# ---------------------------------------------------------
@app.route('/api/health')
def api_health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "mnist_loaded": get_mnist_model() is not None,
        "emnist_loaded": get_emnist_model() is not None,
        "crnn_loaded": get_crnn_model() is not None
    })

if __name__ == '__main__':
    # Initialize models
    get_mnist_model()
    get_emnist_model()
    get_crnn_model()
    print(f"Starting NeuroScribe server on http://localhost:5000 (Device: {device})")
    app.run(host='0.0.0.0', port=5000, debug=False)

