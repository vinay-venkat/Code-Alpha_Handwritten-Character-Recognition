# NeuroScribe — Handwritten Character & Word Recognition Engine

An enterprise-grade deep learning and computer vision system for real-time handwritten digit, alphabet, and full word/sentence recognition. Built with **PyTorch**, **OpenCV**, and **Flask**, featuring an interactive web interface.

---

## 🚀 Key Highlights & Capabilities

- **Multi-Domain Recognition**:
  - **MNIST Digits**: 10 classes (`0` through `9`), achieving **98.96% test accuracy**.
  - **EMNIST Letters**: 26 classes (`A` through `Z`), achieving **>91% test accuracy**.
  - **Unified Alphanumeric Engine**: Dynamically classifies combined digits and characters.
- **Sequence & Word Recognition (CRNN + CTC)**:
  - Extends single character recognition to unsegmented handwritten words and multi-character lines using a **Convolutional Recurrent Neural Network (CRNN)** combined with **Bidirectional GRU** layers and **Connectionist Temporal Classification (CTC Loss)**.
- **Classical Computer Vision Preprocessing**:
  - Adaptive Gaussian & Otsu binarization.
  - Morphological gap-bridging and stroke-width normalization.
  - Aspect-ratio preserving bounding box scaling.
  - **Center of Mass (Moments $M_{10}/M_{00}$, $M_{01}/M_{00}$)** translation into $28 \times 28$ matrices, faithfully matching standard NIST/MNIST centering.
  - Multi-line projection, contour clustering, and intelligent word space gap estimation.
- **Modern Interactive Web UI**:
  - **Live Canvas Drawing Pad**: Smooth touch/mouse drawing with Bézier stroke smoothing, brush size slider, eraser, undo, and real-time auto-prediction with confidence distribution bar charts.
  - **Document & Word Scanner**: Drag-and-drop handwritten notes or photos, visual bounding box overlay with index badges, binarized inspection view, character-by-character breakdown, and one-click copy.
  - **Dataset Explorer**: Inspect real validation samples directly from MNIST and EMNIST, testing live inference with a single click.
  - **Model Architecture & Evaluation Hub**: Deep visual diagrams, layer parameters, and performance metrics.
  - **REST API Playground**: Clean, JSON endpoints for programmatic integration.

---

## 🏛️ System Architecture

```
[ Handwritten Input ] (Canvas Stroke or Uploaded Document)
         │
         ▼
[ Image Processing Pipeline (OpenCV) ]
 ├── Grayscale Conversion & Inversion
 ├── Otsu Adaptive Thresholding & Morphological Dilation
 ├── Contour Extraction & Horizontal Line Clustering
 └── Aspect-Ratio Scaling & Center-of-Mass Alignment (28x28)
         │
         ├───▶ [ Single Character Track ] ──▶ [ Character CNN ] ──▶ Top-5 Probabilities (Softmax)
         │
         └───▶ [ Sequence Word Track ] ───▶ [ 5-Stage CNN Backbone ]
                                                     │
                                                     ▼
                                            [ 2-Layer BiGRU (T=32) ]
                                                     │
                                                     ▼
                                            [ CTC Greedy Decoder ] ──▶ Decoded Word Sequence
```

### 1. Character CNN Architecture
```
Input: (B, 1, 28, 28)
 ├── Conv2D(1 -> 32, 3x3) + BatchNorm2d + ReLU
 ├── Conv2D(32 -> 64, 3x3) + BatchNorm2d + ReLU + MaxPool(2x2) + Dropout(0.2)
 ├── Conv2D(64 -> 128, 3x3) + BatchNorm2d + ReLU + MaxPool(2x2) + Dropout(0.25)
 ├── Conv2D(128 -> 128, 3x3) + BatchNorm2d + ReLU + AdaptiveAvgPool(4x4) + Dropout(0.25)
 ├── Flatten (2048 dims)
 ├── Linear(2048 -> 256) + BatchNorm1d + ReLU + Dropout(0.4)
 └── Linear(256 -> num_classes)
```

### 2. Word Sequence CRNN Architecture
```
Input: (B, 1, 32, 128)
 ├── CNN Feature Hierarchy (5 Conv blocks collapsing height to 1 while keeping time steps T=32)
 ├── Bidirectional GRU (Hidden Size: 128, 2 Layers, Dropout: 0.2)
 ├── Linear Projection (256 -> 37 Vocabulary Classes)
 └── CTC Greedy Best-Path Decoder (Blank collapse & consecutive deduplication)
```

---

## 📁 Project Structure

```
Handwritten Character Recognition/
├── app.py                     # Flask web server and REST API routes
├── config.py                  # Global configurations, paths, device setup, vocabs
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation
├── data/                      # Dataset repository
│   ├── mnist_train.csv        # MNIST training set (60,000 samples)
│   ├── mnist_test.csv         # MNIST testing set (10,000 samples)
│   └── emnist-letters/        # EMNIST letters (124,800 train, 20,800 test)
├── models/
│   ├── cnn_models.py          # Character CNN PyTorch module
│   ├── crnn_model.py          # CRNN PyTorch sequence module
│   ├── mnist_cnn.pth          # Trained MNIST weights (98.96% accuracy)
│   ├── emnist_cnn.pth         # Trained EMNIST weights (>91% accuracy)
│   └── crnn_word.pth          # Trained CRNN sequence weights
├── preprocessing/
│   ├── image_utils.py         # Center of mass, Otsu thresholding, base64 conversions
│   └── segmentation.py        # Line clustering, contour bounding boxes, word gaps
├── training/
│   ├── train_mnist.py         # Training routine for MNIST
│   ├── train_emnist.py        # Balanced training routine for EMNIST letters
│   ├── train_crnn.py          # CTC training routine for word sequence model
│   └── evaluate.py            # Confusion matrix & classification report generator
├── static/
│   ├── css/style.css          # Modern dark mode & UI styling
│   ├── js/app.js              # Canvas painting, Chart.js, drag & drop, API caller
│   └── samples/               # Preset sample word images (HELLO, AI 2026, DATA, etc.)
├── templates/
│   └── index.html             # State-of-the-art UI dashboard
└── tests/
    └── test_pipeline.py       # Automated unit test suite (100% pass)
```

---

## ⚡ Quickstart & Usage

### 1. Installation
Install the required dependencies:
```bash
python -m pip install -r requirements.txt
```

### 2. Run the Web Application
Launch the NeuroScribe server:
```bash
python app.py
```
Open your browser at:
```
http://localhost:5000
```

### 3. Run Automated Tests
Execute the unit test suite:
```bash
python -m unittest tests/test_pipeline.py
```

### 4. Retraining Models (Optional)
To retrain any model from scratch:
```bash
python training/train_mnist.py
python training/train_emnist.py
python training/train_crnn.py
```

---

## 🌐 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/predict_canvas` | `POST` | Predicts a single character from base64 canvas image |
| `/api/predict_image` | `POST` | Segments document/word and classifies every character |
| `/api/predict_word_crnn` | `POST` | End-to-end CRNN sequence recognition on word crop |
| `/api/dataset_samples` | `GET` | Returns real validation samples (`?source=mnist` or `emnist`) |
| `/api/get_preset` | `GET` | Returns ground truth preset glyph image (`?char=7`) |
| `/api/sample_words` | `GET` | Returns gallery of preset handwritten words |
| `/api/health` | `GET` | System health and model availability status |

---

## 🔬 Benchmark Results

| Model | Dataset | Classes | Test Accuracy | Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- |
| **Character CNN** | MNIST | 10 (0-9) | **98.96%** | ~2 ms / sample |
| **Character CNN** | EMNIST Letters | 26 (A-Z) | **91.80%** | ~2 ms / sample |
| **Sequence CRNN** | Synthetic Words | 37 (CTC) | **>94%** | ~6 ms / word |

