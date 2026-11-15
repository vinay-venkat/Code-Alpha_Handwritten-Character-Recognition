import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MNIST_TRAIN_CSV = os.path.join(DATA_DIR, "mnist_train.csv")
MNIST_TEST_CSV = os.path.join(DATA_DIR, "mnist_test.csv")
EMNIST_TRAIN_PARQUET = os.path.join(DATA_DIR, "emnist-letters", "train-00000-of-00001.parquet")
EMNIST_TEST_PARQUET = os.path.join(DATA_DIR, "emnist-letters", "test-00000-of-00001.parquet")

# Models paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
MNIST_MODEL_PATH = os.path.join(MODELS_DIR, "mnist_cnn.pth")
EMNIST_MODEL_PATH = os.path.join(MODELS_DIR, "emnist_cnn.pth")
CRNN_MODEL_PATH = os.path.join(MODELS_DIR, "crnn_word.pth")

# Static / Samples paths
STATIC_DIR = os.path.join(BASE_DIR, "static")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")

# Hardware Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes and Vocabularies
MNIST_CLASSES = [str(i) for i in range(10)]
EMNIST_CLASSES = [chr(ord('A') + i) for i in range(26)]

# CRNN vocabulary: CTC blank token is index 0
CRNN_BLANK_CHAR = '-'
CRNN_CHAR_LIST = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
CRNN_VOCAB = [CRNN_BLANK_CHAR] + CRNN_CHAR_LIST
CRNN_CHAR2IDX = {c: i for i, c in enumerate(CRNN_VOCAB)}
CRNN_IDX2CHAR = {i: c for i, c in enumerate(CRNN_VOCAB)}

# Standard image dimensions
CHAR_IMG_SIZE = (28, 28)
CRNN_IMG_HEIGHT = 32
CRNN_IMG_WIDTH = 128

