import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.cnn_models import CharacterCNN
from models.crnn_model import CRNNWordRecognizer
from preprocessing.image_utils import (
    preprocess_single_character,
    preprocess_word_for_crnn,
    center_of_mass_shift,
    cv2_to_base64,
    base64_to_cv2
)
from preprocessing.segmentation import segment_characters_and_words

class TestHandwrittenRecognitionPipeline(unittest.TestCase):
    
    def test_cnn_architecture(self):
        """Verify CharacterCNN forward pass and shapes."""
        model = CharacterCNN(num_classes=10)
        dummy_input = torch.randn(4, 1, 28, 28)
        output = model(dummy_input)
        self.assertEqual(output.shape, (4, 10))
        
        probs = model.predict_probabilities(dummy_input)
        self.assertEqual(probs.shape, (4, 10))
        # Ensure probabilities sum to ~1.0
        sums = probs.sum(dim=1).tolist()
        for s in sums:
            self.assertAlmostEqual(s, 1.0, places=4)

    def test_crnn_architecture(self):
        """Verify CRNN word model forward pass and greedy decoding."""
        num_classes = 37
        crnn = CRNNWordRecognizer(num_classes=num_classes)
        dummy_input = torch.randn(2, 1, 32, 128)
        logits = crnn(dummy_input)
        self.assertEqual(logits.shape, (2, 32, num_classes))
        
        decoded = crnn.decode_greedy(logits, config.CRNN_IDX2CHAR)
        self.assertEqual(len(decoded), 2)
        self.assertIsInstance(decoded[0], str)

    def test_preprocess_single_character(self):
        """Verify single character centering and tensor shape."""
        # Create a 200x200 canvas with a white square in the corner
        test_img = np.zeros((200, 200), dtype=np.uint8)
        test_img[30:70, 30:60] = 255  # draw a rectangle stroke
        
        centered_28, tensor_28 = preprocess_single_character(test_img)
        self.assertEqual(centered_28.shape, (28, 28))
        self.assertEqual(tensor_28.shape, (1, 1, 28, 28))
        self.assertTrue(tensor_28.max() <= 1.0)
        self.assertTrue(tensor_28.min() >= 0.0)

    def test_preprocess_blank_canvas(self):
        """Verify behavior on empty canvas."""
        blank = np.zeros((100, 100), dtype=np.uint8)
        centered_28, tensor_28 = preprocess_single_character(blank)
        self.assertEqual(centered_28.shape, (28, 28))
        self.assertEqual(tensor_28.sum().item(), 0.0)

    def test_word_preprocess_for_crnn(self):
        """Verify word preprocessing aspect scaling to 32x128."""
        word_canvas = np.zeros((60, 150), dtype=np.uint8)
        word_canvas[10:50, 20:130] = 255
        
        canvas_32, tensor_crnn = preprocess_word_for_crnn(word_canvas, target_h=32, target_w=128)
        self.assertEqual(canvas_32.shape, (32, 128))
        self.assertEqual(tensor_crnn.shape, (1, 1, 32, 128))

    def test_segmentation_pipeline(self):
        """Verify contour extraction, line clustering, and reading order sorting."""
        # Create a simulated document image with 2 letters side by side
        doc = np.ones((100, 200, 3), dtype=np.uint8) * 255
        # Draw stroke 1
        doc[20:60, 30:50] = 0
        # Draw stroke 2
        doc[20:60, 80:100] = 0
        
        result = segment_characters_and_words(doc)
        self.assertGreaterEqual(len(result["characters"]), 2)
        # Verify left to right order
        c0_x = result["characters"][0]["bbox"]["x"]
        c1_x = result["characters"][1]["bbox"]["x"]
        self.assertLess(c0_x, c1_x)

    def test_base64_conversions(self):
        """Verify base64 round trip encoding/decoding."""
        original = np.zeros((50, 50, 3), dtype=np.uint8)
        original[10:20, 10:20] = [255, 128, 64]
        b64 = cv2_to_base64(original)
        self.assertTrue(b64.startswith("data:image/png;base64,"))
        decoded = base64_to_cv2(b64)
        self.assertEqual(decoded.shape, original.shape)

if __name__ == "__main__":
    unittest.main()

