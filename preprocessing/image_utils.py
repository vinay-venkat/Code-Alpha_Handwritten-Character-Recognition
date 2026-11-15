import base64
import cv2
import numpy as np
from PIL import Image
import io
import torch

def base64_to_cv2(data_str: str) -> np.ndarray:
    """Decodes a base64 string or data URL into an OpenCV BGR image."""
    if ',' in data_str:
        data_str = data_str.split(',', 1)[1]
    img_bytes = base64.b64decode(data_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    # Handle alpha channel (common with HTML5 canvas)
    if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
        # Check if alpha channel has transparency
        alpha = img[:, :, 3]
        rgb = img[:, :, :3]
        # White background where alpha is 0
        white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
        alpha_factor = alpha[:, :, np.newaxis].astype(np.float32) / 255.0
        blended = (rgb.astype(np.float32) * alpha_factor + white_bg.astype(np.float32) * (1 - alpha_factor)).astype(np.uint8)
        return blended
    elif img is not None and len(img.shape) == 3:
        return img
    elif img is not None and len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def cv2_to_base64(img_np: np.ndarray, ext: str = ".png") -> str:
    """Encodes an OpenCV image to a base64 data URI string."""
    _, buffer = cv2.imencode(ext, img_np)
    encoded = base64.b64encode(buffer).decode('utf-8')
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"

def center_of_mass_shift(img: np.ndarray) -> np.ndarray:
    """
    Shifts a 28x28 binary/grayscale image so its center of mass aligns at (14, 14).
    This strictly replicates the official NIST / MNIST preprocessing specification.
    """
    moments = cv2.moments(img)
    if moments['m00'] == 0:
        return img
    
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    
    shift_x = int(np.round(14.0 - cx))
    shift_y = int(np.round(14.0 - cy))
    
    # Clamp shifts to reasonable range
    shift_x = max(min(shift_x, 6), -6)
    shift_y = max(min(shift_y, 6), -6)
    
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(img, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return shifted

def preprocess_single_character(img_bgr_or_gray: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    """
    Takes an input character image (canvas or crop), extracts the stroke,
    scales aspect-preserved to 20x20 inside 28x28, and aligns center of mass.
    Returns:
        centered_28x28: np.ndarray (28, 28) uint8 with white stroke on black background
        tensor: torch.Tensor of shape (1, 1, 28, 28) normalized to [0, 1]
    """
    if len(img_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray.copy()
        
    # Check if background is mostly light (e.g. black pen on white paper)
    # If mean > 127, invert so stroke is white (255) on dark background (0)
    if np.mean(gray) > 127:
        gray = 255 - gray
        
    # Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean small artifacts with slight opening
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find bounding box of foreground strokes
    coords = cv2.findNonZero(thresh)
    if coords is None:
        # Blank canvas
        blank = np.zeros((28, 28), dtype=np.uint8)
        tensor = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        return blank, tensor
        
    x, y, w, h = cv2.boundingRect(coords)
    crop = thresh[y:y+h, x:x+w]
    
    # Scale aspect-preserved into a 20x20 box
    if w > h:
        new_w = 20
        new_h = max(int(np.round(h * (20.0 / w))), 1)
    else:
        new_h = 20
        new_w = max(int(np.round(w * (20.0 / h))), 1)
        
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center into 28x28 canvas
    canvas_28 = np.zeros((28, 28), dtype=np.uint8)
    pad_top = (28 - new_h) // 2
    pad_left = (28 - new_w) // 2
    canvas_28[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    
    # Align Center of Mass
    centered_28 = center_of_mass_shift(canvas_28)
    
    # Convert to Tensor normalized [0.0, 1.0]
    tensor = torch.from_numpy(centered_28).float().unsqueeze(0).unsqueeze(0) / 255.0
    return centered_28, tensor

def preprocess_word_for_crnn(img_bgr_or_gray: np.ndarray, target_h: int = 32, target_w: int = 128) -> tuple[np.ndarray, torch.Tensor]:
    """
    Prepares a word or multi-character snippet for CRNN input (1, 1, 32, 128).
    """
    if len(img_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray.copy()
        
    if np.mean(gray) > 127:
        gray = 255 - gray
        
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find bounding box
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        thresh = thresh[y:y+h, x:x+w]
    
    h, w = thresh.shape
    scale = target_h / float(max(h, 1))
    new_w = int(w * scale)
    if new_w > target_w:
        new_w = target_w
        resized = cv2.resize(thresh, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(thresh, (new_w, target_h), interpolation=cv2.INTER_AREA)
        
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    canvas[:, :new_w] = resized
    
    tensor = torch.from_numpy(canvas).float().unsqueeze(0).unsqueeze(0) / 255.0
    return canvas, tensor

