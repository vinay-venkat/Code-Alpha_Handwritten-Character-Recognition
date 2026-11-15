import cv2
import numpy as np
import torch
from preprocessing.image_utils import preprocess_single_character, preprocess_word_for_crnn

def segment_characters_and_words(image_bgr: np.ndarray) -> dict:
    """
    Performs full classical computer vision segmentation on an image:
    1. Grayscale & adaptive binarization
    2. Contour detection for characters & word clusters
    3. Sorting by line (top to bottom) and reading order (left to right)
    4. Word gap detection for realistic spacing
    5. Preparation of character tensors and word crops
    """
    orig = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Invert if light background
    if np.mean(gray) > 127:
        inv = 255 - gray
    else:
        inv = gray.copy()
        
    # Otsu thresholding + slight Gaussian blur
    blurred = cv2.GaussianBlur(inv, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological dilation to bridge small gaps within character strokes
    kernel_stroke = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(thresh, kernel_stroke, iterations=1)
    
    # Find external contours
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_boxes = []
    min_area = 25
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 4 or h < 6:
            continue
        raw_boxes.append((x, y, w, h))
        
    if not raw_boxes:
        return {
            "characters": [],
            "lines": [],
            "annotated_bgr": orig,
            "binary_thresh": thresh,
            "full_word_crops": []
        }
        
    # Group boxes into lines
    # Sort boxes primarily by vertical center y + h/2
    raw_boxes = sorted(raw_boxes, key=lambda b: (b[1] + b[3] / 2.0))
    
    # Clustering into horizontal lines
    lines = []
    current_line = [raw_boxes[0]]
    line_y_center = raw_boxes[0][1] + raw_boxes[0][3] / 2.0
    line_height = raw_boxes[0][3]
    
    for box in raw_boxes[1:]:
        bx, by, bw, bh = box
        b_center = by + bh / 2.0
        
        # If vertical center is within 65% of current line's average height, same line
        if abs(b_center - line_y_center) < max(line_height, bh) * 0.65:
            current_line.append(box)
            # update running center
            line_y_center = sum((b[1] + b[3]/2.0) for b in current_line) / len(current_line)
            line_height = max(b[3] for b in current_line)
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
            line_y_center = b_center
            line_height = bh
            
    if current_line:
        lines.append(sorted(current_line, key=lambda b: b[0]))
        
    # Process each line and identify word spaces
    processed_characters = []
    annotated = orig.copy()
    full_word_crops = []
    
    # Color palette for bounding boxes
    colors = [
        (37, 99, 235),   # Blue
        (16, 185, 129),  # Emerald
        (245, 158, 11),  # Amber
        (139, 92, 246),  # Purple
        (236, 72, 153),  # Pink
        (6, 182, 212)    # Cyan
    ]
    
    char_global_idx = 0
    line_structures = []
    
    for line_idx, line in enumerate(lines):
        line_chars = []
        if not line:
            continue
            
        # Calculate median character width for space estimation
        widths = [b[2] for b in line]
        median_w = np.median(widths) if widths else 20
        space_threshold = median_w * 0.8
        
        # Line bounding box for word crop
        min_lx = min(b[0] for b in line)
        min_ly = min(b[1] for b in line)
        max_rx = max(b[0] + b[2] for b in line)
        max_ry = max(b[1] + b[3] for b in line)
        
        pad = 4
        ly1 = max(0, min_ly - pad)
        ly2 = min(orig.shape[0], max_ry + pad)
        lx1 = max(0, min_lx - pad)
        lx2 = min(orig.shape[1], max_rx + pad)
        line_crop = orig[ly1:ly2, lx1:lx2]
        full_word_crops.append(line_crop)
        
        for i, box in enumerate(line):
            x, y, w, h = box
            
            # Check for word break before this character
            is_space_before = False
            if i > 0:
                prev_x = line[i-1][0]
                prev_w = line[i-1][2]
                gap = x - (prev_x + prev_w)
                if gap > space_threshold:
                    is_space_before = True
                    
            # Crop single character with slight margin
            cpad = 2
            cy1 = max(0, y - cpad)
            cy2 = min(orig.shape[0], y + h + cpad)
            cx1 = max(0, x - cpad)
            cx2 = min(orig.shape[1], x + w + cpad)
            char_crop = orig[cy1:cy2, cx1:cx2]
            
            centered_28, tensor_28 = preprocess_single_character(char_crop)
            
            color = colors[char_global_idx % len(colors)]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Put index pill
            badge_text = f"#{char_global_idx + 1}"
            cv2.putText(annotated, badge_text, (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
            char_data = {
                "id": char_global_idx,
                "line_idx": line_idx,
                "is_space_before": is_space_before,
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "centered_28": centered_28,
                "tensor_28": tensor_28
            }
            processed_characters.append(char_data)
            line_chars.append(char_data)
            char_global_idx += 1
            
        line_structures.append(line_chars)
        
    return {
        "characters": processed_characters,
        "lines": line_structures,
        "annotated_bgr": annotated,
        "binary_thresh": thresh,
        "full_word_crops": full_word_crops
    }

