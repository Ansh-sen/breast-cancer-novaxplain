"""
Image Validator for Histopathology (H&E) Detection
Validates that uploaded images are H&E stained tissue, not random photos/objects
Uses color analysis and histogram methods
"""

import numpy as np
import cv2
from PIL import Image


class HistopathologyValidator:
    """
    Validates if an image is histopathology (H&E stained tissue)
    Uses color distribution analysis and texture checks
    """
    
    def __init__(self):
        # H&E Staining Color Ranges (HSV color space)
        # Purple nuclei (Hematoxylin)
        self.PURPLE_H_MIN, self.PURPLE_H_MAX = 220, 280
        self.PURPLE_S_MIN, self.PURPLE_S_MAX = 0.2, 1.0
        self.PURPLE_V_MIN, self.PURPLE_V_MAX = 0.1, 0.9
        
        # Pink cytoplasm (Eosin)
        self.PINK_H_MIN, self.PINK_H_MAX = 330, 20  # wraps around
        self.PINK_S_MIN, self.PINK_S_MAX = 0.1, 0.7
        self.PINK_V_MIN, self.PINK_V_MAX = 0.3, 1.0
        
        # Brown staining (tumor cells in IDC)
        self.BROWN_H_MIN, self.BROWN_H_MAX = 10, 40
        self.BROWN_S_MIN, self.BROWN_S_MAX = 0.2, 0.9
        self.BROWN_V_MIN, self.BROWN_V_MAX = 0.2, 0.9
        
    def validate(self, pil_img):
        """
        Validate if image is histopathology
        
        Args:
            pil_img: PIL Image object
            
        Returns:
            (is_valid, message_type, details)
            - is_valid: True if valid histopathology, False otherwise
            - message_type: 'valid', 'warning', or 'rejected'
            - details: dict with detection results
        """
        
        # Convert PIL to numpy array
        img_array = np.array(pil_img.convert("RGB"))
        
        # Check 1: Image size (minimum 96x96 as per model requirement)
        height, width = img_array.shape[:2]
        if height < 96 or width < 96:
            return False, "rejected", {
                "size": f"{width}x{height}",
                "reason": f"Image too small. Minimum: 96x96, Got: {width}x{height}"
            }
        
        # Check 2: Convert to HSV for color analysis
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = img_hsv[:, :, 0] / 180, img_hsv[:, :, 1] / 255, img_hsv[:, :, 2] / 255
        
        # Detect H&E staining colors
        purple_mask = self._detect_purple(h, s, v)
        pink_mask = self._detect_pink(h, s, v)
        brown_mask = self._detect_brown(h, s, v)
        
        purple_pct = (purple_mask.sum() / purple_mask.size) * 100
        pink_pct = (pink_mask.sum() / pink_mask.size) * 100
        brown_pct = (brown_mask.sum() / brown_mask.size) * 100
        tissue_pct = purple_pct + pink_pct + brown_pct
        
        # Check 3: Color distribution analysis
        histology_score = self._score_histopathology(purple_pct, pink_pct, brown_pct)
        
        # Check 4: Texture analysis (cell-like structures)
        texture_score = self._analyze_texture(img_array)
        
        # Check 5: Detect if it's a face or obvious non-medical image
        is_face = self._detect_possible_face(img_array)
        
        # Determine validation result
        if is_face:
            return False, "rejected", {
                "purple": f"{purple_pct:.1f}%",
                "pink": f"{pink_pct:.1f}%",
                "brown": f"{brown_pct:.1f}%",
                "tissue_score": f"{histology_score:.1f}%",
                "texture_score": f"{texture_score:.1f}%",
                "reason": "Image appears to be a photograph (face/person), not tissue"
            }
        
        if histology_score >= 60 and tissue_pct >= 15:
            # Strong histopathology signature
            return True, "valid", {
                "purple": f"{purple_pct:.1f}%",
                "pink": f"{pink_pct:.1f}%",
                "brown": f"{brown_pct:.1f}%",
                "tissue_score": f"{histology_score:.1f}%",
                "texture_score": f"{texture_score:.1f}%",
                "type": "Histopathology - Valid"
            }
        
        elif histology_score >= 40 and tissue_pct >= 10:
            # Moderate histopathology signature - warn but allow
            return True, "warning", {
                "purple": f"{purple_pct:.1f}%",
                "pink": f"{pink_pct:.1f}%",
                "brown": f"{brown_pct:.1f}%",
                "tissue_score": f"{histology_score:.1f}%",
                "texture_score": f"{texture_score:.1f}%",
                "type": "Possible histopathology - Low quality/staining"
            }
        
        elif tissue_pct >= 5:
            # Some tissue-like colors but weak signal - warning
            return True, "warning", {
                "purple": f"{purple_pct:.1f}%",
                "pink": f"{pink_pct:.1f}%",
                "brown": f"{brown_pct:.1f}%",
                "tissue_score": f"{histology_score:.1f}%",
                "texture_score": f"{texture_score:.1f}%",
                "type": "Unclear - Tissue colors weak"
            }
        
        else:
            # No tissue signature detected
            detected_type = self._classify_image_type(img_array)
            return False, "rejected", {
                "purple": f"{purple_pct:.1f}%",
                "pink": f"{pink_pct:.1f}%",
                "brown": f"{brown_pct:.1f}%",
                "tissue_score": f"{histology_score:.1f}%",
                "texture_score": f"{texture_score:.1f}%",
                "reason": f"Image appears to be: {detected_type}"
            }
    
    def _detect_purple(self, h, s, v):
        """Detect purple nuclei (Hematoxylin)"""
        return (
            ((h >= self.PURPLE_H_MIN / 180) | (h <= self.PURPLE_H_MAX / 180)) &
            (s >= self.PURPLE_S_MIN) & (s <= self.PURPLE_S_MAX) &
            (v >= self.PURPLE_V_MIN) & (v <= self.PURPLE_V_MAX)
        )
    
    def _detect_pink(self, h, s, v):
        """Detect pink cytoplasm (Eosin) - handles hue wraparound"""
        pink_H_unwrapped = (h >= (self.PINK_H_MIN / 180)) | (h <= (self.PINK_H_MAX / 180))
        return (
            pink_H_unwrapped &
            (s >= self.PINK_S_MIN) & (s <= self.PINK_S_MAX) &
            (v >= self.PINK_V_MIN) & (v <= self.PINK_V_MAX)
        )
    
    def _detect_brown(self, h, s, v):
        """Detect brown staining (malignant tumor cells)"""
        return (
            (h >= self.BROWN_H_MIN / 180) & (h <= self.BROWN_H_MAX / 180) &
            (s >= self.BROWN_S_MIN) & (s <= self.BROWN_S_MAX) &
            (v >= self.BROWN_V_MIN) & (v <= self.BROWN_V_MAX)
        )
    
    def _score_histopathology(self, purple_pct, pink_pct, brown_pct):
        """
        Score how likely this is histopathology
        Weighed by typical H&E staining distribution
        """
        # Purple nuclei: 30-50% typical
        purple_score = min(purple_pct / 40, 1.0) * 35
        
        # Pink cytoplasm: 10-30% typical
        pink_score = min(pink_pct / 25, 1.0) * 35
        
        # Brown staining: 0-20% (depends on IDC presence)
        brown_score = min(brown_pct / 15, 1.0) * 30
        
        return purple_score + pink_score + brown_score
    
    def _analyze_texture(self, img_array):
        """
        Analyze texture to detect cell-like structures
        High variance = more detailed tissue
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate local variance using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.var(laplacian)
        
        # Normalize: tissue typically has variance > 100
        # Random photos have different patterns
        texture_score = min(np.sqrt(variance) / 3.0, 100.0)
        return texture_score
    
    def _detect_possible_face(self, img_array):
        """
        Simple face detection using color distribution
        Faces have specific skin tone ranges
        """
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = img_hsv[:, :, 0] / 180, img_hsv[:, :, 1] / 255, img_hsv[:, :, 2] / 255
        
        # Skin tone ranges
        skin_tone = (
            (h >= 0.9) | (h <= 0.1) &  # Red-orange hues
            (s >= 0.1) & (s <= 0.6) &  # Medium saturation
            (v >= 0.3) & (v <= 0.9)    # Medium-high brightness
        )
        
        skin_pct = (skin_tone.sum() / skin_tone.size) * 100
        
        # If too much uniform skin-like color, likely a face
        return skin_pct > 60
    
    def _classify_image_type(self, img_array):
        """Classify what type of image this appears to be"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Check if too uniform (solid color/screenshot)
        variance = np.var(gray)
        if variance < 100:
            return "Solid color or screenshot"
        
        # Check if has natural scene characteristics
        # Natural images have specific color distributions
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h_dist = cv2.calcHist([img_hsv], [0], None, [180], [0, 180])
        
        # If blue/green heavy, likely natural scene
        blue_green = (h_dist[80:130].sum() / h_dist.sum()) > 0.4
        if blue_green:
            return "Natural scene or photograph"
        
        return "Non-medical image"


def validate_histopathology_image(pil_img):
    """
    Convenience function for app integration
    
    Returns:
    {
        'is_valid': bool,
        'is_warning': bool,
        'message': str (user-friendly message),
        'details': dict (technical details)
    }
    """
    validator = HistopathologyValidator()
    is_valid, msg_type, details = validator.validate(pil_img)
    
    if msg_type == "valid":
        return {
            "is_valid": True,
            "is_warning": False,
            "message": "✅ Valid histopathology image detected",
            "details": details
        }
    
    elif msg_type == "warning":
        return {
            "is_valid": True,
            "is_warning": True,
            "message": (
                f"⚠️ Image Validation Warning\n"
                f"Expected: Histopathology image with H&E staining\n"
                f"Issue: {details['type']}\n"
                f"Status: Proceeding with caution (lower confidence)\n"
                f"Note: Check that H&E staining is present (Purple nuclei: {details['purple']}, "
                f"Pink cytoplasm: {details['pink']})"
            ),
            "details": details
        }
    
    else:  # rejected
        return {
            "is_valid": False,
            "is_warning": False,
            "message": (
                f"⚠️ Image Validation Failed\n"
                f"Expected: Histopathology image with H&E staining\n"
                f"Issue: {details['reason']}\n"
                f"Please upload a valid H&E stained tissue sample image"
            ),
            "details": details
        }


if __name__ == "__main__":
    # Test the validator
    print("Image Validator Module")
    print("Use: validate_histopathology_image(pil_img)")
    print("Returns dict with: is_valid, is_warning, message, details")
