import cv2
import numpy as np
from PIL import Image

def detect_tumor_region(original_pil_img, heatmap, threshold=0.5, min_area=50, prediction_prob=0.0):
    """
    Detect tumor region from Grad-CAM heatmap using contour detection.
    Enhanced to show original + heatmap + bounding box + probability.
    
    Args:
        original_pil_img: PIL Image of original specimen
        heatmap: Raw heatmap from Grad-CAM (0-1 normalized float32)
        threshold: Threshold for binarization (0.0-1.0, default 0.5)
        min_area: Minimum area to consider as tumor (default 50 pixels)
        prediction_prob: float representing CNN predicted malignancy probability
    
    Returns:
        dict with images and bounding box details
    """
    try:
        orig = np.array(original_pil_img.convert("RGB"))
        orig_h, orig_w = orig.shape[:2]
        
        # Normalize heatmap to 0-1 range
        if heatmap.max() > 1e-8:
            hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            hm_norm = heatmap
            
        hm_resized = cv2.resize(hm_norm, (orig_w, orig_h))
        hm_uint8 = np.uint8(255 * np.maximum(hm_resized, 0))
        
        # Apply threshold to create binary mask
        _, binary_mask = cv2.threshold(hm_uint8, int(255 * threshold), 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tumor_bbox = None
        largest_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area and area > max_area:
                max_area = area
                largest_contour = contour
        
        # Create improved colored heatmap overlay (with thresholded transparency)
        hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
        
        # Hide low activation areas in heatmap mask
        mask_heatmap = (hm_uint8 < int(255 * 0.3))
        hm_colored[mask_heatmap] = [0, 0, 0]
        
        # Create a transparency mask based on heatmap non-black areas
        alpha_mask = np.any(hm_colored != [0,0,0], axis=-1).astype(np.float32)
        alpha_mask = np.expand_dims(alpha_mask, axis=-1)
        
        # Superimpose GradCAM on original image with 0.5 opacity for heatmap regions
        heatmap_overlay = orig.astype(np.float32) * (1 - alpha_mask * 0.5) + hm_colored.astype(np.float32) * (alpha_mask * 0.5)
        heatmap_overlay = np.clip(heatmap_overlay, 0, 255).astype(np.uint8)
        
        # Start drawing on top of the overlay image
        tumor_box_img = heatmap_overlay.copy()
        has_tumor = False
        
        if largest_contour is not None:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            tumor_bbox = (x, y, w, h)
            
            # Draw bounding box (red rectangle)
            cv2.rectangle(tumor_box_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
            # Display Probability near box
            prob_text = f"Prob: {prediction_prob:.2f}"
            
            # Add semi-transparent background for text visibility
            (text_w, text_h), _ = cv2.getTextSize(prob_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bg_rect_y = y - 10 if y - 10 > 20 else y + 20
            
            sub_img = tumor_box_img[bg_rect_y-text_h-5:bg_rect_y+5, x:x+text_w+10]
            if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
                black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
                tumor_box_img[bg_rect_y-text_h-5:bg_rect_y+5, x:x+text_w+10] = res
            
            cv2.putText(tumor_box_img, prob_text, (x + 5, bg_rect_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
            has_tumor = True
            
        heatmap_pil = Image.fromarray(heatmap_overlay)
        tumor_box_pil = Image.fromarray(tumor_box_img)
        
        return {
            "heatmap_image": heatmap_pil,
            "tumor_box_image": tumor_box_pil,
            "tumor_coordinates": tumor_bbox,
            "has_tumor": has_tumor,
            "largest_area": max_area,
            "binary_mask": binary_mask
        }
    except Exception as e:
        return {
            "heatmap_image": Image.fromarray(np.array(original_pil_img)),
            "tumor_box_image": Image.fromarray(np.array(original_pil_img)),
            "tumor_coordinates": None,
            "has_tumor": False,
            "largest_area": 0,
            "binary_mask": None,
            "error": str(e)
        }
