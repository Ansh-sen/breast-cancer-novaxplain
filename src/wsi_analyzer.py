import cv2
import numpy as np
from PIL import Image
import io

from src.gradcam_utils import load_and_preprocess_image

def split_into_patches(image_pil, patch_size=224, stride=224):
    """
    Extracts patches from a PIL image using a sliding window.
    Resizes large images to prevent excessive processing time.
    """
    # Performance Safety: Limit max dimension
    max_dim = 2000
    width, height = image_pil.size
    
    if width > max_dim or height > max_dim:
        scale = max_dim / max(width, height)
        new_w, new_h = int(width * scale), int(height * scale)
        image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        width, height = image_pil.size
        print(f"[WSI Analyzer] Image resized to {width}x{height} for performance safety.")

    img_array = np.array(image_pil)
    patches = []
    
    # Slide window
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch_array = img_array[y:y+patch_size, x:x+patch_size]
            patch_pil = Image.fromarray(patch_array)
            patches.append({
                "patch": patch_pil,
                "x": x,
                "y": y
            })
            
    # Include edge cases (if image isn't perfectly divisible by stride)
    # Could add logic for remaining edges but typical WSI processing either pads or ignores edges
    # For simplicity and speed, we stick to full patches only.

    return patches, width, height


def analyze_patches(model, patches):
    """
    Runs CNN prediction on each patch.
    """
    results = []
    for p in patches:
        patch_pil = p["patch"]
        x = p["x"]
        y = p["y"]
        
        try:
            # Reusing the existing preprocessing function
            # Note: load_and_preprocess_image usually takes a file or BytesIO, 
            # so we'll convert the PIL image to a byte stream.
            img_byte_arr = io.BytesIO()
            patch_pil.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            img_array, _ = load_and_preprocess_image(img_byte_arr, model=model)
            
            # Predict using the loaded model
            if img_array is not None:
                prob = float(model.predict(img_array, verbose=0)[0][0])
            else:
                prob = 0.0
                
        except Exception as e:
            print(f"[WSI Analyzer] Error predicting patch ({x},{y}): {str(e)}")
            prob = 0.0
            
        results.append({
            "x": x,
            "y": y,
            "prob": prob
        })
        
    return results


def run_wsi_analysis(image_pil, model, patch_size=224, stride=224, tumor_threshold=0.7):
    """
    Orchestrates the WSI patch-based analysis.
    """
    print("[WSI Analyzer] Starting patch extraction...")
    patches, img_w, img_h = split_into_patches(image_pil, patch_size, stride)
    total_patches = len(patches)
    
    if total_patches == 0:
        print("[WSI Analyzer] Image too small for patch extraction.")
        return None
    
    print(f"[WSI Analyzer] Analyzing {total_patches} patches...")
    patch_results = analyze_patches(model, patches)
    
    # Tumor detection metrics
    tumor_patches = sum(1 for p in patch_results if p["prob"] > tumor_threshold)
    tumor_percentage = (tumor_patches / total_patches) * 100 if total_patches > 0 else 0
    
    # Generate Heatmap
    heatmap_pil = _generate_heatmap(img_w, img_h, patch_results, patch_size, stride, image_pil)
    
    # Tumor Region Grouping Detection
    tumor_regions, largest_region = _detect_tumor_regions(img_w, img_h, patch_results, patch_size, stride, tumor_threshold)
    
    return {
        "total_patches": total_patches,
        "tumor_patches": tumor_patches,
        "tumor_percentage": round(tumor_percentage, 2),
        "tumor_regions": tumor_regions,
        "largest_region": largest_region,
        "heatmap_image": heatmap_pil
    }


def _generate_heatmap(img_w, img_h, patch_results, patch_size, stride, original_pil):
    """
    Creates a tumor probability heatmap overlay.
    Color mapping: 0.0->blue, 0.5->yellow, 1.0->red
    """
    # Create empty probability map (1 channel)
    prob_map = np.zeros((img_h, img_w), dtype=np.float32)
    # Keep track of overlap counts if stride < patch_size
    counts = np.zeros((img_h, img_w), dtype=np.float32)
    
    for p in patch_results:
        x, y, prob = p["x"], p["y"], p["prob"]
        prob_map[y:y+patch_size, x:x+patch_size] += prob
        counts[y:y+patch_size, x:x+patch_size] += 1
        
    # Average probabilities where patches overlap
    counts[counts == 0] = 1 # Avoid division by zero
    prob_map = prob_map / counts
    
    # Convert probability (0.0 - 1.0) to uint8 (0 - 255)
    hm_uint8 = np.uint8(255 * prob_map)
    
    # Apply JET colormap (0=blue, 127=green/yellow, 255=red)
    heatmap_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    # Note: original image might have been resized inside split_into_patches
    # So we resize original_pil to match heatmap dimensions just in case.
    orig_resized = np.array(original_pil.resize((img_w, img_h), Image.Resampling.LANCZOS).convert("RGB"))
    
    overlay = cv2.addWeighted(
        orig_resized.astype(np.float32), 0.6,
        heatmap_color.astype(np.float32), 0.4,
        0
    ).astype(np.uint8)
    
    return Image.fromarray(overlay)


def _detect_tumor_regions(img_w, img_h, patch_results, patch_size, stride, threshold):
    """
    Groups neighboring suspicious patches using connected components to detect distinct tumor regions.
    """
    # Create a simplified binary grid representing patches
    grid_w = (img_w - patch_size) // stride + 1
    grid_h = (img_h - patch_size) // stride + 1
    
    if grid_w <= 0 or grid_h <= 0:
        return 0, 0
        
    binary_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    
    for p in patch_results:
        if p["prob"] > threshold:
            col = p["x"] // stride
            row = p["y"] // stride
            if row < grid_h and col < grid_w:
                binary_grid[row, col] = 255
                
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_grid, connectivity=8)
    
    # num_labels includes background (0), so actual regions is num_labels - 1
    tumor_regions = max(0, num_labels - 1)
    
    largest_region = 0
    if tumor_regions > 0:
        # stats matrix columns: [x, y, w, h, area]
        # label 0 is background, so we slice from 1
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) > 0:
            largest_region = int(np.max(areas))
            
    return tumor_regions, largest_region
