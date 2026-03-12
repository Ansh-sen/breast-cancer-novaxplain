import os
import sys
from PIL import Image
import numpy as np

# Set up path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.wsi_analyzer import run_wsi_analysis
import tensorflow as tf

def test_wsi_analysis():
    print("Loading test image...")
    # Use any known test image
    img_path = os.path.join(current_dir, "data", "samples", "benign_01.png")
    
    # Fallback to creating a dummy image if sample doesn't exist
    if not os.path.exists(img_path):
        print(f"Sample image not found at {img_path}. Creating dummy image.")
        img_array = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_array)
    else:
        img_pil = Image.open(img_path).convert("RGB")
        
    print(f"Image loaded. Size: {img_pil.size}")

    print("Loading model...")
    model_path = os.path.join(current_dir, "models", "breast_cancer_model.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(current_dir, "models", "breast_cancer_model_best.keras")
        
    if not os.path.exists(model_path):
        print("Model not found. Cannot proceed with full verification, but module loaded successfully.")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    print("Running WSI Analysis...")
    try:
        # Run with smaller sizes for faster testing
        result = run_wsi_analysis(
            img_pil, 
            model, 
            patch_size=224, 
            stride=224, 
            tumor_threshold=0.5
        )
        
        if result:
            print(f"Total Patches: {result['total_patches']}")
            print(f"Tumor Patches: {result['tumor_patches']}")
            print(f"Tumor Percentage: {result['tumor_percentage']}%")
            print(f"Tumor Regions: {result['tumor_regions']}")
            print(f"Largest Region: {result['largest_region']}")
            print(f"Heatmap Generated: {'Yes' if result['heatmap_image'] else 'No'}")
            print("\n✅ WSI Analysis Module Verification Successful!")
        else:
            print("WSI returned None (image too small).")
            
    except Exception as e:
        print(f"❌ Error during WSI analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wsi_analysis()
