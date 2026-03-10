"""
batch_processing.py — Batch Image Diagnosis

Handles:
- Multiple image uploads (2-10 images)
- Individual predictions for each image
- Majority voting for final diagnosis
- Average confidence score calculation
- Batch result aggregation
"""

import numpy as np
from typing import List, Dict, Tuple
from PIL import Image
from src.gradcam_utils import (load_and_preprocess_image,
                                get_gradcam_heatmap,
                                create_gradcam_figure,
                                create_improved_gradcam_heatmap)


class BatchAnalyzer:
    """
    Analyzes multiple histopathology images and provides batch diagnosis.
    """
    
    def __init__(self, model, img_size=(96, 96), threshold=0.50):
        """
        Initialize batch analyzer.
        
        Args:
            model: Loaded TensorFlow model
            img_size: Tuple of (height, width) for model input
            threshold: Classification threshold (default 0.50)
        """
        self.model = model
        self.img_size = img_size
        self.threshold = threshold
    
    def analyze_single_image(self, uploaded_file, file_name: str) -> Dict:
        """
        Analyze a single image and return prediction + heatmap.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            file_name: Name of the file (for display)
        
        Returns:
            dict with keys:
                - file_name: str
                - pil_img: PIL Image
                - img_array: preprocessed numpy array
                - raw_pred: float (0-1, probability of malignant)
                - label: str ("MALIGNANT" or "BENIGN")
                - confidence: float (0-1)
                - heatmap: numpy array
                - figure: PIL Image (grad-cam visualization)
                - status: str ("success", "error")
                - error_msg: str (if error)
        """
        try:
            # Load and preprocess
            img_array, pil_img = load_and_preprocess_image(
                uploaded_file, self.img_size, self.model
            )
            
            # Make prediction
            raw_pred = float(self.model.predict(img_array, verbose=0)[0][0])
            
            # Classify based on threshold
            label = "MALIGNANT" if raw_pred > self.threshold else "BENIGN"
            confidence = raw_pred if label == "MALIGNANT" else 1.0 - raw_pred
            
            # Generate heatmap
            heatmap = get_gradcam_heatmap(img_array, self.model)
            
            # Create visualization
            uncertain = 0.40 < raw_pred < 0.60
            figure = create_gradcam_figure(
                heatmap, pil_img, label, confidence, uncertain, self.img_size
            )
            
            return {
                "file_name": file_name,
                "pil_img": pil_img,
                "img_array": img_array,
                "raw_pred": raw_pred,
                "label": label,
                "confidence": confidence,
                "heatmap": heatmap,
                "figure": figure,
                "uncertain": uncertain,
                "status": "success",
                "error_msg": None
            }
        
        except Exception as e:
            return {
                "file_name": file_name,
                "pil_img": None,
                "img_array": None,
                "raw_pred": None,
                "label": None,
                "confidence": None,
                "heatmap": None,
                "figure": None,
                "uncertain": None,
                "status": "error",
                "error_msg": str(e)
            }
    
    def analyze_batch(self, uploaded_files: List) -> Dict:
        """
        Analyze multiple images and compute batch statistics.
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
        
        Returns:
            dict with keys:
                - results: List[dict] - Individual image results
                - final_diagnosis: str ("MALIGNANT" or "BENIGN")
                - avg_confidence: float (0-1)
                - malignant_count: int
                - benign_count: int
                - uncertain_count: int
                - success_count: int
                - error_count: int
        """
        results = []
        malignant_count = 0
        benign_count = 0
        uncertain_count = 0
        success_count = 0
        error_count = 0
        confidences = []
        
        # Process each image
        for uploaded_file in uploaded_files:
            result = self.analyze_single_image(uploaded_file, uploaded_file.name)
            # Validate result is not None and is a dictionary
            if result is not None and isinstance(result, dict):
                results.append(result)
            else:
                # Create error result if analyze_single_image failed
                results.append({
                    "file_name": uploaded_file.name,
                    "pil_img": None,
                    "img_array": None,
                    "raw_pred": None,
                    "label": None,
                    "confidence": None,
                    "heatmap": None,
                    "figure": None,
                    "uncertain": None,
                    "status": "error",
                    "error_msg": "Failed to process image"
                })
                error_count += 1
                continue
            
            if result["status"] == "success":
                success_count += 1
                
                if result["uncertain"]:
                    uncertain_count += 1
                
                if result["label"] == "MALIGNANT":
                    malignant_count += 1
                else:
                    benign_count += 1
                
                confidences.append(result["confidence"])
            else:
                error_count += 1
        
        # Compute final diagnosis using majority voting
        if success_count == 0:
            final_diagnosis = "ERROR"
            avg_confidence = 0.0
        else:
            # Majority voting
            if malignant_count > benign_count:
                final_diagnosis = "MALIGNANT"
            elif benign_count > malignant_count:
                final_diagnosis = "BENIGN"
            else:
                # Equal split - use average probability
                avg_prob = np.mean([r["raw_pred"] for r in results 
                                   if r["status"] == "success"])
                final_diagnosis = "MALIGNANT" if avg_prob > 0.5 else "BENIGN"
            
            # Average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Calculate percentages
        total_diagnosed = malignant_count + benign_count
        malignant_pct = (malignant_count / total_diagnosed * 100) if total_diagnosed > 0 else 0
        benign_pct = (benign_count / total_diagnosed * 100) if total_diagnosed > 0 else 0
        
        return {
            "results": results,
            "final_diagnosis": final_diagnosis,
            "avg_confidence": avg_confidence,
            "malignant_count": malignant_count,
            "benign_count": benign_count,
            "malignant_pct": malignant_pct,
            "benign_pct": benign_pct,
            "uncertain_count": uncertain_count,
            "success_count": success_count,
            "error_count": error_count,
            "total_count": len(uploaded_files)
        }
    
    def get_batch_summary_stats(self, batch_result: Dict) -> Dict:
        """
        Extract summary statistics from batch result.
        
        Args:
            batch_result: Output from analyze_batch()
        
        Returns:
            dict with summary statistics
        """
        return {
            "final_diagnosis": batch_result["final_diagnosis"],
            "avg_confidence": batch_result["avg_confidence"],
            "total_images": batch_result["total_count"],
            "successful": batch_result["success_count"],
            "failed": batch_result["error_count"],
            "malignant": batch_result["malignant_count"],
            "benign": batch_result["benign_count"],
            "borderline": batch_result["uncertain_count"],
            "malignant_pct": (batch_result["malignant_count"] / batch_result["success_count"] * 100 
                             if batch_result["success_count"] > 0 else 0),
            "benign_pct": (batch_result["benign_count"] / batch_result["success_count"] * 100 
                          if batch_result["success_count"] > 0 else 0)
        }


def validate_batch_files(uploaded_files: List) -> Tuple[bool, str]:
    """
    Validate batch file uploads.
    
    Args:
        uploaded_files: List of uploaded files
    
    Returns:
        (is_valid, message) tuple
    """
    if not uploaded_files:
        return False, "No images uploaded"
    
    if len(uploaded_files) < 2:
        return False, "At least 2 images required for batch analysis"
    
    if len(uploaded_files) > 10:
        return False, "Maximum 10 images allowed per batch"
    
    return True, "Batch files valid"
