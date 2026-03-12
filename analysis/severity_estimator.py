def estimate_severity(malignant_prob, tumor_area, heatmap_max_intensity):
    """
    Estimate severity level: LOW RISK, MEDIUM RISK, HIGH RISK
    Based on model confidence, tumor area size, and heatmap intensity.
    
    Args:
        malignant_prob (float): Model predicted malignancy probability (0.0 - 1.0)
        tumor_area (float): Pixel area of the detected tumor bounding box
        heatmap_max_intensity (float): Max value in the generated GradCAM heatmap
        
    Returns:
        str: "LOW RISK", "MEDIUM RISK", or "HIGH RISK"
    """
    # Base risk directly off high predicted malignancy probability
    if malignant_prob < 0.5:
        return "LOW RISK"
        
    # Start basic score (0 to 10)
    risk_score = malignant_prob * 10.0
    
    # Increase risk if large tumor area detected or high heatmap intensity
    if tumor_area > 1500 or heatmap_max_intensity > 0.8:
        risk_score += 2.0
        
    if risk_score > 8.5:
        return "HIGH RISK"
    elif risk_score >= 6.0:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"
