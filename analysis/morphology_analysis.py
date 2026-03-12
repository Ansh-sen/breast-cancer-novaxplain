"""
Cell Morphology Analysis Module

Analyzes microscopic cell structures to provide biological evidence supporting AI predictions.
Detects cell nuclei, calculates morphological features, and identifies abnormal patterns.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Any, List


def analyze_cell_morphology(image) -> Dict[str, Any]:
    """
    Analyzes cell morphology features from a histopathology image.
    
    Detects cell nuclei, calculates density and irregularity metrics,
    identifies clusters, and returns annotated visualization.
    
    Args:
        image: PIL Image or numpy array of the histopathology image
        
    Returns:
        dict containing:
            - cell_count: int, number of detected nuclei
            - cell_density: float, cells per pixel area
            - irregular_nuclei_ratio: float, proportion of irregular cells
            - cluster_count: int, number of detected clusters
            - largest_cluster: int, size of largest cluster
            - suspicion_level: str, "Low" | "Moderate" | "High" | "Unknown"
            - annotated_image: PIL Image with detected contours and clusters
    """
    
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Ensure we have a valid image
        if image_np is None or image_np.size == 0:
            return _get_safe_defaults(None)
        
        # Step 1: Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 2.5: Calculate advanced morphology features
        # Texture irregularity using Laplacian variance (edge intensity/blurriness metric)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_irregularity = float(round(laplacian_var, 2))
        
        # Tissue uniformity using standard deviation of pixel intensities
        # (Inverse relationship: higher std -> lower uniformity)
        std_intensity = np.std(gray)
        tissue_uniformity = float(round(max(0, 100.0 - std_intensity), 2)) # 0 to 100 roughly

        
        # Step 3: Apply adaptive thresholding to highlight nuclei
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
        # Step 4: Detect contours representing cell nuclei
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return _get_safe_defaults(image_np)
        
        # Step 5: Filter small noise contours and calculate morphology
        valid_contours = []
        circularity_values = []
        contour_areas = []
        centroids = []
        
        min_area = 10  # Minimum nucleus size in pixels
        max_area = 5000  # Maximum nucleus size in pixels
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size
            if area < min_area or area > max_area:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            
            # Handle degenerate cases
            if perimeter == 0:
                continue
            
            # Calculate circularity (1.0 = perfect circle)
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            valid_contours.append(contour)
            circularity_values.append(circularity)
            contour_areas.append(area)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
            else:
                centroids.append((0, 0))
        
        # Return safe defaults if no valid contours found
        if not valid_contours:
            return _get_safe_defaults(image_np)
        
        # Compute metrics
        cell_count = len(valid_contours)
        image_area = image_np.shape[0] * image_np.shape[1]
        cell_density = cell_count / max(image_area, 1)
        
        # Count irregular nuclei
        irregular_count = sum(1 for circ in circularity_values if circ < 0.7)
        irregular_nuclei_ratio = irregular_count / max(cell_count, 1)
        
        # Detect clusters using centroid distances
        cluster_count, largest_cluster = _detect_clusters(centroids)
        
        # Determine suspicion level
        suspicion_level = _calculate_suspicion_level(
            cell_density,
            irregular_nuclei_ratio,
            image_area
        )
        
        # Create annotated image
        annotated_image = _create_annotated_image(
            image_np,
            valid_contours,
            centroids,
            cluster_count,
            circularity_values
        )
        
        # Nucleus density is cell density scaled conventionally
        nucleus_density = round(cell_density * 1e6, 2)
        
        return {
            "cell_count": cell_count,
            "cell_density": nucleus_density,
            "irregular_nuclei_ratio": round(irregular_nuclei_ratio, 3),
            "cluster_count": cluster_count,
            "largest_cluster": largest_cluster,
            "suspicion_level": suspicion_level,
            "annotated_image": annotated_image,
            "texture_irregularity": texture_irregularity,
            "nucleus_density": nucleus_density,
            "tissue_uniformity": tissue_uniformity
        }
        
    except Exception as e:
        print(f"Error in morphology analysis: {str(e)}")
        return _get_safe_defaults(None)


def _detect_clusters(centroids: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Detects cell clusters by measuring distances between centroids.
    
    Args:
        centroids: list of (x, y) centroid positions
        
    Returns:
        tuple of (cluster_count, largest_cluster_size)
    """
    if not centroids or len(centroids) < 2:
        return 0, len(centroids) if centroids else 0
    
    try:
        # Distance threshold for considering cells as neighbors (pixels)
        cluster_distance_threshold = 40
        
        visited = set()
        clusters = []
        
        def dfs_cluster(idx: int, cluster: List[int]):
            """Depth-first search to find connected cell clusters."""
            if idx in visited:
                return
            visited.add(idx)
            cluster.append(idx)
            
            cx, cy = centroids[idx]
            for j, (nx, ny) in enumerate(centroids):
                if j in visited:
                    continue
                distance = np.sqrt((cx - nx) ** 2 + (cy - ny) ** 2)
                if distance < cluster_distance_threshold:
                    dfs_cluster(j, cluster)
        
        # Find all clusters
        for i in range(len(centroids)):
            if i not in visited:
                cluster = []
                dfs_cluster(i, cluster)
                if len(cluster) >= 2:  # Only count as cluster if 2+ cells
                    clusters.append(cluster)
        
        cluster_count = len(clusters)
        largest_cluster = max(len(c) for c in clusters) if clusters else 0
        
        return cluster_count, largest_cluster
        
    except Exception as e:
        print(f"Error detecting clusters: {str(e)}")
        return 0, 0


def _calculate_suspicion_level(
    cell_density: float,
    irregular_nuclei_ratio: float,
    image_area: int
) -> str:
    """
    Determines suspicion level based on morphological features.
    
    Args:
        cell_density: cells per pixel
        irregular_nuclei_ratio: proportion of irregular nuclei
        image_area: total image area in pixels
        
    Returns:
        str: "Low", "Moderate", "High", or "Unknown"
    """
    try:
        # Scale density thresholds based on image area
        # Larger images naturally have lower average density
        density_threshold_high = 200e-6    # High density threshold
        density_threshold_moderate = 100e-6  # Moderate density threshold
        
        # Determine density level
        high_density = cell_density > density_threshold_high
        moderate_density = density_threshold_moderate <= cell_density <= density_threshold_high
        
        # Apply suspicion rules
        if high_density and irregular_nuclei_ratio > 0.4:
            return "High"
        elif moderate_density or (0.2 <= irregular_nuclei_ratio <= 0.4):
            return "Moderate"
        elif irregular_nuclei_ratio < 0.2:
            return "Low"
        else:
            return "Moderate"
    
    except Exception:
        return "Unknown"


def _create_annotated_image(
    image_np: np.ndarray,
    contours: List,
    centroids: List[Tuple[int, int]],
    cluster_count: int,
    circularity_values: List[float]
) -> Image.Image:
    """
    Creates an annotated image showing detected nuclei and clusters.
    
    Args:
        image_np: original image as numpy array
        contours: list of detected contours
        centroids: list of centroid coordinates
        cluster_count: number of detected clusters
        circularity_values: circularity values for each contour
        
    Returns:
        PIL Image with annotations
    """
    try:
        # Create a copy for annotation
        annotated = image_np.copy()
        
        if len(annotated.shape) == 2:
            # Convert grayscale to RGB for colored annotations
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2RGB)
        
        # Draw contours with color coding based on regularity
        for contour, circularity, centroid in zip(contours, circularity_values, centroids):
            # Color: green for regular, red for irregular
            if circularity >= 0.7:
                color = (0, 255, 0)  # Green - regular
            else:
                color = (255, 0, 0)  # Red - irregular
            
            cv2.drawContours(annotated, [contour], 0, color, 1)
            
            # Draw centroid
            cv2.circle(annotated, centroid, 3, color, -1)
        
        # Convert back to PIL Image
        if len(annotated.shape) == 3 and annotated.shape[2] == 3:
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        else:
            annotated_pil = Image.fromarray(annotated)
        
        return annotated_pil
        
    except Exception as e:
        print(f"Error creating annotated image: {str(e)}")
        # Return original image as fallback
        if isinstance(image_np, np.ndarray):
            return Image.fromarray(image_np)
        return Image.new('RGB', (100, 100))


def _get_safe_defaults(image_np: Any) -> Dict[str, Any]:
    """
    Returns safe default values when analysis fails.
    
    Args:
        image_np: original image array (used as fallback for annotated image)
        
    Returns:
        dict with safe default values
    """
    # Create blank fallback image
    if image_np is not None and isinstance(image_np, np.ndarray):
        fallback_image = Image.fromarray(image_np)
    else:
        # Create a blank image as ultimate fallback
        fallback_image = Image.new('RGB', (200, 200), color=(200, 200, 200))
    
    return {
        "cell_count": 0,
        "cell_density": 0.0,
        "irregular_nuclei_ratio": 0.0,
        "cluster_count": 0,
        "largest_cluster": 0,
        "suspicion_level": "Unknown",
        "annotated_image": fallback_image,
        "texture_irregularity": 0.0,
        "nucleus_density": 0.0,
        "tissue_uniformity": 0.0
    }
