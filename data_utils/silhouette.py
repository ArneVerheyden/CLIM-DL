from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_samples
import math
import numpy as np


## data: array with dimensions [width, height, frames]
## buffer_width, width of the buffer zone in pixels
def get_silhouette_scores(video_data: np.ndarray, labels: np.ndarray, window_factor: int, buffer_width: int) -> np.ndarray:
    """
    video_data: Numpy array with dimensions (width, height, frames)
    window_factor: int that the determines the size of the windowed region (e.g. a factor of 8 will make the window (width/8xheight/8))
    buffer_width: Width of pixels that are included around the windowed region, a larger value will make the end result more acurate but will take longer to compute
    """
    width = video_data.shape[0]
    height = video_data.shape[1]

    assert width == labels.shape[0]
    assert height == labels.shape[1]
    
    # Calculate window dimensions
    max_window_width = math.ceil(width/window_factor)
    max_window_height = math.ceil(height/window_factor)
    
    # Initialize result array
    result = np.zeros((width, height))
    
    # Process each window
    for i in range(0, width, max_window_width):
        # Calculate buffer boundaries for x dimension
        buffer_width_start = max(0, i - buffer_width)
        buffer_width_end = min(width, i + max_window_width + buffer_width)
        
        for j in range(0, height, max_window_height):
            # Calculate buffer boundaries for y dimension
            buffer_height_start = max(0, j - buffer_width)
            buffer_height_end = min(height, j + max_window_height + buffer_width)
            
            # Extract buffer data
            buffer_data = video_data[buffer_width_start:buffer_width_end, buffer_height_start:buffer_height_end, :]
            buffer_labels = labels[buffer_width_start:buffer_width_end, buffer_height_start:buffer_height_end]
            
            # Reshape for silhouette calculation
            buffer_shape = buffer_data.shape
            buffer_flat = buffer_data.reshape(-1, buffer_data.shape[2])
            buffer_labels_flat = buffer_labels.flatten()
            
            # If there are no 2 clusters in this part we skip it
            if len(np.unique(buffer_labels_flat)) < 2:
                continue

            # Calculate silhouette scores
            silhouette = silhouette_samples(buffer_flat, buffer_labels_flat, metric='correlation', n_jobs=10)
            
            # Reshape back to 2D
            silhouette_reshaped = silhouette.reshape(buffer_shape[0], buffer_shape[1])
            
            # Calculate ROI within the buffer
            roi_i_start = i - buffer_width_start  # Offset in buffer coordinates
            roi_j_start = j - buffer_height_start
            
            # Make sure we don't go out of bounds
            roi_i_end = min(roi_i_start + max_window_width, buffer_shape[0])
            roi_j_end = min(roi_j_start + max_window_height, buffer_shape[1])
            
            # Adjust ROI start positions to handle edge cases
            roi_i_start = max(0, roi_i_start)
            roi_j_start = max(0, roi_j_start)
            
            # Extract ROI silhouette scores
            roi_silhouette = silhouette_reshaped[roi_i_start:roi_i_end, roi_j_start:roi_j_end]
            
            # Calculate target region in the result array
            result_i_end = min(i + max_window_width, width)
            result_j_end = min(j + max_window_height, height)
            
            # Copy ROI to result
            result[i:result_i_end, j:result_j_end] = roi_silhouette
            
    return result
