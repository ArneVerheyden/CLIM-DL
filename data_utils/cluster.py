from skimage import measure, morphology
import matplotlib.colors as mcolors

import numpy as np
import torch

from skimage.measure import block_reduce
from scipy import stats

def gen_random_cmap(input):
    num_clusters = len(np.unique(input.flatten()))
    random_colors = np.random.rand(num_clusters, 4)
    random_colors[0, :] = 0
    random_colors[1:, 3] = 1

    return mcolors.ListedColormap(random_colors)

def find_largest_cluster(clustered_image):
    # Get unique cluster values and their counts
    unique_values, counts = np.unique(clustered_image[clustered_image > 0], return_counts=True)
    
    # Find the index of the maximum count
    largest_cluster_idx = np.argmax(counts)
    
    # Get the value of the largest cluster
    largest_cluster_value = unique_values[largest_cluster_idx]
    
    # Get the size of the largest cluster
    largest_cluster_size = counts[largest_cluster_idx]
    
    # Create a mask for the largest cluster
    largest_cluster_mask = (clustered_image == largest_cluster_value)
    
    return largest_cluster_value, largest_cluster_size, largest_cluster_mask

def border_to_clusters(border_image, threshold=0.5):
    ## Padding to remove large erroneous clusters in the background
    pad_width = 3
    border_image = np.pad(border_image, pad_width=pad_width, mode='constant', constant_values=0)

    # Convert PyTorch tensor to NumPy if needed
    if isinstance(border_image, torch.Tensor):
        border_image = border_image.detach().cpu().numpy()
    
    # Invert the image since borders are 1 and inside is 0
    inverted = 1 - border_image
    
    # Apply binary threshold to separate regions
    binary = inverted > threshold
    
    # Make sure it's a binary NumPy array with bool dtype
    binary = np.array(binary, dtype=bool)
    
    # Optional: Clean up small holes and artifacts
    cleaned = morphology.remove_small_holes(binary)
    # cleaned = morphology.remove_small_objects(cleaned, min_size=5)

    # Label connected components (clusters)
    labeled_clusters = measure.label(cleaned)

    ## The largest cluster usually represents the entire background which will skew the silhoutte score 
    ## Because they are not really clusters
    _, _, mask = find_largest_cluster(labeled_clusters)
    labeled_clusters[mask] = 0

    # Return clusters with the padding removed
    return labeled_clusters[pad_width:-pad_width, pad_width:-pad_width]

def downscale_cluster_map(cluster_map, factor=2):
    """
    Downscale a cluster map by taking the most frequent value in each block.
    
    Parameters:
    -----------
    cluster_map : numpy.ndarray
        The input cluster map where each value is a cluster ID
    factor : int
        The downscaling factor (default: 2)
        
    Returns:
    --------
    numpy.ndarray
        The downscaled cluster map
    """
    h, w = cluster_map.shape
    new_h, new_w = h // factor, w // factor
    downscaled = np.zeros((new_h, new_w), dtype=cluster_map.dtype)
    
    for i in range(new_h):
        for j in range(new_w):
            # Extract the block
            block = cluster_map[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            # Find the most common value
            values, counts = np.unique(block, return_counts=True)
            downscaled[i, j] = values[np.argmax(counts)]
    
    return downscaled