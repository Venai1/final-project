'''
feature extractions that we perform on the data:

1. Quadrant intensity: We divide the image into four quadrants and sum intensity of each quadrant. This gives us a feature vector of size 4.
2. Horizontal and vertical symmetry: We check if the image is symmetric along the horizontal and vertical axes. This gives us a feature vector of size 2.
3: Projection features: We compute the sum of pixel intensities in the top and bottom halves of the image, and also compute the standard deviation of pixel intensities across rows and columns. This gives us a feature vector of size 4.
4. Edge transitions: We count the number of horizontal and vertical transitions in the image. This gives us a feature vector of size 2. 
'''

import numpy as np


def extract_features(X):
    """
    Extracts quadrant intensity and symmetry features from all images in X.
    
    Args:
        X (np.ndarray): shape (n_samples, 256)
    
    Returns:
        np.ndarray: shape (n_samples, 6)
    """
    feature_list = []
    for img in X:
        q_feat = quadrant_intensity(img)         # shape (4,)
        s_feat = symmetry_features(img)          # shape (2,)
        p_feat = projection_features(img)         # shape (4,)
        e_feat = edge_transitions(img)         # (2,)
        features = np.concatenate([q_feat, s_feat, p_feat, e_feat])  # (12,)
        feature_list.append(features)
    
    return np.array(feature_list)

def quadrant_intensity(image_flat):
    """
    Given a 1D image vector of length 256, return the sum of intensities in each quadrant.
    
    Args:
        image_flat (np.ndarray): shape (256,)
        
    Returns:
        features (np.ndarray): shape (4,) — [Q1, Q2, Q3, Q4]
    """
    image_2d = image_flat.reshape(16, 16)

    # Top-left (Q1)
    q1 = image_2d[:8, :8].sum()
    # Top-right (Q2)
    q2 = image_2d[:8, 8:].sum()
    # Bottom-left (Q3)
    q3 = image_2d[8:, :8].sum()
    # Bottom-right (Q4)
    q4 = image_2d[8:, 8:].sum()

    return np.array([q1, q2, q3, q4])


def symmetry_features(image_flat):
    """
    Compute horizontal and vertical symmetry of a 16x16 image.

    Args:
        image_flat (np.ndarray): shape (256,)
        
    Returns:
        np.ndarray: [horizontal_symmetry, vertical_symmetry]
    """
    image = image_flat.reshape(16, 16)

    # Horizontal Symmetry (left vs. right)
    left = image[:, :8]
    right = np.fliplr(image[:, 8:])
    h_diff = np.abs(left - right)
    h_symmetry = 1 - (np.sum(h_diff) / (np.sum(left + right) + 1e-5))  # avoid div by 0

    # Vertical Symmetry (top vs. bottom)
    top = image[:8, :]
    bottom = np.flipud(image[8:, :])
    v_diff = np.abs(top - bottom)
    v_symmetry = 1 - (np.sum(v_diff) / (np.sum(top + bottom) + 1e-5))

    return np.array([h_symmetry, v_symmetry])

def projection_features(image_flat):
    """
    Computes row/column projection features from a 16x16 image.
    
    Returns:
        np.ndarray: shape (4,) - [top_half_sum, bottom_half_sum, row_std, col_std]
    """
    image = image_flat.reshape(16, 16)

    # Row and column sums
    row_sums = image.sum(axis=1)  # shape (16,)
    col_sums = image.sum(axis=0)  # shape (16,)

    # Top vs bottom row sums
    top_half_sum = np.sum(row_sums[:8])
    bottom_half_sum = np.sum(row_sums[8:])

    # Spread of pixel mass
    row_std = np.std(row_sums)
    col_std = np.std(col_sums)

    return np.array([top_half_sum, bottom_half_sum, row_std, col_std])

def edge_transitions(image_flat):
    """
    Computes number of horizontal and vertical pixel transitions.
    """
    image = image_flat.reshape(16, 16)
    horiz = np.sum(np.abs(np.diff((image > 0).astype(int), axis=1)))
    vert = np.sum(np.abs(np.diff((image > 0).astype(int), axis=0)))
    return np.array([horiz, vert])
