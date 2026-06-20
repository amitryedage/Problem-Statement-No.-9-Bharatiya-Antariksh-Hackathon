import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.centroiding import (extract_subapertures, centroid_com,
                                   centroid_all_patches,
                                   extract_slopes_from_frame,
                                   compute_reference_centroids)
from config import N_LENSLETS, N_SUBAPERTURES, N_SLOPES, GRID_SIZE

def test_subaperture_count():
    frame = np.random.rand(128, 128).astype(np.float32)
    patches, ph, pw, pos = extract_subapertures(frame, N_LENSLETS)
    assert len(patches) == N_SUBAPERTURES
    assert patches.shape == (N_SUBAPERTURES, ph, pw)
    print("  test_subaperture_count")

def test_centroid_known_position():
    """Centroid of a Gaussian blob at known position"""
    patch = np.zeros((16, 16), dtype=np.float32)
    patch[8, 10] = 100.0   # bright pixel at (x=10, y=8)
    cx, cy = centroid_com(patch)
    assert abs(cx - 10.0) < 0.5, f"Expected cx≈10, got {cx}"
    assert abs(cy - 8.0)  < 0.5, f"Expected cy≈8, got {cy}"
    print("   test_centroid_known_position")



if __name__ == '__main__':
    print("Running centroiding tests...")
    test_subaperture_count()
    test_centroid_known_position()
    test_slopes_shape()
    test_zero_slopes_same_frame()
    print("All tests passed!")
