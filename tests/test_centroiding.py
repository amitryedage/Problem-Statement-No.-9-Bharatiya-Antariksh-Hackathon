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

def test_slopes_shape():
    frame = np.random.rand(128, 128).astype(np.float32)
    ref   = compute_reference_centroids(frame, N_LENSLETS)
    slopes= extract_slopes_from_frame(frame, ref, N_LENSLETS)
    assert slopes.shape == (N_SLOPES,), f"Expected ({N_SLOPES},), got {slopes.shape}"
    print("   test_slopes_shape")

def test_zero_slopes_same_frame():
    """Same frame as reference should give ~zero slopes"""
    frame  = np.random.rand(128, 128).astype(np.float32)
    ref    = compute_reference_centroids(frame, N_LENSLETS)
    slopes = extract_slopes_from_frame(frame, ref, N_LENSLETS)
    assert np.allclose(slopes, 0, atol=1e-10), "Slopes should be ~0 for reference frame"
    print("   test_zero_slopes_same_frame")

if __name__ == '__main__':
    print("Running centroiding tests...")
    test_subaperture_count()
    test_centroid_known_position()
    test_slopes_shape()
    test_zero_slopes_same_frame()
    print("All tests passed!")
