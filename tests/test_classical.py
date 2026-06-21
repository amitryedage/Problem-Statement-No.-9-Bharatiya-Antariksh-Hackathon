"""Unit tests for src/reconstruction/classical.py"""
"""Unit tests for src/reconstruction/classical.py"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.reconstruction.classical import (modal_reconstruct, zonal_reconstruct,
                                           direct_integrate, compute_slope_curl)
from config import N_LENSLETS, GRID_SIZE
# Create the dummy data to check the test cases 
def make_dummy_slopes(seed=42):
    np.random.seed(seed)
    return np.random.randn(N_LENSLETS**2 * 2).astype(np.float32) * 0.001

def test_modal_output_shape():
    slopes = make_dummy_slopes()
    D_pinv = np.random.randn(36, N_LENSLETS**2*2).astype(np.float32) * 0.01
    pm, c  = modal_reconstruct(slopes, D_pinv, grid_size=64, n_modes=36)
    assert pm.shape == (64, 64), f"Expected (64,64) got {pm.shape}"
    assert c.shape  == (36,),    f"Expected (36,) got {c.shape}"
    print("test_modal_output_shape")

def test_zonal_output_shape():
    slopes = make_dummy_slopes()
    pz     = zonal_reconstruct(slopes, N_LENSLETS)
    assert pz.shape == (N_LENSLETS, N_LENSLETS)
    print("test_zonal_output_shape")

def test_direct_output_shape():
    slopes = make_dummy_slopes()
    pd     = direct_integrate(slopes, N_LENSLETS, grid_size=64)
    assert pd.shape == (64, 64)
    print("test_direct_output_shape")

def test_zero_slopes_near_zero_phase():
    """Near-zero slopes should give near-zero phase"""
    slopes = np.zeros(N_LENSLETS**2 * 2, dtype=np.float32)
    pd     = direct_integrate(slopes, N_LENSLETS, grid_size=32)
    assert np.abs(pd).max() < 1e-3, f"Expected ~0 phase, got max={np.abs(pd).max()}"
    print("test_zero_slopes_near_zero_phase")




