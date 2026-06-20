"""Unit tests for src/reconstruction/zernike.py"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.reconstruction.zernike import (get_zernike_basis,
                                         decompose_into_zernike,
                                         reconstruct_from_coefficients)
from config import N_ZERNIKE_MODES, GRID_SIZE

def test_basis_shape():
    Z = get_zernike_basis(N_ZERNIKE_MODES, 64)
    assert Z.shape == (N_ZERNIKE_MODES, 64, 64)
    print("   test_basis_shape")

def test_roundtrip():
    """Decompose then reconstruct — should recover most of signal"""
    import aotools
    Z    = aotools.zernikeArray(N_ZERNIKE_MODES, 64)
    # Build a simple test wavefront from known coefficients
    true_coeffs = np.zeros(N_ZERNIKE_MODES, dtype=np.float32)
    true_coeffs[1] = 50.0  # 50nm tip
    true_coeffs[3] = 30.0  # 30nm defocus
    phase = np.tensordot(true_coeffs, Z, axes=[[0],[0]]).astype(np.float32)
    est_coeffs = decompose_into_zernike(phase, N_ZERNIKE_MODES, 64)
    recon      = reconstruct_from_coefficients(est_coeffs, 64)
    rms        = float(np.sqrt(np.mean((phase-recon)**2)))
    assert rms < 1.0, f"Roundtrip RMS too high: {rms:.2f}nm"
    print(f"   test_roundtrip (RMS={rms:.3f}nm)")

def test_coeffs_shape():
    Z     = get_zernike_basis(N_ZERNIKE_MODES, GRID_SIZE)
    dummy = Z[1] * 50 + Z[3] * 20
    c     = decompose_into_zernike(dummy.astype(np.float32), N_ZERNIKE_MODES, GRID_SIZE)
    assert c.shape == (N_ZERNIKE_MODES,)
    print("   test_coeffs_shape")

if __name__ == '__main__':
    print("Running Zernike tests...")
    test_basis_shape()
    test_roundtrip()
    test_coeffs_shape()
    print("All tests passed!")
