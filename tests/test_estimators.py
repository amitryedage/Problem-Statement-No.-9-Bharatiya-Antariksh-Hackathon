"""Unit tests for src/reconstruction/estimators.py"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.turbulence.estimators import (estimate_r0, estimate_tau0,
                                        estimate_wind_speed,
                                        compute_tau0_from_formula)
from config import WAVELENGTH, D
def make_synthetic_zernike_series(r0_true, T=200, n_modes=36,
                                   wavelength=WAVELENGTH, D_ap=D):
    """Generate Zernike series with known r0 statistics."""
    C_noll   = 0.4490
    var_rad2 = C_noll * (D_ap / r0_true) ** (5/3)
    rad_to_nm= wavelength * 1e9 / (2 * np.pi)
    sigma_nm = float(np.sqrt(var_rad2)) * rad_to_nm
    coeffs   = np.zeros((T, n_modes), dtype=np.float32)
    np.random.seed(42)
    coeffs[:, 1] = np.random.randn(T) * sigma_nm   # tip
    coeffs[:, 2] = np.random.randn(T) * sigma_nm   # tilt
    return coeffs

def test_r0_estimation_accuracy():
    """r0 estimate should be within 20% of true value"""
    for r0_true in [0.05, 0.10, 0.15, 0.20]:
        coeffs    = make_synthetic_zernike_series(r0_true, T=500)
        r0_est    = estimate_r0(coeffs, WAVELENGTH, D)
        err_pct   = abs(r0_est - r0_true) / r0_true * 100
        assert err_pct < 25, f"r0={r0_true*100:.0f}cm: error {err_pct:.1f}% > 25%"
    print(" test_r0_estimation_accuracy")

def test_r0_ordering():
    """Higher turbulence (lower r0) must give lower estimated r0"""
    r0_estimates = []
    for r0_true in [0.20, 0.10, 0.05]:
        coeffs = make_synthetic_zernike_series(r0_true, T=300)
        r0_estimates.append(estimate_r0(coeffs, WAVELENGTH, D))
    assert r0_estimates[0] > r0_estimates[1] > r0_estimates[2], \
        f"r0 ordering wrong: {[r*100 for r in r0_estimates]}"
    print("test_r0_ordering")

def test_tau0_returns_positive():
    slopes = np.random.randn(50, 200).astype(np.float32) * 0.001
    tau0, lag, ac = estimate_tau0(slopes, dt_seconds=0.002)
    assert tau0 >= 0, f"tau0 should be positive, got {tau0}"
    assert lag  >= 0, f"lag should be non-negative, got {lag}"
    print("test_tau0_returns_positive")

def test_wind_speed_formula():
    r0, tau0_ms = 0.10, 3.14
    v = estimate_wind_speed(r0, tau0_ms)
    v_expected = 0.314 * r0 / (tau0_ms/1000)
    assert abs(v - v_expected) < 1e-6
    print("test_wind_speed_formula")

def test_tau0_formula_roundtrip():
    r0, v = 0.10, 10.0
    tau0  = compute_tau0_from_formula(r0, v)
    v_back= estimate_wind_speed(r0, tau0)
    assert abs(v_back - v) < 0.01, f"Expected {v}, got {v_back}"
    print("  test_tau0_formula_roundtrip")

if __name__ == '__main__':
    print("Running turbulence estimator tests...")
    test_r0_estimation_accuracy()
    test_r0_ordering()
    test_tau0_returns_positive()
    test_wind_speed_formula()
    test_tau0_formula_roundtrip()
    print("All tests passed!")
