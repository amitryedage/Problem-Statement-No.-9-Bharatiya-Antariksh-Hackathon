"""Unit tests for src/utils/metrics.py"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.metrics import compute_rms, compute_strehl, compute_r0_error

def test_rms_perfect():
    p = np.random.randn(128, 128).astype(np.float32)
    assert compute_rms(p, p) < 1e-6
    print("  ✅ test_rms_perfect")

def test_strehl_at_zero_rms():
    assert abs(compute_strehl(0.0) - 1.0) < 1e-10
    print("  ✅ test_strehl_at_zero_rms")

def test_strehl_range():
    for rms in [0, 50, 100, 200, 500]:
        s = compute_strehl(rms)
        assert 0 <= s <= 1.0
    print("  ✅ test_strehl_range")

def test_r0_error():
    err = compute_r0_error(r0_pred=0.110, r0_true=0.100)
    assert abs(err - 10.0) < 0.1
    print("  ✅ test_r0_error")

if __name__ == '__main__':
    print("Running metrics tests...")
    test_rms_perfect()
    test_strehl_at_zero_rms()
    test_strehl_range()
    test_r0_error()
    print("All tests passed!")
