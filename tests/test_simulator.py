"""Unit tests for src/data/simulator.py"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.simulator import simulate_wavefront_phase, generate_training_pair
from config import GRID_SIZE, N_SLOPES

def test_wavefront_shape():
    phase, _, _, _ = simulate_wavefront_phase(r0=0.10, seed=42)
    assert phase.shape == (GRID_SIZE, GRID_SIZE)
    print("  ✅ test_wavefront_shape")

def test_rms_increases_with_turbulence():
    rms = []
    for r0 in [0.20, 0.10, 0.05, 0.03]:
        phase, _, _, _ = simulate_wavefront_phase(r0=r0, seed=42)
        rms.append(float(np.nanstd(phase)))
    assert rms[0] < rms[1] < rms[2] < rms[3]
    print("  ✅ test_rms_increases_with_turbulence")

def test_training_pair_shapes():
    spot, slopes, phase, r0 = generate_training_pair(r0=0.10, seed=42)
    assert spot.dtype  == np.uint8
    assert slopes.shape == (N_SLOPES,)
    assert phase.shape  == (GRID_SIZE, GRID_SIZE)
    print("  ✅ test_training_pair_shapes")

def test_slopes_larger_for_stronger_turbulence():
    vars_ = []
    for r0 in [0.20, 0.10, 0.05]:
        _, slopes, _, _ = generate_training_pair(r0=r0, seed=42)
        vars_.append(float(np.var(slopes)))
    assert vars_[0] < vars_[1] < vars_[2]
    print("  ✅ test_slopes_larger_for_stronger_turbulence")

if __name__ == '__main__':
    print("Running simulator tests...")
    test_wavefront_shape()
    test_rms_increases_with_turbulence()
    test_training_pair_shapes()
    test_slopes_larger_for_stronger_turbulence()
    print("All tests passed!")
