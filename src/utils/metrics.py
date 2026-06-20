import numpy as np
import time


def compute_rms(pred_phase, true_phase):
    
    diff  = np.array(pred_phase, dtype=np.float32) - np.array(true_phase, dtype=np.float32)
    valid = ~(np.isnan(diff))
    if valid.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.mean(diff[valid] ** 2)))


def compute_strehl(rms_nm, wavelength_nm=500.0):
    sigma_rad = (2 * np.pi * rms_nm) / wavelength_nm
    return float(np.exp(-sigma_rad ** 2))


def compute_r0_error(r0_pred, r0_true):
    return float(abs(r0_pred - r0_true) / r0_true * 100)


def compute_tau0_error(tau0_pred, tau0_true):
    return float(abs(tau0_pred - tau0_true) / tau0_true * 100)




if __name__ == '__main__':
    pred  = np.random.randn(128, 128).astype(np.float32) * 50
    true  = np.random.randn(128, 128).astype(np.float32) * 50
    rms   = compute_rms(pred, true)
    s     = compute_strehl(rms)
    print(f"Test RMS: {rms:.1f}nm | Strehl: {s:.3f}")
    r0_err = compute_r0_error(0.108, 0.100)
    print(f"r0 error: {r0_err:.1f}%")
    ms, std, fps, ok = benchmark_speed(np.linalg.pinv, np.random.randn(100,100))
    print(f"Benchmark: {ms:.2f}ms | {fps:.0f}fps | {'PASS' if ok else 'FAIL'}")
    print("metrics.py OK")
