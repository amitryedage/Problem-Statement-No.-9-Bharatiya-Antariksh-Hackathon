"""estimators.py — Was a stub. Now fully implemented with 5 functions:
estimate_r0() — Noll 1976 formula from tip/tilt Zernike variance
estimate_r0_sliding() — sliding window r₀(t) time-series
estimate_tau0() — slope temporal autocorrelation at 1/e
estimate_wind_speed() — from r₀ and τ₀
compute_tau0_from_formula() — theoretical τ₀ for validation"""
"""
PS09 BAH 2026 — Turbulence Parameter Estimators
Replicated from: notebooks/day3/day3_classical_baselines.py

PURPOSE:
  ISRO Outputs 3 and 4:
    Output 3: r0  (Fried parameter) — from Noll (1976) formula
    Output 4: tau0 (coherence time) — from slope autocorrelation

PHYSICS:
  r0   : Var(a_tip) = 0.449 × (D/r0)^(5/3) × (λ/2π)²
  tau0 : lag where slope autocorrelation drops to 1/e
  wind : v = 0.314 × r0 / tau0
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import D, WAVELENGTH, N_LENSLETS, FRAME_DT_MS


def estimate_r0(zernike_series, wavelength=WAVELENGTH, D_aperture=D):
    """
    Estimate Fried parameter r0 from temporal variance of Zernike modes.
    Uses Noll (1976) formula for tip (Z2) and tilt (Z3) modes.
    
    Noll formula: Var(a_j) = C_j × (D/r0)^(5/3) × (λ/2π)²
    C2 = C3 = 0.4490 (tip/tilt Noll coefficients)

    Why tip/tilt: largest variance, most accurately measured,
    Noll coefficients well-established in literature.

    Args:
        zernike_series : (T, N_modes) float32 — Zernike coefficients in nm
        wavelength     : float — wavelength in metres
        D_aperture     : float — pupil diameter in metres

    Returns:
        r0_metres : float — estimated Fried parameter in metres
    """
    if zernike_series.shape[0] < 5:
        raise ValueError("Need at least 5 frames for reliable r0 estimation")

    # Z2 (tip) and Z3 (tilt) are at 0-indexed positions 1 and 2
    tip_nm   = zernike_series[:, 1].astype(np.float64)
    tilt_nm  = zernike_series[:, 2].astype(np.float64)

    # Convert nm variance to rad² variance
    nm_to_rad = (2 * np.pi) / (wavelength * 1e9)
    var_tip_rad2  = float(np.var(tip_nm))  * nm_to_rad**2
    var_tilt_rad2 = float(np.var(tilt_nm)) * nm_to_rad**2
    mean_var_rad2 = (var_tip_rad2 + var_tilt_rad2) / 2.0

    if mean_var_rad2 < 1e-14:
        return float('nan')

    C_noll = 0.4490
    r0     = D_aperture / (mean_var_rad2 / C_noll) ** (3.0 / 5.0)

    return float(r0)


def estimate_r0_sliding(zernike_series, frame_times,
                         window_frames=10, wavelength=WAVELENGTH,
                         D_aperture=D):
    """
    Estimate r0 as a time-series using a sliding window.
    Produces r0(t) — how turbulence strength varies over time.

    Args:
        zernike_series : (T, N_modes) — Zernike time-series
        frame_times    : (T,) — timestamps in seconds
        window_frames  : frames per estimation window

    Returns:
        r0_series   : (T_windows,) — r0 estimates in metres
        t_midpoints : (T_windows,) — window centre timestamps
    """
    T         = len(zernike_series)
    r0_list   = []
    t_list    = []

    for start in range(0, T - window_frames + 1):
        window  = zernike_series[start:start + window_frames]
        r0_w    = estimate_r0(window, wavelength, D_aperture)
        r0_list.append(r0_w)
        t_mid   = float(frame_times[start + window_frames // 2])
        t_list.append(t_mid)

    return np.array(r0_list, dtype=np.float32), np.array(t_list, dtype=np.float32)


def estimate_tau0(slopes_series, dt_seconds, n_avg_subs=10):
    """
    Estimate coherence time tau0 from slope temporal autocorrelation.
    Algorithm:
      1. Average autocorrelation across central subapertures
      2. Find first lag where mean_autocorr drops below 1/e
      3. tau0 = that_lag × dt_seconds
    Physics: Taylor's frozen turbulence — atmosphere moves as rigid
    screen at wind speed v. tau0 = time for screen to move r0.
    Args:
        slopes_series : (T, 2*N_sub) float32 — slope time-series
        dt_seconds    : float — time between frames in seconds
        n_avg_subs    : int — number of subapertures to average over
    Returns:
        tau0_ms      : float — coherence time in milliseconds
        tau0_lag     : int   — frame lag at 1/e crossing
        autocorr     : (T,) — mean normalised autocorrelation
    """
    from scipy.signal import correlate

    T     = slopes_series.shape[0]
    N_sub = slopes_series.shape[1] // 2
    n_use = min(n_avg_subs, N_sub)
    centre= N_sub // 2
    subs  = list(range(centre - n_use//2, centre + n_use//2))

    autocorrs = []
    for idx in subs:
        s = slopes_series[:, idx].astype(np.float64)
        s = s - s.mean()
        if np.std(s) < 1e-14:
            continue
        ac      = correlate(s, s, mode='full')
        ac      = ac[len(ac)//2:]
        ac_norm = ac / (ac[0] + 1e-14)
        autocorrs.append(ac_norm)

    if len(autocorrs) == 0:
        return float('nan'), -1, np.array([])

    mean_ac   = np.mean(autocorrs, axis=0)
    threshold = 1.0 / np.e
    crossings = np.where(mean_ac < threshold)[0]
    tau0_lag  = int(crossings[0]) if len(crossings) > 0 else T - 1
    tau0_ms   = float(tau0_lag * dt_seconds * 1000)

    return tau0_ms, tau0_lag, mean_ac.astype(np.float32)


def estimate_wind_speed(r0_metres, tau0_ms):
    """
    Estimate wind speed from r0 and tau0.
    Roddier (1981): tau0 = 0.314 × r0 / v_wind
    Args:
        r0_metres : float — Fried parameter in metres
        tau0_ms   : float — coherence time in milliseconds
    Returns:
        v_wind_ms : float — wind speed in m/s
    """
    tau0_s = tau0_ms / 1000.0
    if tau0_s < 1e-10:
        return float('nan')
    return float(0.314 * r0_metres / tau0_s)


def compute_tau0_from_formula(r0_metres, wind_speed_ms):
    """
    Theoretical tau0 from r0 and wind speed.
    Used to validate estimate_tau0() results.

    Returns:
        tau0_ms : float — theoretical coherence time in ms
    """
    return float(0.314 * r0_metres / wind_speed_ms * 1000)


if __name__ == '__main__':
    print("Testing estimators.py...")
    # Synthetic Zernike series with known statistics
    np.random.seed(42)
    T, n_modes = 100, 36
    r0_true    = 0.10   # 10cm
    D_ap       = 2.0
    wl         = 500e-9

    # Generate synthetic tip/tilt variance matching Noll formula
    C_noll    = 0.4490
    expected_var_rad2 = C_noll * (D_ap / r0_true) ** (5/3)
    # Convert to nm variance
    rad_to_nm = wl * 1e9 / (2 * np.pi)
    expected_var_nm2 = expected_var_rad2 * rad_to_nm**2

    coeffs = np.zeros((T, n_modes), dtype=np.float32)
    # Set tip/tilt variance to match expected
    sigma_nm = float(np.sqrt(expected_var_nm2))
    coeffs[:, 1] = np.random.randn(T) * sigma_nm
    coeffs[:, 2] = np.random.randn(T) * sigma_nm

    r0_est = estimate_r0(coeffs, wl, D_ap)
    err    = abs(r0_est - r0_true) / r0_true * 100
    print(f"  r0 true={r0_true*100:.0f}cm | estimated={r0_est*100:.1f}cm | error={err:.1f}%")
    print(f"  {'OK' if err < 20 else 'NOT OK'} r0 estimation")

    # Test tau0
    slopes_ts = np.random.randn(50, 200).astype(np.float32) * 0.001
    tau0_ms, lag, ac = estimate_tau0(slopes_ts, dt_seconds=0.002)
    print(f"  tau0 estimated: {tau0_ms:.2f}ms (lag={lag})")
    print(" estimators.py OK")
