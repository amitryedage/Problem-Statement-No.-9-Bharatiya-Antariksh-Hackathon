import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (D, WAVELENGTH, GRID_SIZE, N_LENSLETS,
                    MLA_FOCAL_LENGTH, L0, WIND_SPEED, N_SLOPES)


def simulate_wavefront_phase(r0, D=D, wavelength=WAVELENGTH,
                              grid_size=GRID_SIZE, wind_speed=WIND_SPEED,
                              outer_scale=L0, seed=None):
    try:
        import hcipy
    except ImportError:
        raise ImportError("Run: pip install hcipy")

    if seed is not None:
        np.random.seed(seed)

    pupil_grid  = hcipy.make_pupil_grid(grid_size, D)
    aperture    = hcipy.circular_aperture(D)(pupil_grid)
    Cn2         = hcipy.Cn_squared_from_fried_parameter(r0, wavelength)
    layer       = hcipy.InfiniteAtmosphericLayer(
                      pupil_grid, Cn2,
                      outer_scale,
                      [wind_speed, 0], 1)

    wf_flat  = hcipy.Wavefront(aperture, wavelength)
    wf_turb  = layer.forward(wf_flat)

    phase_rad = wf_turb.phase.shaped
    phase_nm  = (phase_rad * (wavelength * 1e9) / (2 * np.pi)).astype(np.float32)

    # Mask outside circular aperture
    mask = aperture.shaped > 0.5
    phase_nm[~mask] = np.nan

    return phase_nm, wf_turb, pupil_grid, aperture


def simulate_shwfs_frame(wf_turbulent, pupil_grid, aperture,
                          n_lenslets=N_LENSLETS,
                          mla_focal_length=MLA_FOCAL_LENGTH,
                          D=D, add_noise=False, noise_std=0.01):
    try:
        import hcipy
    except ImportError:
        raise ImportError("Run: pip install hcipy")

    shwfs_optics = hcipy.SquareShackHartmannWavefrontSensorOptics(
        pupil_grid.scaled(mla_focal_length / D),
        mla_focal_length, n_lenslets, mla_focal_length
    )
    estimator = hcipy.ShackHartmannWavefrontSensorEstimator(
        shwfs_optics.mla_grid,
        shwfs_optics.micro_lens_array.mla_index
    )

    camera = hcipy.NoiselessDetector()
    camera.integrate(shwfs_optics.forward(wf_turbulent), 1)
    raw_image = camera.read_out()

    # Convert to uint8 (same format as BMP)
    spot = raw_image.shaped.astype(np.float32)
    smin, smax = spot.min(), spot.max()
    if smax > smin:
        spot_norm = ((spot - smin) / (smax - smin) * 255).astype(np.uint8)
    else:
        spot_norm = np.zeros_like(spot, dtype=np.uint8)

    if add_noise:
        noise = np.random.normal(0, noise_std * 255, spot_norm.shape)
        spot_norm = np.clip(spot_norm.astype(float) + noise, 0, 255).astype(np.uint8)

    slopes = np.array(estimator.estimate([raw_image])).flatten().astype(np.float32)
    camera.clear()

    return spot_norm, slopes


def generate_training_pair(r0, seed=None, **kwargs):
    """
    Generate one complete training pair for the CNN.
    Called 50,000 times in Day 4 dataset generation.

    Args:
        r0   : Fried parameter in metres
        seed : Random seed

    Returns:
        spot_image : (H, W) uint8  — CNN primary input
        slopes     : (2*N_sub,) float32 — CNN secondary input
        phase_nm   : (H, W) float32 — CNN ground truth label
        r0         : float — turbulence level tag
    """
    phase_nm, wf_turb, pupil_grid, aperture = simulate_wavefront_phase(
        r0=r0, seed=seed, **kwargs
    )
    spot_image, slopes = simulate_shwfs_frame(
        wf_turb, pupil_grid, aperture
    )
    return spot_image, slopes, phase_nm, float(r0)


