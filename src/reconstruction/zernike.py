# 4 Task are excuted in this section 
# get_zernike_basis() — 36 modes on NxN grid using aotools
# decompose_into_zernike() — phase map → 36 coefficients (CNN training labels)
# reconstruct_from_coefficients() — 36 coefficients → phase map (CNN output decoder)
# build_interaction_matrix() — builds D matrix and D_pinv for Day 3 classical reconstruction
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import N_ZERNIKE_MODES, GRID_SIZE, N_LENSLETS, D


def get_zernike_basis(n_modes=N_ZERNIKE_MODES, grid_size=GRID_SIZE):
    Z = get_zernike_basis(n_modes, grid_size)
    Z_flat    = Z.reshape(n_modes, -1)
    phase_flat= phase_map.flatten()

    # Mask NaN (outside aperture)
    valid     = ~np.isnan(phase_flat)
    coeffs, _, _, _ = np.linalg.lstsq(
        Z_flat[:, valid].T, phase_flat[valid], rcond=None)

    return coeffs.astype(np.float32)



